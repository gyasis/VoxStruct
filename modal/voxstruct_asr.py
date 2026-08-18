#!/usr/bin/env python3
"""VoxStruct ASR on a Modal GPU — the lane that works when no sidecar is reachable.

    modal deploy modal/voxstruct_asr.py

VoxStruct's `remote` backend targets an ASR sidecar on the LAN (the Atelier whisper
box). When that host is off the network there is no ASR at all, and the `builtin`
path needs a local torch + whisper install that a low-compute client deliberately
does not carry. This puts whisper large-v3 -- and optional pyannote diarization --
on a rented L4 that spins up per call and scales to zero after, so a recording can
be transcribed with nothing but this machine and internet.

Design notes, both learned the hard way:

  1. TRANSCRIBE ONCE, ASSIGN SPEAKERS BY OVERLAP. The obvious implementation
     decodes each diarized turn separately. That is necessary only when the audio
     is bilingual and Whisper's single up-front language detection would mangle the
     losing language (see chiron's il_centro_asr.py, which must do exactly that).
     For monolingual audio it is pure cost: hundreds of separate decodes across a
     long recording. One pass with word timestamps, then max-overlap assignment,
     is far faster and no less accurate.

  2. WEIGHTS LIVE IN A VOLUME. whisper large-v3 plus pyannote 3.1 is ~4 GB; a cold
     pull on every invocation would dominate runtime.
"""
import modal

app = modal.App("voxstruct-asr")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "faster-whisper==1.1.0",
        "pyannote.audio==3.3.2",
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "numpy<2",
        # huggingface_hub 1.0 REMOVED the `use_auth_token` kwarg that pyannote 3.3.2
        # still passes into hf_hub_download; unpinned resolves to 1.27 and dies at
        # from_pretrained -- after the image builds and the GPU is already warm.
        "huggingface_hub<1.0",
    )
)

cache = modal.Volume.from_name("voxstruct-asr-cache", create_if_missing=True)
CACHE = "/cache"
DEFAULT_MODEL = "large-v3"
DIARIZE = "pyannote/speaker-diarization-3.1"

# Diarization is optional so the function still runs without an HF token; pyannote
# 3.1 is a gated repo and only that path needs the secret.
_secrets = [modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])]


@app.function(
    image=image, gpu="L4", volumes={CACHE: cache}, secrets=_secrets,
    timeout=60 * 120, scaledown_window=120,
)
def transcribe(
    audio: bytes,
    ext: str = "wav",
    language: str | None = None,
    word_timestamps: bool = False,
    model: str = DEFAULT_MODEL,
    diarize: bool = False,
    num_speakers: int | None = None,
    vad_filter: bool = False,
) -> dict:
    """Return the VoxStruct sidecar JSON shape.

    {"text", "language", "segments":[{start,end,text,speaker?}], "words":[...]?,
     "meta": {...}}
    """
    import os, subprocess, tempfile, time
    os.environ.setdefault("HF_HOME", CACHE)
    t0 = time.perf_counter()

    d = tempfile.mkdtemp()
    src, wav = f"{d}/in.{ext}", f"{d}/in.wav"
    with open(src, "wb") as fh:
        fh.write(audio)
    # 16k mono once here so neither model has to resample.
    subprocess.run(["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", wav],
                   check=True, capture_output=True)

    turns = []
    t_diar = 0.0
    if diarize:
        import torch
        from pyannote.audio import Pipeline
        pipe = Pipeline.from_pretrained(DIARIZE, use_auth_token=os.environ["HF_TOKEN"])
        pipe.to(torch.device("cuda"))
        # num_speakers is EXACT when given. Passing None lets pyannote cluster, which
        # can split one voice into phantom speakers -- pass the real count if known.
        diar = pipe(wav, num_speakers=num_speakers)
        turns = [{"start": float(t.start), "end": float(t.end), "speaker": str(spk)}
                 for t, _, spk in diar.itertracks(yield_label=True)]
        t_diar = time.perf_counter() - t0

    from faster_whisper import WhisperModel
    asr = WhisperModel(model, device="cuda", compute_type="float16", download_root=CACHE)
    need_words = word_timestamps or diarize        # diarization splits on word bounds
    segs, info = asr.transcribe(
        wav, language=language, beam_size=5,
        word_timestamps=need_words, vad_filter=vad_filter,
        condition_on_previous_text=False,
    )
    segs = list(segs)

    def overlap_tally(a, b):
        """Speaker -> seconds of diarized overlap across [a, b]."""
        tally = {}
        for t in turns:
            ov = min(b, t["end"]) - max(a, t["start"])
            if ov > 0:
                tally[t["speaker"]] = tally.get(t["speaker"], 0.0) + ov
        return tally

    def nearest_speaker(a, b):
        """Closest turn by time. Used instead of an UNKNOWN bucket: a word landing in
        a diarization gap (breath, crosstalk, timestamp drift) still belongs to
        whoever was speaking around it."""
        best, dist = None, float("inf")
        for t in turns:
            d = 0.0 if (b > t["start"] and a < t["end"]) else min(abs(a - t["end"]),
                                                                  abs(t["start"] - b))
            if d < dist:
                dist, best = d, t["speaker"]
        return best or "SPEAKER_00"

    def speaker_of(a, b):
        tally = overlap_tally(a, b)
        return max(tally, key=tally.get) if tally else nearest_speaker(a, b)

    words = []
    for s in segs:
        for w in (s.words or []):
            item = {"start": float(w.start), "end": float(w.end),
                    "word": w.word, "probability": float(w.probability)}
            if turns:
                item["speaker"] = speaker_of(float(w.start), float(w.end))
            words.append(item)

    if turns:
        # Assign at SEGMENT level, not word level. Whisper's word timestamps drift by
        # a few hundred ms, so per-word assignment flips speaker mid-sentence and
        # shreds one person's turn into alternating fragments. A whisper segment is
        # phrase-length, which matches diarization granularity.
        MIXED_SHARE = 0.7      # below this the segment really does span a handover
        MIN_RUN_S = 0.6        # runs shorter than this are timestamp noise, not a turn
        MERGE_GAP_S = 2.0      # never glue across a real pause
        MAX_SEG_S = 30.0       # keep segments navigable -- see note below
        pieces = []
        for s in segs:
            a, b = float(s.start), float(s.end)
            tally = overlap_tally(a, b)
            total = sum(tally.values())
            dominant = max(tally, key=tally.get) if tally else nearest_speaker(a, b)
            sw = list(s.words or [])
            if not sw or not total or tally.get(dominant, 0.0) / total >= MIXED_SHARE:
                pieces.append({"start": a, "end": b, "speaker": dominant,
                               "text": s.text.strip()})
                continue
            # genuine handover inside this segment -> split on word boundaries
            run = None
            for w in sw:
                spk = speaker_of(float(w.start), float(w.end))
                if run and run["speaker"] == spk:
                    run["text"] += w.word
                    run["end"] = float(w.end)
                else:
                    if run:
                        pieces.append(run)
                    run = {"start": float(w.start), "end": float(w.end),
                           "speaker": spk, "text": w.word}
            if run:
                pieces.append(run)

        # absorb sub-threshold runs into the neighbour, then merge adjacent same-speaker
        cleaned = []
        for p in pieces:
            if (cleaned and p["end"] - p["start"] < MIN_RUN_S
                    and p["speaker"] != cleaned[-1]["speaker"]):
                p = {**p, "speaker": cleaned[-1]["speaker"]}
            # Merging adjacent same-speaker pieces is required (diarization splits
            # mid-sentence), but merging them UNCONDITIONALLY collapses a monologue
            # into one untimestamped block -- a 2h meeting came back as 70 segments,
            # one of them ten minutes long, which is unusable for navigation. Cap on
            # both pause length and total duration.
            if (cleaned and cleaned[-1]["speaker"] == p["speaker"]
                    and p["start"] - cleaned[-1]["end"] <= MERGE_GAP_S
                    and p["end"] - cleaned[-1]["start"] <= MAX_SEG_S):
                cleaned[-1]["text"] = (cleaned[-1]["text"] + " " + p["text"].strip()).strip()
                cleaned[-1]["end"] = p["end"]
            else:
                cleaned.append({**p, "text": p["text"].strip()})
        segments = [c for c in cleaned if c["text"]]
    else:
        segments = [{"start": float(s.start), "end": float(s.end),
                     "text": s.text.strip()} for s in segs]

    text = " ".join(s["text"] for s in segments).strip()
    spk_time: dict = {}
    for s in segments:
        if "speaker" in s:
            spk_time[s["speaker"]] = round(
                spk_time.get(s["speaker"], 0.0) + s["end"] - s["start"], 1)

    return {
        "text": text,
        "language": info.language,
        "segments": segments,
        "words": words or None,
        "meta": {"duration_s": round(float(info.duration), 1),
                 "n_segments": len(segments), "n_words": len(words),
                 "model": model, "diarized": bool(turns), "vad_filter": vad_filter,
                 "diarizer": DIARIZE if turns else None,
                 "speaker_time_s": spk_time or None,
                 "diarize_s": round(t_diar, 1),
                 "total_s": round(time.perf_counter() - t0, 1)},
    }


@app.local_entrypoint()
def smoke(path: str, diarize: bool = False, num_speakers: int = 0, language: str = ""):
    r = transcribe.remote(
        open(path, "rb").read(), path.rsplit(".", 1)[-1],
        language or None, False, DEFAULT_MODEL, diarize, num_speakers or None,
    )
    m = r["meta"]
    print(f"{m['n_segments']} segments · {m['duration_s']}s audio · {m['total_s']}s compute")
    if m["speaker_time_s"]:
        print(f"speakers: {m['speaker_time_s']}")
    for s in r["segments"][:6]:
        who = f"{s.get('speaker','')} " if s.get("speaker") else ""
        print(f"  [{s['start']:7.1f}] {who}{s['text'][:70]}")
