# Remote / offline ASR backend

VoxStruct can offload transcription (and LLM structuring) to another host over
HTTP, so the machine running VoxStruct stays low-compute. The capability ships in
the repo; the actual endpoint is supplied via environment config and is never
committed.

## Backends

| `VOXSTRUCT_ASR_BACKEND` | Behavior | Needs |
|---|---|---|
| `builtin` (default) | In-process Whisper/Vosk — unchanged legacy path | local torch + whisper |
| `remote` | Uploads the audio to an HTTP ASR sidecar and uses its transcript | a reachable sidecar host |
| `modal` | Runs whisper large-v3 (+ optional diarization) on a rented GPU | a Modal account |

Pick `remote` when your sidecar is up — it is warm, free, and already holds the
weights. Pick `modal` when it is **not**: travelling, the sidecar box powered
down, or a machine with no GPU and no local weights. Both emit the same
`TranscriptResult`, so nothing downstream changes.

## Configuration

Copy `.env.example` to `.env` (gitignored) and set:

```env
VOXSTRUCT_ASR_BACKEND=remote
VOXSTRUCT_ASR_URL=http://<host>:8766     # your ASR sidecar — PRIVATE, your box only
VOXSTRUCT_ASR_TOKEN=                     # optional bearer token (if the sidecar sets one)
VOXSTRUCT_ASR_MODEL=turbo                # optional model alias / HF repo

# Run structuring off-box too (optional): point litellm at a LAN LLM
VOXSTRUCT_LLM_API_BASE=http://<host>:11434   # e.g. an Ollama host
```

Then:

```bash
# Remote ASR, no LLM:
voxstruct your_audio.mp3 --no-llm

# Remote ASR + structuring on a LAN Ollama:
voxstruct your_audio.mp3 --llm-model ollama/llama3.2:1b
```

## Sidecar contract (what `remote` expects)

The backend POSTs `multipart/form-data` to `<VOXSTRUCT_ASR_URL>/transcribe`:

- `file` — the audio upload (we upload because `path`/`url` resolve on the
  sidecar's host, not ours)
- `model`, `language`, `word_timestamps`, `response_format=json`
- optional `Authorization: Bearer <token>`

Expected JSON response (lean `json` shape):

```json
{ "text": "…", "language": "en", "segments": [ {"start": 0.0, "end": 2.1, "text": "…"} ] }
```

`GET /healthz` should return 2xx when the sidecar is up. This matches the
Atelier MLX-whisper sidecar (`/transcribe`, `/healthz`, `/readyz`, plus
`/transcribe/batch` + `/jobs/{id}/stream` for long audio — a future addition).

## Code

- `voxstruct/asr/base.py` — `ASRBackend` protocol + `TranscriptResult`
- `voxstruct/asr/remote_http.py` — `RemoteHTTPBackend`
- `voxstruct/asr/__init__.py` — `get_asr_backend()` factory (reads env)


## `modal` backend

Transcription on a Modal GPU. Nothing is installed locally and no host on your
network has to be awake — the tradeoff is that audio leaves the machine and the
GPU is billed per second (an L4 scales to zero ~2 min after the call).

### Deploy once

```bash
pip install modal && modal setup
# Diarization only: pyannote 3.1 is a gated repo, so it needs an HF token.
# Accept the terms at huggingface.co/pyannote/speaker-diarization-3.1 first.
modal secret create huggingface HF_TOKEN=hf_...
modal deploy modal/voxstruct_asr.py
```

### Configure

```env
VOXSTRUCT_ASR_BACKEND=modal
VOXSTRUCT_ASR_MODEL=large-v3     # optional; default large-v3
VOXSTRUCT_MODAL_APP=voxstruct-asr        # optional, matches the deployed app
VOXSTRUCT_MODAL_FUNCTION=transcribe      # optional
VOXSTRUCT_DIARIZE=true           # optional — adds `speaker` to each segment
VOXSTRUCT_NUM_SPEAKERS=2         # exact count; see the warning below
```

```bash
voxstruct meeting.m4a --language en
modal run modal/voxstruct_asr.py --path meeting.m4a --diarize --num-speakers 2  # direct smoke test
```

### Diarization

With `VOXSTRUCT_DIARIZE=true` every segment carries a `speaker` key and
`raw["meta"]["speaker_time_s"]` reports talk time per speaker.

**`VOXSTRUCT_NUM_SPEAKERS` is an exact count, not a hint.** Give the real number
when you know it — letting the clusterer guess is how one person gets split
across two or three phantom speakers. Leave it unset only when you genuinely
don't know.

### VAD is OFF by default, deliberately

`faster-whisper`'s `vad_filter` looks like a free win and is a trap on quiet or
far-field audio. Measured on a 10-minute slice of a real meeting recorded at
-22 to -16 dB mean, against a known-good transcript of the same audio (1,557 words):

| variant | words | vs truth |
|---|---|---|
| `vad_filter=True` | 778 | 50% |
| `vad_filter=False` | 1,617 | ok |
| `vad_filter=False` + `speechnorm` | 1,609 | ok |
| `vad_filter=True` + `speechnorm` | 610 | 39% |

Silero VAD scores quiet speech below its 0.5 threshold and drops it, and the loss
is *uniform across the file* -- so it reads as a plausible transcript rather than
an error. The first full run of this backend lost 79% of a 2h meeting that way.
Normalization does not rescue it (and made VAD worse), which is why there is no
normalization step here. Pass `vad_filter=True` only for audio you know is clean
and close-mic'd.

### Why there is no chunk loop here

`remote` slices long audio because the HTTP sidecar's `/transcribe` is documented
for short clips. Modal has no such limit, and a single pass is what keeps speaker
assignment coherent across a whole recording, so `modal` sends the file whole.

### Why one pass instead of per-turn decoding

Decoding each diarized turn separately is only necessary for **bilingual** audio,
where Whisper's single up-front language detection mangles the losing language.
For monolingual audio it costs hundreds of extra decodes for no gain, so this
backend transcribes once with word timestamps and assigns speakers by maximum
temporal overlap. If you need the bilingual behaviour, pass `--language` per run
or transcribe the languages separately.
