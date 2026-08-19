# VoxStruct — guide for coding agents

Audio → **structured** transcript. ASR produces a lowercase, unpunctuated wall of text;
the LLM pass turns it into a document (headers, paragraphs, lists, punctuation, casing).
The structuring is the point of the tool.

**Read `docs/REMOTE_ASR.md` before choosing a backend.** A fuller operator runbook lives
outside this repo at `~/Documents/code/codebook/voxstruct.md`.

## Pick an ASR backend — `VOXSTRUCT_ASR_BACKEND`

| backend | needs | when |
|---|---|---|
| `builtin` (default) | local `torch` + `whisper` | rarely — **not installed in this repo's venv** |
| `remote` | an ASR sidecar reachable on the network | **preferred**: warm, free, weights resident |
| `modal` | a Modal account | the sidecar host is down / travelling / no local GPU |

```bash
VOXSTRUCT_ASR_BACKEND=remote VOXSTRUCT_ASR_URL=http://<host>:8766 voxstruct audio.m4a
VOXSTRUCT_ASR_BACKEND=modal VOXSTRUCT_DIARIZE=true VOXSTRUCT_NUM_SPEAKERS=2 voxstruct audio.m4a
```

All three return the same `TranscriptResult`, so downstream code never branches on backend.

```python
from voxstruct.asr import get_asr_backend
b = get_asr_backend()          # None => use the builtin in-process path
b.health()                     # cheap reachability probe, never raises
r = b.transcribe("a.mp3", language="en", word_timestamps=True)
```

## Three ways this bites you — two fail SILENTLY

**1. `vad_filter` must stay off for quiet or far-field audio.** Silero VAD scores quiet
speech below its 0.5 threshold and discards it *uniformly across the file*, so the output
reads as a plausible transcript rather than an error. It cost **79% of the words** of a real
2h meeting. Measured against a known-good reference (1,557 words): VAD on → 778, VAD off →
1,617. Normalization does not rescue it. **Sanity-check word count: ~110–150 words per
minute of speech.**

**2. `validate_and_improve_transcript` returns its INPUT UNCHANGED on any exception.** A
dead key, an exhausted quota or a bad model name is indistinguishable from success — only a
line on stdout marks it. Twelve failed calls once produced a file labelled "structured"
containing no structure. Guard every call:

```python
if structured.strip() == raw.strip():
    raise RuntimeError("LLM call failed silently — refusing to write")
```

**3. The API key must match the model's provider.** `LLMSupervisor` assigns whatever key it
receives to the provider named in the model string, so a Gemini key with `openai/...` gives
`Incorrect API key provided: AIzaSy...`. Select the key from the model string.

## Other things worth knowing

- **`--focus` DROPS off-topic content.** Great for a targeted extract, wrong when the
  transcript is the record of a meeting. Omit it to structure everything.
- **Structuring must be chunked** — `max_tokens=16000` per call; ~10-minute blocks work.
- **`VOXSTRUCT_NUM_SPEAKERS` is exact, not a hint.** Guessing splits one person into
  phantom speakers. ASR does not diarize by itself; pyannote is a separate step.
- **Pass `word_timestamps=True`** if anything downstream must locate a word in the audio.
  Diarization computes them anyway; not saving them means re-running the whole job.
- Raw ASR is lowercase/unpunctuated because `condition_on_previous_text` is off (the guard
  against repetition-loop hallucinations on long audio). The LLM pass fixes it — don't flip
  the flag.

## Tests

```bash
python3 -m pytest tests/test_asr_backends.py -q     # backend selection, no network
```
