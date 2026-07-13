# Remote / offline ASR backend

VoxStruct can offload transcription (and LLM structuring) to another host over
HTTP, so the machine running VoxStruct stays low-compute. The capability ships in
the repo; the actual endpoint is supplied via environment config and is never
committed.

## Backends

| `VOXSTRUCT_ASR_BACKEND` | Behavior |
|---|---|
| `builtin` (default) | In-process Whisper/Vosk — unchanged legacy path |
| `remote` | Uploads the audio to an HTTP ASR sidecar and uses its transcript |

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
