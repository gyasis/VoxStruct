"""Pluggable ASR backends for VoxStruct.

A backend turns an audio file into a `TranscriptResult`. The default (`builtin`)
keeps VoxStruct's in-process Whisper/Vosk path; `remote` offloads transcription
to an HTTP ASR service (e.g. the Atelier MLX-whisper sidecar) so this machine
stays low-compute. The running endpoint is supplied via environment config and
is never committed.
"""

import os

from .base import ASRBackend, TranscriptResult
from .remote_http import RemoteHTTPBackend


def get_asr_backend():
    """Return an ASRBackend per env config, or None to use the builtin path.

    Env:
      VOXSTRUCT_ASR_BACKEND   builtin (default) | remote
      VOXSTRUCT_ASR_URL       base URL of the remote sidecar (remote only)
      VOXSTRUCT_ASR_TOKEN     optional bearer token
      VOXSTRUCT_ASR_MODEL     optional model alias / HF repo (e.g. turbo)
    """
    backend = os.getenv("VOXSTRUCT_ASR_BACKEND", "builtin").strip().lower()
    if backend in ("", "builtin", "local-builtin"):
        return None
    if backend == "remote":
        url = os.getenv("VOXSTRUCT_ASR_URL")
        if not url:
            raise ValueError(
                "VOXSTRUCT_ASR_BACKEND=remote requires VOXSTRUCT_ASR_URL "
                "(e.g. http://<host>:8766)"
            )
        return RemoteHTTPBackend(
            base_url=url,
            token=os.getenv("VOXSTRUCT_ASR_TOKEN"),
            model=os.getenv("VOXSTRUCT_ASR_MODEL"),
        )
    raise ValueError(f"unknown VOXSTRUCT_ASR_BACKEND: {backend!r}")


__all__ = ["ASRBackend", "TranscriptResult", "RemoteHTTPBackend", "get_asr_backend"]
