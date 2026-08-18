"""Modal GPU ASR backend — transcription on a rented GPU, no local weights.

`remote` needs an ASR sidecar reachable on the network; `builtin` needs torch +
whisper installed here. This backend needs neither: it calls a deployed Modal
function (see `modal/voxstruct_asr.py`) which holds whisper large-v3 and optional
pyannote diarization on an L4 that scales to zero between calls.

The whole file is sent in one call -- Modal has no "short audio only" limit the way
the HTTP sidecar's /transcribe does, and a single pass is what makes speaker
assignment coherent across the recording -- so there is deliberately no chunk loop
here (contrast RemoteHTTPBackend, which must slice).

Deploy once, then set VOXSTRUCT_ASR_BACKEND=modal. See docs/REMOTE_ASR.md.
"""

import os
from typing import Optional

from .base import TranscriptResult

DEFAULT_APP = "voxstruct-asr"
DEFAULT_FUNCTION = "transcribe"


class ModalBackend:
    def __init__(
        self,
        app_name: str = DEFAULT_APP,
        function_name: str = DEFAULT_FUNCTION,
        model: Optional[str] = None,
        diarize: bool = False,
        num_speakers: Optional[int] = None,
    ):
        self.app_name = app_name
        self.function_name = function_name
        self.model = model
        self.diarize = diarize
        self.num_speakers = num_speakers
        self._fn = None

    def _function(self):
        """Resolve the deployed Modal function (lazily — importing modal is slow)."""
        if self._fn is not None:
            return self._fn
        try:
            import modal
        except ImportError as e:
            raise ImportError(
                "VOXSTRUCT_ASR_BACKEND=modal requires the modal client: pip install modal"
            ) from e
        try:
            self._fn = modal.Function.from_name(self.app_name, self.function_name)
        except Exception as e:
            raise RuntimeError(
                f"could not resolve Modal function {self.app_name}/{self.function_name}. "
                f"Deploy it first:  modal deploy modal/voxstruct_asr.py   ({e})"
            ) from e
        return self._fn

    def health(self) -> bool:
        """True if the deployed function can be resolved (does not spin up a GPU)."""
        try:
            self._function()
            return True
        except Exception:
            return False

    def transcribe(
        self,
        audio_path: str,
        *,
        language: Optional[str] = None,
        word_timestamps: bool = False,
    ) -> TranscriptResult:
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"audio file not found: {audio_path}")

        with open(audio_path, "rb") as fh:
            audio = fh.read()

        kwargs = {
            "ext": os.path.splitext(audio_path)[1].lstrip(".").lower() or "wav",
            "language": language,
            "word_timestamps": word_timestamps,
            "diarize": self.diarize,
            "num_speakers": self.num_speakers,
        }
        if self.model:
            kwargs["model"] = self.model

        body = self._function().remote(audio, **kwargs)

        return TranscriptResult(
            text=(body.get("text") or "").strip(),
            segments=body.get("segments") or [],
            language=body.get("language") or language,
            words=body.get("words"),
            raw=body,
        )
