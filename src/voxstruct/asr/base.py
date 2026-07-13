"""ASR backend interface + the common result shape every backend emits."""

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, runtime_checkable


@dataclass
class TranscriptResult:
    """Engine-agnostic transcription result.

    Attributes:
        text: full transcript text.
        segments: list of {"start": float_s, "end": float_s, "text": str} (may be empty).
        language: detected/declared language code, if known.
        words: optional word-level list when word_timestamps was requested.
        raw: the backend's raw response, for debugging / extra fields.
    """

    text: str
    segments: List[dict] = field(default_factory=list)
    language: Optional[str] = None
    words: Optional[List[dict]] = None
    raw: Optional[dict] = None


@runtime_checkable
class ASRBackend(Protocol):
    """A source VoxStruct can hand an audio file to and get text back."""

    def transcribe(
        self,
        audio_path: str,
        *,
        language: Optional[str] = None,
        word_timestamps: bool = False,
    ) -> TranscriptResult:
        ...
