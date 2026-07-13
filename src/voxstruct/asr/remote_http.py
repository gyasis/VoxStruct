"""Remote HTTP ASR backend — offload transcription to an external sidecar.

Targets an Atelier-style whisper sidecar (`POST /transcribe`). The audio file is
uploaded as multipart (`file=`), because the sidecar's `path=`/`url=` fields
resolve on the *sidecar's* host, not ours. The endpoint URL/token live in env
config (never committed), so the capability ships in the repo but only the owner
can point it at their box.

Sidecar contract (default response_format=json):
    {"text": str, "language": str|None, "segments": [{"start","end","text"}], "quality": {...}}
"""

import os
from typing import Optional

import requests

from .base import TranscriptResult

# Whole-file sync transcription can take a while on long audio; the sidecar warns
# /transcribe is for "under a few minutes". A long, generous timeout keeps short
# clips simple. (Batch + SSE for long audio is a future addition — see issue #21.)
DEFAULT_TIMEOUT_S = 1800


class RemoteHTTPBackend:
    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.model = model
        self.timeout = timeout

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def health(self) -> bool:
        """True if the sidecar answers /healthz with 2xx."""
        try:
            r = requests.get(self.base_url + "/healthz", headers=self._headers(), timeout=10)
            return r.ok
        except requests.RequestException:
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

        # The sidecar's /transcribe is meant for short audio ("under a few
        # minutes"). For long files we slice into bounded segments locally and
        # stitch the results back with absolute time offsets — mirroring the
        # builtin chunk loop so remote mode is safe on long audio.
        try:
            chunk_s = float(os.getenv("VOXSTRUCT_REMOTE_CHUNK_S", "180") or 180)
        except ValueError:
            chunk_s = 180.0

        audio = None
        duration_s = 0.0
        if chunk_s > 0:
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(audio_path)
                duration_s = len(audio) / 1000.0
            except Exception:
                audio = None  # pydub/ffmpeg unavailable → single-shot below

        if audio is None or duration_s <= chunk_s:
            return self._transcribe_file(
                audio_path, language=language, word_timestamps=word_timestamps
            )

        import tempfile

        chunk_ms = int(chunk_s * 1000)
        n = (len(audio) + chunk_ms - 1) // chunk_ms
        texts: list = []
        segments: list = []
        words: list = []
        detected_lang = None
        with tempfile.TemporaryDirectory(prefix="voxstruct_remote_") as td:
            for idx, i in enumerate(range(0, len(audio), chunk_ms), start=1):
                offset_s = i / 1000.0
                seg_audio = audio[i:i + chunk_ms]
                seg_path = os.path.join(td, f"chunk_{idx:04d}.wav")
                seg_audio.export(seg_path, format="wav")
                print(
                    f"\r  remote ASR chunk {idx}/{n} "
                    f"({offset_s:.0f}s-{offset_s + len(seg_audio) / 1000.0:.0f}s)...",
                    end="", flush=True,
                )
                part = self._transcribe_file(
                    seg_path, language=language, word_timestamps=word_timestamps
                )
                if part.text:
                    texts.append(part.text)
                detected_lang = detected_lang or part.language
                for s in (part.segments or []):
                    s2 = dict(s)
                    if s2.get("start") is not None:
                        s2["start"] = float(s2["start"]) + offset_s
                    if s2.get("end") is not None:
                        s2["end"] = float(s2["end"]) + offset_s
                    segments.append(s2)
                for w in (part.words or []):
                    w2 = dict(w)
                    if w2.get("start") is not None:
                        w2["start"] = float(w2["start"]) + offset_s
                    if w2.get("end") is not None:
                        w2["end"] = float(w2["end"]) + offset_s
                    words.append(w2)
            print()

        return TranscriptResult(
            text=" ".join(texts).strip(),
            segments=segments,
            language=detected_lang or language,
            words=words or None,
            raw={"chunked": True, "chunk_seconds": chunk_s, "chunks": n},
        )

    def _transcribe_file(
        self,
        audio_path: str,
        *,
        language: Optional[str] = None,
        word_timestamps: bool = False,
    ) -> TranscriptResult:
        """POST a single (short) audio file to the sidecar's /transcribe."""
        data = {
            "response_format": "json",
            "word_timestamps": "true" if word_timestamps else "false",
        }
        if self.model:
            data["model"] = self.model
        if language:
            data["language"] = language

        url = self.base_url + "/transcribe"
        with open(audio_path, "rb") as fh:
            files = {"file": (os.path.basename(audio_path), fh)}
            resp = requests.post(
                url, data=data, files=files,
                headers=self._headers(), timeout=self.timeout,
            )

        if resp.status_code == 401:
            raise PermissionError(
                f"ASR sidecar rejected the token (401) at {url}. "
                "Set VOXSTRUCT_ASR_TOKEN to a valid bearer token."
            )
        resp.raise_for_status()
        body = resp.json()

        return TranscriptResult(
            text=(body.get("text") or "").strip(),
            segments=body.get("segments") or [],
            language=body.get("language"),
            words=body.get("words"),
            raw=body,
        )
