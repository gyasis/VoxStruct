"""Backend selection in voxstruct.asr.get_asr_backend.

These are pure config tests -- none of them reach the network or spin up a GPU.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from voxstruct.asr import ASRBackend, ModalBackend, RemoteHTTPBackend, get_asr_backend


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("VOXSTRUCT_ASR_BACKEND", "VOXSTRUCT_ASR_URL", "VOXSTRUCT_ASR_TOKEN",
              "VOXSTRUCT_ASR_MODEL", "VOXSTRUCT_MODAL_APP", "VOXSTRUCT_MODAL_FUNCTION",
              "VOXSTRUCT_DIARIZE", "VOXSTRUCT_NUM_SPEAKERS"):
        monkeypatch.delenv(k, raising=False)


def test_default_is_builtin():
    """Unset env must keep the legacy in-process path (None = builtin)."""
    assert get_asr_backend() is None


def test_remote_requires_url(monkeypatch):
    monkeypatch.setenv("VOXSTRUCT_ASR_BACKEND", "remote")
    with pytest.raises(ValueError, match="VOXSTRUCT_ASR_URL"):
        get_asr_backend()


def test_remote_selected(monkeypatch):
    monkeypatch.setenv("VOXSTRUCT_ASR_BACKEND", "remote")
    monkeypatch.setenv("VOXSTRUCT_ASR_URL", "http://host:8766")
    assert isinstance(get_asr_backend(), RemoteHTTPBackend)


def test_modal_selected_with_defaults(monkeypatch):
    monkeypatch.setenv("VOXSTRUCT_ASR_BACKEND", "modal")
    b = get_asr_backend()
    assert isinstance(b, ModalBackend)
    assert b.app_name == "voxstruct-asr"
    assert b.function_name == "transcribe"
    assert b.diarize is False
    assert b.num_speakers is None


def test_modal_diarization_config(monkeypatch):
    monkeypatch.setenv("VOXSTRUCT_ASR_BACKEND", "modal")
    monkeypatch.setenv("VOXSTRUCT_DIARIZE", "true")
    monkeypatch.setenv("VOXSTRUCT_NUM_SPEAKERS", "3")
    b = get_asr_backend()
    assert b.diarize is True
    assert b.num_speakers == 3


def test_modal_ignores_non_numeric_speaker_count(monkeypatch):
    """A bad count must fall back to 'let pyannote cluster', not crash."""
    monkeypatch.setenv("VOXSTRUCT_ASR_BACKEND", "modal")
    monkeypatch.setenv("VOXSTRUCT_NUM_SPEAKERS", "two")
    assert get_asr_backend().num_speakers is None


def test_backends_satisfy_protocol(monkeypatch):
    monkeypatch.setenv("VOXSTRUCT_ASR_BACKEND", "modal")
    assert isinstance(get_asr_backend(), ASRBackend)


def test_unknown_backend_raises(monkeypatch):
    monkeypatch.setenv("VOXSTRUCT_ASR_BACKEND", "bogus")
    with pytest.raises(ValueError, match="unknown VOXSTRUCT_ASR_BACKEND"):
        get_asr_backend()


def test_modal_missing_file(tmp_path):
    b = ModalBackend()
    with pytest.raises(FileNotFoundError):
        b.transcribe(str(tmp_path / "nope.wav"))
