"""
VoxStruct - A tool for structured audio transcription with LLM supervision
"""

from .utils.audio_processor import AudioProcessor
from .utils.audio_library import AudioLibrary
from .utils.speech_recognizer import SpeechRecognizer
from .utils.pause_detector import PauseDetector
from .utils.transcript_builder import TranscriptBuilder
from .utils.llm_supervisory import LLMSupervisor

__all__ = [
    'AudioProcessor',
    'AudioLibrary',
    'SpeechRecognizer',
    'PauseDetector',
    'TranscriptBuilder',
    'LLMSupervisor',
]

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
