"""
VoxStruct utilities package.
Contains helper classes and functions for audio processing and transcription.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from .audio_processor import AudioProcessor
from .audio_library import AudioLibrary
from .pause_detector import PauseDetector
from .transcript_builder import TranscriptBuilder
from .youtube_downloader import YoutubeDownloader

__all__ = [
    'AudioProcessor',
    'AudioLibrary',
    'PauseDetector',
    'TranscriptBuilder',
    'YoutubeDownloader'
]
