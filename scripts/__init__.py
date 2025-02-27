"""
VoxStruct scripts package.
Contains audio processing and transcription utilities.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from .audio_converter import AudioConverter
from .whisper_transcriber import transcribe_audio as whisper_transcribe
from .vosk_transcriber import transcribe_audio as vosk_transcribe

# Try to import Coqui STT, but don't fail if it's not available
try:
    from .coqui_transcriber import transcribe_audio as coqui_transcribe
    COQUI_AVAILABLE = True
except ImportError:
    coqui_transcribe = None
    COQUI_AVAILABLE = False
    print("Note: Coqui STT not available. Skipping Coqui STT support.")

__all__ = [
    'AudioConverter',
    'whisper_transcribe',
    'vosk_transcribe',
    'COQUI_AVAILABLE'
]

if COQUI_AVAILABLE:
    __all__.append('coqui_transcribe') 