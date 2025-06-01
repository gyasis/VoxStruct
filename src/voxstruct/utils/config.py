"""
Audio processing configuration parameters.
These values can be adjusted to fine-tune the audio processing behavior.
"""

# Audio chunk processing
CHUNK_SIZE = 30000  # in milliseconds

# Silence detection parameters
SILENCE_THRESHOLD = -40  # in dB
MIN_SILENCE_LENGTH = 500  # in milliseconds

# Optional processing parameters
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH = 2 