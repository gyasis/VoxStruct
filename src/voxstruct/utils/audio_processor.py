# util/audio_processor.py
from pydub import AudioSegment, silence
from typing import List, Optional
from .config import CHUNK_SIZE, DEFAULT_SAMPLE_RATE

class AudioProcessor:
    def __init__(self, file_path, sample_rate=DEFAULT_SAMPLE_RATE, chunk_size=CHUNK_SIZE):
        """
        Initialize the audio processor.
        
        Args:
            file_path: Path to the audio file
            sample_rate: Sample rate in Hz (default from config)
            chunk_size: Size of each chunk in milliseconds (default from config)
        """
        self.file_path = file_path
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size  # in milliseconds
        self.audio = None

    def load_audio(self):
        """Loads the audio file into an AudioSegment."""
        try:
            self.audio = AudioSegment.from_file(self.file_path)
            print(f"Loaded audio file: {self.file_path}")
        except Exception as e:
            print("Error loading audio file:", e)
        return self.audio

    def get_chunks(self):
        """Splits the audio into fixed-size chunks."""
        if not self.audio:
            raise ValueError("Audio not loaded. Call load_audio() first.")
        chunks = []
        duration_ms = len(self.audio)
        for i in range(0, duration_ms, self.chunk_size):
            chunk = self.audio[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks
