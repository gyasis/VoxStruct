"""
Detect natural pauses in audio for better transcript structuring.
"""
from pydub import AudioSegment
from pydub.silence import detect_silence
from typing import List
from .config import SILENCE_THRESHOLD, MIN_SILENCE_LENGTH

class PauseDetector:
    def __init__(self, silence_thresh=SILENCE_THRESHOLD, min_silence_len=MIN_SILENCE_LENGTH):
        """
        Initialize the pause detector.
        
        Args:
            silence_thresh: Silence threshold in dB (default: -40)
            min_silence_len: Minimum silence length in ms (default: 500)
        """
        self.silence_thresh = silence_thresh
        self.min_silence_len = min_silence_len
    
    def detect_pauses(self, audio_segment: AudioSegment) -> List[float]:
        """
        Detect natural pauses in the audio.
        
        Args:
            audio_segment: The audio to analyze
            
        Returns:
            list: List of timestamps (in seconds) where pauses occur
        """
        try:
            # Detect silence ranges
            silence_ranges = detect_silence(
                audio_segment,
                min_silence_len=self.min_silence_len,
                silence_thresh=self.silence_thresh
            )
            
            # Convert to seconds and get midpoints of pauses
            pause_points = []
            for start, end in silence_ranges:
                # Convert from milliseconds to seconds
                pause_point = (start + end) / 2000  # Midpoint in seconds
                pause_points.append(pause_point)
            
            return pause_points
            
        except Exception as e:
            print(f"Error detecting pauses: {e}")
            return []
    
    def get_pause_ranges(self, audio_segment: AudioSegment) -> List[tuple]:
        """
        Get the full ranges of pauses.
        
        Args:
            audio_segment: The audio to analyze
            
        Returns:
            list: List of (start, end) tuples in seconds
        """
        try:
            # Detect silence ranges
            silence_ranges = detect_silence(
                audio_segment,
                min_silence_len=self.min_silence_len,
                silence_thresh=self.silence_thresh
            )
            
            # Convert to seconds
            return [(start/1000, end/1000) for start, end in silence_ranges]
            
        except Exception as e:
            print(f"Error getting pause ranges: {e}")
            return []
    
    def get_speech_segments(self, audio_segment: AudioSegment) -> List[tuple]:
        """
        Get the ranges of speech (non-silence) segments.
        
        Args:
            audio_segment: The audio to analyze
            
        Returns:
            list: List of (start, end) tuples in seconds for speech segments
        """
        try:
            # Get pause ranges
            pause_ranges = self.get_pause_ranges(audio_segment)
            
            # Convert to speech segments
            speech_segments = []
            current_pos = 0
            
            for pause_start, pause_end in pause_ranges:
                if current_pos < pause_start:
                    speech_segments.append((current_pos, pause_start))
                current_pos = pause_end
            
            # Add final segment if needed
            duration = len(audio_segment) / 1000  # Convert to seconds
            if current_pos < duration:
                speech_segments.append((current_pos, duration))
            
            return speech_segments
            
        except Exception as e:
            print(f"Error getting speech segments: {e}")
            return []
