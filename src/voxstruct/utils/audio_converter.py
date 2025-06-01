"""
Utility script for converting audio files to formats compatible with all speech recognition engines.

Use Cases:
1. Command Line Usage:
   # Convert a single audio file to all supported formats
   python audio_converter.py input.mp3
   
   # Specify custom output directories
   python audio_converter.py input.mp3 --output-dir my_converted --segment-dir my_segments
   
   # Customize segmentation parameters
   python audio_converter.py input.mp3 --segment-length 45000 --min-silence 1500 --silence-thresh -45

2. As an Imported Module:
   # Basic usage
   from audio_converter import AudioConverter
   converter = AudioConverter()
   converted_files, segments = converter.process_file("input.mp3")
   
   # Custom configuration
   converter = AudioConverter(
       output_dir="custom_output",
       segment_dir="custom_segments",
       segment_length_ms=45000,
       min_silence_ms=1500,
       silence_thresh=-45
   )
   
   # Process multiple files with the same settings
   for audio_file in audio_files:
       converter.process_file(audio_file)
   
   # Convert without segmenting
   converter = AudioConverter()
   converted_files = converter.convert_to_wav("input.mp3")
   
   # Only segment an existing WAV file
   converter = AudioConverter()
   segments = converter.split_audio("input.wav")

Features:
- Converts audio files to multiple formats optimized for different speech recognition engines
- Automatically segments long audio files based on silence detection
- Maintains natural speech patterns by including silence at segment boundaries
- Supports various input formats: mp3, ogg, flac, m4a, wma, wav
- Creates standardized WAV files (16kHz, mono, 16-bit)
- Handles temporary files and cleanup automatically
"""
import os
import argparse
import tempfile
import shutil
from pathlib import Path
from pydub import AudioSegment
import soundfile as sf
import numpy as np
from pydub.silence import detect_silence

class AudioConverter:
    """
    A class to handle audio file conversion and segmentation for speech recognition.
    
    Attributes:
        output_dir (str): Directory for converted files
        segment_dir (str): Directory for segmented files
        temp_dir (str): Directory for temporary files
        segment_length_ms (int): Maximum length of segments in milliseconds
        min_silence_ms (int): Minimum length of silence to split on
        silence_thresh (int): Silence threshold in dB
        keep_temp (bool): Whether to keep temporary files
    """
    
    def __init__(self, 
                 output_dir="converted",
                 segment_dir="segments",
                 temp_dir=None,
                 segment_length_ms=30000,
                 min_silence_ms=1000,
                 silence_thresh=-40,
                 keep_temp=False):
        """Initialize the AudioConverter with the given parameters."""
        self.output_dir = output_dir
        self.segment_dir = segment_dir
        self.temp_dir = temp_dir or self._get_temp_dir()
        self.segment_length_ms = segment_length_ms
        self.min_silence_ms = min_silence_ms
        self.silence_thresh = silence_thresh
        self.keep_temp = keep_temp
        
        # Create necessary directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.segment_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def _get_temp_dir(self):
        """Create and return a temporary directory for processing files."""
        temp_dir = os.path.join(tempfile.gettempdir(), 'voxstruct_temp')
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir
    
    def cleanup_temp_files(self):
        """Clean up temporary files after processing."""
        try:
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary files in: {self.temp_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up temporary files: {e}")
    
    def _get_silence_portion(self, silence_start, silence_end, max_silence_ms=500):
        """Get a portion of silence to include in segments."""
        silence_length = silence_end - silence_start
        if silence_length <= max_silence_ms:
            return silence_start, silence_end
        else:
            # Take half from the beginning and half from the end of silence
            half_silence = max_silence_ms / 2
            return (silence_start, silence_start + half_silence), (silence_end - half_silence, silence_end)
    
    def convert_to_wav(self, input_path):
        """
        Convert any audio file to WAV format with proper settings for speech recognition.
        Supports: mp3, ogg, flac, m4a, wma, etc.
        
        Args:
            input_path (str): Path to input audio file
            
        Returns:
            dict: Dictionary of output files for each engine
        """
        # Get file extension
        _, ext = os.path.splitext(input_path)
        filename = os.path.basename(input_path).replace(ext, '')
        
        # Load audio file
        print(f"Converting: {input_path}")
        try:
            if ext.lower() in ['.wav']:
                audio = AudioSegment.from_wav(input_path)
            elif ext.lower() in ['.mp3']:
                audio = AudioSegment.from_mp3(input_path)
            elif ext.lower() in ['.ogg']:
                audio = AudioSegment.from_ogg(input_path)
            elif ext.lower() in ['.flac']:
                audio = AudioSegment.from_file(input_path, format="flac")
            elif ext.lower() in ['.m4a']:
                audio = AudioSegment.from_file(input_path, format="m4a")
            else:
                audio = AudioSegment.from_file(input_path)
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return None
        
        # Create different versions optimized for each engine
        conversions = {
            'standard': {
                'sample_rate': 16000,
                'channels': 1,
                'sample_width': 2,
            },
            'vosk': {
                'sample_rate': 16000,
                'channels': 1,
                'sample_width': 2,
            },
            'deepspeech': {
                'sample_rate': 16000,
                'channels': 1,
                'sample_width': 2,
            },
            'whisper': {
                'sample_rate': 16000,
                'channels': 1,
                'sample_width': 2,
            }
        }
        
        output_files = {}
        
        for engine, settings in conversions.items():
            try:
                # First save to temp directory
                temp_path = os.path.join(self.temp_dir, f"{filename}_{engine}_temp.wav")
                
                # Apply conversion settings
                converted = audio.set_frame_rate(settings['sample_rate'])
                converted = converted.set_channels(settings['channels'])
                converted = converted.set_sample_width(settings['sample_width'])
                
                # Export to temp first
                converted.export(temp_path, format="wav")
                
                # Move to final destination
                final_path = os.path.join(self.output_dir, f"{filename}_{engine}.wav")
                shutil.move(temp_path, final_path)
                
                output_files[engine] = final_path
                print(f"Created {engine} version: {final_path}")
            except Exception as e:
                print(f"Error converting {engine} version: {e}")
                continue
        
        return output_files
    
    def split_audio(self, input_path):
        """
        Split long audio file into smaller segments based on silence detection.
        Includes a portion of silence at segment boundaries for natural pauses.
        
        Args:
            input_path (str): Path to input audio file
            
        Returns:
            list: List of paths to created segments
        """
        try:
            # Load audio
            audio = AudioSegment.from_file(input_path)
            
            # Find silence points
            silence_points = detect_silence(
                audio, 
                min_silence_len=self.min_silence_ms,
                silence_thresh=self.silence_thresh
            )
            
            segments = []
            current_start = 0
            next_start = 0  # Track where the next segment should start
            
            # If no silence points found or only one segment needed
            if not silence_points or len(audio) <= self.segment_length_ms:
                # Save to temp first
                temp_path = os.path.join(self.temp_dir, f"segment_000_temp.wav")
                audio.export(temp_path, format="wav")
                
                # Move to final destination
                final_path = os.path.join(self.segment_dir, f"segment_000.wav")
                shutil.move(temp_path, final_path)
                segments.append(final_path)
                print(f"Created segment: {final_path} ({len(audio)/1000:.1f}s)")
                return segments
            
            # Process segments using silence points
            for i, (silence_start, silence_end) in enumerate(silence_points):
                # Skip if this silence is before our next starting point
                if silence_end <= next_start:
                    continue
                    
                # If current segment would be too long, split at max length
                if silence_start - current_start > self.segment_length_ms:
                    segment_end = current_start + self.segment_length_ms
                    # Find the next silence point after this forced split
                    next_silence = None
                    for s_start, s_end in silence_points:
                        if s_start > segment_end:
                            next_silence = (s_start, s_end)
                            break
                    # If we found a silence point after the split, include some of it
                    if next_silence and next_silence[0] - segment_end < 1000:  # If silence is within 1 second
                        # Get portion of silence to include
                        pre_silence, post_silence = self._get_silence_portion(next_silence[0], next_silence[1])
                        segment_end = pre_silence[1]  # Include first half of silence
                        next_start = post_silence[0]  # Start next segment at second half
                    else:
                        next_start = segment_end  # No silence found, start next segment at the split point
                else:
                    # Get portion of silence to include
                    pre_silence, post_silence = self._get_silence_portion(silence_start, silence_end)
                    segment_end = pre_silence[1]  # Include first half of silence
                    next_start = post_silence[0]  # Start next segment at second half
                
                # Only create segment if it has content and reasonable length
                if segment_end > current_start and segment_end - current_start >= self.min_silence_ms:
                    # Extract segment (including silence portion)
                    segment = audio[current_start:segment_end]
                    
                    # Save to temp first
                    temp_path = os.path.join(self.temp_dir, f"segment_{len(segments):03d}_temp.wav")
                    segment.export(temp_path, format="wav")
                    
                    # Move to final destination
                    final_path = os.path.join(self.segment_dir, f"segment_{len(segments):03d}.wav")
                    shutil.move(temp_path, final_path)
                    
                    segments.append(final_path)
                    print(f"Created segment: {final_path} ({len(segment)/1000:.1f}s)")
                
                current_start = next_start  # Start next segment from the clean point
            
            # Handle remaining audio if any
            if current_start < len(audio):
                segment = audio[current_start:]
                
                # Try to find a final silence
                final_silence = None
                for s_start, s_end in reversed(silence_points):
                    if s_start > current_start and s_end - current_start <= self.segment_length_ms:
                        final_silence = (s_start, s_end)
                        break
                
                # If found, include portion of the silence
                if final_silence:
                    pre_silence, _ = self._get_silence_portion(final_silence[0], final_silence[1])
                    segment = audio[current_start:pre_silence[1]]
                
                # Only create segment if it has content and reasonable length
                if len(segment) >= self.min_silence_ms:
                    # Save to temp first
                    temp_path = os.path.join(self.temp_dir, f"segment_{len(segments):03d}_temp.wav")
                    segment.export(temp_path, format="wav")
                    
                    # Move to final destination
                    final_path = os.path.join(self.segment_dir, f"segment_{len(segments):03d}.wav")
                    shutil.move(temp_path, final_path)
                    
                    segments.append(final_path)
                    print(f"Created segment: {final_path} ({len(segment)/1000:.1f}s)")
            
            return segments
        except Exception as e:
            print(f"Error splitting audio: {e}")
            return None
    
    def process_file(self, input_file):
        """
        Process an audio file: convert to WAV and split into segments if needed.
        
        Args:
            input_file (str): Path to input audio file
            
        Returns:
            tuple: (converted_files, segments)
        """
        try:
            # Convert to different formats
            converted_files = self.convert_to_wav(input_file)
            segments = None
            
            if converted_files:
                # Only split the standard version, since all versions have the same content
                standard_path = converted_files['standard']
                if os.path.getsize(standard_path) > 10 * 1024 * 1024:  # If larger than 10MB
                    print(f"\nSplitting audio into segments...")
                    segments = self.split_audio(standard_path)
            
            # Cleanup temp files unless keep_temp is True
            if not self.keep_temp:
                self.cleanup_temp_files()
                
            return converted_files, segments
            
        except Exception as e:
            print(f"Error during processing: {e}")
            if not self.keep_temp:
                self.cleanup_temp_files()
            return None, None

def main():
    """Command line interface for AudioConverter."""
    parser = argparse.ArgumentParser(description='Convert and prepare audio files for speech recognition.')
    parser.add_argument('input_file', help='Path to the input audio file')
    parser.add_argument('--output-dir', default='converted', help='Directory for converted files')
    parser.add_argument('--segment-dir', default='segments', help='Directory for segmented files')
    parser.add_argument('--temp-dir', help='Directory for temporary files (optional)')
    parser.add_argument('--segment-length', type=int, default=30000, 
                      help='Maximum length of segments in milliseconds (default: 30000)')
    parser.add_argument('--min-silence', type=int, default=1000,
                      help='Minimum length of silence to split on in milliseconds (default: 1000)')
    parser.add_argument('--silence-thresh', type=int, default=-40,
                      help='Silence threshold in dB (default: -40)')
    parser.add_argument('--keep-temp', action='store_true', 
                      help='Keep temporary files after processing')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return
    
    # Create converter instance with command line arguments
    converter = AudioConverter(
        output_dir=args.output_dir,
        segment_dir=args.segment_dir,
        temp_dir=args.temp_dir,
        segment_length_ms=args.segment_length,
        min_silence_ms=args.min_silence,
        silence_thresh=args.silence_thresh,
        keep_temp=args.keep_temp
    )
    
    # Process the file
    converter.process_file(args.input_file)

if __name__ == "__main__":
    main() 