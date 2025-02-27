"""
Generate detailed JSON output with word-level timing for precise audio editing.
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

class TranscriptTimingGenerator:
    def __init__(self, output_dir: str = "json_output"):
        """Initialize the generator with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_timing_json(self, 
                           transcript_data: Dict[str, Any], 
                           audio_file: str) -> str:
        """
        Generate detailed JSON with word-level timing information.
        
        Structure:
        {
            "metadata": {
                "audio_file": "path/to/file.mp3",
                "duration": 123.45,
                "timestamp": "2024-01-20T15:30:00",
                "sample_rate": 16000,
                "channels": 1
            },
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 2.5,
                    "words": [
                        {
                            "word": "Hello",
                            "start": 0.0,
                            "end": 0.4,
                            "confidence": 0.98,
                            "speaker": "speaker_1",
                            "next_word_gap": 0.1,  # Time gap until next word
                            "previous_word_gap": 0.0,  # Time gap from previous word
                            "punctuation_after": ",",
                            "index": 0,  # Position in full transcript
                            "segment_index": 0  # Position in current segment
                        },
                        // ... more words
                    ],
                    "speaker": "speaker_1",
                    "language": "en"
                },
                // ... more segments
            ],
            "edit_points": {
                "natural_pauses": [
                    {"start": 2.5, "end": 2.8, "duration": 0.3}
                ],
                "sentence_boundaries": [
                    {"index": 15, "time": 2.5, "confidence": 0.95}
                ],
                "speaker_changes": [
                    {"time": 5.0, "speaker_before": "speaker_1", "speaker_after": "speaker_2"}
                ]
            },
            "statistics": {
                "word_count": 150,
                "speaker_count": 2,
                "average_word_length": 0.3,
                "average_gap_between_words": 0.1
            }
        }
        """
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            self.output_dir, 
            f"transcript_timing_{os.path.basename(audio_file)}_{timestamp}.json"
        )
        
        # Process transcript data and add timing information
        processed_data = self._process_transcript_data(transcript_data, audio_file)
        
        # Add edit points and statistics
        processed_data["edit_points"] = self._generate_edit_points(processed_data)
        processed_data["statistics"] = self._generate_statistics(processed_data)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        return output_file

    def _process_transcript_data(self, 
                               transcript_data: Dict[str, Any], 
                               audio_file: str) -> Dict[str, Any]:
        """Process raw transcript data into our detailed format."""
        processed_data = {
            "metadata": {
                "audio_file": audio_file,
                "duration": transcript_data.get("duration", 0),
                "timestamp": datetime.now().isoformat(),
                "sample_rate": transcript_data.get("sample_rate", 16000),
                "channels": transcript_data.get("channels", 1)
            },
            "segments": []
        }
        
        # Process segments and words
        word_index = 0
        for segment_index, segment in enumerate(transcript_data.get("segments", [])):
            processed_segment = {
                "id": segment_index,
                "start": segment["start"],
                "end": segment["end"],
                "words": []
            }
            
            # Process words in segment
            for word_data in segment.get("words", []):
                processed_word = self._process_word(
                    word_data, 
                    word_index,
                    len(processed_segment["words"]),
                    segment
                )
                processed_segment["words"].append(processed_word)
                word_index += 1
            
            processed_data["segments"].append(processed_segment)
        
        return processed_data

    def _process_word(self,
                     word_data: Dict[str, Any],
                     global_index: int,
                     segment_index: int,
                     segment: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual word data."""
        return {
            "word": word_data["word"],
            "start": word_data["start"],
            "end": word_data["end"],
            "confidence": word_data.get("confidence", 1.0),
            "speaker": word_data.get("speaker", "unknown"),
            "next_word_gap": self._calculate_next_word_gap(word_data, segment),
            "previous_word_gap": self._calculate_previous_word_gap(word_data, segment),
            "punctuation_after": self._detect_punctuation(word_data["word"]),
            "index": global_index,
            "segment_index": segment_index
        }

    def _calculate_next_word_gap(self,
                                word_data: Dict[str, Any],
                                segment: Dict[str, Any]) -> float:
        """Calculate time gap until next word."""
        # Implementation depends on specific transcript format
        return 0.1  # Placeholder

    def _calculate_previous_word_gap(self,
                                   word_data: Dict[str, Any],
                                   segment: Dict[str, Any]) -> float:
        """Calculate time gap from previous word."""
        # Implementation depends on specific transcript format
        return 0.1  # Placeholder

    def _detect_punctuation(self, word: str) -> str:
        """Detect punctuation after word."""
        punctuation_marks = ".,:;!?"
        for mark in punctuation_marks:
            if word.endswith(mark):
                return mark
        return ""

    def _generate_edit_points(self, 
                            processed_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate edit points for natural breaks in speech."""
        edit_points = {
            "natural_pauses": [],
            "sentence_boundaries": [],
            "speaker_changes": []
        }
        
        # Implementation would analyze processed_data to find:
        # - Natural pauses between words/segments
        # - Sentence boundaries
        # - Speaker changes
        # - Other potential edit points
        
        return edit_points

    def _generate_statistics(self, 
                           processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistics about the transcript."""
        # Calculate various statistics about the transcript
        return {
            "word_count": sum(len(seg["words"]) for seg in processed_data["segments"]),
            "speaker_count": len(set(word["speaker"] 
                                   for seg in processed_data["segments"] 
                                   for word in seg["words"])),
            "average_word_length": 0.3,  # Placeholder
            "average_gap_between_words": 0.1  # Placeholder
        }

if __name__ == "__main__":
    # Example usage
    generator = TranscriptTimingGenerator()
    
    # Example transcript data (would come from speech recognition)
    example_data = {
        "duration": 10.5,
        "segments": [
            {
                "start": 0.0,
                "end": 2.5,
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.4},
                    {"word": "world", "start": 0.5, "end": 0.9}
                ]
            }
        ]
    }
    
    output_file = generator.generate_timing_json(example_data, "example.wav")
    print(f"Generated timing JSON: {output_file}") 