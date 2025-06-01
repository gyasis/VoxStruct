# util/transcript_builder.py

"""
Build and structure transcripts with proper timing and formatting.
"""
from typing import List, Dict, Any, Optional, Union

class TranscriptBuilder:
    def __init__(self, granularity: str = "chunk"):
        """
        Initialize the transcript builder.

        Args:
            granularity (str): The level of detail for timestamps ('chunk' or 'word'). 
                               Defaults to 'chunk'.
        """
        if granularity not in ["chunk", "word"]:
            raise ValueError("Granularity must be 'chunk' or 'word'")
        self.granularity = granularity
        # Segments list will store dicts based on granularity
        # chunk: {'text': str, 'start_time': float, 'end_time': float, ...}
        # word: {'word': str, 'start_time': float, 'end_time': float, 'confidence': float, ...}
        self.segments: List[Dict[str, Any]] = [] 
        self.metadata = {
            "total_duration": 0, # Stored in milliseconds
            "pause_points": [],  # Stored in milliseconds
            "speaker_changes": [], # Stored in milliseconds
            "confidence_scores": [] # List of confidences (word or chunk level)
        }
    
    def add_chunk_segment(self, 
                          text: str, 
                          start_time: float, 
                          end_time: float,
                          speaker: Optional[str] = None,
                          confidence: Optional[float] = None):
        """
        Add a transcript segment representing a larger chunk of audio.
        Only use when granularity is 'chunk'. Times in milliseconds.
        """
        if self.granularity != "chunk":
            raise TypeError("Cannot add chunk segment when granularity is 'word'")
        
        segment = {
            "text": text.strip(),
            "start_time": start_time, # ms
            "end_time": end_time,     # ms
            "speaker": speaker,
            "confidence": confidence
        }
        self.segments.append(segment)
        
        # Update metadata (using end_time directly as it's already ms)
        self.metadata["total_duration"] = max(self.metadata["total_duration"], end_time)
        if confidence is not None:
            self.metadata["confidence_scores"].append(confidence)
        # Speaker change logic remains the same for chunk granularity
        if speaker and (len(self.segments) <= 1 or self.segments[-2].get("speaker") != speaker):
             self.metadata["speaker_changes"].append(start_time)

    def add_word_segments(self, 
                          word_segment_list: List[Dict[str, Any]], 
                          chunk_start_time_ms: float,
                          speaker: Optional[str] = None):
        """
        Add a list of word segments, adjusting their times relative to the chunk start.
        Only use when granularity is 'word'.
        Expects word segments like {'word': str, 'start': float, 'end': float, 'confidence': float}.
        Input times ('start', 'end') are assumed to be in seconds relative to the chunk start.
        Times stored internally will be absolute milliseconds from the audio start.
        """
        if self.granularity != "word":
            raise TypeError("Cannot add word segments when granularity is 'chunk'")
        
        for word_info in word_segment_list:
            word = word_info.get("word", "").strip()
            start_sec = word_info.get("start")
            end_sec = word_info.get("end")
            confidence = word_info.get("confidence")

            if not word or start_sec is None or end_sec is None:
                continue # Skip words without necessary info

            # Convert relative seconds to absolute milliseconds
            abs_start_time_ms = chunk_start_time_ms + (start_sec * 1000)
            abs_end_time_ms = chunk_start_time_ms + (end_sec * 1000)

            segment = {
                "word": word,
                "start_time": abs_start_time_ms, # ms
                "end_time": abs_end_time_ms,     # ms
                "speaker": speaker, # Apply speaker to all words in this chunk for now
                "confidence": confidence
            }
            self.segments.append(segment)

            # Update metadata
            self.metadata["total_duration"] = max(self.metadata["total_duration"], abs_end_time_ms)
            if confidence is not None:
                self.metadata["confidence_scores"].append(confidence)
            # Note: Speaker change detection might be less meaningful at word level here,
            # unless speaker diarization provides word-level speaker info.
            # We'll keep the chunk-based logic for now if speaker is provided for the whole list.
            if speaker and (len(self.segments) <= 1 or self.segments[-2].get("speaker") != speaker):
                 self.metadata["speaker_changes"].append(abs_start_time_ms) # Use first word's time

    def add_pause_point(self, timestamp_ms: float):
        """Add a detected pause point timestamp in milliseconds."""
        self.metadata["pause_points"].append(timestamp_ms)
    
    def build_transcript(self, format_type: str = "simple") -> str:
        """
        Build the final transcript in the specified format. Handles both chunk and word granularity.
        
        Args:
            format_type: Type of formatting to apply
                - "simple": Basic punctuation (joins words/chunks based on timing)
                - "detailed": Include speaker attribution and word/chunk timestamps (seconds)
                - "raw": Just the text joined by spaces
        
        Returns:
            str: The formatted transcript
        """
        if not self.segments:
            return ""

        if format_type == "raw":
            if self.granularity == "word":
                return " ".join(seg.get("word", "") for seg in self.segments)
            else: # chunk
                return " ".join(seg.get("text", "") for seg in self.segments)

        transcript = ""
        
        if self.granularity == "word":
            for i, segment in enumerate(self.segments):
                word = segment.get("word", "")
                speaker = segment.get("speaker")
                start_time_ms = segment.get("start_time")
                end_time_ms = segment.get("end_time")
                
                # Add speaker attribution if available and changed
                is_new_speaker = speaker and (i == 0 or self.segments[i-1].get("speaker") != speaker)
                if format_type == "detailed" and is_new_speaker:
                     # Check if previous character was newline for cleaner formatting
                     if transcript and not transcript.endswith(("\\n", "\\n ")) : transcript += " "
                     transcript += f"\\n[{speaker}]:"
                
                # Add space before the word if needed
                if i > 0 and not is_new_speaker and not transcript.endswith(("\\n", "\\n ")):
                    transcript += " " 
                    
                transcript += word
                
                # Add timing for detailed format
                if format_type == "detailed" and start_time_ms is not None:
                    transcript += f" [{start_time_ms/1000:.2f}s]" # Use more precision for words
                
                # Add punctuation based on pause after word (simplified for word level)
                if i < len(self.segments) - 1:
                    next_start_ms = self.segments[i+1].get("start_time")
                    if next_start_ms is not None and end_time_ms is not None:
                        pause_duration_ms = next_start_ms - end_time_ms
                        # Simple punctuation based on pause length
                        if pause_duration_ms > 800: # Longer pause -> period, maybe newline later
                            transcript += "."
                        elif pause_duration_ms > 300: # Medium pause -> comma
                            transcript += ","
                        # else: short pause -> just space (already added)
                else:
                     transcript += "." # End of transcript

            # Post-process for better paragraphing based on punctuation/pauses (optional refinement)
            # Could replace ". " with ".\\n" for pauses > 1000ms etc.
            # For now, keep it simple.

        else: # Granularity is "chunk" - original logic mostly applies
            for i, segment in enumerate(self.segments):
                text = segment.get("text", "")
                speaker = segment.get("speaker")
                start_time_ms = segment.get("start_time")
                end_time_ms = segment.get("end_time")

                # Add speaker attribution if available and changed
                is_new_speaker = speaker and (i == 0 or self.segments[i-1].get("speaker") != speaker)
                if format_type == "detailed" and is_new_speaker:
                    if transcript and not transcript.endswith(("\\n", "\\n ")): transcript += " "
                    transcript += f"\\n[{speaker}]: "
                
                # Add space before the text if needed
                if i > 0 and not is_new_speaker and not transcript.endswith(("\\n", "\\n ")):
                     transcript += " "

                transcript += text
                
                # Add timing for detailed format (chunk start time)
                if format_type == "detailed" and start_time_ms is not None:
                    transcript += f" [{start_time_ms/1000:.1f}s]"
                
                # Punctuation based on pauses between chunks
                if i < len(self.segments) - 1:
                    next_start_ms = self.segments[i+1].get("start_time")
                    if next_start_ms is not None and end_time_ms is not None:
                        pause_duration_ms = next_start_ms - end_time_ms
                        is_major_pause = any(
                            abs(p - next_start_ms) < 150 # Wider tolerance for chunk alignment
                            for p in self.metadata["pause_points"]
                        )
                        
                        if is_major_pause or pause_duration_ms > 1000:
                            transcript += ".\\n" # Use newline for major pauses/long silences
                        elif pause_duration_ms > 400:
                            transcript += ". "
                        else:
                             transcript += ", " # Treat closer chunks like clauses
                else:
                    transcript += "." # End of transcript

        return transcript.strip() # Remove leading/trailing whitespace
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the transcript."""
        avg_conf = None
        if self.metadata["confidence_scores"]:
             valid_scores = [s for s in self.metadata["confidence_scores"] if s is not None]
             if valid_scores:
                  avg_conf = sum(valid_scores) / len(valid_scores)

        return {
            "granularity": self.granularity, # Add granularity info
            "duration_seconds": self.metadata["total_duration"] / 1000,
            "pause_count": len(self.metadata["pause_points"]),
            "speaker_changes": len(self.metadata["speaker_changes"]),
            "average_confidence": avg_conf,
            "segment_count": len(self.segments) # Count of words or chunks
        }
    
    def get_segments_for_llm(self) -> List[Dict[str, Any]]:
        """
        Get segments in a format suitable for LLM processing.
        For 'word' granularity, provides word list. 
        For 'chunk' granularity, provides chunk list.
        Times are in milliseconds.
        """
        if self.granularity == "word":
             # LLMs usually prefer coherent text blocks. Reconstructing text from words.
             # This simple version just joins words; a more complex one could group by pauses.
             # We pass the detailed word segments directly for now.
             return self.segments # Pass the list of word dicts
        else: # Granularity is "chunk"
             formatted_segments = []
             for i, segment in enumerate(self.segments):
                 # Calculate pause after this segment
                 pause_after_ms = None
                 is_major_pause_after = False
                 if i < len(self.segments) - 1:
                     next_start_ms = self.segments[i+1].get("start_time")
                     end_time_ms = segment.get("end_time")
                     if next_start_ms is not None and end_time_ms is not None:
                         pause_after_ms = next_start_ms - end_time_ms
                         is_major_pause_after = any(
                             abs(p - next_start_ms) < 150 
                             for p in self.metadata["pause_points"]
                         )

                 formatted_segments.append({
                     "text": segment.get("text", ""),
                     "start_time": segment.get("start_time"), # ms
                     "end_time": segment.get("end_time"),     # ms
                     "speaker": segment.get("speaker"),
                     "pause_after_ms": pause_after_ms,
                     "is_major_pause": is_major_pause_after or (pause_after_ms is not None and pause_after_ms > 1000)
                 })
             return formatted_segments
