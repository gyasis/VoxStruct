# util/transcript_builder.py

"""
Build and structure transcripts with proper timing and formatting.
"""
from typing import List, Dict, Any, Optional

class TranscriptBuilder:
    def __init__(self):
        """Initialize the transcript builder."""
        self.segments = []  # each segment is a dict with 'text', 'start_time', 'end_time'
        self.metadata = {
            "total_duration": 0,
            "pause_points": [],
            "speaker_changes": [],
            "confidence_scores": []
        }
    
    def add_segment(self, 
                   text: str, 
                   start_time: float, 
                   end_time: float,
                   speaker: str = None,
                   confidence: float = None):
        """
        Add a transcript segment with timing and metadata.
        
        Args:
            text: The transcribed text
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            speaker: Optional speaker identifier
            confidence: Optional confidence score
        """
        segment = {
            "text": text.strip(),
            "start_time": start_time,
            "end_time": end_time,
            "speaker": speaker,
            "confidence": confidence
        }
        self.segments.append(segment)
        
        # Update metadata
        self.metadata["total_duration"] = max(
            self.metadata["total_duration"],
            end_time
        )
        if confidence is not None:
            self.metadata["confidence_scores"].append(confidence)
        if speaker and (not self.segments[:-1] or 
                       self.segments[-2]["speaker"] != speaker):
            self.metadata["speaker_changes"].append(start_time)
    
    def add_pause_point(self, timestamp: float):
        """Add a detected pause point."""
        self.metadata["pause_points"].append(timestamp)
    
    def build_transcript(self, format_type: str = "simple") -> str:
        """
        Build the final transcript in the specified format.
        
        Args:
            format_type: Type of formatting to apply
                - "simple": Basic punctuation
                - "detailed": Include speaker attribution and timestamps
                - "raw": Just the text without formatting
        
        Returns:
            str: The formatted transcript
        """
        if format_type == "raw":
            return " ".join(seg["text"] for seg in self.segments)
        
        transcript = ""
        for i, segment in enumerate(self.segments):
            # Add speaker attribution if available and changed
            if (format_type == "detailed" and segment["speaker"] and 
                (i == 0 or self.segments[i-1]["speaker"] != segment["speaker"])):
                transcript += f"\n[{segment['speaker']}]: "
            
            # Add the text
            transcript += segment["text"]
            
            # Add timing for detailed format
            if format_type == "detailed":
                transcript += f" [{segment['start_time']/1000:.1f}s]"
            
            # If not the last segment, add appropriate punctuation
            if i < len(self.segments) - 1:
                next_start = self.segments[i+1]["start_time"]
                pause_duration = next_start - segment["end_time"]
                
                # Check if this pause point is in our detected pauses
                is_major_pause = any(
                    abs(p - next_start) < 100  # Within 100ms
                    for p in self.metadata["pause_points"]
                )
                
                # Decide punctuation
                if is_major_pause or pause_duration > 1000:
                    transcript += ".\n"
                elif pause_duration > 400:
                    transcript += ". "
                else:
                    transcript += ", "
            else:
                transcript += "."
        
        return transcript
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the transcript."""
        return {
            "duration_seconds": self.metadata["total_duration"] / 1000,
            "pause_count": len(self.metadata["pause_points"]),
            "speaker_changes": len(self.metadata["speaker_changes"]),
            "average_confidence": (
                sum(self.metadata["confidence_scores"]) / 
                len(self.metadata["confidence_scores"])
                if self.metadata["confidence_scores"] else None
            ),
            "segment_count": len(self.segments)
        }
    
    def get_segments_for_llm(self) -> List[Dict[str, Any]]:
        """
        Get segments in a format suitable for LLM processing.
        Includes pause information and speaker changes.
        """
        formatted_segments = []
        for i, segment in enumerate(self.segments):
            # Calculate pause after this segment
            pause_after = None
            if i < len(self.segments) - 1:
                pause_after = self.segments[i+1]["start_time"] - segment["end_time"]
            
            formatted_segments.append({
                "text": segment["text"],
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "speaker": segment["speaker"],
                "pause_after_ms": pause_after,
                "is_major_pause": any(
                    abs(p - segment["end_time"]) < 100
                    for p in self.metadata["pause_points"]
                ) if pause_after else False
            })
        
        return formatted_segments
