"""
Speech recognition script using OpenAI's Whisper model (offline version).
"""
import os
import whisper
from datetime import datetime
import json
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

def transcribe_audio(
    audio_path: str,
    model_name: str = "base",
    language: str = None,
    translate_to: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Transcribe audio using Whisper model.
    
    Args:
        audio_path: Path to audio file
        model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        language: Source language code (e.g., 'en', 'de', None for auto-detection)
        translate_to: Target language for translation (if needed)
        **kwargs: Additional arguments passed to whisper
        
    Returns:
        dict: Transcription results including text, language, and timing info
    """
    try:
        print(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name)
        
        # Prepare transcription options
        options = {
            "language": language,
            "task": "translate" if translate_to else "transcribe",
            "verbose": True,  # Enable detailed output
        }
        
        # Add any additional options from kwargs
        options.update(kwargs)
        
        print(f"Transcribing: {audio_path}")
        result = model.transcribe(audio_path, **options)
        
        # Format the result
        output = {
            "text": result["text"],
            "language": result.get("language", "unknown"),
            "duration": result.get("duration", 0),
            "segments": result.get("segments", []),
        }
        
        # Add detailed information if available
        if "segments" in result:
            output["detailed"] = {
                "segments": []
            }
            
            for seg in result["segments"]:
                segment_info = {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                }
                
                # Calculate segment-level confidence if available
                segment_info["confidence"] = seg.get("confidence", 0)
                output["detailed"]["segments"].append(segment_info)
            
            # Calculate overall confidence score
            if output["detailed"]["segments"]:
                total_conf = sum(seg["confidence"] for seg in output["detailed"]["segments"])
                output["confidence"] = total_conf / len(output["detailed"]["segments"])
            else:
                output["confidence"] = 0
        
        return output
        
    except Exception as e:
        print(f"Error in Whisper transcription: {e}")
        return {
            "text": "",
            "error": str(e),
            "language": "unknown",
            "duration": 0,
            "confidence": 0
        }

def batch_transcribe(input_dir, model_name="base", output_dir="output"):
    """
    Batch process multiple audio files in a directory.
    """
    supported_formats = (".wav", ".mp3", ".m4a", ".ogg")
    results = {}
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_dir, filename)
            print(f"\nProcessing: {filename}")
            try:
                results[filename] = transcribe_audio(input_path, model_name, output_dir)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                results[filename] = {"error": str(e)}
    
    # Save batch results
    batch_results_path = os.path.join(output_dir, "batch_results.json")
    with open(batch_results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    # Single file transcription
    audio_path = "sample_audio.wav"  # Replace with your audio file
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        exit(1)
    
    # You can choose model size: "tiny", "base", "small", "medium", "large"
    results = transcribe_audio(audio_path, model_name="base")
    
    # For batch processing, uncomment these lines:
    # input_directory = "audio_files"
    # batch_results = batch_transcribe(input_directory, model_name="base") 