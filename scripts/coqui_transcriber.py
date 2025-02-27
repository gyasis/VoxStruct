"""
Speech recognition script using Coqui STT (formerly DeepSpeech) for offline transcription.
"""
import os
import json
from datetime import datetime
import numpy as np
from stt import Model, Stream
import wave
from typing import Dict, Any

def load_coqui_model(model_path: str, scorer_path: str = None) -> Model:
    """
    Load Coqui STT model with optional scorer.
    
    Args:
        model_path: Path to the model file (.tflite)
        scorer_path: Optional path to the scorer file
        
    Returns:
        Model: Loaded Coqui STT model
    """
    model = Model(model_path)
    if scorer_path:
        model.enableExternalScorer(scorer_path)
    return model

def transcribe_audio(
    audio_path: str,
    model_path: str = None,
    scorer_path: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Transcribe audio using Coqui STT model.
    
    Args:
        audio_path: Path to audio file
        model_path: Path to Coqui STT model file
        scorer_path: Optional path to language model scorer
        **kwargs: Additional arguments
        
    Returns:
        dict: Transcription results including text and timing info
    """
    try:
        if not model_path:
            raise ValueError("Coqui STT model path is required")
            
        print(f"Loading Coqui STT model from: {model_path}")
        model = load_coqui_model(model_path, scorer_path)
        
        print(f"Transcribing audio file: {audio_path}")
        wf = wave.open(audio_path, "rb")
        
        # Check audio format
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            raise ValueError("Audio file must be WAV format mono PCM.")
        
        # Process audio in chunks for memory efficiency
        chunk_size = int(wf.getframerate() * 0.5)  # 0.5 second chunks
        buffer_size = chunk_size * 2  # 16-bit audio = 2 bytes per sample
        
        segments = []
        audio_length = 0
        
        while True:
            chunk = wf.readframes(chunk_size)
            if len(chunk) == 0:
                break
                
            if len(chunk) < buffer_size:
                chunk = chunk + b'\0' * (buffer_size - len(chunk))
                
            audio_length += len(chunk) / 2
            
            # Create stream for this chunk
            stream = Model.createStream()
            stream.feedAudioContent(np.frombuffer(chunk, np.int16))
            
            # Get intermediate results
            text = stream.intermediateDecode()
            if text.strip():
                metadata = stream.intermediateDecodeMetadata()
                
                # Extract word-level information
                words = []
                for token in metadata.tokens:
                    words.append({
                        "word": token.text,
                        "start": token.start_time,
                        "end": token.end_time,
                        "confidence": token.confidence
                    })
                
                # Calculate segment confidence as average of word confidences
                confidence = sum(word["confidence"] for word in words) / len(words) if words else 0
                
                segments.append({
                    "text": text,
                    "words": words,
                    "confidence": confidence,
                    "start": audio_length - len(chunk)/2,
                    "end": audio_length
                })
        
        # Combine all text
        full_text = " ".join(seg["text"] for seg in segments)
        
        # Calculate overall confidence
        overall_confidence = sum(seg["confidence"] for seg in segments) / len(segments) if segments else 0
        
        output = {
            "text": full_text,
            "language": kwargs.get("language", "unknown"),
            "duration": audio_length / wf.getframerate(),
            "confidence": overall_confidence,
            "detailed": {
                "segments": segments
            }
        }
        
        return output
        
    except Exception as e:
        print(f"Error in Coqui STT transcription: {e}")
        return {
            "text": "",
            "error": str(e),
            "language": "unknown",
            "duration": 0,
            "confidence": 0
        }

def batch_transcribe(input_dir, model_path, scorer_path=None, output_dir="output"):
    """
    Batch process multiple audio files in a directory.
    """
    results = {}
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.wav'):
            input_path = os.path.join(input_dir, filename)
            print(f"\nProcessing: {filename}")
            try:
                results[filename] = transcribe_audio(
                    input_path,
                    model_path=model_path,
                    scorer_path=scorer_path
                )
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                results[filename] = {"error": str(e)}
    
    # Save batch results
    batch_results_path = os.path.join(output_dir, "coqui_batch_results.json")
    with open(batch_results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    audio_path = "sample_audio.wav"  # Replace with your audio file
    model_path = "path/to/model.tflite"  # Replace with your model path
    scorer_path = "path/to/scorer.scorer"  # Optional
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        exit(1)
    
    results = transcribe_audio(audio_path, model_path, scorer_path) 