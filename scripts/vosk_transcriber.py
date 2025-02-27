"""
Speech recognition script using Vosk for offline transcription.
"""
import os
import wave
import json
from datetime import datetime
from vosk import Model, KaldiRecognizer, SetLogLevel
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

def download_model(model_name="vosk-model-small-en-us"):
    """
    Download Vosk model if not already present.
    Available models: https://alphacephei.com/vosk/models
    """
    import wget
    models_dir = "models"
    model_path = os.path.join(models_dir, model_name)
    
    if not os.path.exists(model_path):
        os.makedirs(models_dir, exist_ok=True)
        print(f"Downloading model {model_name}...")
        wget.download(f"https://alphacephei.com/vosk/models/{model_name}.zip")
        
        # Unzip the model
        import zipfile
        with zipfile.ZipFile(f"{model_name}.zip", 'r') as zip_ref:
            zip_ref.extractall(models_dir)
        
        # Clean up
        os.remove(f"{model_name}.zip")
    
    return model_path

def transcribe_audio(
    audio_path: str,
    model_path: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Transcribe audio using Vosk model.
    
    Args:
        audio_path: Path to audio file
        model_path: Path to Vosk model directory
        **kwargs: Additional arguments
        
    Returns:
        dict: Transcription results including text and timing info
    """
    try:
        if not model_path:
            raise ValueError("Vosk model path is required")
            
        print(f"Loading Vosk model from: {model_path}")
        model = Model(model_path)
        
        print(f"Transcribing audio file: {audio_path}")
        wf = wave.open(audio_path, "rb")
        
        # Check audio format
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            raise ValueError("Audio file must be WAV format mono PCM.")
            
        # Create recognizer
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)  # Enable word timing
        
        # Process audio
        results = []
        total_duration = 0
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                part_result = json.loads(rec.Result())
                results.append(part_result)
                
        # Get final result
        part_result = json.loads(rec.FinalResult())
        if part_result.get("text", "").strip():
            results.append(part_result)
            
        # Combine all results
        all_text = []
        all_segments = []
        
        for res in results:
            text = res.get("text", "").strip()
            if text:
                all_text.append(text)
                if "result" in res:
                    all_segments.extend(res["result"])
                    
        combined_text = " ".join(all_text)
        
        # Format output to match standard format
        output = {
            "text": combined_text,
            "language": kwargs.get("language", "unknown"),
            "duration": wf.getnframes() / wf.getframerate(),
            "segments": []
        }
        
        # Add detailed segment information
        if all_segments:
            output["detailed"] = {
                "segments": [
                    {
                        "start": segment.get("start", 0),
                        "end": segment.get("end", 0),
                        "text": segment.get("word", ""),
                        "confidence": segment.get("conf", 0)
                    }
                    for segment in all_segments
                ]
            }
        
        return output
        
    except Exception as e:
        print(f"Error in Vosk transcription: {e}")
        return {
            "text": "",
            "error": str(e),
            "language": "unknown",
            "duration": 0
        }

def batch_transcribe(input_dir, model_path=None, output_dir="output"):
    """
    Batch process multiple audio files in a directory.
    """
    results = {}
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.wav'):
            input_path = os.path.join(input_dir, filename)
            print(f"\nProcessing: {filename}")
            try:
                results[filename] = transcribe_audio(input_path, model_path, output_dir)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                results[filename] = {"error": str(e)}
    
    # Save batch results
    batch_results_path = os.path.join(output_dir, "vosk_batch_results.json")
    with open(batch_results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    audio_path = "sample_audio.wav"  # Replace with your audio file
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        exit(1)
    
    results = transcribe_audio(audio_path)
    
    # For batch processing, uncomment these lines:
    # input_directory = "audio_files"
    # batch_results = batch_transcribe(input_directory) 