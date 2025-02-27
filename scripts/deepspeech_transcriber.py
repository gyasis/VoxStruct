"""
Speech recognition script using Mozilla DeepSpeech for offline transcription.
"""
import os
import wave
import json
import numpy as np
from datetime import datetime
from deepspeech import Model
from dotenv import load_dotenv

load_dotenv()

def download_model(model_dir="models"):
    """
    Download DeepSpeech model if not already present.
    """
    import wget
    
    os.makedirs(model_dir, exist_ok=True)
    base_url = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3"
    
    model_file = "deepspeech-0.9.3-models.pbmm"
    scorer_file = "deepspeech-0.9.3-models.scorer"
    
    model_path = os.path.join(model_dir, model_file)
    scorer_path = os.path.join(model_dir, scorer_file)
    
    # Download model if needed
    if not os.path.exists(model_path):
        print("Downloading model file...")
        wget.download(f"{base_url}/{model_file}", model_path)
    
    # Download scorer if needed
    if not os.path.exists(scorer_path):
        print("\nDownloading scorer file...")
        wget.download(f"{base_url}/{scorer_file}", scorer_path)
    
    return model_path, scorer_path

def create_model(model_path, scorer_path):
    """
    Create and configure DeepSpeech model.
    """
    print("Loading model...")
    ds = Model(model_path)
    ds.enableExternalScorer(scorer_path)
    
    return ds

def process_audio(audio_path):
    """
    Convert audio file to format required by DeepSpeech.
    """
    with wave.open(audio_path, 'rb') as w:
        rate = w.getframerate()
        frames = w.getnframes()
        buffer = w.readframes(frames)
    
    data16 = np.frombuffer(buffer, dtype=np.int16)
    return data16, rate

def transcribe_audio(input_path, model_dir="models", output_dir="output"):
    """
    Transcribe audio using DeepSpeech with various options.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download or use existing model
    model_path, scorer_path = download_model(model_dir)
    
    # Create model
    model = create_model(model_path, scorer_path)
    
    print(f"Transcribing audio file: {input_path}")
    
    # Process audio
    audio, sample_rate = process_audio(input_path)
    
    # Get metadata to enable word timing
    metadata = model.sttWithMetadata(audio)
    
    # Extract words with timing
    word_list = []
    for item in metadata.transcripts[0].tokens:
        word_list.append({
            "word": item.text,
            "start_time": item.start_time,
            "duration": item.duration
        })
    
    # Get full transcript
    transcript = model.stt(audio)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save plain text transcript
    transcript_path = os.path.join(output_dir, f"deepspeech_transcript_{timestamp}.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    
    # Save detailed JSON results
    json_path = os.path.join(output_dir, f"deepspeech_detailed_{timestamp}.json")
    results = {
        "transcript": transcript,
        "words": word_list,
        "sample_rate": sample_rate
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nTranscription saved to: {transcript_path}")
    print(f"Detailed JSON results saved to: {json_path}")
    
    # Calculate statistics
    stats = {
        "word_count": len(word_list),
        "duration": sum(word["duration"] for word in word_list),
        "sample_rate": sample_rate,
        "output_files": [transcript_path, json_path]
    }
    
    print("\nTranscription Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return stats

def batch_transcribe(input_dir, model_dir="models", output_dir="output"):
    """
    Batch process multiple audio files in a directory.
    """
    results = {}
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.wav'):
            input_path = os.path.join(input_dir, filename)
            print(f"\nProcessing: {filename}")
            try:
                results[filename] = transcribe_audio(input_path, model_dir, output_dir)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                results[filename] = {"error": str(e)}
    
    # Save batch results
    batch_results_path = os.path.join(output_dir, "deepspeech_batch_results.json")
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