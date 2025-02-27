# main.py
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import argparse
from dotenv import load_dotenv
import datetime

# Import from utils package
from utils.audio_processor import AudioProcessor
from utils.audio_library import AudioLibrary
from utils.speech_recognizer import SpeechRecognizer
from utils.pause_detector import PauseDetector
from utils.transcript_builder import TranscriptBuilder
from utils.llm_supervisory import LLMSupervisor
from utils.config import SILENCE_THRESHOLD, MIN_SILENCE_LENGTH
import json

def create_markdown_header(metadata: dict) -> str:
    """Create a markdown header with metadata information."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = [
        "# Audio Transcript",
        "",
        "## Metadata",
        f"- **Date**: {now}",
        f"- **File**: {os.path.basename(metadata.get('audio_file', 'unknown'))}",
        f"- **Duration**: {metadata.get('duration', 0)/1000:.2f} seconds",
        f"- **Speech Recognition Engine**: {metadata.get('engine', 'unknown')}",
        f"- **Model**: {metadata.get('model', 'unknown')}"
    ]
    
    if metadata.get('language'):
        header.append(f"- **Language**: {metadata['language']}")
    
    if metadata.get('llm_model'):
        header.append(f"- **LLM Model**: {metadata['llm_model']}")
    
    header.extend(["", "---", ""])  # Add separator
    return "\n".join(header)

def verify_llm_model(api_key: str, model: str) -> bool:
    """
    Verify that the specified LLM model is working correctly.
    
    Args:
        api_key: API key for the model provider
        model: Model identifier (e.g., openai/gpt-4)
        
    Returns:
        bool: True if verification succeeded, False otherwise
    """
    try:
        llm_supervisor = LLMSupervisor(api_key, model)
        success, message = llm_supervisor.verify_model()
        
        if success:
            print(f"\nModel verification successful. Using {model}")
            print(f"Model response: {message}")
            return True
        else:
            print(f"\nModel verification failed: {message}")
            return False
            
    except Exception as e:
        print(f"\nError during model verification: {e}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Transcribe audio using various engines")
    parser.add_argument("audio_file", help="Path to the audio file to transcribe")
    parser.add_argument("--engine", default="whisper", choices=["whisper", "vosk", "coqui"],
                      help="Speech recognition engine to use")
    parser.add_argument("--model", default="base",
                      help="Model name/size (for Whisper) or path (for Vosk/Coqui)")
    parser.add_argument("--language", help="Source language code (e.g., en, de)")
    parser.add_argument("--scorer-path", help="Path to scorer file (Coqui only)")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM supervision")
    parser.add_argument("--llm-model", default="gpt4all/gpt4all-j",
                      help="LLM model to use (e.g., openai/gpt-4, anthropic/claude-3, ollama/mistral)")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    
    # Determine which API key to use based on the selected model
    api_key = None
    if "openai" in args.llm_model:
        api_key = openai_api_key
    elif "anthropic" in args.llm_model:
        api_key = anthropic_api_key
    
    # Verify LLM model if LLM supervision is enabled
    if not args.no_llm:
        if not verify_llm_model(api_key, args.llm_model):
            print("LLM verification failed. Proceeding without LLM supervision.")
            args.no_llm = True
    
    try:
        # === Audio Processing ===
        print(f"\nProcessing audio file: {args.audio_file}")
        audio_processor = AudioProcessor(args.audio_file)
        audio_segment = audio_processor.load_audio()
        if audio_segment is None:
            print("Failed to load audio. Exiting.")
            return
        
        # === Pause Detection ===
        silence_thresh = SILENCE_THRESHOLD
        min_silence_len = MIN_SILENCE_LENGTH
        pause_detector = PauseDetector(silence_thresh=silence_thresh, min_silence_len=min_silence_len)
        pause_timestamps = pause_detector.detect_pauses(audio_segment)
        print("\nDetected pause timestamps (ms):", pause_timestamps)
        
        # === Speech Recognition Configuration ===
        config = {
            "language": args.language
        }
        
        if args.engine == "whisper":
            config["model_name"] = args.model
        else:  # vosk or coqui
            config["model_path"] = args.model
            if args.engine == "coqui" and args.scorer_path:
                config["scorer_path"] = args.scorer_path
        
        # Initialize speech recognizer and audio library
        speech_recognizer = SpeechRecognizer(args.engine, **config)
        audio_lib = AudioLibrary(args.engine, **config)
        
        # Initialize transcript builder
        transcript_builder = TranscriptBuilder()
        
        # === Process Audio in Chunks ===
        chunks = audio_processor.get_chunks()
        current_time = 0
        
        print(f"\nTranscribing with {args.engine.upper()}...")
        for i, chunk in enumerate(chunks, 1):
            print(f"\rProcessing chunk {i}/{len(chunks)}...", end="", flush=True)
            
            # Transcribe chunk
            result = speech_recognizer.transcribe_audio(chunk)
            
            # Add to transcript with timing
            if result and result["text"]:
                start_time = current_time
                end_time = current_time + len(chunk)
                transcript_builder.add_segment(result["text"], start_time, end_time)
            
            current_time += len(chunk)
        
        print("\n")  # New line after progress
        
        # === Build Raw Transcript ===
        raw_transcript = transcript_builder.build_transcript()
        print("\nRaw Transcript:")
        print("-" * 80)
        print(raw_transcript)
        print("-" * 80)
        
        # === LLM Supervision ===
        if not args.no_llm:
            print(f"\nImproving transcript with LLM supervision using {args.llm_model}...")
            llm_supervisor = LLMSupervisor(api_key, args.llm_model)
            final_transcript = llm_supervisor.validate_and_improve_transcript(raw_transcript)
        else:
            final_transcript = raw_transcript
        
        # === Save Output ===
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw transcript as txt
        raw_output_file = os.path.join(output_dir, f"raw_transcript_{os.path.basename(args.audio_file)}.txt")
        with open(raw_output_file, "w", encoding="utf-8") as f:
            f.write(raw_transcript)
        
        # Prepare metadata for markdown
        metadata = {
            "audio_file": args.audio_file,
            "engine": args.engine,
            "model": args.model,
            "language": args.language,
            "llm_model": args.llm_model if not args.no_llm else None,
            "pause_timestamps": pause_timestamps,
            "chunk_count": len(chunks),
            "duration": len(audio_segment),
            "sample_rate": audio_segment.frame_rate,
            "channels": audio_segment.channels
        }
        
        # Create markdown content with metadata header
        markdown_content = create_markdown_header(metadata) + final_transcript
        
        # Save final transcript as markdown
        final_output_file = os.path.join(output_dir, f"transcript_{os.path.basename(args.audio_file)}.md")
        with open(final_output_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        print(f"\nRaw transcript saved to: {raw_output_file}")
        print(f"Final transcript (markdown) saved to: {final_output_file}")
        
        if final_transcript != raw_transcript:
            print("\nFinal Transcript (LLM Improved):")
            print("-" * 80)
            print(final_transcript)
            print("-" * 80)
        
        # Save detailed metadata as JSON
        metadata_file = os.path.join(output_dir, f"metadata_{os.path.basename(args.audio_file)}.json")
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nMetadata saved to: {metadata_file}")
        
    except Exception as e:
        print(f"\nError during transcription: {e}")

if __name__ == "__main__":
    main()
