# main.py
import os
import sys
# Print sys.path immediately and exit for debugging # REMOVED
# print("sys.path when running main.py:\\n", sys.path) # REMOVED
# sys.exit() # REMOVED

from pathlib import Path

# # Add the project root to Python path # REMOVED
# project_root = str(Path(__file__).parent.parent)
# sys.path.append(project_root)

import argparse
from dotenv import load_dotenv
import datetime
import json # Moved json import higher
import shutil # Added for cleaning up temp youtube downloads

# Import using absolute paths assuming 'voxstruct' is the root package
from voxstruct.utils.audio_processor import AudioProcessor
# from voxstruct.utils.audio_library import AudioLibrary # Keep commented if unused
from voxstruct.utils.speech_recognizer import SpeechRecognizer
from voxstruct.utils.pause_detector import PauseDetector
from voxstruct.utils.transcript_builder import TranscriptBuilder
from voxstruct.utils.llm_supervisory import LLMSupervisor
from voxstruct.utils.youtube_downloader import YoutubeDownloader # Added YoutubeDownloader
from voxstruct.utils.config import SILENCE_THRESHOLD, MIN_SILENCE_LENGTH
# Removed duplicate json import

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
    parser.add_argument("audio_file", help="Path to the audio file to transcribe, or a YouTube URL") # Updated help text
    parser.add_argument("--engine", default="whisper", choices=["whisper", "vosk"],
                      help="Speech recognition engine to use")
    parser.add_argument("--model", default="base",
                      help="Model name/size (for Whisper) or path (for Vosk)")
    parser.add_argument("--language", help="Source language code (e.g., en, de)")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM supervision")
    parser.add_argument("--llm-model", default="openai/gpt-4o-mini",
                      help="LLM model to use (e.g., openai/gpt-4, anthropic/claude-3, ollama/mistral)")
    parser.add_argument("--timestamp-granularity", default="chunk", choices=["chunk", "word"],
                      help="Level of detail for timestamps ('chunk' or 'word')")
    args = parser.parse_args()
    
    # Load environment variables, overriding existing ones if present in .env
    load_dotenv('/home/gyasis/Documents/code/VoxStruct/.env',override=True)
    
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
    
    processed_audio_file_path = args.audio_file
    youtube_temp_dir = None # To store temp dir path if YouTube download occurs
    output_basename = None # To store the base name for output files

    try:
        # === YouTube Download (if URL is provided) ===
        if "youtube.com/" in args.audio_file or "youtu.be/" in args.audio_file:
            print(f"\nDetected YouTube URL: {args.audio_file}")
            print("Attempting to download audio...")
            downloader = YoutubeDownloader()
            download_result = downloader.download_audio_from_youtube(args.audio_file)
            
            if download_result:
                processed_audio_file_path, youtube_temp_dir, output_basename = download_result
                print(f"YouTube audio downloaded to: {processed_audio_file_path}")
            else:
                print("Failed to download audio from YouTube. Exiting.")
                return
        else:
            # For local files, use the original filename as the base
            output_basename = os.path.splitext(os.path.basename(processed_audio_file_path))[0]
        
        if not output_basename: # Fallback if basename couldn't be determined
            output_basename = "transcription_output"
            print(f"Warning: Could not determine a base name for output files. Using '{output_basename}'.")
        
        # === Audio Processing ===
        print(f"\nProcessing audio file: {processed_audio_file_path}")
        audio_processor = AudioProcessor(processed_audio_file_path)
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
        else:  # vosk (only remaining option)
            config["model_path"] = args.model
        
        # Initialize speech recognizer, passing granularity
        speech_recognizer = SpeechRecognizer(
            args.engine, 
            granularity=args.timestamp_granularity, 
            **config
        )
        # audio_lib = AudioLibrary(args.engine, **config) # This seems unused, commenting out for now
        
        # Initialize transcript builder with specified granularity
        transcript_builder = TranscriptBuilder(granularity=args.timestamp_granularity)
        
        # === Process Audio in Chunks ===
        chunks = audio_processor.get_chunks()
        current_time_ms = 0 # Keep track of absolute time in milliseconds
        
        print(f"\nTranscribing with {args.engine.upper()} (Granularity: {args.timestamp_granularity})...")
        for i, chunk in enumerate(chunks, 1):
            print(f"\rProcessing chunk {i}/{len(chunks)}...", end="", flush=True)
            
            # Transcribe chunk
            result = speech_recognizer.transcribe_audio(chunk)
            
            # Add to transcript based on granularity
            if result and result.get("text"): # Check if transcription was successful
                if args.timestamp_granularity == "word":
                    word_segments = result.get("segments", [])
                    # Check if we actually got word segments (start time is not None)
                    if word_segments and word_segments[0].get("start") is not None:
                       transcript_builder.add_word_segments(word_segments, current_time_ms)
                    else:
                       # Fallback to chunk if word timings weren't available from recognizer
                       print(f"Warning: Word timestamps not available for chunk {i}, adding as chunk.", file=sys.stderr) # Warn user
                       end_time_ms = current_time_ms + len(chunk)
                       transcript_builder.add_chunk_segment(result["text"], current_time_ms, end_time_ms, confidence=result.get("confidence"))
                else: # Granularity is "chunk"
                    start_time_ms = current_time_ms
                    end_time_ms = current_time_ms + len(chunk)
                    transcript_builder.add_chunk_segment(result["text"], start_time_ms, end_time_ms, confidence=result.get("confidence"))
            elif result and result.get("error"):
                 print(f"\nError transcribing chunk {i}: {result.get('error')}", file=sys.stderr)
            
            current_time_ms += len(chunk) # Advance time by chunk duration
        
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
        raw_output_file = os.path.join(output_dir, f"raw_transcript_{output_basename}.txt")
        with open(raw_output_file, "w", encoding="utf-8") as f:
            f.write(raw_transcript)
        
        # Get metadata from builder and add execution specifics
        builder_metadata = transcript_builder.get_metadata()
        metadata = {
            # Execution parameters
            "audio_file": processed_audio_file_path, # This will be the path to the temp downloaded file for YouTube
            "original_source": args.audio_file, # Keep track of the original input (URL or local path)
            "output_basename": output_basename,
            "engine": args.engine,
            "model": args.model,
            "language": args.language,
            "llm_model": args.llm_model if not args.no_llm else None,
            "pause_timestamps": pause_timestamps,
            "chunk_count": len(chunks),
            "duration": len(audio_segment),
            "sample_rate": audio_segment.frame_rate,
            "channels": audio_segment.channels,
            # Transcript details from builder
            "timestamp_granularity": builder_metadata.get("granularity"),
            "transcript_duration_seconds": builder_metadata.get("duration_seconds"),
            "segment_count": builder_metadata.get("segment_count"),
            "average_confidence": builder_metadata.get("average_confidence")
        }
        
        # Create markdown content with metadata header
        markdown_content = create_markdown_header(metadata) + final_transcript
        
        # Save final transcript as markdown
        final_output_file = os.path.join(output_dir, f"transcript_{output_basename}.md")
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
        metadata_file = os.path.join(output_dir, f"metadata_{output_basename}.json")
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nMetadata saved to: {metadata_file}")
        
        # === Save Detailed Timestamps (if word granularity) ===
        if args.timestamp_granularity == "word":
            # Call the existing get_segments_for_llm() method
            # that returns the list of segment dictionaries
            try:
                detailed_segments = transcript_builder.get_segments_for_llm()
                timestamps_file = os.path.join(output_dir, f"timestamps_{output_basename}.json")
                with open(timestamps_file, "w", encoding="utf-8") as f:
                    json.dump(detailed_segments, f, indent=2)
                print(f"Detailed word timestamps saved to: {timestamps_file}")
            except AttributeError:
                print("Warning: Could not save detailed timestamps. TranscriptBuilder might lack get_segments_for_llm() method.", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Error saving detailed timestamps: {e}", file=sys.stderr)
        
    except Exception as e:
        print(f"\nError during transcription: {e}")
    finally:
        if youtube_temp_dir and os.path.exists(youtube_temp_dir):
            try:
                shutil.rmtree(youtube_temp_dir)
                print(f"Cleaned up temporary YouTube download directory: {youtube_temp_dir}")
            except Exception as e_clean:
                print(f"Error cleaning up YouTube temporary directory {youtube_temp_dir}: {e_clean}")

if __name__ == "__main__":
    main()
