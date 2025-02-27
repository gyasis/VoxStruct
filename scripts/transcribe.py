"""
Unified transcription script that supports multiple engines without comparison.

DESCRIPTION
-----------
This script provides a unified interface for transcribing audio files using different
speech recognition engines (Whisper, Vosk, or optionally Coqui STT). It handles audio format
conversion automatically and provides consistent output formatting.

USAGE
-----
Basic Usage:
    python transcribe.py audio_file.mp3
    python transcribe.py audio_file.mp3 --engine whisper
    python transcribe.py audio_file.mp3 --engine vosk --model-path path/to/model
    python transcribe.py audio_file.mp3 --engine coqui --model-path path/to/model  # If Coqui STT is installed

ARGUMENTS
---------
Required Arguments:
    audio_file              Path to the input audio file (supports mp3, wav, ogg, flac, m4a)

Engine Selection:
    --engine               Speech recognition engine to use
                          Choices: [whisper, vosk, coqui]
                          Default: whisper

Whisper Options:
    --model               Model size for Whisper
                          Choices: [tiny, base, small, medium, large]
                          Default: base
    --language           Source language code (e.g., en, de, fr)
                          Optional: Whisper will auto-detect if not specified
    --translate-to       Target language for translation
                          Optional: Translate output to specified language

Vosk Options:
    --model-path         Path to Vosk model file
                          Required when using Vosk engine
                          Download models from: https://alphacephei.com/vosk/models

Coqui STT Options:
    --model-path         Path to Coqui STT model file
                          Required when using Coqui engine
    --scorer-path        Path to Coqui STT scorer file
                          Optional: Improves accuracy when provided

Output Options:
    --output-format      Format for saving transcription
                          Choices: [json, txt]
                          Default: json

OUTPUT FORMATS
-------------
JSON Output (--output-format json):
    {
        "engine": "whisper/vosk/coqui",
        "text": "transcribed text",
        "language": "detected language",
        "duration": seconds,
        "confidence": 0.0-1.0,
        "timestamp": "YYYYMMDD_HHMMSS",
        "segments": [
            {
                "start": start_time,
                "end": end_time,
                "text": "segment text",
                "confidence": 0.0-1.0
            },
            ...
        ]
    }

Text Output (--output-format txt):
    Plain transcribed text without metadata

EXAMPLES
--------
1. Basic Whisper Transcription:
   python transcribe.py audio.mp3

2. Whisper with Large Model:
   python transcribe.py audio.mp3 --engine whisper --model large

3. Whisper with Language Specification:
   python transcribe.py audio.mp3 --engine whisper --language en

4. Whisper with Translation:
   python transcribe.py audio.mp3 --engine whisper --translate-to en

5. Vosk Transcription:
   python transcribe.py audio.mp3 --engine vosk --model-path models/vosk-model-en-us

6. Coqui STT Transcription:
   python transcribe.py audio.mp3 --engine coqui \\
       --model-path models/coqui-model.tflite \\
       --scorer-path models/coqui-scorer.scorer

7. Save as Text File:
   python transcribe.py audio.mp3 --output-format txt

NOTES
-----
- The script automatically converts audio to the appropriate format for each engine
- Temporary files are automatically cleaned up after processing
- Output files are saved in the 'output' directory with timestamps
- For best results:
  * Whisper: Good for multiple languages and general accuracy
  * Vosk: Good for real-time and lightweight processing
  * Coqui STT: Good for offline processing with custom models

REQUIREMENTS
------------
- ffmpeg must be installed for audio conversion
- Each engine requires its own models:
  * Whisper: Downloads automatically
  * Vosk: Download manually from https://alphacephei.com/vosk/models
  * Coqui STT: Download manually from https://coqui.ai/models
"""
import os
import argparse
import json
from datetime import datetime

# Import transcribers with error handling
try:
    from whisper_transcriber import transcribe_audio as whisper_transcribe
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: Whisper not available. To use Whisper, install openai-whisper")

try:
    from vosk_transcriber import transcribe_audio as vosk_transcribe
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    print("Warning: Vosk not available. To use Vosk, install vosk")

try:
    from coqui_transcriber import transcribe_audio as coqui_transcribe
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False
    print("Warning: Coqui STT not available. To use Coqui STT, install coqui-stt (pip install coqui-stt)")

from audio_converter import AudioConverter

def format_output(result, engine_name):
    """Format the transcription output consistently."""
    output = {
        "engine": engine_name,
        "text": result.get("text", ""),
        "language": result.get("language", "unknown"),
        "duration": result.get("duration", 0),
        "confidence": result.get("confidence", 0),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Add detailed information if available
    if "detailed" in result:
        output["segments"] = result["detailed"].get("segments", [])
    
    return output

def save_output(output, input_file, output_format="json"):
    """Save the transcription output to a file."""
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    if output_format == "json":
        output_path = os.path.join(output_dir, f"{base_name}_{timestamp}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    elif output_format == "txt":
        output_path = os.path.join(output_dir, f"{base_name}_{timestamp}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output["text"])
    
    print(f"\nOutput saved to: {output_path}")
    return output_path

def transcribe(input_file, engine="whisper", **kwargs):
    """
    Transcribe audio using the specified engine.
    
    Args:
        input_file: Path to the audio file
        engine: Name of the engine to use (whisper/vosk/coqui)
        **kwargs: Additional arguments for the specific engine
    
    Returns:
        dict: Formatted transcription results
    """
    # Check if selected engine is available
    if engine == "whisper" and not WHISPER_AVAILABLE:
        print("Error: Whisper is not installed. Install with: pip install openai-whisper")
        return None
    elif engine == "vosk" and not VOSK_AVAILABLE:
        print("Error: Vosk is not installed. Install with: pip install vosk")
        return None
    elif engine == "coqui" and not COQUI_AVAILABLE:
        print("Error: Coqui STT is not installed. Install with: pip install coqui-stt")
        return None

    # Initialize audio converter
    converter = AudioConverter()
    
    # Convert audio file to appropriate format
    print(f"Converting audio file: {input_file}")
    
    # Define conversion settings for the selected engine
    conversion_settings = {
        'sample_rate': 16000,
        'channels': 1,
        'sample_width': 2
    }
    
    try:
        # Convert audio to WAV format
        print(f"Converting: {input_file}")
        converted_files = converter.convert_to_wav(input_file)
        if not converted_files:
            raise ValueError("Failed to convert audio file")
            
        # Use the standard WAV file for processing
        output_path = converted_files['standard']
        print(f"Created WAV file: {output_path}")
        
        # Process with selected engine
        try:
            if engine == "whisper":
                result = whisper_transcribe(
                    output_path,
                    model_name=kwargs.get("model", "base"),
                    language=kwargs.get("language"),
                    translate_to=kwargs.get("translate_to")
                )
            elif engine == "vosk":
                result = vosk_transcribe(
                    output_path,
                    model_path=kwargs.get("model_path"),
                    language=kwargs.get("language")
                )
            elif engine == "coqui":
                result = coqui_transcribe(
                    output_path,
                    model_path=kwargs.get("model_path"),
                    scorer_path=kwargs.get("scorer_path")
                )
            else:
                raise ValueError(f"Unsupported engine: {engine}")
            
            # Format and return results
            return format_output(result, engine)
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None
            
    except Exception as e:
        print(f"Error converting audio file: {e}")
        return None
    finally:
        # Clean up temporary files
        converter.cleanup_temp_files()

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using various engines.")
    
    # Required arguments
    parser.add_argument("input_file", help="Path to the input audio file")
    
    # Engine selection and options
    available_engines = []
    if WHISPER_AVAILABLE: available_engines.append("whisper")
    if VOSK_AVAILABLE: available_engines.append("vosk")
    if COQUI_AVAILABLE: available_engines.append("coqui")
    
    if not available_engines:
        print("Error: No transcription engines available. Please install at least one of:")
        print("- Whisper: pip install openai-whisper")
        print("- Vosk: pip install vosk")
        if not COQUI_AVAILABLE:
            print("- Coqui STT: pip install coqui-stt (optional)")
        return

    parser.add_argument("--engine", default="whisper" if "whisper" in available_engines else available_engines[0],
                      choices=available_engines,
                      help="Speech recognition engine to use")
    
    # Whisper options
    parser.add_argument("--model", default="base",
                      choices=["tiny", "base", "small", "medium", "large"],
                      help="Model size for Whisper")
    parser.add_argument("--language", help="Source language code (e.g., en, de)")
    parser.add_argument("--translate-to", help="Target language for translation")
    
    # Vosk/Coqui options
    parser.add_argument("--model-path", help="Path to model file for Vosk/Coqui")
    if COQUI_AVAILABLE:
        parser.add_argument("--scorer-path", help="Path to scorer file for Coqui")
    
    # Output options
    parser.add_argument("--output-format", default="json", choices=["json", "txt"],
                      help="Output format for transcription")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return
    
    # Check if Coqui is selected but not available
    if args.engine == "coqui" and not COQUI_AVAILABLE:
        print("Error: Coqui STT is not installed. Please install it with: pip install coqui-stt")
        print("Or choose a different engine (whisper or vosk)")
        return
    
    # Prepare kwargs for the transcribe function
    kwargs = {
        "model": args.model,
        "language": args.language,
        "translate_to": args.translate_to,
        "model_path": args.model_path,
    }
    
    if COQUI_AVAILABLE and args.engine == "coqui":
        kwargs["scorer_path"] = args.scorer_path
    
    # Run transcription
    print(f"\nTranscribing with {args.engine.upper()}...")
    result = transcribe(args.input_file, args.engine, **kwargs)
    
    if result:
        # Save output
        save_output(result, args.input_file, args.output_format)
        
        # Print summary
        print("\nTranscription Summary:")
        print(f"Engine: {result['engine']}")
        print(f"Duration: {result['duration']:.2f} seconds")
        print(f"Language: {result['language']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("\nTranscribed Text:")
        print("-" * 80)
        print(result['text'])
        print("-" * 80)

if __name__ == "__main__":
    main() 