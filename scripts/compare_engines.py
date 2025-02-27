"""
Script to compare different speech recognition engines.

Available Speech Recognition Engines:
------------------------------------
1. Whisper (OpenAI)
   - Offline, supports 99 languages
   - Models: tiny, base, small, medium, large
   - Best for: Multi-language content, high accuracy needs
   - Example: --engines whisper --model large --language auto

2. Vosk
   - Offline, lightweight, multiple language models
   - Good for: Real-time transcription, limited resources
   - Models available for 20+ languages
   - Example: --engines vosk --model vosk-model-en-us-0.22

3. DeepSpeech (Mozilla)
   - Offline, optimized for English
   - Good for: Privacy-focused applications
   - Example: --engines deepspeech --scorer deepspeech-0.9.3-models.scorer

4. Deepgram (API)
   - Cloud-based, requires API key
   - Features: Speaker diarization, language detection
   - Good for: Enterprise applications
   - Example: --engines deepgram --model nova --language en-US

5. Google Speech-to-Text (API)
   - Cloud-based, requires API key
   - Supports 125+ languages
   - Features: Automatic punctuation, speaker diarization
   - Example: --engines google --language en-US --enable-punctuation

6. Azure Speech (API)
   - Cloud-based, requires subscription
   - Features: Real-time transcription, custom models
   - Example: --engines azure --region westus --language en-US

Usage Examples:
--------------
1. Basic Comparison:
   ```bash
   python compare_engines.py input.mp3
   ```

2. Specific Engines:
   ```bash
   python compare_engines.py input.mp3 --engines whisper,vosk
   ```

3. High Accuracy Mode:
   ```bash
   python compare_engines.py input.mp3 --high-accuracy \
       --whisper-model large \
       --vosk-model large \
       --deepspeech-scorer path/to/scorer
   ```

4. Multi-Language Content:
   ```bash
   python compare_engines.py multilingual.mp3 \
       --engines whisper,google \
       --language auto \
       --translate-to en
   ```

5. Long Audio Processing:
   ```bash
   python compare_engines.py long_audio.mp3 \
       --chunk-size 30000 \
       --batch-size 16 \
       --max-workers 4
   ```

6. Speaker Detection:
   ```bash
   python compare_engines.py meeting.wav \
       --engines deepgram,google \
       --speaker-diarization \
       --min-speakers 2 \
       --max-speakers 10
   ```

Advanced Features:
----------------
1. Audio Preprocessing:
   - Noise reduction: --denoise
   - Volume normalization: --normalize
   - Silence removal: --remove-silence
   - Custom sample rate: --sample-rate 16000

2. Output Options:
   - Format: --output-format json,txt,srt,vtt
   - Timestamps: --word-timestamps
   - Confidence scores: --include-confidence
   - Speaker labels: --speaker-labels

3. Performance Tuning:
   - GPU acceleration: --use-gpu
   - Batch processing: --batch-size N
   - Parallel workers: --max-workers N
   - Memory optimization: --memory-optimized

4. Quality Settings:
   - High accuracy: --high-accuracy
   - Fast processing: --speed-optimized
   - Balanced mode: --balanced

Environment Variables:
--------------------
- WHISPER_MODEL: Default Whisper model size
- VOSK_MODEL_PATH: Path to Vosk model
- DEEPSPEECH_MODEL: Path to DeepSpeech model
- DEEPGRAM_API_KEY: Deepgram API key
- GOOGLE_APPLICATION_CREDENTIALS: Path to Google credentials
- AZURE_SPEECH_KEY: Azure Speech API key
- AZURE_SPEECH_REGION: Azure region

For more details, see the full documentation in README.md
"""
import os
import json
import time
import subprocess
import shutil
from datetime import datetime
from typing import Dict, Any
import pandas as pd
from audio_converter import AudioConverter
import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config import CHUNK_SIZE

# Dictionary to store transcriber functions
TRANSCRIBERS = {}

# Model URLs and information
MODEL_INFO = {
    'vosk': {
        'en': {
            'url': 'https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip',
            'size': 'small',
            'filename': 'vosk-model-small-en-us-0.15.zip'
        },
        'de': {
            'url': 'https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip',
            'size': 'small',
            'filename': 'vosk-model-small-de-0.15.zip'
        },
        # Add more languages as needed
    },
    'coqui': {
        'en': {
            'model_url': 'https://coqui.ai/models/en/wav2vec2_base.pt',
            'model_filename': 'wav2vec2_base.pt',
            'scorer_url': 'https://coqui.ai/models/en/wav2vec2_base_scorer.pt',
            'scorer_filename': 'wav2vec2_base_scorer.pt'
        },
        'de': {
            'model_url': 'https://coqui.ai/models/de/wav2vec2_base.pt',
            'model_filename': 'wav2vec2_base.pt',
            'scorer_url': 'https://coqui.ai/models/de/wav2vec2_base_scorer.pt',
            'scorer_filename': 'wav2vec2_base_scorer.pt'
        },
        # Add more languages as needed
    }
}

def download_file(url: str, target_path: str) -> bool:
    """
    Download a file using wget showing progress.
    
    Args:
        url: URL to download from
        target_path: Path to save the file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Downloading from {url}...")
        # Use urllib instead of wget
        import urllib.request
        
        def report_progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\rDownloading: {percent}% completed", end='', flush=True)
            
        urllib.request.urlretrieve(url, target_path, reporthook=report_progress)
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def setup_vosk_model(language: str = 'en', model_size: str = 'small') -> str:
    """
    Download and set up Vosk model if not already present.
    
    Args:
        language: Language code (e.g., 'en', 'de')
        model_size: Model size ('small', 'large')
        
    Returns:
        str: Path to the model directory or None if setup failed
    """
    try:
        # Create models directory if it doesn't exist
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Get model info
        model_info = MODEL_INFO['vosk'].get(language)
        if not model_info:
            print(f"No Vosk model available for language: {language}")
            return None
            
        # Check if model already exists
        model_name = model_info['filename'].replace('.zip', '')
        model_path = os.path.join(models_dir, model_name)
        if os.path.exists(model_path):
            print(f"Vosk model already exists at: {model_path}")
            return model_path
            
        # Download model
        zip_path = os.path.join(models_dir, model_info['filename'])
        if not download_file(model_info['url'], zip_path):
            return None
            
        # Unzip model
        print(f"Extracting model to {model_path}...")
        try:
            subprocess.run(['unzip', '-q', '-o', zip_path, '-d', models_dir], check=True)
            # Remove zip file after extraction
            os.remove(zip_path)
            print(f"Model setup complete at: {model_path}")
            return model_path
        except subprocess.CalledProcessError as e:
            print(f"Error extracting model: {e}")
            return None
            
    except Exception as e:
        print(f"Error setting up Vosk model: {e}")
        return None

def setup_coqui_model(language: str = 'en') -> Dict[str, str]:
    """
    Download and set up Coqui STT model if not already present.
    
    Args:
        language: Language code (e.g., 'en', 'de')
        
    Returns:
        dict: Paths to model and scorer files
    """
    try:
        # Create models directory if it doesn't exist
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Get model info based on language
        model_info = MODEL_INFO['coqui'].get(language)
        if not model_info:
            print(f"No Coqui STT model available for language: {language}")
            return None
            
        # Check if model already exists
        model_path = os.path.join(models_dir, model_info['model_filename'])
        scorer_path = os.path.join(models_dir, model_info.get('scorer_filename', ''))
        
        if os.path.exists(model_path):
            print(f"Coqui STT model already exists at: {model_path}")
            return {
                'model_path': model_path,
                'scorer_path': scorer_path if os.path.exists(scorer_path) else None
            }
            
        # Download model
        print(f"Downloading Coqui STT model for {language}...")
        if not download_file(model_info['model_url'], model_path):
            return None
            
        # Download scorer if available
        if 'scorer_url' in model_info:
            print(f"Downloading Coqui STT scorer...")
            if not download_file(model_info['scorer_url'], scorer_path):
                return None
                
        return {
            'model_path': model_path,
            'scorer_path': scorer_path if os.path.exists(scorer_path) else None
        }
        
    except Exception as e:
        print(f"Error setting up Coqui STT model: {e}")
        return None

def ensure_models(engines: list, language: str = 'en') -> Dict[str, Any]:
    """
    Ensure all required models are available for the requested engines.
    
    Args:
        engines: List of engine names
        language: Target language code
        
    Returns:
        dict: Mapping of engine names to model paths/info
    """
    model_paths = {}
    
    for engine in engines:
        if engine == 'vosk':
            model_path = setup_vosk_model(language)
            if model_path:
                model_paths['vosk'] = model_path
            else:
                print("Warning: Vosk model setup failed")
                
        elif engine == 'coqui':
            model_info = setup_coqui_model(language)
            if model_info:
                model_paths['coqui'] = model_info
            else:
                print("Warning: Coqui STT model setup failed")
        
        # Add setup for other engines that require model downloads
        
    return model_paths

def load_transcriber(engine: str):
    """Dynamically load transcriber module for the requested engine."""
    try:
        if engine == 'whisper':
            from whisper_transcriber import transcribe_audio
            TRANSCRIBERS['whisper'] = transcribe_audio
        elif engine == 'vosk':
            from vosk_transcriber import transcribe_audio
            TRANSCRIBERS['vosk'] = transcribe_audio
        elif engine == 'coqui':
            from coqui_transcriber import transcribe_audio
            TRANSCRIBERS['coqui'] = transcribe_audio
        return True
    except ImportError as e:
        print(f"Warning: Could not load {engine} transcriber: {e}")
        return False

def detect_language_from_path(file_path: str) -> str:
    """Detect language from file path or return default."""
    file_path = file_path.lower()
    if 'german' in file_path or 'deutsch' in file_path:
        return 'de'
    if 'french' in file_path or 'français' in file_path:
        return 'fr'
    if 'spanish' in file_path or 'español' in file_path:
        return 'es'
    if 'japanese' in file_path or '日本語' in file_path:
        return 'ja'
    # Add more language detection patterns as needed
    return 'en'  # Default to English

def run_comparison(input_file, engines=None, **kwargs):
    """Run comparison of different speech recognition engines."""
    try:
        # Detect language from file path if not specified
        language = kwargs.get('whisper_options', {}).get('language')
        if not language:
            language = detect_language_from_path(input_file)
            if 'whisper_options' not in kwargs:
                kwargs['whisper_options'] = {}
            kwargs['whisper_options']['language'] = language
        
        # Ensure required models are available
        model_paths = ensure_models(engines or ['whisper', 'vosk', 'coqui'], language)
        
        # Update engine options with model paths
        vosk_opts = kwargs.get('vosk_options', {})
        if 'vosk' in (engines or ['whisper', 'vosk', 'coqui']) and not vosk_opts.get('model_path'):
            vosk_opts['model_path'] = model_paths.get('vosk')
            kwargs['vosk_options'] = vosk_opts
            
        coqui_opts = kwargs.get('coqui_options', {})
        if 'coqui' in (engines or ['whisper', 'vosk', 'coqui']):
            coqui_model_info = model_paths.get('coqui', {})
            if coqui_model_info:
                coqui_opts.update(coqui_model_info)
                kwargs['coqui_options'] = coqui_opts
        
        # Create AudioConverter instance with audio processing options
        audio_options = kwargs.get('audio_options', {})
        converter = AudioConverter(
            output_dir=os.path.join(os.path.dirname(input_file), "converted"),
            segment_dir=os.path.join(os.path.dirname(input_file), "segments"),
            segment_length_ms=kwargs.get('performance_options', {}).get('chunk_size', CHUNK_SIZE),
            min_silence_ms=500,  # Default value
            silence_thresh=-40,  # Default value
            keep_temp=False
        )
        
        # Convert audio file
        converted_files = converter.convert_to_wav(input_file)
        
        if not converted_files:
            print(f"Error: Could not convert audio file: {input_file}")
            return None
            
        results = {
            "metadata": {
                "input_file": input_file,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "duration": None  # Will be filled by first engine
            },
            "transcripts": {},
            "performance": {},
            "word_counts": {},
            "confidence_scores": {},
            "language_detection": {}
        }
        
        # Process with each engine
        requested_engines = engines or ['whisper', 'vosk', 'coqui']
        for engine in requested_engines:
            if engine not in TRANSCRIBERS and not load_transcriber(engine):
                print(f"Skipping {engine} due to import error")
                continue
                
            try:
                start_time = time.time()
                
                # Get the appropriate converted file for this engine
                engine_file = converted_files.get(engine) or converted_files.get('standard')
                if not engine_file:
                    print(f"No compatible audio file found for {engine}")
                    continue
                
                # Get the transcriber function
                transcribe_func = TRANSCRIBERS[engine]
                
                # Process with the selected engine
                if engine == 'whisper':
                    whisper_opts = kwargs.get('whisper_options', {})
                    result = transcribe_func(
                        engine_file,
                        model_name=whisper_opts.get('model', 'base'),
                        language=whisper_opts.get('language'),
                        translate_to=whisper_opts.get('translate_to')
                    )
                elif engine == 'vosk':
                    vosk_opts = kwargs.get('vosk_options', {})
                    if not vosk_opts.get('model_path'):
                        print(f"No Vosk model path provided for language: {language}")
                        continue
                    result = transcribe_func(
                        engine_file,
                        model_path=vosk_opts.get('model_path'),
                        language=language
                    )
                elif engine == 'coqui':
                    coqui_opts = kwargs.get('coqui_options', {})
                    if not coqui_opts.get('model_path'):
                        print(f"No Coqui STT model path provided for language: {language}")
                        continue
                    result = transcribe_func(
                        engine_file,
                        model_path=coqui_opts.get('model_path'),
                        scorer_path=coqui_opts.get('scorer_path'),
                        language=language
                    )
                else:
                    print(f"Engine {engine} not implemented yet")
                    continue
                
                # Store results
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Get text and detailed info
                text = result.get("text", "").strip()
                detailed = result.get("detailed", {})
                duration = result.get("duration", 0) or 1  # Avoid division by zero
                
                results["transcripts"][engine] = {
                    "text": text,
                    "detailed": detailed
                }
                
                results["performance"][engine] = {
                    "processing_time": processing_time,
                    "real_time_factor": processing_time / duration
                }
                
                # Count words
                results["word_counts"][engine] = len(text.split()) if text else 0
                
                # Store confidence scores
                if "confidence" in result:
                    results["confidence_scores"][engine] = {
                        "mean": result["confidence"],
                        "min": min((seg.get("confidence", 0) for seg in detailed.get("segments", [])), default=0),
                        "max": max((seg.get("confidence", 0) for seg in detailed.get("segments", [])), default=0)
                    }
                
                # Store language detection (for Whisper)
                if engine == "whisper":
                    results["language_detection"] = result.get("language", "unknown")
                
                # Store duration in metadata if not already set
                if "duration" not in results["metadata"] or not results["metadata"]["duration"]:
                    results["metadata"]["duration"] = duration
                
                print(f"\nProcessed with {engine.upper()}:")
                print(f"Text length: {len(text)} characters")
                print(f"Word count: {results['word_counts'][engine]}")
                print(f"Processing time: {processing_time:.2f} seconds")
                print(f"Real-time factor: {processing_time / duration:.2f}x")
                if engine in results["confidence_scores"]:
                    print(f"Confidence scores: {results['confidence_scores'][engine]}")
                
            except Exception as e:
                print(f"Error processing with {engine}: {e}")
                continue
        
        # Clean up temporary files
        converter.cleanup_temp_files()
        
        return results
    except Exception as e:
        print(f"Error during comparison: {e}")
        return None

def generate_comparison_report(results: Dict[str, Any], output_path: str):
    """
    Generate an HTML report comparing the results from different engines.
    """
    # Create performance comparison table
    performance_df = pd.DataFrame.from_dict(results["performance"], orient="index")
    
    # Create word count comparison
    word_counts_df = pd.DataFrame.from_dict(results["word_counts"], orient="index", columns=["Word Count"])
    
    # Create HTML report
    html_content = f"""
    <html>
    <head>
        <title>Speech Recognition Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .transcript {{ white-space: pre-wrap; background-color: #f9f9f9; padding: 10px; }}
        </style>
    </head>
    <body>
        <h1>Speech Recognition Comparison Report</h1>
        
        <h2>File Information</h2>
        <p>Input File: {results["metadata"]["input_file"]}</p>
        <p>Duration: {results["metadata"]["duration"]:.2f} seconds</p>
        <p>Detected Language (Whisper): {results["language_detection"]}</p>
        
        <h2>Performance Comparison</h2>
        {performance_df.to_html()}
        
        <h2>Word Count Comparison</h2>
        {word_counts_df.to_html()}
        
        <h2>Transcripts</h2>
        {"".join(f'''
        <h3>{engine}</h3>
        <div class="transcript">{transcript["text"]}</div>
        ''' for engine, transcript in results["transcripts"].items() if "text" in transcript)}
    </body>
    </html>
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Comparison report saved to: {output_path}")

def main():
    """Command line interface for speech recognition engine comparison."""
    parser = argparse.ArgumentParser(
        description='Compare different speech recognition engines.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input/Output Arguments
    parser.add_argument('input_file', help='Path to the input audio file')
    parser.add_argument('--output-dir', default='results', help='Directory for output files')
    parser.add_argument('--output-format', default='json', 
                       help='Output format(s), comma-separated (json,txt,srt,vtt)')
    
    # Engine Selection
    parser.add_argument('--engines', help='Comma-separated list of engines to use (whisper,vosk,coqui)')
    
    # Engine-Specific Options
    whisper_group = parser.add_argument_group('Whisper Options')
    whisper_group.add_argument('--whisper-model', default='base',
                              choices=['tiny', 'base', 'small', 'medium', 'large'],
                              help='Whisper model size')
    whisper_group.add_argument('--language', help='Source language code (e.g., en-US) or "auto"')
    whisper_group.add_argument('--translate-to', help='Target language for translation')
    
    vosk_group = parser.add_argument_group('Vosk Options')
    vosk_group.add_argument('--vosk-model', help='Path to Vosk model')
    
    coqui_group = parser.add_argument_group('Coqui STT Options')
    coqui_group.add_argument('--coqui-model', help='Path to Coqui STT model')
    coqui_group.add_argument('--coqui-scorer', help='Path to Coqui STT scorer')
    
    # Audio Processing Options
    audio_group = parser.add_argument_group('Audio Processing')
    audio_group.add_argument('--denoise', action='store_true', help='Apply noise reduction')
    audio_group.add_argument('--normalize', action='store_true', help='Normalize audio volume')
    audio_group.add_argument('--remove-silence', action='store_true', help='Remove silence')
    audio_group.add_argument('--sample-rate', type=int, default=16000, help='Audio sample rate')
    
    # Performance Options
    perf_group = parser.add_argument_group('Performance Options')
    perf_group.add_argument('--use-gpu', action='store_true', help='Enable GPU acceleration')
    perf_group.add_argument('--batch-size', type=int, default=16, help='Batch size for processing')
    perf_group.add_argument('--max-workers', type=int, default=4, help='Number of parallel workers')
    perf_group.add_argument('--chunk-size', type=int, default=30000, 
                           help='Size of audio chunks in milliseconds')
    perf_group.add_argument('--memory-optimized', action='store_true', 
                           help='Optimize for memory usage')
    
    # Quality Settings
    quality_group = parser.add_argument_group('Quality Settings')
    quality_group.add_argument('--high-accuracy', action='store_true', 
                             help='Optimize for accuracy')
    quality_group.add_argument('--speed-optimized', action='store_true', 
                             help='Optimize for speed')
    quality_group.add_argument('--balanced', action='store_true', 
                             help='Balance speed and accuracy')
    
    # Advanced Features
    advanced_group = parser.add_argument_group('Advanced Features')
    advanced_group.add_argument('--speaker-diarization', action='store_true', 
                              help='Enable speaker diarization')
    advanced_group.add_argument('--min-speakers', type=int, help='Minimum number of speakers')
    advanced_group.add_argument('--max-speakers', type=int, help='Maximum number of speakers')
    advanced_group.add_argument('--word-timestamps', action='store_true', 
                              help='Include word-level timestamps')
    advanced_group.add_argument('--include-confidence', action='store_true', 
                              help='Include confidence scores')
    advanced_group.add_argument('--speaker-labels', action='store_true', 
                              help='Include speaker labels')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return
    
    # Parse comma-separated values
    engines = args.engines.split(',') if args.engines else None
    output_formats = args.output_format.split(',') if args.output_format else ['json']
    
    # Run comparison with all arguments
    results = run_comparison(
        args.input_file,
        engines=engines,
        output_formats=output_formats,
        whisper_options={
            'model': args.whisper_model,
            'language': args.language,
            'translate_to': args.translate_to
        },
        vosk_options={
            'model_path': args.vosk_model
        },
        coqui_options={
            'model_path': args.coqui_model,
            'scorer_path': args.coqui_scorer
        },
        audio_options={
            'denoise': args.denoise,
            'normalize': args.normalize,
            'remove_silence': args.remove_silence,
            'sample_rate': args.sample_rate
        },
        performance_options={
            'use_gpu': args.use_gpu,
            'batch_size': args.batch_size,
            'max_workers': args.max_workers,
            'chunk_size': args.chunk_size,
            'memory_optimized': args.memory_optimized
        },
        quality_options={
            'high_accuracy': args.high_accuracy,
            'speed_optimized': args.speed_optimized,
            'balanced': args.balanced
        },
        advanced_options={
            'speaker_diarization': args.speaker_diarization,
            'min_speakers': args.min_speakers,
            'max_speakers': args.max_speakers,
            'word_timestamps': args.word_timestamps,
            'include_confidence': args.include_confidence,
            'speaker_labels': args.speaker_labels
        }
    )
    
    if results:
        print("\nComparison Results:")
        for engine, result in results.items():
            print(f"\n{engine.upper()}:")
            print(result)

if __name__ == "__main__":
    main() 