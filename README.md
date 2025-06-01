# VoxStruct

A powerful tool for structured audio transcription with LLM supervision. VoxStruct combines state-of-the-art speech recognition with LLM-powered transcript structuring and improvement.

## Features

- Multiple speech recognition engines support (Whisper, Vosk, Coqui)
- LLM supervision for transcript improvement and structuring
- Support for multiple LLM providers (OpenAI, Anthropic, local models via Ollama)
- Automatic pause detection and segmentation
- Markdown-formatted output
- Detailed metadata tracking

## Installation

You can install VoxStruct using pip:

```bash
pip install voxstruct
```

For development installation:

```bash
git clone https://github.com/gyasis/VoxStruct.git
cd VoxStruct
pip install -e ".[dev]"
```

## Usage

### As a Command Line Tool

```bash
# Basic usage with default settings
voxstruct your_audio.mp3

# Using specific speech recognition engine and model
voxstruct your_audio.mp3 --engine whisper --model base

# Using a specific LLM for supervision
voxstruct your_audio.mp3 --llm-model openai/gpt-4

# Using a local LLM via Ollama
voxstruct your_audio.mp3 --llm-model ollama/mistral

# NEW: Transcribe directly from a YouTube video link
voxstruct 'https://www.youtube.com/watch?v=YOUR_VIDEO_ID'
```

### As a Python Package

```python
from voxstruct import AudioProcessor, LLMSupervisor, TranscriptBuilder

# Process audio file
processor = AudioProcessor("your_audio.mp3")
audio_segment = processor.load_audio()

# Build transcript
builder = TranscriptBuilder()
transcript = builder.build_transcript()

# Improve with LLM
supervisor = LLMSupervisor(model="openai/gpt-4")
improved_transcript = supervisor.validate_and_improve_transcript(transcript)

print(improved_transcript)
```

## Configuration

VoxStruct uses environment variables for API keys and configuration. Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

## Development

1. Clone the repository
2. Install development dependencies: `pip install -e ".[dev]"`
3. Run tests: `pytest`
4. Format code: `black src tests`
5. Sort imports: `isort src tests`
6. Run linter: `flake8 src tests`

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
