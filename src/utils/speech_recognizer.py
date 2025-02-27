# util/speech_recognizer.py
import time
import requests

class SpeechRecognizer:
    def __init__(self, engine="whisper", **kwargs):
        """
        Initializes the recognizer with the specified engine.
        
        Args:
            engine: The speech recognition engine to use ('whisper', 'vosk', 'coqui')
            **kwargs: Engine-specific configuration (model paths, API keys, etc.)
        """
        self.engine = engine.lower()
        self.config = kwargs
        
        # Initialize the selected engine
        if self.engine == "whisper":
            import whisper
            self.model = whisper.load_model(kwargs.get("model_name", "base"))
        elif self.engine == "vosk":
            from vosk import Model, KaldiRecognizer
            model_path = kwargs.get("model_path")
            if not model_path:
                raise ValueError("Model path required for Vosk")
            self.model = Model(model_path)
            self.KaldiRecognizer = KaldiRecognizer
        elif self.engine == "coqui":
            from stt import Model
            model_path = kwargs.get("model_path")
            if not model_path:
                raise ValueError("Model path required for Coqui STT")
            self.model = Model(model_path)
            if kwargs.get("scorer_path"):
                self.model.enableExternalScorer(kwargs["scorer_path"])
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")
    
    def transcribe_audio(self, audio_segment):
        """
        Transcribes an audio segment using the selected engine.
        
        Args:
            audio_segment: A pydub AudioSegment object
            
        Returns:
            dict: Transcription results including text, timing, and confidence
        """
        try:
            # Export audio segment to temporary WAV file
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                audio_segment.export(temp_file.name, format="wav")
                
                if self.engine == "whisper":
                    result = self.model.transcribe(
                        temp_file.name,
                        language=self.config.get("language"),
                        task=self.config.get("task", "transcribe"),
                        verbose=True
                    )
                    
                elif self.engine == "vosk":
                    import wave
                    wf = wave.open(temp_file.name, "rb")
                    rec = self.KaldiRecognizer(self.model, wf.getframerate())
                    
                    while True:
                        data = wf.readframes(4000)
                        if len(data) == 0:
                            break
                        rec.AcceptWaveform(data)
                    
                    result = rec.FinalResult()
                    
                elif self.engine == "coqui":
                    import wave
                    wf = wave.open(temp_file.name, "rb")
                    audio = wf.readframes(wf.getnframes())
                    text = self.model.stt(audio)
                    result = {"text": text}
                
            # Clean up temp file
            os.unlink(temp_file.name)
            
            # Format result consistently
            return self._format_result(result)
            
        except Exception as e:
            print(f"Error in {self.engine} transcription:", e)
            return {
                "text": "",
                "error": str(e),
                "confidence": 0,
                "segments": []
            }
    
    def _format_result(self, result):
        """Format the engine-specific result into a consistent structure."""
        if self.engine == "whisper":
            return {
                "text": result["text"],
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", []),
                "confidence": sum(s.get("confidence", 0) for s in result.get("segments", [])) / 
                            len(result.get("segments", [1])) if result.get("segments") else 0
            }
        elif self.engine == "vosk":
            import json
            result = json.loads(result)
            return {
                "text": result.get("text", ""),
                "confidence": result.get("confidence", 0),
                "segments": [{"text": result.get("text", "")}]
            }
        elif self.engine == "coqui":
            return {
                "text": result["text"],
                "confidence": 1.0,  # Coqui doesn't provide confidence scores
                "segments": [{"text": result["text"]}]
            }
    
    def dummy_transcribe_audio(self, audio_segment):
        """
        Dummy transcription function for demonstration purposes.
        Replace this with transcribe_audio() when integrating Deepgram.
        """
        time.sleep(0.5)  # simulate processing delay
        return "this is a dummy transcription of the audio chunk"
