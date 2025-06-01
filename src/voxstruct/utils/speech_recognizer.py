# util/speech_recognizer.py
import time
import requests
import sys

class SpeechRecognizer:
    def __init__(self, engine="whisper", granularity="chunk", **kwargs):
        """
        Initializes the recognizer with the specified engine.
        
        Args:
            engine: The speech recognition engine to use ('whisper', 'vosk')
            granularity: The desired timestamp granularity ('chunk' or 'word').
            **kwargs: Engine-specific configuration (model paths, API keys, etc.)
        """
        self.engine = engine.lower()
        self.granularity = granularity # Store granularity
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
        else:
            # Simplified error message as only whisper/vosk are expected
            raise ValueError(f"Unsupported engine: {self.engine}. Expected 'whisper' or 'vosk'.")
    
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
                
                # Use whisper-timestamped for word granularity, else use standard whisper
                if self.engine == "whisper":
                    if self.granularity == "word":
                        try:
                            import whisper_timestamped
                            # Use whisper_timestamped - it takes the loaded model and audio path
                            # Pass relevant options like language if needed
                            # Note: whisper-timestamped might have slightly different param names/options
                            result = whisper_timestamped.transcribe(
                                self.model, 
                                temp_file.name,
                                language=self.config.get("language"),
                                # Add other whisper-timestamped specific options if desired
                                # e.g., detect_disfluencies=True, vad=True 
                            )
                        except ImportError:
                            print("Error: whisper-timestamped package not found. Install it to use word granularity.", file=sys.stderr)
                            # Fallback or raise error?
                            raise # Re-raise the ImportError for now
                    else: # Chunk granularity
                        # Use standard whisper transcribe
                        result = self.model.transcribe(
                            temp_file.name,
                            language=self.config.get("language"),
                            task=self.config.get("task", "transcribe"),
                            verbose=True 
                        )
                    
                elif self.engine == "vosk":
                    import wave
                    wf = wave.open(temp_file.name, "rb")
                    # Ensure KaldiRecognizer is available (initialized in __init__)
                    if not hasattr(self, 'KaldiRecognizer'):
                         raise RuntimeError("Vosk KaldiRecognizer not initialized.")
                    rec = self.KaldiRecognizer(self.model, wf.getframerate())
                    
                    while True:
                        data = wf.readframes(4000)
                        if len(data) == 0:
                            break
                        rec.AcceptWaveform(data)
                    
                    result = rec.FinalResult()
                else:
                    # This should ideally not be reached if __init__ validates engine
                    raise ValueError(f"Transcription logic not implemented for engine: {self.engine}")
                
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
        """
        Format the engine-specific result into a consistent structure.
        Ensures 'segments' contains word-level detail where available.
        Each segment should ideally have 'word', 'start', 'end', 'confidence'.
        Times are in seconds relative to the start of the processed audio chunk.
        """
        if self.engine == "whisper":
            word_segments = []
            full_text = result.get("text", "")
            overall_confidence = 0
            word_count = 0
            
            # Process output based on granularity used during transcription
            if self.granularity == "word":
                # whisper-timestamped output structure:
                # result['segments'] is a list of segments
                # each segment has a 'words' list: [{'text': ..., 'start': ..., 'end': ..., 'confidence': ...}]
                whisper_segments = result.get("segments", [])
                for segment in whisper_segments:
                    segment_words = segment.get("words", [])
                    if isinstance(segment_words, list):
                        for word_info in segment_words:
                             # Basic check for essential keys
                            if word_info.get("text") is not None and word_info.get("start") is not None and word_info.get("end") is not None:
                                word_segments.append({
                                    "word": word_info.get("text", "").strip(),
                                    "start": word_info.get("start"), # In seconds relative to chunk
                                    "end": word_info.get("end"),
                                    "confidence": word_info.get("confidence") 
                                })
                                conf = word_info.get("confidence")
                                if conf is not None:
                                    overall_confidence += conf
                                    word_count += 1
            
            else: # Chunk granularity - use the previous logic for standard whisper output
                # This part remains the same as before for processing standard whisper output
                found_word_level_data = False # Flag to track if we got word timings
                whisper_segments = result.get("segments", [])
                
                # Iterate through all segments from Whisper to find word data
                for segment in whisper_segments:
                    segment_words = segment.get("words") # Check for 'words' key
                    # Ensure segment_words is a list and contains dictionaries (word info)
                    if isinstance(segment_words, list) and segment_words and isinstance(segment_words[0], dict):
                        found_word_level_data = True # Mark that we found actual word data
                        for word_info in segment_words:
                            # Basic check for essential keys
                            if word_info.get("word") is not None and word_info.get("start") is not None and word_info.get("end") is not None:
                                word_segments.append({
                                    "word": word_info.get("word", "").strip(),
                                    "start": word_info.get("start"), # In seconds relative to chunk
                                    "end": word_info.get("end"),
                                    "confidence": word_info.get("probability", word_info.get("confidence")) # Handle diff key names
                                })
                                conf = word_info.get("probability", word_info.get("confidence"))
                                if conf is not None:
                                    overall_confidence += conf
                                    word_count += 1
                
                # Fallback logic for chunk granularity (unchanged from previous version)
                if not found_word_level_data and whisper_segments and whisper_segments[0].get("start") is not None:
                     print("Warning: Word-level timestamps not found in Whisper output, using phrase segments.", file=sys.stderr)
                     word_segments = [
                         {
                             "word": seg.get("text", "").strip(), 
                             "start": seg.get("start"), 
                             "end": seg.get("end"),
                             "confidence": seg.get("confidence", seg.get("avg_logprob"))
                         } for seg in whisper_segments
                     ]
                     overall_confidence = sum(s.get("confidence", 0) for s in word_segments if s.get("confidence"))
                     word_count = len(word_segments)
                elif not word_segments:
                     print("Warning: No segments found in Whisper output.", file=sys.stderr)
                     word_segments = [{"word": full_text, "start": None, "end": None, "confidence": None}]
                     word_count = 1 # Avoid division by zero

            # Calculate average confidence (common logic)
            avg_confidence = (overall_confidence / word_count) if word_count > 0 else 0

            return {
                "text": full_text,
                "language": result.get("language", "unknown"),
                "segments": word_segments, # List of word dicts (now truly word-level if requested)
                "confidence": avg_confidence 
            }
            
        elif self.engine == "vosk":
            import json
            try:
                res = json.loads(result)
                word_segments = []
                full_text = res.get("text", "")
                
                # Vosk often returns word timings in a 'result' list
                if "result" in res and isinstance(res["result"], list):
                    word_segments = [
                        {
                            "word": word_info.get("word", "").strip(),
                            "start": word_info.get("start"),
                            "end": word_info.get("end"),
                            "confidence": word_info.get("conf") 
                        } 
                        for word_info in res["result"] if "start" in word_info
                    ]
                    # Recalculate full_text from words if possible
                    if word_segments:
                       full_text = " ".join(w["word"] for w in word_segments)

                # Fallback if 'result' isn't present or structured as expected
                else:
                    word_segments = [{"word": full_text, "start": None, "end": None, "confidence": None}]

                # Vosk might provide overall confidence differently, TBD if needed
                avg_confidence = sum(w.get("confidence", 0) for w in word_segments if w.get("confidence")) / len(word_segments) if word_segments else 0

                return {
                    "text": full_text,
                    "confidence": avg_confidence,
                    "segments": word_segments
                }
            except json.JSONDecodeError:
                 print("Vosk result was not valid JSON")
                 return {"text": "", "confidence": 0, "segments": []}

        else: # Fallback for unknown engines (should not happen)
             return {
                 "text": str(result), # Attempt to convert result to string
                 "confidence": None,
                 "segments": [{"word": str(result), "start": None, "end": None, "confidence": None}]
             }

    def dummy_transcribe_audio(self, audio_segment):
        """
        Dummy transcription function for demonstration purposes.
        Replace this with transcribe_audio() when integrating Deepgram.
        """
        time.sleep(0.5)  # simulate processing delay
        return "this is a dummy transcription of the audio chunk"
