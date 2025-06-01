# util/audio_library.py

import os
from .audio_converter import AudioConverter

# Remove all the complex dynamic loading code and replace with simple import

class AudioLibrary:
    def __init__(self, option, **kwargs):
        """
        Initializes the AudioLibrary with the specified speech-to-text engine.
        
        :param option: A string indicating which library to use. Acceptable values:
            "whisper", "deepgram", "google", "azure", "deepspeech", "vosk". # Removed coqui
        :param kwargs: Configuration parameters (e.g., API keys, file paths, etc.)
        """
        self.option = option.lower()
        self.config = kwargs
        self.converter = AudioConverter(
            output_dir=kwargs.get("output_dir", "converted"),
            segment_dir=kwargs.get("segment_dir", "segments"),
            keep_temp=False  # Always clean up temp files
        )
        
        if self.option == "whisper":
            import whisper
            self.model = whisper.load_model(kwargs.get("model_name", "base"))
        
        elif self.option == "deepgram":
            import deepgram
            api_key = kwargs.get("api_key") or os.getenv("DEEPGRAM_API_KEY")
            if not api_key:
                raise ValueError("Deepgram API key is required.")
            self.dg = deepgram.Deepgram(api_key)
        
        elif self.option == "google":
            from google.cloud import speech_v1p1beta1 as speech
            self.speech = speech
            self.client = speech.SpeechClient()
        
        elif self.option == "azure":
            import azure.cognitiveservices.speech as speechsdk
            subscription = kwargs.get("subscription")
            region = kwargs.get("region")
            if not subscription or not region:
                raise ValueError("Azure subscription and region are required.")
            self.speechsdk = speechsdk
            self.speech_config = speechsdk.SpeechConfig(subscription=subscription, region=region)
            self.azure_audio_file = kwargs.get("audio_file")
        
        elif self.option == "deepspeech":
            import deepspeech
            model_file_path = kwargs.get("model_file_path")
            if not model_file_path:
                raise ValueError("Model file path is required for DeepSpeech.")
            self.deepspeech = deepspeech
            self.model = deepspeech.Model(model_file_path)
        
        elif self.option == "vosk":
            from vosk import Model, KaldiRecognizer
            model_path = kwargs.get("model_path")
            if not model_path:
                raise ValueError("Model path is required for Vosk.")
            self.vosk_model = Model(model_path)
            self.KaldiRecognizer = KaldiRecognizer
        
        else:
            # Adjusted error message
            raise ValueError("Unsupported library option. Choose one of: whisper, deepgram, google, azure, deepspeech, vosk.")

    def transcribe(self, audio_data):
        """
        Transcribes the provided audio data using the selected library.
        
        :param audio_data: For API-based libraries, this is expected to be raw audio bytes.
                           For file-based libraries, this should be the file path.
        :return: The transcription result (format varies by library).
        """
        try:
            # First, ensure audio is in the correct format for the engine
            converted_files = self.converter.convert_to_wav(audio_data)
            if not converted_files:
                raise ValueError("Failed to convert audio file")
            
            results = None
            
            # Use the appropriate version for each engine
            if self.option == "whisper":
                audio_file = converted_files["standard"]
                results = self.model.transcribe(audio_file)
                
            elif self.option == "deepgram":
                audio_file = converted_files["standard"]
                # Deepgram processing...
                
            elif self.option == "google":
                audio_file = converted_files["standard"]
                # Google processing...
                
            elif self.option == "azure":
                audio_file = converted_files["standard"]
                # Azure processing...
                
            elif self.option == "deepspeech":
                audio_file = converted_files["deepspeech"]
                # DeepSpeech processing...
                
            elif self.option == "vosk":
                audio_file = converted_files["vosk"]
                # Vosk processing...
                
            else:
                # Adjusted error message
                raise ValueError("Unsupported library option.")
                
            # Clean up temporary files
            self.converter.cleanup_temp_files()
            
            return results
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            self.converter.cleanup_temp_files()
            return None
