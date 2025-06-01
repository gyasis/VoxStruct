"""
LLM Supervisor for intelligent transcript structuring.

This module uses Litellm to support multiple LLM providers for transcript analysis:
- OpenAI models (gpt-4, gpt-3.5-turbo)
- Anthropic models (claude-3, claude-2)
- Local models via Ollama (mistral, llama2)
- And others supported by Litellm
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from litellm import completion
import litellm # Import litellm to set verbose mode

litellm.set_verbose = True # Enable verbose logging for litellm

class LLMSupervisor:
    def __init__(self, api_key: str = None, model: str = "gpt4all/gpt4all-j"):
        """
        Initialize the LLM supervisor with API key and model selection.
        
        Args:
            api_key: API key for the selected provider
            model: Model identifier in format 'provider/model-name'. Examples:
                  - 'openai/gpt-4'
                  - 'anthropic/claude-3'
                  - 'ollama/mistral'
                  - 'gpt4all/gpt4all-j'
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key and not model.startswith(("ollama/", "gpt4all/")):
            raise ValueError("API key is required for cloud LLM providers")
        
        self.model = model
        # Set environment variables for the chosen provider
        if "anthropic" in model:
            os.environ["ANTHROPIC_API_KEY"] = self.api_key
        elif "openai" in model:
            os.environ["OPENAI_API_KEY"] = self.api_key

    def verify_model(self) -> Tuple[bool, str]:
        """
        Verify that the model is accessible and working by requesting a simple introduction.
        
        Returns:
            Tuple[bool, str]: (success, message)
                - success: True if model is working, False otherwise
                - message: Model's response or error message
        """
        try:
            # Simple verification prompt
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant. Please introduce yourself briefly."},
                    {"role": "user", "content": "Please introduce yourself and confirm you can help with transcript formatting."}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            return True, response.choices[0].message.content.strip()
            
        except Exception as e:
            error_msg = f"Model verification failed: {str(e)}"
            # Add provider-specific error handling
            if "API key" in str(e):
                error_msg += "\nPlease check your API key configuration."
            elif "not found" in str(e):
                error_msg += "\nThe specified model may not exist or be available."
            elif "rate limit" in str(e).lower():
                error_msg += "\nRate limit exceeded. Please try again later."
            
            return False, error_msg

    def validate_and_improve_transcript(self, 
                                      raw_transcript: str,
                                      pause_timestamps: Optional[List[float]] = None,
                                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Analyze and structure the transcript using the selected LLM.
        
        Args:
            raw_transcript: The unformatted transcript text
            pause_timestamps: List of timestamps where pauses were detected
            metadata: Additional information about the audio/transcript
            
        Returns:
            str: Properly formatted and structured transcript
        """
        # Prepare the prompt with context and instructions
        prompt = self._build_structuring_prompt(raw_transcript, pause_timestamps, metadata)
        
        try:
            # Call LLM using litellm
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": """You are an expert at structuring transcripts.
Your task is to analyze the transcript and structure it with:
1. Proper paragraphs based on topic changes and natural pauses
2. Lists (ordered/unordered) when appropriate
3. Headers for major topic changes
4. Proper punctuation and formatting
5. Correction of obvious speech-to-text errors
6. Maintain medical/technical accuracy"""},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent formatting
                max_tokens=16000 # Increased max_tokens for longer transcript outputs
            )
            
            # Extract the improved transcript from the response
            improved_transcript = response.choices[0].message.content.strip()
            return improved_transcript
            
        except Exception as e:
            print(f"Error during LLM processing: {str(e)}")
            return raw_transcript  # Return original if processing fails

    def analyze_subject_matter(self, transcript: str) -> Dict[str, Any]:
        """Analyze the subject matter of the transcript to identify key topics."""
        try:
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": """Analyze this transcript and identify:
1. Main topics discussed
2. Key points for each topic
3. Any action items or important conclusions
4. Technical terms or jargon used
5. Overall subject matter category"""},
                    {"role": "user", "content": transcript}
                ],
                temperature=0.3
            )
            
            return {
                "analysis": response.choices[0].message.content,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error during subject matter analysis: {e}")
            return {"error": str(e)}

    def suggest_formatting_improvements(self, 
                                     transcript: str,
                                     current_format: str) -> Dict[str, Any]:
        """Suggest improvements to transcript formatting."""
        try:
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": """Analyze this transcript's formatting and suggest improvements for:
1. Paragraph structure
2. List usage (ordered vs unordered)
3. Header placement and hierarchy
4. Punctuation and spacing
5. Overall readability"""},
                    {"role": "user", "content": f"Current format: {current_format}\n\nTranscript:\n{transcript}"}
                ]
            )
            
            return {
                "suggestions": response.choices[0].message.content,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error generating formatting suggestions: {e}")
            return {"error": str(e)}

    def _build_structuring_prompt(self,
                                raw_transcript: str,
                                pause_timestamps: Optional[List[float]] = None,
                                metadata: Optional[Dict[str, Any]] = None) -> str:
        """Build the prompt for the LLM with all available context."""
        prompt_parts = [
            "Please structure and format this transcript.",
            "\nOriginal transcript:",
            raw_transcript,
            "\nConsider the following:"
        ]
        
        if pause_timestamps:
            prompt_parts.append("\nNatural pauses were detected at these points (in seconds):")
            pause_str = ", ".join(f"{p:.1f}" for p in pause_timestamps)
            prompt_parts.append(pause_str)
        
        if metadata:
            prompt_parts.append("\nAdditional context:")
            if "duration" in metadata:
                prompt_parts.append(f"- Audio duration: {metadata['duration']:.1f} seconds")
            if "language" in metadata:
                prompt_parts.append(f"- Language: {metadata['language']}")
            if "speakers" in metadata:
                prompt_parts.append(f"- Number of speakers: {metadata['speakers']}")
        
        prompt_parts.extend([
            "\nPlease:",
            "1. Split into logical paragraphs based on topic changes and pauses",
            "2. Use ordered lists (1., 2., etc.) for sequential steps or instructions",
            "3. Use unordered lists (•) for related items or examples",
            "4. Add headers (using #, ##, etc.) for major topic changes",
            "5. Ensure proper punctuation and capitalization",
            "6. Preserve any speaker attributions if present",
            "7. Use markdown formatting",
            "8. Identify any words or phrases that switch to a different language from the main transcript language. Wrap these foreign language switches in italics using single asterisks (*foreign phrase*).",
            "\nStructured transcript:"
        ])
        
        return "\n".join(prompt_parts)
