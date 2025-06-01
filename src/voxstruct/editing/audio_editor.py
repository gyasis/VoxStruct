import re
import os
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from pydub import AudioSegment
import string # For punctuation stripping
from collections import defaultdict

class AudioEdit:
    """Provides methods for editing and analyzing audio transcripts."""

    def __init__(self):
        pass  # No specific initialization needed for now

    def extract_italicized_text(self, markdown_file_path: str) -> List[str]:
        """
        Extracts text wrapped in single asterisks (*) from a markdown file.

        If the extracted text contains sentence-ending punctuation (.!?),
        it attempts to split it into individual sentences. Otherwise, it's
        treated as a single word or phrase.

        Args:
            markdown_file_path: The path to the markdown file.

        Returns:
            A list of extracted words, phrases, or sentences.
        """
        if not os.path.exists(markdown_file_path):
            raise FileNotFoundError(f"Markdown file not found: {markdown_file_path}")

        extracted_items: List[str] = []
        try:
            with open(markdown_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Regex to find text enclosed in single asterisks (non-greedy)
            # It avoids matching across paragraphs by not matching double newlines
            # It also avoids matching '**bold**' or '***bold-italic***'
            # Matches *italic* but not **bold** or ***bolditalic*** or **** etc.
            # Matches content between single asterisks.
            pattern = r'(?<!\*)\*([^*]+)\*(?!\*)'
            matches = re.finditer(pattern, content)

            for match in matches:
                text = match.group(1).strip()
                # Basic check for sentence-like structure
                if re.search(r'[.!?]\s*$', text) and len(text.split()) > 3: # Check for ending punctuation and more than 3 words as a heuristic for sentences
                    # Attempt to split into sentences (basic approach)
                    # Splits on '.', '!', '?' followed by optional whitespace and a capital letter or end of string
                    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z]|$)', text)
                    extracted_items.extend([s.strip() for s in sentences if s.strip()])
                else:
                    # Treat as a single word/phrase
                    extracted_items.append(text)

        except Exception as e:
            print(f"Error processing file {markdown_file_path}: {e}")
            # Optionally re-raise or return empty list depending on desired error handling
            raise

        return extracted_items

    def _sanitize_filename(self, text: str, max_length: int = 50) -> str:
        """Sanitizes text to be used as a filename."""
        # Remove punctuation
        sanitized = text.translate(str.maketrans('', '', string.punctuation))
        # Replace spaces with underscores
        sanitized = sanitized.replace(' ', '_')
        # Keep only alphanumeric and underscore
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '', sanitized)
        # Truncate to max_length
        return sanitized[:max_length]

    def extract_audio_snippets(
        self,
        target_phrases_from_markdown: List[str],
        timestamp_file_path: str,
        original_audio_path: str,
        output_dir: str,
        output_format: str = "wav"
    ) -> Dict[str, str]:
        """
        Extracts audio snippets corresponding to target items using timestamps.
        - Single words are extracted globally based on the timestamp file.
        - Multi-word phrases are extracted sequentially based on their order
          in the target_phrases_from_markdown list.

        Args:
            target_phrases_from_markdown: List of words/phrases from markdown italics.
            timestamp_file_path: Path to the JSON timestamp file.
            original_audio_path: Path to the original audio file.
            output_dir: Directory to save extracted snippets.
            output_format: Format for saving snippets (e.g., "wav", "mp3").

        Returns:
            A dictionary mapping extracted item identifiers (e.g., "word_1", "phrase_1")
            to their saved snippet file paths.
        """
        # --- 1. Input Validation and Loading --- 
        if not os.path.exists(timestamp_file_path):
            print(f"Error: Timestamp file not found: {timestamp_file_path}")
            return {}
        if not os.path.exists(original_audio_path):
            print(f"Error: Original audio file not found: {original_audio_path}")
            return {}

        try:
            with open(timestamp_file_path, 'r', encoding='utf-8') as f:
                original_timestamps: List[Dict[str, Any]] = json.load(f)
        except Exception as e:
            print(f"Error loading/parsing timestamp file {timestamp_file_path}: {e}")
            return {}

        try:
            audio = AudioSegment.from_file(original_audio_path)
            print(f"Loaded original audio: {original_audio_path}")
        except Exception as e:
            print(f"Error loading original audio file {original_audio_path}: {e}")
            return {}

        os.makedirs(output_dir, exist_ok=True)
        extracted_snippets: Dict[str, str] = {}

        # --- 2. Preprocessing and Categorization --- 
        translator = str.maketrans('', '', string.punctuation)
        
        def clean_word(word): 
            if not isinstance(word, str): return ""
            return word.translate(translator).lower()

        cleaned_timestamps = [
            {
                **ts,
                'cleaned_word': clean_word(ts.get('word'))
            } for ts in original_timestamps
        ]

        unique_single_words: Set[str] = set()
        multi_word_phrases_targets: List[Tuple[str, List[str]]] = [] # (original_text, [cleaned_words])

        for phrase_text in target_phrases_from_markdown:
            if not isinstance(phrase_text, str) or len(phrase_text.strip()) <= 1:
                continue # Skip invalid or single chars
            
            cleaned_phrase_words = [clean_word(w) for w in phrase_text.split() if clean_word(w)]
            
            if len(cleaned_phrase_words) == 1:
                unique_single_words.add(cleaned_phrase_words[0])
            elif len(cleaned_phrase_words) > 1:
                 multi_word_phrases_targets.append((phrase_text, cleaned_phrase_words))

        print(f"Identified {len(unique_single_words)} unique single words and {len(multi_word_phrases_targets)} multi-word phrase targets.")

        # --- 3. Process Single Words (Globally) --- 
        print("\nProcessing single words globally...")
        single_word_counters = defaultdict(int)
        for i, ts_data in enumerate(cleaned_timestamps):
            word_cleaned = ts_data['cleaned_word']
            if word_cleaned in unique_single_words:
                start_time_ms = original_timestamps[i].get('start_time')
                end_time_ms = original_timestamps[i].get('end_time')

                if isinstance(start_time_ms, (int, float)) and isinstance(end_time_ms, (int, float)):
                    try:
                        snippet = audio[int(start_time_ms):int(end_time_ms)]
                        single_word_counters[word_cleaned] += 1
                        count = single_word_counters[word_cleaned]
                        
                        sanitized_word = self._sanitize_filename(word_cleaned)
                        snippet_filename = f"{sanitized_word}_{count}.{output_format}"
                        snippet_path = os.path.join(output_dir, snippet_filename)
                        
                        snippet.export(snippet_path, format=output_format)
                        extracted_snippets[f"{word_cleaned}_{count}"] = snippet_path
                        # print(f"  Extracted: {word_cleaned} (occurrence {count}) -> {snippet_path}")
                    except Exception as e:
                        print(f"  Error extracting/saving snippet for '{word_cleaned}' (occurrence {count}): {e}")
                else:
                    print(f"  Warning: Invalid time data for single word '{word_cleaned}' at index {i}. Skipping.")

        # --- 4. Process Multi-Word Phrases (Sequentially) --- 
        print("\nProcessing multi-word phrases sequentially...")
        timestamp_search_index = 0 # Tracks search position in timestamp list for phrases
        phrase_instance_counters = defaultdict(int)
        
        for original_phrase_text, phrase_words_to_match in multi_word_phrases_targets:
            found_this_instance = False
            phrase_len = len(phrase_words_to_match)
            
            # Search for the phrase starting from the current timestamp_search_index
            for i in range(timestamp_search_index, len(cleaned_timestamps) - phrase_len + 1):
                # Check if the sequence matches
                sequence_match = True
                for j in range(phrase_len):
                    if cleaned_timestamps[i+j]['cleaned_word'] != phrase_words_to_match[j]:
                        sequence_match = False
                        break
                
                if sequence_match:
                    start_time_ms = original_timestamps[i].get('start_time')
                    end_time_ms = original_timestamps[i + phrase_len - 1].get('end_time')

                    if isinstance(start_time_ms, (int, float)) and isinstance(end_time_ms, (int, float)):
                        try:
                            snippet = audio[int(start_time_ms):int(end_time_ms)]
                            
                            phrase_instance_counters[original_phrase_text] += 1
                            count = phrase_instance_counters[original_phrase_text]

                            sanitized_phrase = self._sanitize_filename(original_phrase_text)
                            snippet_filename = f"{sanitized_phrase}_{count}.{output_format}"
                            snippet_path = os.path.join(output_dir, snippet_filename)

                            snippet.export(snippet_path, format=output_format)
                            extracted_snippets[f"{original_phrase_text}_{count}"] = snippet_path
                            # print(f"  Extracted: '{original_phrase_text}' (instance {count}) -> {snippet_path}")
                            
                            timestamp_search_index = i + phrase_len # Move index past this match
                            found_this_instance = True
                            break # Stop searching for this specific instance, move to next target phrase
                        except Exception as e:
                             print(f"  Error extracting/saving snippet for phrase '{original_phrase_text}' (instance {count}): {e}")
                             # Don't advance timestamp_search_index if saving failed, maybe retry?
                             # For now, just report error and break the inner loop.
                             break 
                    else:
                        print(f"  Warning: Invalid time data for phrase '{original_phrase_text}' found at index {i}. Skipping sequence.")
                        # Continue searching for the phrase sequence from the next timestamp
                        
            if not found_this_instance:
                 print(f"  Warning: Could not find sequential match for phrase instance: '{original_phrase_text}'")

        total_extracted = len(extracted_snippets)
        print(f"\nFinished extraction. {total_extracted} snippets saved to {output_dir}")
        # print(f"Unique single words processed: {len(single_word_counters)}")
        # print(f"Multi-word phrase instances found: {sum(phrase_instance_counters.values())}")
        return extracted_snippets


# Updated Example Usage
if __name__ == '__main__':
    markdown_file = "output/transcript_How_to_pronounce_every_Italian_sound_in_18_Minutes.mp3.md"
    timestamp_file = "output/timestamps_How_to_pronounce_every_Italian_sound_in_18_Minutes.mp3.json"
    # --- Use the user-provided absolute path --- #
    original_audio = "/home/gyasis/Documents/code/Applied_AI/How_to_pronounce_every_Italian_sound_in_18_Minutes.mp3" # UPDATED PATH
    snippet_output_dir = "output/audio_snippets"

    if not os.path.exists(markdown_file):
        print(f"Error: Markdown transcript file not found at {markdown_file}")
    elif not os.path.exists(timestamp_file):
        print(f"Error: Timestamp file not found at {timestamp_file}")
    elif not os.path.exists(original_audio):
         print(f"Error: Original audio file not found at {original_audio}")
         print("Please verify the path to the original audio file.")
    else:
        editor = AudioEdit()
        try:
            print(f"Attempting to extract italicized text from: {markdown_file}...")
            italicized_phrases = editor.extract_italicized_text(markdown_file)
            print(f"Found {len(italicized_phrases)} italicized items originally.")

            if italicized_phrases:
                print(f"\nAttempting to extract audio snippets with new logic...")
                saved_snippets = editor.extract_audio_snippets(
                    target_phrases_from_markdown=italicized_phrases,
                    timestamp_file_path=timestamp_file,
                    original_audio_path=original_audio,
                    output_dir=snippet_output_dir
                )
                # print("\nSaved Snippets Map:")
                # for item_id, path in saved_snippets.items():
                #     print(f"- '{item_id}' -> {path}")

        except Exception as e:
            print(f"An unexpected error occurred: {e}") 