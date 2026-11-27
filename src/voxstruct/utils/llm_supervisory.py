"""
LLM Supervisor for intelligent transcript structuring.

This module uses Litellm to support multiple LLM providers for transcript analysis:
- OpenAI models (gpt-4, gpt-3.5-turbo)
- Anthropic models (claude-3, claude-2)
- Local models via Ollama (mistral, llama2)
- And others supported by Litellm
"""

import os
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from litellm import completion

class LLMSupervisor:
    def __init__(self, api_key: str = None, model: str = "gpt4all/gpt4all-j", api_base: str = None,
                 completion_kwargs: Optional[Dict[str, Any]] = None):
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
        # Optional override of the LLM endpoint (e.g. a LAN Ollama at
        # http://<host>:11434) so structuring runs off-box. None = provider default.
        self.api_base = api_base or os.getenv("VOXSTRUCT_LLM_API_BASE")
        # Extra litellm completion() arguments, e.g. {"reasoning_effort": "none"} to turn
        # thinking off on an OpenAI-compatible endpoint.
        self.completion_kwargs = dict(completion_kwargs or {})
        # The key is passed to each completion() call; the process environment is never
        # changed, so other clients in the same process keep their own keys.

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
                api_base=self.api_base,
                api_key=self.api_key,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant. Please introduce yourself briefly."},
                    {"role": "user", "content": "Please introduce yourself and confirm you can help with transcript formatting."}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            # Reasoning/thinking models (e.g. Gemini Pro) may spend the small
            # verify budget on internal thinking and return empty content. A
            # completed call with no exception still proves the model is
            # reachable + authorized, so treat empty as success (don't trigger
            # the silent no-LLM fallback).
            if content:
                return True, content.strip()
            return True, "(model reachable — empty verify response, likely a reasoning model)"

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
                                      metadata: Optional[Dict[str, Any]] = None,
                                      focus: Optional[str] = None) -> str:
        """
        Analyze and structure the transcript using the selected LLM.

        Args:
            raw_transcript: The unformatted transcript text
            pause_timestamps: List of timestamps where pauses were detected
            metadata: Additional information about the audio/transcript
            focus: Optional free-text scope. When set, the output is restricted to
                   content relevant to this focus and off-topic chatter / gossip is
                   dropped (see _focus_directive).

        Returns:
            str: Properly formatted and structured transcript
        """
        # Prepare the prompt with context and instructions
        prompt = self._build_structuring_prompt(raw_transcript, pause_timestamps, metadata, focus)

        system_content = """You are an expert at structuring transcripts.
Your task is to analyze the transcript and structure it with:
1. Proper paragraphs based on topic changes and natural pauses
2. Lists (ordered/unordered) when appropriate
3. Headers for major topic changes
4. Proper punctuation and formatting
5. Correction of obvious speech-to-text errors
6. Maintain medical/technical accuracy"""
        if focus:
            system_content += "\n" + self._focus_directive(focus)

        try:
            # Call LLM using litellm
            response = completion(
                model=self.model,
                api_base=self.api_base,
                api_key=self.api_key,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent formatting
                **self.completion_kwargs,
            )

            choice = response.choices[0]
            improved_transcript = (choice.message.content or "").strip()
            if choice.finish_reason == "length" or not improved_transcript:
                print(f"LLM structuring returned a truncated or empty reply "
                      f"(finish_reason={choice.finish_reason}); keeping the source transcript")
                return raw_transcript
            return improved_transcript

        except Exception as e:
            print(f"Error during LLM processing: {str(e)}")
            return raw_transcript  # Return original if processing fails

    _TS_RE = re.compile(r"^\[(\d\d:\d\d:\d\d)\]\s*")
    _NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
    _WORD_RE = re.compile(r"(?<![\w-])[A-Za-z][\w]*(?:[-.][\w]+)*")
    _IDENT_SKIP = {"OK", "TV", "US", "UK"}

    @staticmethod
    def _is_identifier(word: str) -> bool:
        """A digit (gpt-4o, RAON-1), CamelCase inside the word (OpenBot), or ALL-CAPS (RTX)."""
        return (any(c.isdigit() for c in word)
                or any(c.isupper() for c in word[1:]) and any(c.islower() for c in word)
                or (len(word) >= 2 and word.isupper()))

    @classmethod
    def facts(cls, text: str) -> Dict[str, List[str]]:
        """Hard facts of a text that a rewrite must keep.

        numbers: numeric cores ("70b" -> "70", "1.5B" -> "1.5"), so "1.5 billion" also counts.
        terms:   identifiers — words with a digit (gpt-4o, RTX5090), CamelCase, ALL-CAPS.
        [hh:mm:ss] markers are not facts.
        """
        text = re.sub(r"\[\d\d:\d\d:\d\d\]", " ", text)
        nums = list(dict.fromkeys(m.group(0).rstrip(".,") for m in cls._NUM_RE.finditer(text)))
        terms = [t for t in cls._WORD_RE.findall(text) if cls._is_identifier(t)]
        terms = list(dict.fromkeys(t for t in terms if t not in cls._IDENT_SKIP))
        return {"numbers": nums, "terms": terms}

    @classmethod
    def missing_facts(cls, source: str, output: str) -> List[str]:
        """Facts of `source` that are absent from `output`.

        A number is kept when the output states the same value in digits or words, and
        "N and a half" is kept by N.5. Terms match case-insensitively.
        """
        f = cls.facts(source)
        out_nums = set(cls.facts(output)["numbers"]) | cls._number_values(output)
        low = output.lower()

        def kept(n):
            if n in out_nums:
                return True
            # "12 and a half" in the source is kept by "12.5" in the output.
            if re.search(r"(?<![\d.])" + re.escape(n) + r"\s+and\s+a\s+half\b", source, re.I):
                try:
                    return cls._norm_num(float(n.replace(",", "")) + 0.5) in out_nums
                except ValueError:
                    return False
            return False

        missing = [n for n in f["numbers"] if not kept(n)]
        missing += [t for t in f["terms"]
                    if not re.search(r"(?<![\w])" + re.escape(t.lower()) + r"(?![\w])", low)]
        return missing

    _UNITS = {w: i for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen".split())}
    _TENS = {w: 10 * i for i, w in enumerate(
        "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
    _SCALES = {"hundred": 100, "thousand": 1000, "million": 10**6, "billion": 10**9, "trillion": 10**12}

    @staticmethod
    def _norm_num(v: float) -> str:
        return str(int(v)) if float(v).is_integer() else f"{v:.6f}".rstrip("0").rstrip(".")

    @classmethod
    def _number_values(cls, text: str) -> set:
        """Every number a text states, written as digits or as words, normalised ("six point
        three" -> "6.3", "twelve and a half" -> "12.5"). Scale words multiply ("two hundred")
        and a written number also counts at each scale ("1.5 billion" -> "1.5", "1500000000")."""
        text = re.sub(r"\[\d\d:\d\d:\d\d\]", " ", text.lower())
        vals = set()
        for m in cls._NUM_RE.finditer(text):
            core = m.group(0).rstrip(".,").replace(",", "")
            try:
                vals.add(cls._norm_num(float(core)))
            except ValueError:
                pass
            if re.match(r"\s+and\s+a\s+half\b", text[m.end():]):
                vals.add(cls._norm_num(float(core) + 0.5))
        words = re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.replace("-", " "))
        i = 0
        while i < len(words):
            w = words[i]
            if w not in cls._UNITS and w not in cls._TENS:
                i += 1
                continue
            total, cur = 0, 0
            while i < len(words):
                w = words[i]
                if w in cls._UNITS:
                    cur += cls._UNITS[w]
                elif w in cls._TENS:
                    cur += cls._TENS[w]
                elif w in cls._SCALES:
                    cur = max(cur, 1) * cls._SCALES[w]
                    if cls._SCALES[w] > 100:
                        total, cur = total + cur, 0
                elif w == "and" and i + 1 < len(words) and (words[i + 1] in cls._UNITS or words[i + 1] in cls._TENS):
                    pass
                else:
                    break
                i += 1
            value = total + cur
            if i + 1 < len(words) and words[i] == "point" and words[i + 1] in cls._UNITS:
                digits = []
                i += 1
                while i < len(words) and words[i] in cls._UNITS and cls._UNITS[words[i]] < 10:
                    digits.append(str(cls._UNITS[words[i]]))
                    i += 1
                value = float(f"{value}.{''.join(digits)}")
            elif words[i:i + 3] == ["and", "a", "half"]:
                vals.add(cls._norm_num(value))
                value += 0.5
            vals.add(cls._norm_num(value))
        return vals

    @classmethod
    def added_numbers(cls, source: str, output: str) -> List[str]:
        """Numbers the output states that the source does not (digits or words) — invented facts."""
        src = cls._number_values(source)
        out = cls._number_values(output)
        return sorted(out - src, key=lambda x: float(x))

    def _complete(self, prompt: str, temperature: float) -> str:
        """One completion; raises on a truncated or empty reply."""
        response = completion(
            model=self.model,
            api_base=self.api_base,
            api_key=self.api_key,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            **self.completion_kwargs,
        )
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        if choice.finish_reason == "length" or not text:
            raise ValueError(f"truncated or empty reply (finish_reason={choice.finish_reason})")
        return text

    def find_sections(self, paragraphs: List[str], window_chars: int = 40000) -> Dict[int, str]:
        """Topic sections of a transcript: {index of the first paragraph: section title}.

        The LLM reads numbered paragraphs and returns only section starts and titles.
        Paragraphs are sent in windows of about `window_chars` characters to keep each
        prompt small; indices are global. A window whose reply cannot be parsed gets no
        sections. Paragraph 0 always starts a section.
        """
        windows, cur, size = [], [], 0
        for i, p in enumerate(paragraphs):
            if cur and size + len(p) > window_chars:
                windows.append(cur)
                cur, size = [], 0
            cur.append(i)
            size += len(p)
        if cur:
            windows.append(cur)

        heads: Dict[int, str] = {}
        for w in windows:
            numbered = "\n\n".join(f"<{i}> {paragraphs[i]}" for i in w)
            prompt = (
                "Below are numbered paragraphs of a transcript. Divide them into sections by "
                "topic. Return ONLY a JSON array, no prose and no code fence:\n"
                '[{"index": <number of the paragraph where the section starts>, '
                '"title": "<3-8 word section header>"}]\n'
                f"The first section starts at paragraph {w[0]}. A section covers at least two "
                "paragraphs unless the topic really changes.\n\n" + numbered
            )
            try:
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", self._complete(prompt, 0.2))
                valid = set(w)
                for sec in json.loads(raw):
                    idx, title = sec.get("index"), str(sec.get("title") or "").strip()
                    if isinstance(idx, int) and idx in valid and title:
                        heads[idx] = title
            except Exception as e:
                print(f"Section finding: window {w[0]}-{w[-1]} got no sections ({e})")
        if paragraphs and 0 not in heads:
            heads[0] = "Introduction"
        return dict(sorted(heads.items()))

    def structure_lossless(self, paragraphs: List[str], window_chars: int = 40000,
                           header_level: int = 2) -> Tuple[str, List[Dict[str, Any]]]:
        """Add section headers to a transcript without rewriting a single word.

        Returns (markdown, sections) where sections = [{"index": int, "title": str}];
        the paragraphs come back byte-for-byte.
        """
        heads = self.find_sections(paragraphs, window_chars)
        out = []
        for i, p in enumerate(paragraphs):
            if i in heads:
                out.append(f"{'#' * header_level} {heads[i]}")
            out.append(p)
        return "\n\n".join(out), [{"index": i, "title": t} for i, t in heads.items()]

    _REPORT_PROMPT = (
        "Rewrite this part of a spoken transcript as a compact technical report.\n"
        "Keep every fact, claim, number, name, comparison, step, caveat and opinion. "
        "Say whose opinion it is when the speaker gives one.\n"
        "Remove filler words, repetition, greetings, and requests to like or subscribe.\n"
        "Write short sentences (20 words or fewer) in the active voice, one fact per sentence. "
        "Use a bullet list for a list of items or steps. Do not add anything that is not in "
        "the text. Do not write a heading.\n"
        "These terms and numbers MUST appear in your text, written as here: {facts}\n\n"
        "TRANSCRIPT:\n{text}"
    )

    def structure_report(self, paragraphs: List[str], window_chars: int = 40000,
                         header_level: int = 2, retries: int = 1
                         ) -> Tuple[str, List[Dict[str, Any]]]:
        """Turn a spoken transcript into a compact sectioned report that keeps every hard fact.

        1. find_sections() splits the transcript by topic.
        2. Each section is rewritten on its own (small prompts lose less).
        3. The rewrite must contain every fact of the section's source text (see facts()) and
           state no number the source does not (see added_numbers()). A failing rewrite is
           retried `retries` times, naming what is wrong. If it still fails, the section's
           source text is kept verbatim.

        A paragraph that starts with a [hh:mm:ss] marker puts that time in its section heading.
        Returns (markdown, sections), each section:
        {"index", "title", "start", "facts", "missing", "added", "attempts", "verbatim"}.
        """
        heads = self.find_sections(paragraphs, window_chars)
        starts = list(heads)
        md, report = [], []
        for n, start in enumerate(starts):
            end = starts[n + 1] if n + 1 < len(starts) else len(paragraphs)
            source = "\n\n".join(paragraphs[start:end])
            m = self._TS_RE.match(paragraphs[start])
            stamp = m.group(1) if m else None
            f = self.facts(source)
            wanted = f["numbers"] + f["terms"]
            body, missing, added, attempts = None, wanted, [], 0
            prompt = self._REPORT_PROMPT.format(facts=", ".join(wanted) or "(none)", text=source)
            while attempts <= retries:
                attempts += 1
                try:
                    draft = self._complete(prompt, 0.2)
                except Exception as e:
                    print(f"Report section '{heads[start]}': rewrite failed ({e})")
                    continue
                missing = self.missing_facts(source, draft)
                added = self.added_numbers(source, draft)
                if not missing and not added:
                    body = draft
                    break
                fix = []
                if missing:
                    fix.append("Your last version left out: " + ", ".join(missing) + ". Include them.")
                if added:
                    fix.append("Your last version stated numbers that are not in the transcript: "
                               + ", ".join(added) + ". Remove them.")
                prompt = (self._REPORT_PROMPT.format(facts=", ".join(wanted), text=source)
                          + "\n\n" + " ".join(fix))
            verbatim = body is None
            if verbatim:
                print(f"Report section '{heads[start]}': kept source text, rewrite left out "
                      f"[{', '.join(missing)}] and added [{', '.join(added)}]")
                body = source
            title = heads[start] + (f" [{stamp}]" if stamp else "")
            md.append(f"{'#' * header_level} {title}\n\n{body}")
            report.append({"index": start, "title": heads[start], "start": stamp,
                           "facts": len(wanted), "missing": missing if verbatim else [],
                           "added": added if verbatim else [],
                           "attempts": attempts, "verbatim": verbatim})
        return "\n\n".join(md), report

    def analyze_subject_matter(self, transcript: str) -> Dict[str, Any]:
        """Analyze the subject matter of the transcript to identify key topics."""
        try:
            response = completion(
                model=self.model,
                api_base=self.api_base,
                api_key=self.api_key,
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
                api_base=self.api_base,
                api_key=self.api_key,
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

    @staticmethod
    def _focus_directive(focus: str) -> str:
        """Instruction block that restricts the output to a given scope.

        Completeness is deliberately overridden: only content relevant to `focus`
        is kept, and anything ambiguous between substance and gossip is dropped.
        """
        return (
            "\nCONTENT FOCUS (this overrides completeness — do NOT include everything):\n"
            f"Keep ONLY material relevant to: {focus}\n"
            "- Omit unrelated conversation, personal or company gossip, side chatter, "
            "complaints about people, and names mentioned only in passing.\n"
            "- When it is a judgment call whether something is on-scope substance "
            "(e.g. a requirement, decision, task, or fact) versus incidental gossip, "
            "LEAVE IT OUT.\n"
            "- Do NOT invent, infer, or embellish content that is not in the transcript. "
            "If little is on-scope, return only the little that is."
        )

    def _build_structuring_prompt(self,
                                raw_transcript: str,
                                pause_timestamps: Optional[List[float]] = None,
                                metadata: Optional[Dict[str, Any]] = None,
                                focus: Optional[str] = None) -> str:
        """Build the prompt for the LLM with all available context."""
        prompt_parts = [
            "Please structure and format this transcript.",
            "\nOriginal transcript:",
            raw_transcript,
            "\nConsider the following:"
        ]
        if focus:
            prompt_parts.append(self._focus_directive(focus))
        
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
