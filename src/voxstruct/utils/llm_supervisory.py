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
    # Not facts: common capitalised words, and the platform a talk is published on (it appears
    # in "like and subscribe" asides that the report is told to drop).
    _IDENT_SKIP = {"OK", "TV", "US", "UK", "YouTube", "YouTuber", "YouTubers"}

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
    def _number_values(cls, text: str, words: bool = True) -> set:
        """Every number a text states, written as digits or as words, normalised ("six point
        three" -> "6.3", "twelve and a half" -> "12.5"). Scale words multiply ("two hundred").
        words=False reads digits only."""
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
        if not words:
            return vals
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
        """Numbers the output writes in digits that the source does not state (in digits or
        words) — invented facts. Number words in the output are not read: prose uses "one" as a
        pronoun far more often than as a figure. Numbered-list markers are not read either."""
        src = cls._number_values(source)
        output = re.sub(r"(?m)^\s*\d+[.)]\s", " ", output)  # list numbering is not a fact
        out = cls._number_values(output, words=False)
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

    # Document styles for structure_report(). Each style sets:
    #   outline: None = topic sections in talk order; a list = fixed sections the content is
    #            regrouped into (each item "Name: what belongs there").
    #   rank:    order topic sections by importance instead of talk order.
    #   front:   (label, instruction) for the opening block written from the finished sections.
    #   voice:   how each section is written.
    #   stamps:  "heading" puts the section's start time in its heading; "footer" lists the
    #            transcript times a regrouped section draws on.
    STYLES: Dict[str, Dict[str, Any]] = {
        "article": {
            "outline": None, "rank": False, "stamps": "heading",
            "front": ("standfirst", "Write the standfirst of this magazine article: 2 or 3 "
                      "sentences under the headline that tell the reader what the piece is "
                      "about and why it matters."),
            "voice": "a magazine feature: engaging but exact, explaining terms a curious "
                     "non-specialist reader would not know",
        },
        "news": {
            "outline": None, "rank": True, "stamps": "heading",
            "front": ("lede", "Write the lede and nut graf of this news story: a first "
                      "paragraph with the single most important news (who, what, when, why), "
                      "then a second paragraph that says why it matters."),
            "voice": "a newspaper story: the most important facts first, short paragraphs, "
                     "every claim attributed",
        },
        "paper": {
            "outline": ["Background: the problem, context and prior work the talk describes",
                        "Approach: the method, design, architecture or process",
                        "Findings: results, measurements, benchmarks and observations",
                        "Discussion: interpretation, comparisons, implications and opinions",
                        "Limitations: weaknesses, caveats, risks and open questions"],
            "rank": False, "stamps": "footer",
            "front": ("Abstract", "Write the abstract of this paper: one paragraph of 5 to 7 "
                      "sentences covering the problem, the approach, the main findings and the "
                      "conclusion."),
            "voice": "a research paper: formal, precise and impersonal, with each claim tied "
                     "to its evidence",
        },
        "report": {
            "outline": ["Context: background and why the topic came up",
                        "Details: the substance, topic by topic",
                        "Decisions and actions: what was decided, recommended or must be done, "
                        "and by whom",
                        "Open questions: unresolved points, risks and follow-ups"],
            "rank": False, "stamps": "footer",
            "front": ("Summary", "Write the executive summary of this report: one paragraph of "
                      "4 to 6 sentences with the key points and conclusions a busy reader needs."),
            "voice": "a business or technical briefing report: direct, scannable and complete",
        },
    }

    _SECTION_PROMPT = (
        "You are writing one section of {voice}. The source is a spoken transcript.\n"
        "Document: {title}\nSections in order: {outline}\nWrite the section: {section}\n\n"
        "Write finished prose for a reader, not notes. Use paragraphs of 3 to 6 connected "
        "sentences. Build each paragraph around one idea and open it with a sentence that "
        "states that idea. Link the ideas with transitions. Use a bulleted or numbered list "
        "only for a real list (steps, specifications, options), and a markdown table only when "
        "several things are compared on the same attributes.\n"
        "State facts directly. Attribute opinions, judgements and predictions to the speaker. "
        "Call the speaker \"the speaker\" or by the name the transcript gives; never guess a "
        "gender. You may quote the speaker, but a quotation must copy the transcript words "
        "exactly.\n"
        "Leave out filler, false starts, repetition, greetings, sponsor reads, self-promotion, "
        "requests to like or subscribe, and remarks about the recording itself.\n"
        "Keep every fact, number, name, comparison, step and caveat. Do not add anything that "
        "is not in the transcript. Do not write a heading.\n"
        "Use plain technical English: active voice, sentences under 25 words, one term for "
        "one concept.\n"
        "These terms and numbers MUST appear in your text, written as here: {facts}\n\n"
        "TRANSCRIPT:\n{text}"
    )

    _FRONT_PROMPT = (
        "{instruction} Use only what the document below says. Do not write a heading.\n\n"
        "DOCUMENT: {title}\n\n{body}"
    )

    _QUOTE_RE = re.compile(r"[\"\u201c]([^\"\u201c\u201d\n]*)[\"\u201d]")

    @classmethod
    def unfaithful_quotes(cls, source: str, output: str) -> List[str]:
        """Quotations in `output` (20+ characters) whose words do not appear, in order, in `source`."""
        def norm(t):
            return " ".join(re.findall(r"[a-z0-9']+", re.sub(r"\[\d\d:\d\d:\d\d\]", " ", t.lower())))
        src = norm(source)
        quotes = [q for q in cls._QUOTE_RE.findall(output) if len(q) >= 20]
        return [q for q in quotes if norm(q) and norm(q) not in src]

    def assign_sections(self, paragraphs: List[str], outline: List[str]) -> Dict[str, List[int]]:
        """Regroup paragraphs into a fixed outline: {section name: [paragraph indices]}.

        The LLM only assigns paragraph numbers. Each paragraph goes to exactly one section (the
        first that claims it); a paragraph nobody claims goes to "Details" if the outline has it,
        else to the last section. Nothing is dropped.
        """
        names = [o.split(":", 1)[0].strip() for o in outline]
        numbered = "\n\n".join(f"<{i}> {p}" for i, p in enumerate(paragraphs))
        prompt = (
            "Assign every numbered paragraph of this transcript to ONE section of the outline. "
            "Return ONLY a JSON object, no prose and no code fence: "
            '{"<section name>": [paragraph numbers], ...}\n\nOUTLINE:\n'
            + "\n".join(f"- {o}" for o in outline) + "\n\nTRANSCRIPT:\n" + numbered
        )
        groups: Dict[str, List[int]] = {n: [] for n in names}
        taken = set()
        try:
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", self._complete(prompt, 0.2))
            for name, idxs in json.loads(raw).items():
                key = next((n for n in names if n.lower() == str(name).strip().lower()), None)
                if key is None:
                    continue
                for i in idxs if isinstance(idxs, list) else []:
                    if isinstance(i, int) and 0 <= i < len(paragraphs) and i not in taken:
                        groups[key].append(i)
                        taken.add(i)
        except Exception as e:
            print(f"Section assignment failed ({e}); unassigned paragraphs go to one section")
        rest = [i for i in range(len(paragraphs)) if i not in taken]
        if rest:
            groups["Details" if "Details" in groups else names[-1]].extend(rest)
        return {n: sorted(v) for n, v in groups.items() if v}

    def _rank_sections(self, titles: List[str], title: str) -> List[int]:
        """Order of section positions by news importance; talk order if the reply is unusable."""
        prompt = ("Rank these sections of a news story about \"" + title + "\" from most to least "
                  "newsworthy. Return ONLY a JSON array of their numbers.\n"
                  + "\n".join(f"{i}: {t}" for i, t in enumerate(titles)))
        try:
            order = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", self._complete(prompt, 0.2)))
            order = [i for i in dict.fromkeys(order) if isinstance(i, int) and 0 <= i < len(titles)]
            return order + [i for i in range(len(titles)) if i not in order]
        except Exception as e:
            print(f"Section ranking failed ({e}); keeping talk order")
            return list(range(len(titles)))

    def _write_section(self, voice, doc_title, outline, name, source, retries, known=None):
        """One section as prose; (body or None, facts wanted, missing, added, bad quotes, attempts).

        `known` is the text a number must appear in to not count as invented (default: source)."""
        known = known or source
        f = self.facts(source)
        wanted = f["numbers"] + f["terms"]
        base = self._SECTION_PROMPT.format(voice=voice, title=doc_title, outline=outline,
                                           section=name, facts=", ".join(wanted) or "(none)",
                                           text=source)
        prompt, missing, added, quotes, attempts = base, wanted, [], [], 0
        while attempts <= retries:
            attempts += 1
            try:
                draft = self._complete(prompt, 0.2)
            except Exception as e:
                print(f"Section '{name}': writing failed ({e})")
                continue
            missing = self.missing_facts(source, draft)
            # Invented = said nowhere in the talk or title. Checking only this section rejected
            # correct names: captions write "Inkling Too Mini" and say "Inkling 2" once, elsewhere.
            added = self.added_numbers(known, draft)
            quotes = self.unfaithful_quotes(source, draft)
            if not (missing or added or quotes):
                return draft, wanted, [], [], [], attempts
            fix = []
            if missing:
                fix.append("Your last version left out: " + ", ".join(missing) + ". Include them.")
            if added:
                fix.append("Your last version stated numbers that are not in the transcript: "
                           + ", ".join(added) + ". Remove them.")
            if quotes:
                fix.append("These quotations are not the transcript's exact words: "
                           + " | ".join(quotes) + ". Quote exactly or paraphrase without quotes.")
            prompt = base + "\n\n" + " ".join(fix)
        return None, wanted, missing, added, quotes, attempts

    _STYLE_PROMPT = (
        "Pick the written form that best fits this spoken transcript. Answer with ONE word:\n"
        "article - commentary, explainer, review, opinion, tutorial or interview for a general reader\n"
        "news - an announcement or report of a recent event, release or result\n"
        "paper - a lecture, research talk or technical deep-dive on a method and its evidence\n"
        "report - a meeting, briefing, status update or discussion that ends in decisions or actions\n"
        "\nTitle: {title}\n\nTRANSCRIPT (opening):\n{text}"
    )

    def choose_style(self, paragraphs: List[str], title: Optional[str] = None,
                     sample_chars: int = 12000) -> str:
        """Pick the document style for a transcript from its title and opening; "article" if
        the reply is not one of the styles."""
        text = "\n\n".join(paragraphs)[:sample_chars]
        try:
            reply = self._complete(self._STYLE_PROMPT.format(title=title or "(none)", text=text), 0.0)
            word = re.findall(r"[a-z]+", reply.lower())
            pick = next((w for w in word if w in self.STYLES), None)
        except Exception as e:
            print(f"Style choice failed ({e}); using article")
            pick = None
        return pick or "article"

    def structure_report(self, paragraphs: List[str], window_chars: int = 40000,
                         header_level: int = 2, retries: int = 1, title: Optional[str] = None,
                         style: str = "article", section_chars: int = 6000
                         ) -> Tuple[str, List[Dict[str, Any]]]:
        """Turn a spoken transcript into a written document that keeps every hard fact.

        `style` picks the document shape (see STYLES), or "auto" to let choose_style() pick it:
        "article" (magazine feature, talk order),
        "news" (lede first, sections by importance), "paper" (Abstract, Background, Approach,
        Findings, Discussion, Limitations) or "report" (Summary, Context, Details, Decisions and
        actions, Open questions).

        1. Sections: topic sections from find_sections(), or the style's fixed outline filled by
           assign_sections(). Every paragraph lands in exactly one section.
        2. Each section is written as prose on its own, with the title and outline as context.
           A section longer than `section_chars` is split into topic subsections (###), because
           one long rewrite drops facts.
        3. A section is accepted only if it keeps every number and identifier of its source
           (facts()), states no number the source does not (added_numbers()) and quotes only the
           transcript's exact words (unfaithful_quotes()). It is retried `retries` times naming
           the problem; if it still fails, its source text is kept verbatim.
        4. The opening block (standfirst, lede, abstract or summary) is written from the finished
           sections and dropped if it states a number the transcript does not.

        Returns (markdown, sections), each section:
        {"style", "title", "paragraphs", "start", "facts", "missing", "added", "quotes", "attempts",
         "verbatim"}.
        """
        if style == "auto":
            style = self.choose_style(paragraphs, title)
            print(f"Document style chosen: {style}")
        if style not in self.STYLES:
            raise ValueError(f"unknown style {style!r}; choose from {sorted(self.STYLES)} or 'auto'")
        st = dict(self.STYLES[style], name=style)
        doc_title = title or "(untitled talk)"

        if st["outline"]:
            groups = self.assign_sections(paragraphs, st["outline"])
            plan = list(groups.items())
        else:
            heads = self.find_sections(paragraphs, window_chars)
            starts = list(heads)
            plan = [(heads[s], list(range(s, starts[n + 1] if n + 1 < len(starts) else len(paragraphs))))
                    for n, s in enumerate(starts)]
            if st["rank"] and len(plan) > 1:
                plan = [plan[i] for i in self._rank_sections([p[0] for p in plan], doc_title)]
        outline = " | ".join(name for name, _ in plan)

        md, report = [], []
        for name, idxs in plan:
            parts = self._split_large(paragraphs, idxs, section_chars, window_chars)
            if len(parts) > 1:
                md.append(f"{'#' * header_level} {name}")
                for sub, sub_idxs in parts:
                    self._emit(md, report, st, doc_title, outline, f"{name} / {sub}", sub, sub_idxs,
                               paragraphs, header_level + 1, retries)
                continue
            self._emit(md, report, st, doc_title, outline, name, name, idxs, paragraphs,
                       header_level, retries)
        return self._finish(md, report, st, title, doc_title, paragraphs, header_level)

    def _split_large(self, paragraphs, idxs, section_chars, window_chars):
        """[(subsection title, indices)] for a section over `section_chars`; one item otherwise.

        Topic split of the section's own paragraphs; if that yields one topic, size chunks."""
        if sum(len(paragraphs[i]) for i in idxs) <= section_chars or len(idxs) < 2:
            return [(None, idxs)]
        sub = self.find_sections([paragraphs[i] for i in idxs], window_chars)
        starts = list(sub)
        parts = [(sub[s], idxs[s:starts[n + 1] if n + 1 < len(starts) else len(idxs)])
                 for n, s in enumerate(starts)]
        if len(parts) > 1:
            return parts
        chunks, cur, size = [], [], 0
        for i in idxs:
            if cur and size + len(paragraphs[i]) > section_chars:
                chunks.append(cur)
                cur, size = [], 0
            cur.append(i)
            size += len(paragraphs[i])
        chunks.append(cur)
        return [(f"Part {n}", c) for n, c in enumerate(chunks, 1)]

    def _emit(self, md, report, st, doc_title, outline, name, heading_name, idxs, paragraphs,
              level, retries):
        """Write one section (or subsection) and append its markdown and audit record."""
        source = "\n\n".join(paragraphs[i] for i in idxs)
        stamps = [m.group(1) for i in idxs if (m := self._TS_RE.match(paragraphs[i]))]
        body, wanted, missing, added, quotes, attempts = self._write_section(
            st["voice"], doc_title, outline, name, source, retries,
            known="\n\n".join(paragraphs) + "\n" + doc_title)
        verbatim = body is None
        if verbatim:
            print(f"Section '{name}': kept source text; left out {missing}, added {added}, "
                  f"inexact quotes {quotes}")
            body = source
        heading = heading_name
        if stamps and st["stamps"] == "heading":
            heading += f" [{stamps[0]}]"
        elif stamps and st["stamps"] == "footer":
            body += "\n\n*Transcript: " + ", ".join(f"[{t}]" for t in stamps) + "*"
        md.append(f"{'#' * level} {heading}\n\n{body}")
        report.append({"style": st["name"], "title": name, "paragraphs": idxs, "start": stamps[0] if stamps else None,
                       "facts": len(wanted), "missing": missing if verbatim else [],
                       "added": added if verbatim else [], "quotes": quotes if verbatim else [],
                       "attempts": attempts, "verbatim": verbatim})

    def _finish(self, md, report, st, title, doc_title, paragraphs, header_level):
        """Prepend the title and the style's opening block."""
        front = []
        if title:
            front = [f"# {title}"]
        label, instruction = st["front"]
        text = self._front(instruction, doc_title, md, "\n\n".join(paragraphs))
        if text:
            if label[0].isupper():
                front.append(f"{'#' * header_level} {label}\n\n{text}")
            elif label == "standfirst":
                front.append(f"*{text}*")
            else:
                front.append(text)
        return "\n\n".join(front + md), report

    def _front(self, instruction: str, title: str, sections_md: List[str], source: str) -> Optional[str]:
        """Opening block written from the finished sections; None if it fails or invents a number."""
        try:
            text = self._complete(self._FRONT_PROMPT.format(
                instruction=instruction, title=title, body="\n\n".join(sections_md)), 0.2)
        except Exception as e:
            print(f"Opening block: failed ({e})")
            return None
        added = self.added_numbers(source + "\n" + title, text)
        if added:
            print(f"Opening block: dropped, it states numbers not in the transcript: {added}")
            return None
        return text

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
