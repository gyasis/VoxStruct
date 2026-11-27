# VoxStruct — guide for coding agents

Audio → **structured** transcript. ASR produces a lowercase, unpunctuated wall of text;
the LLM pass turns it into a document (headers, paragraphs, lists, punctuation, casing).
The structuring is the point of the tool.

**Read `docs/REMOTE_ASR.md` before choosing a backend.** A fuller operator runbook lives
outside this repo at `~/Documents/code/codebook/voxstruct.md`.

## Pick an ASR backend — `VOXSTRUCT_ASR_BACKEND`

| backend | needs | when |
|---|---|---|
| `builtin` (default) | local `torch` + `whisper` | rarely — **not installed in this repo's venv** |
| `remote` | an ASR sidecar reachable on the network | **preferred**: warm, free, weights resident |
| `modal` | a Modal account | the sidecar host is down / travelling / no local GPU |

```bash
VOXSTRUCT_ASR_BACKEND=remote VOXSTRUCT_ASR_URL=http://<host>:8766 voxstruct audio.m4a
VOXSTRUCT_ASR_BACKEND=modal VOXSTRUCT_DIARIZE=true VOXSTRUCT_NUM_SPEAKERS=2 voxstruct audio.m4a
```

All three return the same `TranscriptResult`, so downstream code never branches on backend.

```python
from voxstruct.asr import get_asr_backend
b = get_asr_backend()          # None => use the builtin in-process path
b.health()                     # cheap reachability probe, never raises
r = b.transcribe("a.mp3", language="en", word_timestamps=True)
```

## Three ways this bites you — two fail SILENTLY

**1. `vad_filter` must stay off for quiet or far-field audio.** Silero VAD scores quiet
speech below its 0.5 threshold and discards it *uniformly across the file*, so the output
reads as a plausible transcript rather than an error. It cost **79% of the words** of a real
2h meeting. Measured against a known-good reference (1,557 words): VAD on → 778, VAD off →
1,617. Normalization does not rescue it. **Sanity-check word count: ~110–150 words per
minute of speech.**

**2. `validate_and_improve_transcript` returns its INPUT UNCHANGED on any exception** (and on
a truncated or empty reply). A dead key, an exhausted quota or a bad model name is
indistinguishable from success — only a line on stdout marks it. Twelve failed calls once produced a file labelled "structured"
containing no structure. Guard every call:

```python
if structured.strip() == raw.strip():
    raise RuntimeError("LLM call failed silently — refusing to write")
```

**3. The API key must match the model's provider.** `LLMSupervisor` passes its key to every
call for the provider named in the model string, so a Gemini key with `openai/...` gives
`Incorrect API key provided: AIzaSy...`. Select the key from the model string. It never
writes the key into the process environment, so other clients keep their own keys.

## Speech to a document — `structure_report(paragraphs, title=, style=)`

`validate_and_improve_transcript` rewrites a whole block in one call and drops facts: on two
YouTube talks it kept 42–79% of the numbers. To turn speech into a document, use
`structure_report`. It writes finished prose (paragraphs with topic sentences, lists and tables
only where the content is one) in one of four shapes:

| `style` | Shape | Section order |
|---|---|---|
| `article` (default) | `# Title`, italic standfirst, `## Topic [hh:mm:ss]` | talk order |
| `news` | `# Title`, lede + nut graf, `## Topic [hh:mm:ss]` | by news importance |
| `paper` | `# Title`, `## Abstract`, Background, Approach, Findings, Discussion, Limitations | content regrouped by role |
| `report` | `# Title`, `## Summary`, Context, Details, Decisions and actions, Open questions | content regrouped by role |

`style="auto"` lets `choose_style()` pick one from the title and the transcript opening (one
call; "article" if the reply is unusable). Checked on samples: meeting → report, lecture →
paper, announcement → news, commentary/review → article.

Every paragraph lands in exactly one section. A section over `section_chars` (6000) is split
into `###` topic subsections, because one long rewrite drops facts. Regrouped sections end with
`*Transcript: [hh:mm:ss], ...*`.

**Checks (by code, per section):** `missing_facts` — every number and identifier of the source is
in the rewrite (spoken numbers count: "six point three" = 6.3, "8 and a half" = 8.5);
`added_numbers` — no digit the speaker never said (list numbering ignored); `unfaithful_quotes` —
a quotation must be the transcript's exact words. A failing section is retried once with the
problem named, then kept verbatim. The opening block is dropped if it invents a number. The
checks do not cover the meaning of every sentence. Judge output on them, never on word count:
removing filler is the point.

`structure_lossless(paragraphs)` adds `##` headers and leaves every word unchanged.

## Other things worth knowing

- **`--focus` DROPS off-topic content.** Great for a targeted extract, wrong when the
  transcript is the record of a meeting. Omit it to structure everything.
- **Structuring must be chunked** — one block per call; ~10-minute blocks work. No output cap
  is sent, so a reply that hits the model's own limit returns the input (see 2).
- **Thinking off on an OpenAI-compatible gateway:** `completion_kwargs={"reasoning_effort":
  "none", "allowed_openai_params": ["reasoning_effort"]}` (litellm rejects the parameter for
  an unknown model without the second key).
- **`VOXSTRUCT_NUM_SPEAKERS` is exact, not a hint.** Guessing splits one person into
  phantom speakers. ASR does not diarize by itself; pyannote is a separate step.
- **Pass `word_timestamps=True`** if anything downstream must locate a word in the audio.
  Diarization computes them anyway; not saving them means re-running the whole job.
- Raw ASR is lowercase/unpunctuated because `condition_on_previous_text` is off (the guard
  against repetition-loop hallucinations on long audio). The LLM pass fixes it — don't flip
  the flag.

## Tests

```bash
python3 -m pytest tests/test_asr_backends.py -q     # backend selection, no network
python3 -m pytest tests/test_llm_supervisory_lossless.py -q   # structuring checks, no network
```
