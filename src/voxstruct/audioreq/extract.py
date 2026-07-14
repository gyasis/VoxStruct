"""Extract requirement atoms from a transcript (whole-transcript, section-aware)."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Optional

from litellm import completion

from .atoms import AtomSet, atom_from_dict

DEFAULT_MODEL = os.getenv("VOXSTRUCT_AUDIOREQ_MODEL", "openai/gpt-4o-mini")

_SYSTEM = """You are a senior business analyst. You read a transcript of a spoken \
conversation (a meeting, hand-off, or dictated note) and extract the REQUIREMENTS \
in it — what needs to be DONE — as discrete, typed records ("atoms").

Rules:
- Cover the WHOLE transcript. Use its section headers as structure; record each \
atom's source section in provenance.section.
- One atom = one requirement. Split compound statements into separate atoms.
- `type` is exactly one of: task, process-step, decision, constraint, blocker, \
open-question, data-entity.
    * process-step : an ordered step in a procedure.
    * task         : an action item / to-do.
    * decision     : a choice that was made.
    * constraint   : a rule/filter/condition that governs the work.
    * blocker      : something preventing progress.
    * open-question: an unresolved ambiguity that needs a human answer.
    * data-entity  : a system, file, field, or dataset the work touches.
- `confidence`: "stated" if the transcript says it explicitly; "inferred" if you \
had to read between the lines. Set needs_confirmation=true for anything inferred \
or ambiguous.
- `provenance.quote`: a SHORT verbatim snippet (<=120 chars) from the transcript \
that supports the atom.
- For process-step atoms, express order via `dependencies` (the id(s) that must \
come first) when the order is clear.
- Do NOT invent requirements that aren't supported by the text. Skip pure chit-chat.

Return ONLY a JSON object of the form {"atoms": [ ... ]}. Each atom object:
{"id","type","statement","owner","timing","dependencies","provenance":{"section","quote"},"confidence","needs_confirmation"}
"""


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def extract_atoms(
    transcript_text: str,
    source: str,
    model: str = DEFAULT_MODEL,
    focus: Optional[str] = None,
) -> AtomSet:
    """Transcript text -> AtomSet. `focus` restricts extraction to one section."""
    user = transcript_text
    if focus:
        user = f"FOCUS ONLY on the section titled '{focus}'.\n\n{transcript_text}"

    resp = completion(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = resp["choices"][0]["message"]["content"]
    data = _extract_json(content)
    raw = data.get("atoms") or []
    atoms = [atom_from_dict(d, i + 1) for i, d in enumerate(raw)]

    aset = AtomSet(
        source=source,
        model=model,
        generated_at=datetime.now(timezone.utc).isoformat(),
        atoms=atoms,
    )
    aset.meta = {"focus": focus, "atom_count": len(atoms), "counts": aset.counts()}
    return aset
