"""Requirement atoms — the contract between extraction and rendering.

An atom is one typed requirement pulled from a transcript. Renderers are views
over a list of atoms; extraction produces them once (extract-once, render-many).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional

ATOM_TYPES = [
    "task",
    "process-step",
    "decision",
    "constraint",
    "blocker",
    "open-question",
    "data-entity",
]
CONFIDENCE = ["stated", "inferred"]


@dataclass
class Provenance:
    section: Optional[str] = None       # transcript header the atom came from
    quote: Optional[str] = None         # short verbatim snippet supporting it
    timestamp: Optional[float] = None   # seconds into the audio (when available)


@dataclass
class RequirementAtom:
    id: str
    type: str
    statement: str
    owner: Optional[str] = None
    timing: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)
    confidence: str = "stated"          # stated | inferred
    needs_confirmation: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AtomSet:
    source: str                         # transcript path / audio file
    model: str
    generated_at: str
    atoms: List[RequirementAtom] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            {
                "source": self.source,
                "model": self.model,
                "generated_at": self.generated_at,
                "meta": self.meta,
                "atoms": [a.to_dict() for a in self.atoms],
            },
            indent=indent,
            ensure_ascii=False,
        )

    def by_type(self, t: str) -> List[RequirementAtom]:
        return [a for a in self.atoms if a.type == t]

    def counts(self) -> dict:
        out = {}
        for a in self.atoms:
            out[a.type] = out.get(a.type, 0) + 1
        return out


def atom_from_dict(d: dict, idx: int) -> RequirementAtom:
    """Build an atom from loosely-typed LLM JSON, coercing bad values safely."""
    prov = d.get("provenance") or {}
    if not isinstance(prov, dict):
        prov = {}

    t = str(d.get("type") or "task").strip().lower()
    if t not in ATOM_TYPES:
        t = "task"

    conf = str(d.get("confidence") or "stated").strip().lower()
    if conf not in CONFIDENCE:
        conf = "inferred"

    deps = d.get("dependencies") or []
    if isinstance(deps, str):
        deps = [deps]

    return RequirementAtom(
        id=str(d.get("id") or f"R{idx:03d}"),
        type=t,
        statement=str(d.get("statement") or "").strip(),
        owner=d.get("owner") or None,
        timing=d.get("timing") or None,
        dependencies=[str(x) for x in deps],
        provenance=Provenance(
            section=prov.get("section"),
            quote=prov.get("quote"),
            timestamp=prov.get("timestamp"),
        ),
        confidence=conf,
        needs_confirmation=bool(d.get("needs_confirmation", conf == "inferred")),
    )
