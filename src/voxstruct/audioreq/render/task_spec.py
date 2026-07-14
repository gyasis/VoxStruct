"""Render an AtomSet as a task-spec markdown document (v1 renderer)."""

from __future__ import annotations

from ..atoms import AtomSet, RequirementAtom


def _flag(a: RequirementAtom) -> str:
    return " ⚠ CONFIRM" if (a.needs_confirmation or a.confidence == "inferred") else ""


def render_task_spec(aset: AtomSet, title: str = "Task Spec") -> str:
    L: list = []
    L.append(f"# {title}")
    L.append("")
    L.append(f"> Source: `{aset.source}`  ·  model: `{aset.model}`  ·  {aset.generated_at}")
    L.append("> ⚠ CONFIRM = inferred, not explicitly stated — verify before automating.")
    counts = ", ".join(f"{k}={v}" for k, v in sorted(aset.counts().items()))
    L.append(f"> Atoms: {len(aset.atoms)}  ({counts})")
    L.append("")

    steps = aset.by_type("process-step")
    tasks = aset.by_type("task")
    decisions = aset.by_type("decision")
    constraints = aset.by_type("constraint")
    blockers = aset.by_type("blocker")
    questions = aset.by_type("open-question")
    entities = aset.by_type("data-entity")

    if steps:
        L.append("## Procedure (ordered)")
        L.append("")
        for i, a in enumerate(steps, 1):
            sec = f"  ·  _{a.provenance.section}_" if a.provenance.section else ""
            L.append(f"- [ ] **{i}. {a.statement}**{_flag(a)}{sec}")
        L.append("")

    if tasks:
        L.append("## Action items")
        L.append("")
        for a in tasks:
            own = f" — _{a.owner}_" if a.owner else ""
            tim = f" ({a.timing})" if a.timing else ""
            L.append(f"- [ ] {a.statement}{own}{tim}{_flag(a)}")
        L.append("")

    if constraints:
        L.append("## Decision rules / constraints")
        L.append("")
        for a in constraints:
            L.append(f"- {a.statement}{_flag(a)}")
        L.append("")

    if decisions:
        L.append("## Decisions made")
        L.append("")
        for a in decisions:
            L.append(f"- {a.statement}{_flag(a)}")
        L.append("")

    if entities:
        L.append("## Systems / data touched")
        L.append("")
        for a in entities:
            L.append(f"- **{a.statement}**{_flag(a)}")
        L.append("")

    if blockers:
        L.append("## Blockers")
        L.append("")
        for a in blockers:
            L.append(f"- ⛔ {a.statement}{_flag(a)}")
        L.append("")

    if questions:
        L.append("## Open questions (need a human answer)")
        L.append("")
        for a in questions:
            L.append(f"- ❓ {a.statement}{_flag(a)}")
        L.append("")

    if steps:
        L.append("## ASCII — flow")
        L.append("")
        L.append("```")
        for i, a in enumerate(steps, 1):
            label = a.statement[:50].rstrip()
            L.append(f"  [{i}] {label}")
            if i < len(steps):
                L.append("       |")
                L.append("       v")
        L.append("```")
        L.append("")

    return "\n".join(L)
