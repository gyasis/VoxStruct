"""audio-req — extract structured requirements from a transcript.

Sits on top of VoxStruct (the transcript producer). Extraction emits typed
"requirement atoms" once; renderers (task-spec, backlog, ...) are views over
those atoms. CLI-first so any agent can call it.
"""

from .atoms import AtomSet, RequirementAtom, Provenance
from .extract import extract_atoms

__all__ = ["AtomSet", "RequirementAtom", "Provenance", "extract_atoms"]
