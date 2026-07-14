"""audio-req — turn a recording or transcript into structured requirements.

Portable CLI (any agent can call it). Front door: audio (-> VoxStruct) OR a
transcript .md. Extracts requirement atoms once, renders a chosen artifact.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from .extract import DEFAULT_MODEL, extract_atoms
from .render.task_spec import render_task_spec

AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".ogg", ".flac", ".mp4", ".webm", ".aac"}
RENDERERS = {"task-spec": render_task_spec}


def _transcribe_if_audio(inp: Path) -> Path:
    """If inp is audio, run voxstruct to produce a transcript md and return it."""
    if inp.suffix.lower() not in AUDIO_EXTS:
        return inp
    print(f"[audio-req] input is audio -> transcribing via voxstruct: {inp.name}")
    subprocess.run(["voxstruct", str(inp)], check=True)
    base = inp.stem
    cand = sorted(
        glob.glob(f"output/transcript_{base}*.md"), key=os.path.getmtime, reverse=True
    ) or sorted(glob.glob("output/transcript_*.md"), key=os.path.getmtime, reverse=True)
    if not cand:
        sys.exit("[audio-req] could not locate a voxstruct transcript under output/")
    return Path(cand[0])


def main(argv=None) -> int:
    load_dotenv()  # repo .env: OPENAI/GEMINI keys, VOXSTRUCT_AUDIOREQ_MODEL
    ap = argparse.ArgumentParser(
        description="Extract structured requirements from a recording or transcript."
    )
    ap.add_argument("input", help="audio file OR a transcript .md")
    ap.add_argument(
        "--render", default="task-spec", choices=sorted(RENDERERS),
        help="artifact to render (default: task-spec)",
    )
    ap.add_argument("--focus", help="only extract from this transcript section")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="extraction LLM (litellm id)")
    ap.add_argument(
        "--out-dir",
        default=os.path.expanduser("~/Documents/generated/audio-req"),
        help="where to write atoms.json + the artifact",
    )
    args = ap.parse_args(argv)

    inp = Path(args.input).expanduser()
    if not inp.is_file():
        sys.exit(f"[audio-req] input not found: {inp}")

    transcript_path = _transcribe_if_audio(inp)
    text = Path(transcript_path).read_text(encoding="utf-8", errors="replace")

    print(
        f"[audio-req] extracting atoms ({args.model})"
        + (f" focus='{args.focus}'" if args.focus else "")
        + " ..."
    )
    aset = extract_atoms(
        text, source=str(transcript_path), model=args.model, focus=args.focus
    )
    print(f"[audio-req] extracted {len(aset.atoms)} atoms: {aset.counts()}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(transcript_path).stem.replace("transcript_", "")

    atoms_path = out / f"atoms_{stem}.json"
    atoms_path.write_text(aset.to_json(), encoding="utf-8")

    art = RENDERERS[args.render](aset, title=f"Requirements — {stem}")
    art_path = out / f"{args.render}_{stem}.md"
    art_path.write_text(art, encoding="utf-8")

    print(f"[audio-req] atoms     -> {atoms_path}")
    print(f"[audio-req] {args.render} -> {art_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
