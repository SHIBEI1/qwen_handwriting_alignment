#!/usr/bin/env python3
"""Build a deterministic, train-only medium-length GRPO smoke subset."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unicodedata


TARGET_LENGTH = 16
SAMPLE_COUNT = 8


def text_length(text: str) -> int:
    return len("".join(unicodedata.normalize("NFKC", text).split()))


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "data/native_subset/grpo/train.jsonl"
    destination = root / "data/native_subset/grpo_smoke_medium"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < SAMPLE_COUNT:
        raise ValueError("GRPO source has too few train-only rows for the smoke subset")

    ranked = sorted(rows, key=lambda row: (abs(text_length(row["target_text"]) - TARGET_LENGTH), row["id"]))
    selected: list[dict] = []
    seen_styles: set[str] = set()
    for row in ranked:
        if row["style_id"] in seen_styles:
            continue
        selected.append(row)
        seen_styles.add(row["style_id"])
        if len(selected) == SAMPLE_COUNT:
            break
    if len(selected) < SAMPLE_COUNT:
        for row in ranked:
            if row not in selected:
                selected.append(row)
            if len(selected) == SAMPLE_COUNT:
                break
    if len(selected) != SAMPLE_COUNT:
        raise RuntimeError("Could not deterministically select the requested GRPO smoke rows")

    for row in selected:
        for image in row["images"]:
            resolved = (destination / image).resolve()
            if root not in resolved.parents or not resolved.is_file():
                raise FileNotFoundError(f"Smoke reference is not a project-local file: {image}")

    train = "\n".join(json.dumps(row, ensure_ascii=False) for row in selected) + "\n"
    manifest = {
        "source": "data/native_subset/grpo/train.jsonl",
        "selection": "train-only rows nearest normalized target length 16; distinct style_id first; id tie-break",
        "sample_count": SAMPLE_COUNT,
        "rows": [{"id": row["id"], "style_id": row["style_id"], "target_length": text_length(row["target_text"])} for row in selected],
    }
    write_atomic(destination / "train.jsonl", train)
    write_atomic(destination / "selection.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("GRPO_SMOKE_MEDIUM_READY " + json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
