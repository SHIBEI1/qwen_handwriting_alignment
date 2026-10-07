#!/usr/bin/env python3
"""Export both backend dataset schemas from the fixed, independent case list."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    data = Path(__file__).resolve().parents[1] / "data/native_subset"
    cases = json.loads((data / "train_cases.json").read_text(encoding="utf-8"))
    smoke_ids = {row["id"] for row in sorted(cases, key=lambda row: (len(row["text"]), row["id"]))[:8]}
    for mode in ("sft", "grpo"):
        lines = []
        for row in cases:
            metadata = {"id": row["id"], "target_text": row["text"], "style_id": row["style_id"]}
            if mode == "sft":
                record = {"schema_version": 2, "input": {"prompt": row["prompt"], "media": [{"type": "image", "path": "../" + row["reference"]}]},
                          "supervision": {"type": "demonstration", "target": {"media": [{"type": "image", "path": "../" + row["target"]}]}}, "metadata": metadata}
            else:
                record = {"prompt": row["prompt"], "images": ["../" + row["reference"]], **metadata}
            lines.append(json.dumps(record, ensure_ascii=False))
        smoke_lines = [line for row, line in zip(cases, lines) if row["id"] in smoke_ids]
        for name, rows in ((mode, lines), (mode + "_smoke", smoke_lines)):
            folder = data / name
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "train.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
            print(f"EXPORTED {name} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
