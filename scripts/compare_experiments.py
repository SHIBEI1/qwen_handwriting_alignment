#!/usr/bin/env python3
"""Compare matched generation records using automatic, reproducible metrics only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("runs/comparison"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source, output = args.input_root.resolve(), args.output.resolve()
    if root not in source.parents or root not in output.parents:
        raise ValueError("Comparison inputs and artifacts must be inside this project")
    output.mkdir(parents=True, exist_ok=True)
    summaries, rows = {}, {}
    for stage in ("base", "sft", "grpo"):
        folder = source / stage / "evaluation"
        summaries[stage] = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        records = json.loads((folder / "cases.json").read_text(encoding="utf-8"))
        rows[stage] = {(row["id"], row["seed"]): row for row in records}
        assert len(rows[stage]) == len(records), "Duplicate evaluation id/seed"
        for row in records:
            image_path = (folder / row["image"]).resolve()
            assert root in image_path.parents and image_path.is_file(), "Missing or external evaluation image"
    settings = ("split", "num_cases", "num_images", "seeds", "resolution", "num_inference_steps", "guidance_scale", "inference_precision", "base_quantization", "base_model_id", "base_model_revision")
    keys = set(rows["base"])
    for stage in ("sft", "grpo"):
        assert keys == set(rows[stage]), "Unmatched evaluation cases"
        assert all(summaries[stage][field] == summaries["base"][field] for field in settings), "Generation settings differ"
        assert all(rows[stage][key]["text"] == rows["base"][key]["text"] and
                   rows[stage][key]["style_id"] == rows["base"][key]["style_id"] and
                   rows[stage][key]["reference_sha256"] == rows["base"][key]["reference_sha256"] for key in keys)
    ids = sorted({key[0] for key in keys})
    characters = np.array([sum(row["evaluation_ocr_metrics"]["characters"] for key, row in rows["base"].items() if key[0] == case_id) for case_id in ids])
    errors = {stage: np.array([sum(row["evaluation_ocr_metrics"]["edits"] for key, row in records.items() if key[0] == case_id) for case_id in ids]) for stage, records in rows.items()}
    rng = np.random.default_rng(20261003)
    samples = rng.integers(0, len(ids), size=(2000, len(ids)))
    differences = {}
    for first, second in (("base", "sft"), ("sft", "grpo"), ("base", "grpo")):
        difference = errors[second] - errors[first]
        bootstrap = difference[samples].sum(axis=1) / characters[samples].sum(axis=1)
        differences[f"{second}_minus_{first}"] = {"evaluation_cer_delta": float(difference.sum() / characters.sum()),
                                                 "paired_case_bootstrap_95_percent": np.quantile(bootstrap, [0.025, 0.975]).tolist()}
    report = {"completed_utc": datetime.now(timezone.utc).isoformat(), "matched_case_settings_verified": True,
              "num_cases": len(ids), "metrics": summaries, "paired_evaluation_cer_differences": differences,
              "automatic_evaluation_only": True, "evaluation_artifact_paths_validated": True,
              "self_test_random_fixture": args.self_test,
              "limitations": ["Single training seed and bounded compute: exploratory, not evidence of algorithm superiority.",
                              "Case bootstrap keeps generation seeds together; it does not measure training-seed variability.",
                              "OCR has measured errors on real handwriting; automatic style is only a stroke-statistics proxy.",
                              "No human preference labels or manual visual review are collected; results remain automatic proxy measures."]}
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("THREE_GROUP_COMPARISON " + json.dumps({"output": str(output), "differences": differences, "self_test": args.self_test}), flush=True)


if __name__ == "__main__":
    main()
