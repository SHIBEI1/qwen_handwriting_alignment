#!/usr/bin/env python3
"""Unit-test orchestration and matched reporting using explicit synthetic fixtures."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from PIL import Image
import yaml

from experiment_pipeline import Experiment
from handwriting_align.config import load_config


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "test_artifacts" / ("pipeline_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(parents=True, exist_ok=False)
    experiment = Experiment(72, 10)
    experiment.state_path = output / "fixture_state.json"
    experiment.state = {"completed_phases": []}
    experiment.update(status="synthetic_unit_test")
    config_path = experiment.derived_config("configs/sft.yaml", str((output / "sft_recipe.yaml").relative_to(root)),
                                             lambda config: config["train"].update(max_epochs=2))
    assert yaml.safe_load((root / config_path).read_text())["train"]["max_epochs"] == 2
    with patch("experiment_pipeline.subprocess.check_output", return_value="24100\n"), patch("experiment_pipeline.psutil.virtual_memory") as memory:
        memory.return_value.available = 29 * 2**30
        experiment.wait_resources()
    parsed = load_config(root / "configs/sft.yaml")
    assert str(root / "cache/flow_factory") in parsed.data_args.cache_dir
    namespace = hashlib.sha256(str(root).encode()).hexdigest()[:16]
    assert parsed.data_args.cache_dir.endswith(namespace)
    for stage, edits in (("base", 2), ("sft", 1), ("grpo", 0)):
        folder = output / "fixture_runs" / stage / "evaluation"
        folder.mkdir(parents=True)
        records = []
        for case in ("fixture_a", "fixture_b"):
            for seed in (43, 20261005):
                filename = f"{case}_{seed}.png"
                Image.new("RGB", (80, 32), (200 + edits * 10, 200, 200)).save(folder / filename)
                records.append({"id": case, "seed": seed, "text": "ab1", "style_id": "fixture_writer",
                                "reference_sha256": "synthetic_fixture_only", "image": filename,
                                "evaluation_ocr_metrics": {"edits": edits, "characters": 3}})
        settings = {"split": "synthetic_fixture", "num_cases": 2, "num_images": 4, "seeds": [43, 20261005],
                    "resolution": [32, 80], "num_inference_steps": 0, "guidance_scale": 1.0,
                    "base_quantization": "no real model; unit fixture only", "base_model_id": "synthetic_fixture",
                    "base_model_revision": "synthetic_fixture"}
        (folder / "metrics.json").write_text(json.dumps(settings), encoding="utf-8")
        (folder / "cases.json").write_text(json.dumps(records), encoding="utf-8")
    subprocess.run([sys.executable, "scripts/compare_experiments.py", "--input-root", str(output / "fixture_runs"),
                    "--output", str(output / "fixture_comparison"), "--self-test"], cwd=root, check=True)
    report = json.loads((output / "fixture_comparison/comparison.json").read_text())
    assert report["self_test_random_fixture"] is True
    assert report["automatic_evaluation_only"] is True
    for key in ("sft_minus_base", "grpo_minus_sft"):
        assert abs(report["paired_evaluation_cer_differences"][key]["evaluation_cer_delta"] + 1 / 3) < 1e-12
    (output / "results.json").write_text(json.dumps({"checks_passed": True,
        "scope": "synthetic reporting and orchestration fixtures, not pretrained quality"}, indent=2) + "\n")
    print(f"PIPELINE_UNIT_TEST_PASSED {output}", flush=True)
    print(f"TEST_ARTIFACT_DIR={output}", flush=True)


if __name__ == "__main__":
    main()
