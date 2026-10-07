#!/usr/bin/env python3
"""Validate partitions, local OCR, reward contracts, and SFT reference swaps."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from accelerate import Accelerator
from PIL import Image
import torch
import yaml
from flow_factory.hparams import Arguments, RewardArguments
from flow_factory.rewards.loader import load_reward_model
from flow_factory.logger.loader import load_logger

from handwriting_align.adapter import SingleGPUQwenAdapter, _QwenImage21RGBAOutputCodec
from handwriting_align.metrics import combined_reward, edit_distance, normalize_text, text_metrics
from handwriting_align.ocr import ChineseOCR


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "test_artifacts" / ("software_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(parents=True, exist_ok=False)
    data = root / "data/native_subset"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["samples"]
    split_sets = {name: {normalize_text(r["text"]) for r in rows if r["split"] == name} for name in manifest["counts"]}
    split_pages = {name: {r["source_page"] for r in rows if r["split"] == name} for name in manifest["counts"]}
    for a, b in (("train", "development"), ("train", "test"), ("development", "test")):
        assert not split_sets[a] & split_sets[b]
        assert not split_pages[a] & split_pages[b]
    for row in rows:
        for field in ("target", "reference", "source_page", "reference_page"):
            path = (data / row[field]).resolve()
            assert root in path.parents and path.is_file() and not path.is_symlink()
        assert row["source_page"] != row["reference_page"]

    # The GRPO smoke prompts are a deterministic subset of training data, not
    # a convenience sample from development or test.  Their longer text makes
    # group-relative OCR rewards observable without weakening the reward.
    smoke_dir = data / "grpo_smoke_medium"
    smoke_rows = [json.loads(line) for line in (smoke_dir / "train.jsonl").read_text(encoding="utf-8").splitlines() if line]
    smoke_selection = json.loads((smoke_dir / "selection.json").read_text(encoding="utf-8"))
    train_by_id = {row["id"]: row for row in rows if row["split"] == "train"}
    assert len(smoke_rows) == smoke_selection["sample_count"] == 8
    assert smoke_selection["source"] == "data/native_subset/grpo/train.jsonl"
    assert len({row["style_id"] for row in smoke_rows}) == len(smoke_rows)
    for row in smoke_rows:
        source_row = train_by_id[row["id"]]
        assert row["target_text"] == source_row["text"]
        assert len(normalize_text(row["target_text"])) >= 12
        for image_path in row["images"]:
            image_file = (smoke_dir / image_path).resolve()
            assert root in image_file.parents and image_file.is_file() and not image_file.is_symlink()
    assert edit_distance("你好！", "你好") == 1
    assert text_metrics("汉字123。", "汉字123。") ["cer"] == 0
    assert normalize_text("Ａa，\n") == "Aa,"
    sample = rows[0]
    image = Image.open(data / sample["target"]).convert("RGB")
    reference = Image.open(data / sample["reference"]).convert("RGB")
    assert combined_reward(sample["text"], sample["text"], image, reference)["reward"] > 0.85
    assert combined_reward(sample["text"], "", image, reference)["reward"] == 0
    assert combined_reward(sample["text"], sample["text"], Image.new("RGB", image.size, "white"), reference)["reward"] == 0

    # Qwen-Image 2.1 converts RGB targets to RGBA only at its native VAE
    # boundary; accepting RGB here would reintroduce the generic-codec bug.
    _QwenImage21RGBAOutputCodec._validate_pixel_values(torch.zeros((1, 4, 32, 64)), 1, 32, 64)
    try:
        _QwenImage21RGBAOutputCodec._validate_pixel_values(torch.zeros((1, 3, 32, 64)), 1, 32, 64)
    except ValueError:
        pass
    else:
        raise AssertionError("Qwen RGBA codec accepted an RGB VAE tensor")

    # Exercise the real snapshot helpers on a tiny, non-zero SFT-like tensor fixture.
    fixture = object.__new__(SingleGPUQwenAdapter)
    fixture.model_args = SimpleNamespace(finetune_type="lora")
    fixture.training_args = SimpleNamespace(requires_ref_model=True)
    fixture.target_module_map = {"transformer": ["weight"]}
    fixture._named_parameters = {}
    module = torch.nn.Linear(2, 1, bias=False)
    module.weight.data.fill_(0.7)
    fixture.has_component = lambda name: name == "transformer"
    fixture.get_component = lambda name: module
    fixture._init_ref_parameters()
    module.weight.data.fill_(1.1)
    with fixture.use_ref_parameters():
        assert torch.allclose(module.weight, torch.full_like(module.weight, 0.7))
    assert torch.allclose(module.weight, torch.full_like(module.weight, 1.1))
    policy_loss = module(torch.ones(1, 2)).square().sum()
    with torch.no_grad(), fixture.use_ref_parameters():
        assert torch.allclose(module(torch.ones(1, 2)), torch.tensor([[1.4]]))
    policy_loss.backward()
    assert torch.allclose(module.weight.grad, torch.full_like(module.weight, 4.4))

    for name in ("sft", "sft_smoke", "grpo", "grpo_smoke", "evaluation"):
        raw = yaml.safe_load((root / f"configs/{name}.yaml").read_text(encoding="utf-8"))
        raw["log"]["save_dir"] = str(output / "parsed_configs")
        path = output / f"{name}.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        parsed = Arguments.load_from_yaml(str(path))
        assert parsed.model_args.lora_rank == 16
        if name.startswith("grpo"):
            assert parsed.training_args.advantage_aggregation == "sum"
            assert parsed.training_args.group_size >= 2
            assert parsed.scheduler_args.num_sde_steps == parsed.training_args.num_inference_steps - 1
        if name == "sft_smoke":
            metric_logger = load_logger(parsed)
            metric_logger.log_data({"test_loss": 1.0}, step=0)

    ocr = ChineseOCR()
    evaluator = ChineseOCR(independent_evaluator=True)
    calibration = []
    for row in [r for r in rows if r["split"] == "development"][:16]:
        target = Image.open(data / row["target"]).convert("RGB")
        recognized, independent = ocr.recognize(target), evaluator.recognize(target)
        calibration.append({"id": row["id"], "text": row["text"], "reward_ocr": recognized,
                            "evaluation_ocr": independent, "reward_ocr_metrics": text_metrics(row["text"], recognized),
                            "evaluation_ocr_metrics": text_metrics(row["text"], independent)})
    reward_config = RewardArguments(name="handwriting", reward_model="handwriting_align.rewards.HandwritingReward",
                                    device="cpu", dtype="float32", batch_size=2)
    reward = load_reward_model(reward_config, Accelerator(cpu=True))
    metadata = json.dumps({"target_text": sample["text"]}, ensure_ascii=False)
    blank = Image.new("RGB", image.size, "white")
    result = reward(image=[image, blank], condition_images=[[reference], [reference]], metadata=[metadata, metadata])
    tail = reward(image=[blank], condition_images=[[reference]], metadata=[metadata])
    assert result.rewards.shape == (2,) and tail.rewards.shape == (1,)
    assert result.rewards[1].item() == tail.rewards[0].item() == 0
    assert torch.isfinite(result.rewards).all()
    try:
        reward(image=[image], condition_images=[[reference]], metadata=[])
    except ValueError:
        pass
    else:
        raise AssertionError("Malformed reward batch was accepted")
    calibration_summary = {}
    for key in ("reward_ocr_metrics", "evaluation_ocr_metrics"):
        calibration_summary[key] = sum(r[key]["edits"] for r in calibration) / sum(r[key]["characters"] for r in calibration)
    calibration_passed = calibration_summary["reward_ocr_metrics"] <= 0.4
    report = {"checks_passed": calibration_passed, "dataset_counts": manifest["counts"], "train_writers": manifest["train_writers"],
              "grpo_smoke_medium_train_only_passed": True,
              "reference_snapshot_swap_passed": True, "reward_full_and_tail_batch_passed": True,
              "real_handwriting_ocr_cer": calibration_summary, "calibration": calibration,
              "limitations": ["Stroke-style score is a low-level proxy, not writer identity or human preference.",
                              "The two OCR readers share the same detector; evaluation is not fully independent."]}
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"SOFTWARE_TEST_RESULTS": str(output), "checks_passed": calibration_passed, "calibration_cer": calibration_summary}, ensure_ascii=False), flush=True)
    if not calibration_passed:
        raise RuntimeError("OCR calibration CER exceeds 0.4; inspect before GRPO")
    print(f"TEST_ARTIFACT_DIR={output}", flush=True)


if __name__ == "__main__":
    main()
