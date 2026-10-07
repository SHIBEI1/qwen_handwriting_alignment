"""Pointwise handwriting reward on the existing Flow-Factory contract."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from flow_factory.rewards.abc import PointwiseRewardModel, RewardModelOutput

from .metrics import combined_reward, on_white
from .ocr import ChineseOCR


class HandwritingReward(PointwiseRewardModel):
    required_fields = ("image", "condition_images", "metadata")
    use_tensor_inputs = False

    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)
        if str(self.device) != "cpu":
            raise ValueError("The handwriting reward is deliberately CPU-only")
        self.ocr = ChineseOCR()
        self.artifact_dir = None
        self.diagnostics = []
        if os.environ.get("HANDWRITING_REWARD_ARTIFACT_DIR"):
            root = Path(__file__).resolve().parents[2]
            self.artifact_dir = Path(os.environ["HANDWRITING_REWARD_ARTIFACT_DIR"]).resolve()
            if root not in self.artifact_dir.parents:
                raise ValueError("Reward diagnostics must remain inside the project")
            self.artifact_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def __call__(self, prompt=None, image=None, condition_images=None, metadata=None, **kwargs):
        if image is None or condition_images is None or metadata is None:
            raise ValueError("Handwriting reward requires images, references, and metadata")
        if not len(image) == len(condition_images) == len(metadata):
            raise ValueError("Reward batch lengths differ")
        scores, details = [], []
        for generated, references, raw in zip(image, condition_images, metadata):
            if len(references) != 1:
                raise ValueError("Exactly one handwriting reference is required")
            data = json.loads(raw) if isinstance(raw, str) else raw
            recognized = self.ocr.recognize(generated)
            result = combined_reward(data["target_text"], recognized, generated, references[0])
            if self.artifact_dir is not None and len(self.diagnostics) < 12:
                index = len(self.diagnostics)
                on_white(generated).save(self.artifact_dir / f"candidate_{index:03d}.png")
                on_white(references[0]).save(self.artifact_dir / f"reference_{index:03d}.png")
                self.diagnostics.append({"index": index, "target_text": data["target_text"], "recognized": recognized, **result})
                (self.artifact_dir / "records.json").write_text(json.dumps(self.diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            scores.append(result["reward"])
            details.append(result)
        tensor = torch.tensor(scores, dtype=torch.float32, device="cpu")
        if not torch.isfinite(tensor).all():
            raise ValueError("Non-finite handwriting reward")
        return RewardModelOutput(rewards=tensor, extra_info={"per_sample": details})
