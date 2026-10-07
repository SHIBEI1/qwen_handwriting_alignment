#!/usr/bin/env python3
"""Measure rollout/replay log-probability agreement without updating weights.

The probe generates one ordinary GRPO acquisition, then compares every stored
transition log-probability against replay in two execution modes: evaluation
without a graph and the real policy-autograd path.  It never calls backward,
optimizer.step, or saves an adapter.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from flow_factory.trainers import load_trainer
from flow_factory.trainers.common.replay_batching import move_and_stack_samples
from handwriting_align.config import load_config


def maximum_ratio_error(current: torch.Tensor, rollout: torch.Tensor) -> float:
    return float((torch.exp(current.float() - rollout.float()) - 1).abs().max().cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.training_args.trainer_type != "grpo":
        raise ValueError("ratio diagnostic requires a GRPO config")

    trainer = load_trainer(config)
    if trainer.training_args.offload_samples_to_cpu is not True:
        raise ValueError("diagnostic requires CPU-offloaded samples for bounded memory")

    with trainer.sampling_context():
        samples = trainer.sample()
    if not samples:
        raise RuntimeError("rollout returned no samples")

    batch = move_and_stack_samples(samples[:1], trainer.accelerator.device)
    results: list[dict[str, object]] = []
    try:
        for raw_step in trainer.adapter.get_train_step_indices():
            step = int(raw_step)
            replay = trainer.adapter.get_replay_step(batch, step)
            old = trainer._require_replay_log_prob(replay, step).detach()

            trainer.adapter.eval()
            with torch.no_grad(), trainer.autocast():
                eval_output = trainer._replay_forward(batch, replay, ("log_prob",))
            eval_log_prob = trainer._require_policy_log_prob(
                eval_output, step, old.shape[0]
            ).detach()
            del eval_output

            trainer.adapter.train()
            with trainer._policy_autograd_storage_context():
                with trainer.autocast():
                    policy_output = trainer._replay_forward(batch, replay, ("log_prob",))
            policy_log_prob = trainer._require_policy_log_prob(
                policy_output, step, old.shape[0]
            ).detach()
            del policy_output

            record = {
                "step": step,
                "rollout_dtype": str(old.dtype),
                "eval_ratio_error": maximum_ratio_error(eval_log_prob, old),
                "policy_ratio_error": maximum_ratio_error(policy_log_prob, old),
                "eval_policy_log_prob_max_abs_difference": float(
                    (eval_log_prob.float() - policy_log_prob.float()).abs().max().cpu()
                ),
            }
            results.append(record)
            print("RATIO_DIAGNOSTIC " + json.dumps(record, sort_keys=True), flush=True)
            del eval_log_prob, policy_log_prob, replay
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        batch.clear()
        batch.samples = []
        trainer.cleanup()

    summary = {
        "algorithm": "grpo_ratio_mode_diagnostic",
        "steps": len(results),
        "max_eval_ratio_error": max(item["eval_ratio_error"] for item in results),
        "max_policy_ratio_error": max(item["policy_ratio_error"] for item in results),
        "max_eval_policy_log_prob_max_abs_difference": max(
            item["eval_policy_log_prob_max_abs_difference"] for item in results
        ),
        "weights_updated": False,
    }
    print("RATIO_DIAGNOSTIC_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
