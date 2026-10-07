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
from contextlib import nullcontext
import logging

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

    logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)
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

            def replay_log_prob(*, rollout_mode: bool, requires_grad: bool) -> torch.Tensor:
                if rollout_mode:
                    trainer.adapter.rollout()
                else:
                    trainer.adapter.train()
                grad_context = torch.enable_grad() if requires_grad else torch.no_grad()
                storage_context = (
                    trainer._policy_autograd_storage_context()
                    if requires_grad
                    else nullcontext()
                )
                with grad_context, storage_context, trainer.autocast():
                    output = trainer._replay_forward(batch, replay, ("log_prob",))
                log_prob = trainer._require_policy_log_prob(output, step, old.shape[0]).detach()
                del output
                return log_prob

            rollout_nograd = replay_log_prob(rollout_mode=True, requires_grad=False)
            rollout_grad = replay_log_prob(rollout_mode=True, requires_grad=True)
            train_nograd = replay_log_prob(rollout_mode=False, requires_grad=False)
            policy_grad = replay_log_prob(rollout_mode=False, requires_grad=True)

            record = {
                "step": step,
                "rollout_dtype": str(old.dtype),
                "rollout_nograd_ratio_error": maximum_ratio_error(rollout_nograd, old),
                "rollout_grad_ratio_error": maximum_ratio_error(rollout_grad, old),
                "train_nograd_ratio_error": maximum_ratio_error(train_nograd, old),
                "policy_grad_ratio_error": maximum_ratio_error(policy_grad, old),
                "rollout_grad_policy_grad_log_prob_max_abs_difference": float(
                    (rollout_grad.float() - policy_grad.float()).abs().max().cpu()
                ),
                "rollout_nograd_rollout_grad_log_prob_max_abs_difference": float(
                    (rollout_nograd.float() - rollout_grad.float()).abs().max().cpu()
                ),
            }
            results.append(record)
            print("RATIO_DIAGNOSTIC " + json.dumps(record, sort_keys=True), flush=True)
            del rollout_nograd, rollout_grad, train_nograd, policy_grad, replay
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        batch.clear()
        batch.samples = []
        trainer.cleanup()

    summary = {
        "algorithm": "grpo_ratio_mode_diagnostic",
        "steps": len(results),
        "max_rollout_nograd_ratio_error": max(item["rollout_nograd_ratio_error"] for item in results),
        "max_rollout_grad_ratio_error": max(item["rollout_grad_ratio_error"] for item in results),
        "max_train_nograd_ratio_error": max(item["train_nograd_ratio_error"] for item in results),
        "max_policy_grad_ratio_error": max(item["policy_grad_ratio_error"] for item in results),
        "max_rollout_grad_policy_grad_log_prob_max_abs_difference": max(
            item["rollout_grad_policy_grad_log_prob_max_abs_difference"] for item in results
        ),
        "max_rollout_nograd_rollout_grad_log_prob_max_abs_difference": max(
            item["rollout_nograd_rollout_grad_log_prob_max_abs_difference"] for item in results
        ),
        "weights_updated": False,
    }
    print("RATIO_DIAGNOSTIC_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
