#!/usr/bin/env python3
"""Create explicit single-GPU smoke and experiment recipes with relative paths."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


def recipe(stage: str, smoke: bool) -> dict:
    mode = "sft" if stage == "sft" else "grpo"
    source = f"data/native_subset/{mode}" + ("_smoke" if smoke else "")
    if mode == "grpo" and smoke:
        # The legacy smoke split was constructed from the globally shortest
        # fragments.  It is useful for plumbing, but four-character outputs
        # yield a degenerate all-zero OCR reward at short rollouts.  This
        # deterministic, train-only split is more representative of the
        # actual GRPO prompt lengths.
        source = "data/native_subset/grpo_smoke_medium"
    config = {
        "launcher": "accelerate", "config_file": None, "num_processes": 1, "mixed_precision": "bf16",
        "data": {"datasets": [{"name": "handwriting", "dataset_dir": source, "image_dir": source,
                                  "train": {"weight": 1}}],
                 "enable_preprocess": True, "preprocessing_batch_size": 1, "dataloader_num_workers": 0,
                 "force_reprocess": False, "cache_dir": "cache/flow_factory", "sampler_type": "auto"},
        "model": {"finetune_type": "lora", "lora_rank": 16, "lora_alpha": 32, "target_modules": "default",
                  "model_name_or_path": "models/qwen_image_2_1", "model_type": "handwriting_align.adapter.SingleGPUQwenAdapter",
                  "trainable_parameters_dtype": "fp32", "component_load_dtypes": "bf16",
                  "resume_path": None, "resume_type": None},
        "log": {"run_name": mode, "project": "Qwen-Handwriting-Alignment",
                "logging_backend": "handwriting_align.logger.MetricJSONLogger", "save_dir": "runs",
                "save_freq": 1 if mode == "sft" else 10, "save_model_only": True, "verbose": True},
        "train": {"trainer_type": mode, "max_epochs": 1 if smoke else (3 if mode == "sft" else 100),
                  "resolution": [256, 768], "condition_image_size": [384, 384], "guidance_scale": 1.0,
                  "per_device_batch_size": 1, "ema_decay": 0.0, "enable_gradient_checkpointing": True,
                  # Native prefix KV caching is lossless, but during LoRA GRPO
                  # replay it co-resides with the policy activation graph.  On the
                  # single 24-GB GPU this exhausts memory before a real update can
                  # occur.  SFT can use it safely; GRPO deliberately uses the
                  # uncached, exact forward path instead.
                  "latent_storage_dtype": "fp32", "use_kv_cache": mode == "sft", "seed": 20261003},
        "eval": {"eval_freq": 0, "resolution": [256, 768], "num_inference_steps": 16, "guidance_scale": 1.0, "seed": 43},
        "optimizers": [{"name": "default", "learning_rate": 5e-5 if mode == "sft" else 1e-5,
                        "weight_decay": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8, "max_grad_norm": 1.0}],
    }
    if mode == "sft":
        config["scheduler"] = {"dynamics_type": "ODE"}
        config["train"].update({"gradient_accumulation_steps": 2, "weighting_scheme": "logit_normal",
                                "num_train_timesteps": 1, "timestep_range": 0.99, "time_shift": 1.0})
    else:
        # Keep the formal recipe structurally identical to the successful
        # smoke run: two samples per prompt are sufficient for a relative
        # advantage, and sixteen steps match the fixed evaluation sampler.
        # The uncached path remains necessary on the 24-GB GPU.
        steps = 16
        group = 2
        config["scheduler"] = {"dynamics_type": "Flow-SDE", "noise_level": 0.7,
                               "num_sde_steps": steps - 1, "sde_steps": list(range(steps - 1)), "seed": 20261003}
        config["train"].update({"num_inference_steps": steps, "group_size": group, "unique_sample_num_per_epoch": 2,
                                "gradient_accumulation_steps": group * (steps - 1), "num_inner_epochs": 1,
                                "shuffle_samples": False, "offload_samples_to_cpu": True,
                                "advantage_aggregation": "sum", "global_std": False,
                                "clip_range": [-1e-4, 1e-4], "adv_clip_range": [-5.0, 5.0],
                                "kl_type": "v-based", "kl_beta": 0.01, "ref_param_device": "cpu"})
        config["rewards"] = [{"name": "handwriting", "reward_model": "handwriting_align.rewards.HandwritingReward",
                              "weight": 1.0, "device": "cpu", "dtype": "float32", "batch_size": 1, "async_reward": False}]
    return config


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    folder = root / "configs"
    folder.mkdir(parents=True, exist_ok=True)
    for stage in ("sft", "grpo"):
        for smoke in (False, True):
            config = recipe(stage, smoke)
            name = stage + ("_smoke" if smoke else "")
            if smoke:
                config["log"].update({"save_dir": "test_artifacts/gpu_smoke", "run_name": name})
            if stage == "grpo":
                config["model"].update({"resume_path": "test_artifacts/gpu_smoke/sft_smoke/export" if smoke else "runs/sft/export", "resume_type": "lora"})
            (folder / f"{name}.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
            print(f"CONFIGURED {name}", flush=True)
    evaluation = copy.deepcopy(recipe("sft", True))
    evaluation["log"].update({"logging_backend": "none", "save_dir": "runs/evaluation", "run_name": "evaluation"})
    (folder / "evaluation.yaml").write_text(yaml.safe_dump(evaluation, allow_unicode=True, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
