#!/usr/bin/env python3
"""Evaluate all stages on identical text/reference cases and generation settings."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time

from accelerate import Accelerator
import numpy as np
from PIL import Image
import torch

from handwriting_align.adapter import SingleGPUQwenAdapter
from handwriting_align.config import load_config
from handwriting_align.metrics import combined_reward, on_white, text_metrics
from handwriting_align.ocr import ChineseOCR


def gpu_memory_snapshot(event: str, case_id: str | None = None) -> None:
    """Record live allocator state around the 24 GB component schedule."""
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    payload = {
        "event": event,
        "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
        "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3),
    }
    if case_id is not None:
        payload["case_id"] = case_id
    print("GPU_MEMORY " + json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["base", "sft", "grpo"], required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=["development", "test"], default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 20261005])
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16",
                        help="Component/autocast precision; use the same value for every compared stage.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("The case limit must be positive and generation seeds must be unique")
    root = Path(__file__).resolve().parents[1]
    output = args.output or (root / f"runs/{args.stage}/evaluation")
    output = output.resolve()
    if root not in output.parents:
        raise ValueError("Evaluation artifacts must stay inside this project")
    if (output / "metrics.json").exists():
        raise FileExistsError("Do not silently overwrite a completed evaluation")
    output.mkdir(parents=True, exist_ok=True)
    data = root / "data/native_subset"
    cases = json.loads((data / f"{args.split}_cases.json").read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[:args.limit]
    config = load_config(root / "configs/evaluation.yaml")
    # Keep the frozen INT8 bases identical while selecting the floating-point
    # compute path explicitly for a matched evaluation set.
    config.model_args.component_load_dtypes = args.precision
    if args.stage != "base":
        checkpoint = args.checkpoint or (root / f"runs/{args.stage}/export")
        checkpoint = checkpoint.resolve()
        if root not in checkpoint.parents or not checkpoint.is_dir():
            raise FileNotFoundError("Expected an adapter from this project")
        config.model_args.resume_path = str(checkpoint)
        config.model_args.resume_type = "lora"
    elif args.checkpoint:
        raise ValueError("The untuned baseline must not load an adapter checkpoint")
    accelerator = Accelerator(mixed_precision=args.precision)
    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    adapter = SingleGPUQwenAdapter(config, accelerator)
    adapter.eval()
    gpu_memory_snapshot("adapter_initialized")
    ocr, independent = ChineseOCR(), ChineseOCR(independent_evaluator=True)
    records = []
    # The baseline disables the freshly initialized LoRA entirely and performs no updates.
    context = adapter.transformer.disable_adapter() if args.stage == "base" else nullcontext()
    with context, torch.no_grad():
        # An INT8 bitsandbytes module owns opaque CUDA state in addition to
        # ordinary parameters. Repeated CPU/GPU migrations of the text encoder
        # can leave those states resident across cases, even after the PyTorch
        # allocator reports a lower tensor total. Keep all required components
        # resident for one evaluation process instead. This schedule is used
        # identically for the base, SFT, and GRPO checkpoints.
        adapter.on_load_components(["text_encoders", "vae"])
        adapter.transformer.to(accelerator.device)
        gc.collect()
        torch.cuda.empty_cache()
        gpu_memory_snapshot("components_resident")
        for case in cases:
            reference = Image.open(data / case["reference"]).convert("RGB")
            encode_start = time.monotonic()
            with torch.autocast("cuda", dtype=autocast_dtype):
                encoded = adapter.preprocess_func(prompt=[case["prompt"]], images=[[reference]],
                                                  condition_image_size=(384, 384), guidance_scale=1.0,
                                                  device=accelerator.device)
            encode_seconds = time.monotonic() - encode_start
            for seed in args.seeds:
                generation_start = time.monotonic()
                with torch.autocast("cuda", dtype=autocast_dtype):
                    sample = adapter.inference(prompt=[case["prompt"]], **encoded,
                                               height=256, width=768, guidance_scale=1.0,
                                               num_inference_steps=16, use_kv_cache=True,
                                               compute_log_prob=False, trajectory_indices=[],
                                               generator=torch.Generator(device="cuda").manual_seed(seed))[0]
                tensor = sample.image.detach().float().cpu().clamp(0, 1)
                raw = Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
                generated = on_white(raw)
                path = output / "images" / f"{case['id']}_seed{seed}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                generated.save(path)
                generation_seconds = time.monotonic() - generation_start
                recognized = ocr.recognize(generated)
                independent_text = independent.recognize(generated)
                record = {"id": case["id"], "style_id": case["style_id"], "original_split": case["original_split"],
                          "seed": seed, "text": case["text"], "reward_ocr_text": recognized,
                          "reference_sha256": hashlib.sha256((data / case["reference"]).read_bytes()).hexdigest(),
                          "evaluation_ocr_text": independent_text, "image": str(path.relative_to(output)),
                          "encoding_seconds": encode_seconds, "generation_seconds": generation_seconds,
                          "evaluation_ocr_metrics": text_metrics(case["text"], independent_text),
                          **combined_reward(case["text"], recognized, generated, reference)}
                records.append(record)
                print("EVALUATION_CASE " + json.dumps(record, ensure_ascii=False), flush=True)
                del sample, tensor, raw, generated
            del encoded
            reference.close()
            gc.collect()
            torch.cuda.empty_cache()
            gpu_memory_snapshot("case_cleanup_complete", case["id"])
    adapter.off_load_components(["text_encoders", "vae"])
    adapter.transformer.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    gpu_memory_snapshot("evaluation_cleanup_complete")
    summary = {"stage": args.stage, "split": args.split, "num_cases": len(cases), "num_images": len(records),
               "seeds": args.seeds, "resolution": [256, 768], "num_inference_steps": 16, "guidance_scale": 1.0,
               "inference_precision": args.precision,
               "base_quantization": "identical frozen bitsandbytes int8 transformer and text encoder in all stages",
               "base_model_id": "Qwen/Qwen-Image-2.1", "base_model_revision": "d26bb61231c349cf6b7896fa83353113880e1ba3",
               "reward_reader_cer": sum(r["edits"] for r in records) / sum(r["characters"] for r in records),
               "evaluation_reader_cer": sum(r["evaluation_ocr_metrics"]["edits"] for r in records) / sum(r["evaluation_ocr_metrics"]["characters"] for r in records),
               "exact_match_rate": sum(r["evaluation_ocr_metrics"]["exact_match"] for r in records) / len(records),
               "mean_stroke_style_proxy": sum(r["stroke_style_proxy"] for r in records) / len(records),
               "blank_rate": sum(not r["nonblank"] for r in records) / len(records),
               "mean_generation_seconds": sum(r["generation_seconds"] for r in records) / len(records),
               "elapsed_seconds": time.monotonic() - start, "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
               "completed_utc": datetime.now(timezone.utc).isoformat(),
               "limitations": ["OCR is not ground truth; its calibration on real handwriting quantifies one source of error.",
                               "The stroke-style proxy is content-sensitive and cannot prove writer identity.",
                               "The evaluation reader is different but shares the detector with the reward reader."]}
    (output / "cases.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("EVALUATION_COMPLETE " + json.dumps(summary, ensure_ascii=False), flush=True)
    if root / "test_artifacts" in output.parents:
        print(f"TEST_ARTIFACT_DIR={output}", flush=True)


if __name__ == "__main__":
    main()
