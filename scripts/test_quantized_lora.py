#!/usr/bin/env python3
"""Check the real Qwen-Image 2.1 CUDA/INT8/LoRA path on random tiny weights.

This is an architecture compatibility test, not a pretrained quality evaluation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path

import bitsandbytes as bnb
from diffusers import BitsAndBytesConfig, QwenImage21Transformer2DModel
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import save_file, load_file
import torch


TARGETS = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
           "img_mlp.proj", "img_mlp.gate_layer", "img_mlp.out"]


def digest(tensors: dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        hasher.update(name.encode())
        hasher.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def main() -> None:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required")
    root = Path(__file__).resolve().parents[1]
    output = root / "test_artifacts" / ("quantized_lora_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(20261003)
    torch.cuda.reset_peak_memory_stats()
    base = QwenImage21Transformer2DModel(num_layers=1, num_attention_heads=1, attention_head_dim=128,
                                       context_in_dim=4096, in_channels=64, out_channels=64,
                                       axes_dims_rope=(16, 56, 56), mlp_ratio=3).to(dtype=torch.bfloat16)
    base.save_pretrained(output / "tiny_random_base", safe_serialization=True)
    del base
    gc.collect()
    model = QwenImage21Transformer2DModel.from_pretrained(
        output / "tiny_random_base", quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.bfloat16, local_files_only=True, device_map={"": 0})
    quantized_count = sum(isinstance(module, bnb.nn.Linear8bitLt) for module in model.modules())
    assert quantized_count > 0, "The model was not actually quantized"
    model.requires_grad_(False)
    model.add_adapter(LoraConfig(r=16, lora_alpha=32, target_modules=TARGETS, init_lora_weights=True))
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and all(p.dtype == torch.float32 for p in trainable)
    # The 24 GB evaluation schedule moves the LoRA-wrapped INT8 transformer
    # between CPU and GPU around prompt encoding.  Verify that this exact
    # quantized architecture supports the round trip before using it on the
    # pretrained weights.
    model.to("cpu")
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    model.to("cuda")
    assert all(parameter.device.type == "cuda" for parameter in model.parameters())
    frozen = {name: p for name, p in model.named_parameters() if not p.requires_grad}
    frozen_before = digest(frozen)
    lora_before = digest(get_peft_model_state_dict(model))
    inputs = {
        "hidden_states": torch.randn(1, 8, 64, device="cuda", dtype=torch.bfloat16),
        "encoder_hidden_states": torch.randn(1, 6, 4096, device="cuda", dtype=torch.bfloat16),
        "timestep": torch.tensor([0.5], device="cuda", dtype=torch.bfloat16),
        "img_shapes": [[(1, 2, 2), (1, 2, 2)]],
        # One VLM reference-image slot and the appended target-image slot.
        "img_mask": torch.tensor([[False, False, True, False, False, False, True]], device="cuda"),
        "encoder_hidden_states_mask": torch.ones(1, 6, device="cuda", dtype=torch.bool),
    }
    def forward(**extra: object) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(**inputs, **extra).sample[:, -4:]

    with torch.no_grad():
        initial = forward()
        model.disable_adapters()
        untuned = forward()
        model.enable_adapters()
        assert torch.equal(initial, untuned), "Zero-initialized LoRA changed the untuned baseline"
        cache = QwenImage21KVCache(num_layers=1)
        extracted = forward(kv_cache=cache, kv_cache_mode="extract")
        cached = forward(kv_cache=cache, kv_cache_mode="cached")
        kv_error = float((extracted.float() - cached.float()).abs().max())
        assert torch.allclose(extracted, cached, atol=0.02, rtol=0.02), "Prefix cache changed the output"
    model.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    loss = forward().float().square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in trainable if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    gradient_norm = float(torch.sqrt(sum(g.float().square().sum() for g in gradients)))
    assert gradient_norm > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state = {name: tensor.detach().cpu().clone().contiguous() for name, tensor in get_peft_model_state_dict(model).items()}
    lora_after = digest(state)
    assert lora_before != lora_after, "The LoRA optimizer did not update any parameter"
    assert frozen_before == digest(frozen), "A frozen base parameter changed"
    save_file(state, output / "adapter_model.safetensors")
    with torch.no_grad():
        trained = forward().detach().cpu()
        for parameter in trainable:
            parameter.zero_()
    loaded = load_file(output / "adapter_model.safetensors")
    set_peft_model_state_dict(model, loaded, adapter_name="default")
    with torch.no_grad():
        reloaded = forward().detach().cpu()
    assert torch.equal(trained, reloaded), "The exported LoRA adapter did not reload identically"
    report = {"checks_passed": True, "weights": "tiny randomly initialized architecture fixture; not pretrained",
              "quantized_linear_layers": quantized_count, "trainable_parameters": sum(p.numel() for p in trainable),
              "loss": float(loss.detach()), "gradient_norm": gradient_norm,
              "zero_lora_baseline_equal": True, "frozen_base_unchanged": True, "adapter_reload_equal": True,
              "explicit_cpu_offload_round_trip": True, "prefix_cache_max_error": kv_error,
              "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
              "before_sha256": lora_before, "after_sha256": lora_after}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("QUANTIZED_LORA_TEST_PASSED " + json.dumps(report), flush=True)
    print(f"TEST_ARTIFACT_DIR={output}", flush=True)


if __name__ == "__main__":
    main()
