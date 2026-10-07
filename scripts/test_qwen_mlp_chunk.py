"""CPU contract check for Qwen 2.1 SwiGLU token-axis forward chunking."""

from __future__ import annotations

import copy
import os

import torch

from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21Attention,
    QwenImage21SwiGLUFeedForward,
    QwenImage21TransformerBlock,
    apply_rotary_emb_qwen,
)


def main() -> None:
    torch.manual_seed(20261005)
    chunk_size = int(os.environ.get("QWEN_TOKEN_CHUNK_SIZE", "256"))
    if chunk_size < 1:
        raise ValueError("QWEN_TOKEN_CHUNK_SIZE must be a positive integer")
    full = QwenImage21SwiGLUFeedForward(hidden_size=32, mlp_hidden_size=96).eval()
    chunked = copy.deepcopy(full).eval()
    chunked.set_chunk_feed_forward(chunk_size=chunk_size, dim=1)

    x_full = torch.randn(1, 769, 32, requires_grad=True)
    x_chunked = x_full.detach().clone().requires_grad_(True)
    y_full = full(x_full)
    y_chunked = chunked(x_chunked)
    # Different GEMM row tiling may alter only last-bit CPU rounding. The
    # tolerance is far below FP16/8-bit runtime precision and verifies the
    # token partition preserves the same numerical function.
    torch.testing.assert_close(y_chunked, y_full, rtol=1e-5, atol=1e-6)

    y_full.square().mean().backward()
    y_chunked.square().mean().backward()
    torch.testing.assert_close(x_chunked.grad, x_full.grad, rtol=1e-5, atol=1e-6)
    for (name_full, parameter_full), (name_chunked, parameter_chunked) in zip(
        full.named_parameters(), chunked.named_parameters(), strict=True
    ):
        assert name_full == name_chunked
        torch.testing.assert_close(parameter_chunked.grad, parameter_full.grad, rtol=1e-5, atol=1e-6)

    attention_full = QwenImage21Attention(dim=32, heads=4, dim_head=8).eval()
    attention_chunked = copy.deepcopy(attention_full).eval()
    attention_chunked.set_token_linear_chunking(chunk_size=chunk_size)
    projections = (
        (attention_full.to_q, attention_chunked.to_q),
        (attention_full.to_k, attention_chunked.to_k),
        (attention_full.to_v, attention_chunked.to_v),
        (attention_full.to_out[0], attention_chunked.to_out[0]),
    )
    for full_projection, chunked_projection in projections:
        x_full = torch.randn(1, 769, 32, requires_grad=True)
        x_chunked = x_full.detach().clone().requires_grad_(True)
        full_output = full_projection(x_full)
        chunked_output = attention_chunked._apply_tokenwise_linear(chunked_projection, x_chunked)
        torch.testing.assert_close(chunked_output, full_output, rtol=1e-5, atol=1e-6)
        full_output.square().mean().backward()
        chunked_output.square().mean().backward()
        torch.testing.assert_close(x_chunked.grad, x_full.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(chunked_projection.weight.grad, full_projection.weight.grad, rtol=1e-5, atol=1e-6)

    for full_norm, chunked_norm in (
        (attention_full.norm_q, attention_chunked.norm_q),
        (attention_full.norm_k, attention_chunked.norm_k),
    ):
        attention_full.zero_grad(set_to_none=True)
        attention_chunked.zero_grad(set_to_none=True)
        x_full = torch.randn(1, 769, 4, 8, requires_grad=True)
        x_chunked = x_full.detach().clone().requires_grad_(True)
        full_output = full_norm(x_full)
        chunked_output = attention_chunked._apply_tokenwise_module(chunked_norm, x_chunked)
        torch.testing.assert_close(chunked_output, full_output, rtol=1e-5, atol=1e-6)
        full_output.square().mean().backward()
        chunked_output.square().mean().backward()
        torch.testing.assert_close(x_chunked.grad, x_full.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(chunked_norm.weight.grad, full_norm.weight.grad, rtol=1e-5, atol=1e-6)

    x_full = torch.randn(1, 769, 4, 8, requires_grad=True)
    x_chunked = x_full.detach().clone().requires_grad_(True)
    angles = torch.randn(769, 4)
    rotary = torch.polar(torch.ones_like(angles), angles)
    full_output = apply_rotary_emb_qwen(x_full, rotary, use_real=False)
    chunked_output = attention_chunked._apply_tokenwise_rotary(x_chunked, rotary)
    torch.testing.assert_close(chunked_output, full_output, rtol=1e-5, atol=1e-6)
    full_output.square().mean().backward()
    chunked_output.square().mean().backward()
    torch.testing.assert_close(x_chunked.grad, x_full.grad, rtol=1e-5, atol=1e-6)

    # The residual path must also avoid a full gate*update temporary without
    # changing the token-wise function or any of its three derivatives.
    hidden_full = torch.randn(1, 769, 32, requires_grad=True)
    gate_full = torch.randn(1, 769, 32, requires_grad=True)
    update_source_full = torch.randn(1, 769, 32, requires_grad=True)
    hidden_chunked = hidden_full.detach().clone().requires_grad_(True)
    gate_chunked = gate_full.detach().clone().requires_grad_(True)
    update_source_chunked = update_source_full.detach().clone().requires_grad_(True)
    # A non-leaf update mirrors the attention/MLP result that is safely reused
    # by the memory-bounded residual path; modulation is frozen in LoRA GRPO.
    update_full = update_source_full * 1.3
    update_chunked = update_source_chunked * 1.3
    residual_full = QwenImage21TransformerBlock._add_gated_residual(
        hidden_full, gate_full, update_full, None
    )
    residual_chunked = QwenImage21TransformerBlock._add_gated_residual(
        hidden_chunked, gate_chunked, update_chunked, chunk_size
    )
    torch.testing.assert_close(residual_chunked, residual_full, rtol=1e-5, atol=1e-6)
    residual_full.square().mean().backward()
    residual_chunked.square().mean().backward()
    for chunked_grad, full_grad in (
        (hidden_chunked.grad, hidden_full.grad),
        (gate_chunked.grad, gate_full.grad),
        (update_source_chunked.grad, update_source_full.grad),
    ):
        torch.testing.assert_close(chunked_grad, full_grad, rtol=1e-5, atol=1e-6)

    # Exercise the exact block-level use, including a full token-axis
    # modulation gate and all trainable block parameter gradients.
    block_full = QwenImage21TransformerBlock(dim=32, num_attention_heads=4, attention_head_dim=8).eval()
    block_chunked = copy.deepcopy(block_full).eval()
    block_chunked.set_chunk_feed_forward(chunk_size=chunk_size, dim=1)
    hidden_full = torch.randn(1, 769, 32, requires_grad=True)
    hidden_chunked = hidden_full.detach().clone().requires_grad_(True)
    modulation = torch.randn(2, 128)  # frozen base modulation, batch_size + t=0 row
    target_mask = torch.rand(769) > 0.4
    output_full = block_full(hidden_full, modulation, target_token_mask=target_mask)
    output_chunked = block_chunked(hidden_chunked, modulation, target_token_mask=target_mask)
    torch.testing.assert_close(output_chunked, output_full, rtol=1e-5, atol=1e-6)
    output_full.square().mean().backward()
    output_chunked.square().mean().backward()
    torch.testing.assert_close(hidden_chunked.grad, hidden_full.grad, rtol=1e-5, atol=1e-6)
    for (full_name, full_parameter), (chunked_name, chunked_parameter) in zip(
        block_full.named_parameters(), block_chunked.named_parameters(), strict=True
    ):
        assert full_name == chunked_name
        torch.testing.assert_close(chunked_parameter.grad, full_parameter.grad, rtol=1e-5, atol=1e-6)
    print(f"QWEN_TOKEN_LINEAR_CHUNK_EQUIVALENCE=passed chunk_size={chunk_size}")


if __name__ == "__main__":
    main()
