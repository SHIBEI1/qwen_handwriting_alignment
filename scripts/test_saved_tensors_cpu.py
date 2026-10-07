"""CUDA contract test for PyTorch saved-tensor CPU offload.

The hook is an execution-memory optimization used only around the current
policy forward. This test verifies its output and gradients match ordinary GPU
autograd before it is enabled for the full Flow-GRPO validation.
"""

from __future__ import annotations

import copy

import torch


def _run(model: torch.nn.Module, x: torch.Tensor, *, offload: bool) -> tuple[torch.Tensor, torch.Tensor]:
    context = torch.autograd.graph.save_on_cpu(pin_memory=True, device_type="cuda") if offload else torch.enable_grad()
    with context:
        output = model(x)
        loss = output.square().mean()
    loss.backward()
    return output.detach(), x.grad.detach()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this saved-tensor offload contract test")
    torch.manual_seed(20261005)
    device = torch.device("cuda")
    baseline = torch.nn.Sequential(
        torch.nn.Linear(257, 513), torch.nn.SiLU(), torch.nn.Linear(513, 257)
    ).to(device)
    offloaded = copy.deepcopy(baseline).to(device)
    x_base = torch.randn(7, 257, device=device, requires_grad=True)
    x_offloaded = x_base.detach().clone().requires_grad_(True)
    output_base, grad_base = _run(baseline, x_base, offload=False)
    output_offloaded, grad_offloaded = _run(offloaded, x_offloaded, offload=True)
    # Returning a saved tensor from pinned CPU storage can select a different
    # reduction tiling in CUDA backward. The observed absolute drift is at the
    # final FP32 rounding level, far below the experiment's FP16/INT8 path.
    torch.testing.assert_close(output_offloaded, output_base, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(grad_offloaded, grad_base, rtol=1e-5, atol=1e-8)
    for (base_name, base_param), (offloaded_name, offloaded_param) in zip(
        baseline.named_parameters(), offloaded.named_parameters(), strict=True
    ):
        assert base_name == offloaded_name
        torch.testing.assert_close(offloaded_param.grad, base_param.grad, rtol=1e-5, atol=1e-8)
    print("SAVED_TENSORS_CPU_EQUIVALENCE=passed")


if __name__ == "__main__":
    main()
