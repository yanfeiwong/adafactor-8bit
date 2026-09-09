"""
BF16 Master Weight Training

BF16 weights use half the parameter storage of FP32 weights. The model
parameters themselves are the master weights, with stochastic rounding helping
small updates accumulate in expectation over many training steps.

This example demonstrates parameter-memory savings and a BF16 training loop.
Total training memory also includes gradients, optimizer states and activations.
"""
import torch
from torch import nn

from adafactor8bit import Adafactor8Bit, build_param_groups


def parameter_bytes(model):
    return sum(param.numel() * param.element_size() for param in model.parameters())


def main():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("This example requires a CUDA GPU with BF16 support.")

    model = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 64)).cuda()
    fp32_bytes = parameter_bytes(model)
    model.to(dtype=torch.bfloat16)
    bf16_bytes = parameter_bytes(model)
    print(f"Parameter storage: {fp32_bytes / 1024:.1f} KiB -> {bf16_bytes / 1024:.1f} KiB")

    groups = build_param_groups(model, preset="adamw")
    optimizer = Adafactor8Bit(
        groups,
        lr=1e-4,
        weight_decay=1e-2,
        stochastic_round_writeback=True,
    )

    for step in range(10):
        optimizer.zero_grad(set_to_none=True)
        inputs = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
        targets = torch.randn(8, 64, device="cuda")
        loss = nn.functional.mse_loss(model(inputs).float(), targets)
        loss.backward()
        optimizer.step()
        print(f"Step {step + 1:2d} | loss {loss.item():.4f}")

    # Save model.state_dict() and optimizer.state_dict() together. The optimizer
    # checkpoint includes the per-parameter stochastic-rounding stream metadata.


if __name__ == "__main__":
    main()
