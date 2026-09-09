"""
Optimizer Configuration from JSON

This example configures a small text encoder from custom_policy.json.
Shared defaults, reusable presets, and parameter-group settings are
organized in a single configuration file.
"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn

from adafactor8bit import Adafactor8Bit, build_param_groups


class TinyTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(256, 64)
        self.position_embed = nn.Embedding(32, 64)
        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(64, 4, 128, dropout=0.0, batch_first=True),
            num_layers=2,
        )
        self.norm = nn.LayerNorm(64)

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embed(tokens) + self.position_embed(positions)
        return self.norm(self.layers(hidden)).mean(dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("custom_policy.json"))
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as file:
        policy = json.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyTextEncoder().to(device)
    groups = build_param_groups(
        model,
        preset="apollo",
        grouping="G2",
        apollo_rank=policy["presets"]["apollo_training"]["apollo_rank"],
        apollo_targets=("self_attn", "linear1", "linear2"),
        policy=policy,
    )
    optimizer = Adafactor8Bit(groups, lr=1e-4)

    print(f"{'Group':<12} {'Elements':>10} {'Rank':>6} {'M format':>10}")
    for group in optimizer.param_groups:
        count = sum(param.numel() for param in group["params"])
        print(f"{group['group_name']:<12} {count:>10,} {group['apollo_rank']:>6} {group['m_quant_type']:>10}")

    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        tokens = torch.randint(0, 256, (4, 16), device=device)
        target = torch.randn(4, 64, device=device)
        loss = nn.functional.mse_loss(model(tokens), target)
        loss.backward()
        optimizer.step()
        print(f"Step {step + 1} | loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
