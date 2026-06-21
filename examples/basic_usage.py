"""
Basic Usage Example for Adafactor8Bit

This script demonstrates the recommended best practices for using the optimizer,
specifically how to use parameter groups to protect sensitive layers 
(Embeddings, Norms, Biases) while aggressively quantizing large weight matrices.
"""
import torch
import torch.nn as nn
from adafactor8bit import Adafactor8Bit


class DummyModel(nn.Module):
    """A minimal model containing layers that require different optimization strategies."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 64)
        self.linear = nn.Linear(64, 128)
        self.norm = nn.LayerNorm(128)
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        x = self.embed(x)
        x = self.linear(x)
        x = self.norm(x)
        return self.head(x)


def get_param_groups(model, weight_decay=1e-2):
    """
    Separates model parameters into two groups:
    1. decay: Large 2D weight matrices (Linear/Conv) -> Quantized to 8-bit.
    2. no_decay: 1D vectors, biases, norms, and embeddings -> Kept in FP32 for stability.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Heuristic: Protect 1D tensors, biases, norms, and embeddings
        if param.ndim <= 1 or "bias" in name or "norm" in name or "embed" in name:
            # Note: Grouping embeddings here works well for layers with dense gradient updates. 
            # However, for massive token embeddings with highly sparse updates (e.g., large vocabularies combined with small batch sizes), 
            # please refer to `advanced_usage.py` to route them to the APOLLO path 
            # to avoid Adafactor's cold-start variance explosion.
            no_decay.append(param)
        else:
            decay.append(param)
            
    return [
        {"params": decay, "weight_decay": weight_decay, "quantize": True},
        {"params": no_decay, "weight_decay": 0.0, "quantize": False}
    ]


def main():
    # Directly use CUDA. Users of this library are expected to have a GPU environment.
    device = torch.device("cuda")
    model = DummyModel().to(device)
    
    # Initialize optimizer with grouped parameters
    optimizer = Adafactor8Bit(
        get_param_groups(model), 
        lr=1e-3, 
        # Uncomment for continual learning with external scheduler
        # relative_step=False,     # Disable internal LR scheduling
        # beta2=0.999,             # Lock EMA window to prevent "blunting" over steps

        # Uncomment to try the new APOLLO Subspace Projection
        # Simulates full-rank adaptive scaling in a low-rank space, improving generalization and potentially accelerating convergence.
        # apollo_rank=256,             # 0 to disable. 256 is the official APOLLO default.
    )
    
    print("Starting dummy training loop...")
    for step in range(10):
        optimizer.zero_grad()
        
        # Generate dummy data
        dummy_input = torch.randint(0, 1000, (4, 16), device=device)
        dummy_target = torch.randn(4, 16, 10, device=device)
        
        # Forward & Backward pass
        output = model(dummy_input)
        loss = nn.MSELoss()(output, dummy_target)
        loss.backward()
        
        # Optimization step (triggers JIT compilation of CUDA kernels on the first run)
        optimizer.step()
        print(f"Step {step+1} | Loss: {loss.item():.4f}")
        
    print("Training completed!")


if __name__ == "__main__":
    main()