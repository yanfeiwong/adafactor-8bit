"""
Basic Usage Example for Adafactor8Bit

This script demonstrates a common practice for using the optimizer: 
using parameter groups to protect sensitive layers (Embeddings, Norms, Biases) 
while applying 8-bit quantization to large weight matrices.
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
            # For massive token embeddings with highly sparse updates, please refer to 
            # `advanced_usage.py` for a more specialized routing strategy 
            # (e.g., using `factored=False` and `scale_parameter=False`) to handle sparse gradients effectively.
            no_decay.append(param)
        else:
            decay.append(param)
            
    return [
        {"params": decay, "weight_decay": weight_decay, "quantize": True},
        {"params": no_decay, "weight_decay": 0.0, "quantize": False}
    ]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DummyModel().to(device)
    
    optimizer = Adafactor8Bit(
        get_param_groups(model), 
        lr=1e-3, 
        relative_step=False,
        beta2=0.999,
    )
    
    for step in range(10):
        optimizer.zero_grad()
        
        dummy_input = torch.randint(0, 1000, (4, 16), device=device)
        dummy_target = torch.randn(4, 16, 10, device=device)
        
        output = model(dummy_input)
        loss = nn.MSELoss()(output, dummy_target)
        loss.backward()
        
        optimizer.step()  # First call triggers JIT compilation
        print(f"Step {step + 1} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()
