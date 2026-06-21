"""
Hybrid Routing and Fira Limiter Example for Adafactor8Bit

This script demonstrates practical optimization strategies for large-scale models:
1. Hybrid Routing: Using APOLLO for 2D weights, and Adafactor for non-2D weights.
2. Fira Limiter: Enabling norm-growth limiting to stabilize training. 
   External gradient clipping can be safely removed when this is enabled.
"""
import torch
import torch.nn as nn
from adafactor8bit import Adafactor8Bit


class DummyModel(nn.Module):
    """A model containing 1D, 2D, and >2D parameters to demonstrate hybrid routing."""
    def __init__(self):
        super().__init__()
        # 1D / Sensitive Parameters (Target: FP32, No WD, Adafactor)
        self.embed = nn.Embedding(1000, 64)
        self.norm = nn.LayerNorm(64)
        
        # 2D Parameters (Target: 8-bit, WD, APOLLO)
        self.linear = nn.Linear(64, 10)
        
        # >2D Parameters (Target: 8-bit, WD, Adafactor)
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.head = nn.Linear(16, 10)

    def forward(self, x_text, x_img):
        text_feat = self.norm(self.embed(x_text)).mean(dim=1)
        text_out = self.linear(text_feat)
        
        img_feat = self.conv(x_img).mean(dim=[2, 3])
        img_out = self.head(img_feat)
        
        return text_out + img_out


def get_param_groups(model, weight_decay, apollo_rank=256):
    """
    Precision routing for different parameter dimensions:
    - Group 1 (1D/Sensitive): Norms, Biases -> FP32, No WD, Adafactor.
    - Group 2 (Embeddings): 2D but sparse -> FP32, No WD, APOLLO (Avoids Adafactor's cold-start explosion).
    - Group 3 (2D Weights): Linear layers -> 8-bit, WD, APOLLO.
    - Group 4 (>2D Weights): Conv layers -> 8-bit, WD, Adafactor.
    """
    group_1d_sensitive = []
    group_embed = []
    group_2d_weights = []
    group_nd_weights = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 1. Intercept 1D sensitive layers (Norms, Biases)
        is_sensitive_1d = param.ndim <= 1 or "bias" in name or "norm" in name
        
        if is_sensitive_1d:
            group_1d_sensitive.append(param)
        # 2. Intercept Embeddings (2D and highly sparse)
        elif "embed" in name.lower():
            group_embed.append(param)
        # 3. Standard 2D Weights (Linear)
        elif param.ndim == 2:
            group_2d_weights.append(param)
        # 4. High-dimensional Weights (Conv)
        else:
            group_nd_weights.append(param)

    return [
        {
            "params": group_1d_sensitive,
            "weight_decay": 0.0,
            "quantize": False,
            "apollo_rank": 0,
        },
        {
            "params": group_embed,
            "weight_decay": 0.0,
            "quantize": False,
            # APOLLO handles sparse gradients gracefully 
            # Apollo 的低秩方差不会逐行孤立，即使嵌入行稀疏，仍能利用全局统计稳定缩放
            "apollo_rank": apollo_rank, 
        },
        {
            "params": group_2d_weights,
            "weight_decay": weight_decay,
            "quantize": True,
            "apollo_rank": apollo_rank,
        },
        {
            "params": group_nd_weights,
            "weight_decay": weight_decay,
            "quantize": True,
            # APOLLO is disabled for >2D tensors to preserve spatial structures 
            # and ensure meaningful channel-wise scaling.
            "apollo_rank": 0, 
        }
    ]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DummyModel().to(device)
    
    optimizer = Adafactor8Bit(
        get_param_groups(model, weight_decay=1e-2, apollo_rank=256),
        lr=1e-3,
        relative_step=False,
        beta2=0.999,
        enable_fira_for_adafactor=True, 
        fira_margin=0.01,               
    )
    
    for step in range(1, 11):
        optimizer.zero_grad()
        
        x_text = torch.randint(0, 1000, (4, 16), device=device)
        x_img = torch.randn(4, 3, 32, 32, device=device)
        target = torch.randn(4, 10, device=device)
        
        output = model(x_text, x_img)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # With Fira Limiter enabled, external gradient clipping is generally 
        # not required and can be safely removed to simplify the training pipeline.
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        print(f"Step {step:02d} | Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()