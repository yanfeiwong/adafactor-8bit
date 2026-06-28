"""
Advanced Usage Example for Adafactor8Bit

This script demonstrates a hybrid grouping strategy for complex architectures 
(e.g., Vision-Language Models, Diffusion UNets) to achieve stable and efficient training.

Strategies applied:
1. 1D / Sensitive Parameters: FP32, no weight decay.
2. Embedding Layers: Element-wise variance (momentum-free Adam style) for fine-grained updates.
3. 2D Weights: 8-bit quantization with APOLLO low-rank projection.
4. >2D Weights: 8-bit quantization with full-rank variance to preserve spatial structures.
"""
import torch
import torch.nn as nn
from adafactor8bit import Adafactor8Bit

# Define learning rates
lr = 1e-3
lr_emb = 1e-4 # For Embedding layers, we use an Adam-style learning rate


class DummyModel(nn.Module):
    """A model containing 1D, 2D, and >2D parameters to demonstrate hybrid routing."""
    def __init__(self):
        super().__init__()
        # 1D / Sensitive Parameters
        self.norm = nn.LayerNorm(64)
        
        # Embeddings
        self.token_embed = nn.Embedding(1000, 64)
        self.pos_embed = nn.Embedding(128, 64) # Position embeddings are dense, routed to 1D/2D logic
        
        # 2D Parameters 
        self.linear = nn.Linear(64, 10)
        
        # >2D Parameters 
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.head = nn.Linear(16, 10)

    def forward(self, x_text, x_img):
        seq_len = x_text.size(1)
        positions = torch.arange(seq_len, device=x_text.device).unsqueeze(0)
        text_feat = self.norm(self.token_embed(x_text) + self.pos_embed(positions)).mean(dim=1)
        text_out = self.linear(text_feat)
        
        img_feat = self.conv(x_img).mean(dim=[2, 3])
        img_out = self.head(img_feat)
        
        return text_out + img_out


def get_param_groups(model, lr_emb, weight_decay, apollo_rank=256):
    """
    Precision routing for different parameter dimensions.
    """
    group_1d, group_embed, group_2d, group_nd = [], [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        is_1d = param.ndim <= 1 or "bias" in name or "norm" in name
        # Match true Token Embeddings, excluding Position and Time Embeddings
        is_embedding = ("embed" in name.lower() 
                        and "position" not in name.lower() 
                        and "pos_embed" not in name.lower()
                        and "time" not in name.lower())
        
        if is_1d:
            group_1d.append(param)
        elif is_embedding:
            group_embed.append(param)
        elif param.ndim == 2:
            group_2d.append(param)
        else:
            group_nd.append(param)

    return [
        # 1. 1D / Sensitive: FP32, No Weight Decay
        {"params": group_1d, "weight_decay": 0.0, "quantize": False, "apollo_rank": 0},
        
        # 2. Embeddings: Recreating a momentum-free Adam
        {
            "params": group_embed, 
            "weight_decay": 0.0, 
            "quantize": False,
            "apollo_rank": 0,
            "factored": False,         # Enable element-wise variance
            "scale_parameter": False,  # Disable internal RMS scaling
            "d": 1e9,                  # Disable global Trust-Region clipping
            "lr": lr_emb               # Override global learning rate
        },
        
        # 3. 2D Weights: 8-bit quantization, Weight Decay, APOLLO low-rank projection
        {
            "params": group_2d, 
            "weight_decay": weight_decay, 
            "quantize": True, 
            "apollo_rank": apollo_rank,
            "beta1":0.9,               # Remove if minimizing optimizer memory is the priority.
        },
        
        # 4. >2D Weights: 8-bit quantization, Weight Decay, Full-Rank
        {
            "params": group_nd, 
            "weight_decay": weight_decay, 
            "quantize": True, 
            "apollo_rank": 0,
            "beta1":0.9,               # Remove if minimizing optimizer memory is the priority.
            "factored": False          # Disables factorization to preserve spatial structures, enabling finer gradient scaling.
                                       # Note: This increases state memory for >2D weights, depending on your model architecture.
                                       # If VRAM is constrained, reverting to factored=True is a safe alternative.
        },
    ]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DummyModel().to(device)
    
    optimizer = Adafactor8Bit(
        get_param_groups(model, lr_emb=lr_emb, weight_decay=1e-2, apollo_rank=256),
        lr=lr,
        # For continual learning or when using an external LR scheduler
        relative_step=False,
        beta2=0.999,
        enable_fira_for_adafactor=True,
        fira_margin=0.01,
    )


    for step in range(10):
        optimizer.zero_grad()
        
        x_text = torch.randint(0, 1000, (4, 16), device=device)
        x_img = torch.randn(4, 3, 32, 32, device=device)
        target = torch.randn(4, 10, device=device)
        
        output = model(x_text, x_img)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # With Fira Limiter enabled, external gradient clipping is generally 
        # not required and can be safely removed.
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        print(f"Step {step:02d} | Loss: {loss.item():.4f}")

    print("Training completed!")


if __name__ == "__main__":
    main()