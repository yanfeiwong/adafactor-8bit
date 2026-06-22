"""
Example of using a dual-optimizer setup with Adafactor8Bit.

In large-scale pretraining or fine-tuning, token embeddings are highly sparse.
This script demonstrates how to split `nn.Embedding` parameters from the rest 
of the network, routing them to a standard FP32 Adam optimizer while the main 
network uses Adafactor8Bit. 

This provides per-token adaptive learning for embeddings, while keeping 
the memory footprint of the main network low.
"""

import torch
import torch.nn as nn
from adafactor8bit import Adafactor8Bit


class DummyModel(nn.Module):
    """A dummy multimodal model for demonstration purposes."""
    def __init__(self):
        super().__init__()
        # Sparse lookup table
        self.token_embed = nn.Embedding(1000, 64)
        
        # Dense layers
        self.norm = nn.LayerNorm(64)
        self.linear = nn.Linear(64, 10)
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.head = nn.Linear(16, 10)

    def forward(self, x_text, x_img):
        text_feat = self.norm(self.token_embed(x_text)).mean(dim=1)
        text_out = self.linear(text_feat)
        
        img_feat = self.conv(x_img).mean(dim=[2, 3])
        img_out = self.head(img_feat)
        
        return text_out + img_out


def split_embedding_params(model):
    """
    Separates Token Embedding weights from other parameters.
    
    Note: Position Embeddings (also nn.Embedding) are explicitly excluded 
    because they are densely updated (due to sequence padding) and should 
    not share the lowered learning rate or sparse-update strategies 
    applied to Token Embeddings.
    """
    emb_params, main_params, seen_ids = [], [], set()
    
    # Match true Token Embedding layers, excluding Position Embeddings
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            # Skip position embeddings (e.g., 'position_embeddings', 'pos_embed')
            if "position" in name.lower() or "pos_embed" in name.lower():
                continue
                
            for p in module.parameters(recurse=False):
                if p.requires_grad and id(p) not in seen_ids:
                    emb_params.append(p)
                    seen_ids.add(id(p))
                    
    # Collect remaining parameters (includes the skipped position embeddings)
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen_ids:
            main_params.append(p)
            seen_ids.add(id(p))
            
    return emb_params, main_params



def get_adafactor_groups(params, weight_decay, apollo_rank=256):
    """Routes parameters to Adafactor8Bit groups based on dimensionality."""
    g_1d, g_2d, g_nd = [], [], []
    for p in params:
        if p.ndim <= 1: g_1d.append(p)
        elif p.ndim == 2: g_2d.append(p)
        else: g_nd.append(p)
        
    return [
        # 1D (Bias, Norm): FP32, No WD, standard Adafactor path
        {"params": g_1d, "weight_decay": 0.0, "quantize": False, "apollo_rank": 0},
        # 2D (Linear weights): 8-bit, WD, APOLLO path
        {"params": g_2d, "weight_decay": weight_decay, "quantize": True, "apollo_rank": apollo_rank},
        # >2D (Conv weights): 8-bit, WD, standard Adafactor path
        {"params": g_nd, "weight_decay": weight_decay, "quantize": True, "apollo_rank": 0},
    ]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DummyModel().to(device)
    
    # 1. Split parameters
    emb_params, main_params = split_embedding_params(model)
    
    # 2. Setup FP32 Adam for Embeddings
    # Maintains element-wise variance for fine-grained, per-token updates.
    optimizer_emb = torch.optim.Adam(
        emb_params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8
    )
    
    # 3. Setup Adafactor8Bit for the Main Network
    optimizer_main = Adafactor8Bit(
        get_adafactor_groups(main_params, weight_decay=1e-2, apollo_rank=256),
        lr=1e-3,
        relative_step=False,
        beta2=0.999,
        enable_fira_for_adafactor=True, 
        fira_margin=0.01,
    )
    
    for step in range(1, 11):
        optimizer_main.zero_grad()
        optimizer_emb.zero_grad()
        
        x_text = torch.randint(0, 1000, (4, 16), device=device)
        x_img = torch.randn(4, 3, 32, 32, device=device)
        target = torch.randn(4, 10, device=device)
        
        output = model(x_text, x_img)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(emb_params, max_norm=1.0)
        
        optimizer_main.step()
        optimizer_emb.step()
        
        print(f"Step {step:02d} | Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
