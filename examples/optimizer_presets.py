"""
Optimizer Presets: Emulating Classic Optimizers with Adafactor8Bit

This script demonstrates how to configure Adafactor8Bit to emulate the mathematical 
behavior of various classic and modern optimizers. 
"""

from adafactor8bit import Adafactor8Bit

# Assume `model` is an instance of `torch.nn.Module`

# Common learning rate baselines
ADAM_STYLE_LR = 1e-4
ADAFACTOR_STYLE_LR = 1e-3

# ==============================================================================
# 1. RMSprop
# Strips momentum, weight decay, RMS scaling, and global clipping.
# ==============================================================================
optimizer_rmsprop = Adafactor8Bit(
    model.parameters(),
    lr=ADAM_STYLE_LR,
    beta1=None,                # Disable first moment
    beta2=0.999,               # Lock second moment EMA window
    weight_decay=0.0,
    scale_parameter=False,     # Disable Adafactor RMS scaling
    d=1e9,                     # Disable Adafactor global RMS clipping
    relative_step=False,       # Use constant external LR
    factored=False,            # Use full-rank variance
)

# ==============================================================================
# 2. Adam
# Adds momentum back to the RMSprop configuration.
# ==============================================================================
optimizer_adam = Adafactor8Bit(
    model.parameters(),
    lr=ADAM_STYLE_LR,
    beta1=0.9,                 # Enable first moment
    beta2=0.999,
    weight_decay=0.0,
    scale_parameter=False,
    d=1e9,
    relative_step=False,
    factored=False,           
)

# ==============================================================================
# 3. AdamW
# Adds decoupled weight decay to the Adam configuration.
# ==============================================================================
optimizer_adamw = Adafactor8Bit(
    model.parameters(),
    lr=ADAM_STYLE_LR,
    beta1=0.9,
    beta2=0.999,
    weight_decay=1e-2,
    scale_weight_decay=False,  # Decouple weight decay from LR scaling
    scale_parameter=False,
    d=1e9,
    relative_step=False,
    factored=False,
)

# ==============================================================================
# 4. Adafactor (Native Default)
# Utilizes Adafactor's native relative step sizing, RMS scaling, and factorization.
# ==============================================================================
optimizer_adafactor = Adafactor8Bit(
    model.parameters(),
    lr=ADAFACTOR_STYLE_LR,
)

# ==============================================================================
# 5. Adafactor (Continual / Lifelong Learning)
# Locks the EMA window and disables internal LR scheduling to prevent "blunting".
# ==============================================================================
optimizer_adafactor_continual = Adafactor8Bit(
    model.parameters(),
    lr=ADAFACTOR_STYLE_LR,
    beta2=0.999,               # Lock EMA window to prevent "blunting" over steps
    relative_step=False,       # Disable internal LR scheduling, use external scheduler
)

# ==============================================================================
# 6. APOLLO
# Low-rank subspace projection with decoupled weight decay and constant learning rate.
# Fira Norm-Growth Limiter is inherently active in the APOLLO path.
# Note: Official APOLLO only applies projection to 'attn' and 'mlp' 2D weights.
# ==============================================================================

# Simple inline grouping to match official APOLLO behavior
apollo_params = []
regular_params = []
for name, param in model.named_parameters():
    if not param.requires_grad: continue
    if param.ndim == 2 and any(t in name for t in ["attn", "mlp"]):
        apollo_params.append(param)
    else:
        regular_params.append(param)

param_groups_apollo = [
    {"params": regular_params}, # Will use default apollo_rank=0 (AdamW behavior)
    {
        "params": apollo_params,
        "apollo_rank": 256,               # Enable APOLLO projection
        "apollo_scale_type": 'channel',   # 'channel' (Standard) or 'tensor' (Mini)
        "apollo_update_proj_gap": 200,    # Steps between random projection matrix refreshes
    }
]

optimizer_apollo = Adafactor8Bit(
    param_groups_apollo,
    lr=ADAM_STYLE_LR,
    beta1=0.9,
    beta2=0.999,
    weight_decay=1e-2,
    scale_weight_decay=False,
    scale_parameter=False,
    d=1e9,
    relative_step=False,
)

# ==============================================================================
# 7. APOLLO-Mini
# Extreme memory savings with rank=1 projection. 
# Relies on momentum (beta1) to smooth projection noise and a heuristic scale factor.
# ==============================================================================

# Re-use the same grouping logic defined above
param_groups_apollo_mini = [
    {"params": regular_params},
    {
        "params": apollo_params,
        "apollo_rank": 1,               # Extreme low-rank projection
        "apollo_scale_type": 'tensor',  # Global norm matching recommended for Mini
        "apollo_scale": 128.0,          # Heuristic multiplier to compensate for rank=1 attenuation
        "apollo_update_proj_gap": 200,
    }
]

optimizer_apollo_mini = Adafactor8Bit(
    param_groups_apollo_mini,
    lr=ADAM_STYLE_LR,
    beta1=0.9,                 # Crucial for smoothing rank=1 projection noise
    beta2=0.999,
    weight_decay=1e-2,
    scale_weight_decay=False,
    scale_parameter=False,
    d=1e9,
    relative_step=False,
)

# ==============================================================================
# 8. CAME
# Confidence-guided adaptive optimization with momentum and decoupled weight decay.
# Requires beta1. Mutually exclusive with apollo_rank.
# ==============================================================================
optimizer_came = Adafactor8Bit(
    model.parameters(),
    lr=ADAM_STYLE_LR,          # Official CAME recommends 0.5-0.9x standard AdamW LR
    beta1=0.9,
    beta2=0.999,
    beta3=0.9999,              # Enable CAME confidence guidance
    eps_came=1e-16,            # Align with official CAME residual epsilon
    weight_decay=1e-2,
    scale_weight_decay=False,
    scale_parameter=False,
    d=1.0,                     # Restore official CAME global RMS clipping (clip_threshold=1.0)
    relative_step=False,
)