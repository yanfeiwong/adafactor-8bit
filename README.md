<div align="center">

# 8-bit Adafactor with Fused CUDA Kernels

**English** | [中文](https://github.com/yanfeiwong/adafactor-8bit/blob/main/README_ZH.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/adafactor8bit.svg)](https://pypi.org/project/adafactor8bit/)
[![Total Downloads](https://static.pepy.tech/badge/adafactor8bit)](https://pepy.tech/project/adafactor8bit)
[![GitHub Stars](https://img.shields.io/github/stars/yanfeiwong/adafactor-8bit?style=social)](https://github.com/yanfeiwong/adafactor-8bit/stargazers)

</div>

An 8-bit Adafactor optimizer featuring fused CUDA kernels and log-space block-wise quantization, designed to further reduce optimizer state memory while maintaining low step overhead and stability — suitable for large models such as LLMs and diffusion models.


## Key Features

- **Log-Space Quantization**: Maps the second moment (variance) to the log2 space before 8-bit quantization. This approach accommodates the long-tail distribution of variances, reducing the risk of small second-moment estimates being truncated to zero and improving overall training stability.
- **Fused CUDA Kernels**: Combines dequantization, EMA updates, Warp-Shuffle reductions, and requantization into single kernels. It utilizes `float4` vectorization to optimize memory bandwidth usage.
- **APOLLO Subspace Projection**: Opt-in random subspace projection that estimates adaptive gradient scaling in a low-rank space, preventing stale second-moment statistics and potentially improving convergence and generalization.
- **Fira Norm-Growth Limiter**: Suppresses destructive gradient spikes by regulating the relative increase of update norms. Originally used for the APOLLO path, it is now available for the standard Adafactor path as well. It improves training stability and often allows the safe removal of external gradient clipping.
- **Zero CPU-GPU Sync**: Eliminates implicit synchronizations (e.g., D2H copies) in the control flow, ensuring the GPU computation pipeline runs without blocking.
- **Cross-Platform JIT**: Uses Just-In-Time (JIT) compilation for straightforward setup across both Windows and Linux environments.

## Performance

- **Memory Footprint**: Due to Adafactor's factorized second-moment estimation and 8-bit quantization, the optimizer state memory usage is generally lower than that of `AdamW8Bit`.
- **Training Speed**: The fused kernel design and reduced synchronization overhead allow it to achieve step times comparable to other mainstream 8-bit optimizers.
- **Quantization Precision**: The second moment (variance) in Adafactor is strictly non-negative and spans multiple orders of magnitude. By mapping it to `UINT8` in log2 space rather than linear space, the optimizer preserves relative precision for small variances, mitigating the instability often caused by outlier gradients in standard 8-bit quantization.

## Installation

This project uses JIT (Just-In-Time) compilation.

Please ensure `torch` and `ninja` are installed, and a CUDA compiler (such as MSVC or GCC) is available in your environment.

If CUDA compilation fails, the optimizer will automatically fall back to the pure PyTorch implementation.

### From PyPI

```bash
pip install -U adafactor8bit
```

### From Source

```bash
pip install git+https://github.com/yanfeiwong/adafactor-8bit.git
```

**Note**: The first time you instantiate the optimizer (or run the example script), it will automatically trigger the JIT compilation of the CUDA source code in the background. This may take anywhere from a few seconds to a couple of minutes depending on your system, and the terminal might appear unresponsive. Once compiled, the binary will be cached, and all subsequent runs will be instantaneous.

## Usage Example

It is recommended to use `param_groups` to keep sensitive layers (Embedding, Norm, Bias) in FP32, enabling 8-bit quantization only for large weight matrices.

```python
import torch
import torch.nn as nn
from adafactor8bit import Adafactor8Bit

def get_param_groups(model, weight_decay=1e-2):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        # Protect 1D tensors, biases, norms, and embeddings
        if param.ndim <= 1 or "bias" in name or "norm" in name or "embed" in name:
            # Note: Grouping embeddings here works well for layers with dense gradient updates. 
            # However, for massive token embeddings with highly sparse updates (e.g., large vocabularies combined with small batch sizes), 
            # please refer to the Advanced Example below to route them to the APOLLO path
            # to avoid Adafactor's cold-start variance explosion.
            no_decay.append(param)
        else:
            decay.append(param)
            
    return [
        {"params": decay, "weight_decay": weight_decay, "quantize": True},
        {"params": no_decay, "weight_decay": 0.0, "quantize": False}
    ]

model = MyModel().cuda()
optimizer = Adafactor8Bit(
    get_param_groups(model), 
    lr=1e-3, 
    # For continual learning with external scheduler
    relative_step=False,     # Disable internal LR scheduling
    beta2=0.999,             # Lock EMA window to prevent "blunting" over steps

    # --- 🚀 Uncomment to try the new APOLLO Subspace Projection ---
    # Simulates full-rank adaptive scaling in a low-rank space, improving generalization and potentially accelerating convergence.
    # apollo_rank=256,             # 0 to disable. 256 is the official APOLLO default.
)

# Training loop...
```

## Advanced Example: Hybrid Routing & Fira Limiter

For complex architectures (e.g., Vision-Language Models, Diffusion UNets) containing a mixture of 2D weight matrices and high-dimensional tensors (e.g., convolutions), we recommend a **hybrid routing** strategy.  

While APOLLO works exceptionally well for 2D matrices, applying it to >2D tensors requires reshaping, which might destroy inherent spatial structures and disrupt channel-wise scaling. Therefore, we route 2D weights to the APOLLO path, and >2D/1D parameters to the standard Adafactor path.

📌**Embedding Layers**: Gradients for embeddings are extremely sparse. Adafactor tracks variance locally per row, meaning unactivated rows retain near-zero variance and can cause extreme scaling factors upon first activation (the "cold-start" problem). APOLLO's low-rank projection, however, mixes information across all rows, providing globally backed scaling factors that are inherently stable for sparse gradients. Thus, we route Embeddings to the APOLLO path.

Additionally, enabling the **Fira Limiter** for the Adafactor path helps suppress destructive gradient spikes, often allowing you to safely remove external gradient clipping (`torch.nn.utils.clip_grad_norm_`).

```python
def get_hybrid_param_groups(model, weight_decay, apollo_rank=256):
    group_1d_sensitive, group_embed, group_2d_weights, group_nd_weights = [], [], [], []
    
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        
        is_sensitive = param.ndim <= 1 or "bias" in name or "norm" in name
        is_embedding = "embed" in name.lower()
        
        if is_sensitive:
            group_1d_sensitive.append(param)
        elif is_embedding:
            group_embed.append(param)
        elif param.ndim == 2:
            group_2d_weights.append(param)
        else:
            group_nd_weights.append(param)

    return [
        # 1D / Sensitive: FP32, No WD, Adafactor
        {"params": group_1d_sensitive, "weight_decay": 0.0, "quantize": False, "apollo_rank": 0},
        # Embeddings: FP32, No WD, APOLLO
        {"params": group_embed, "weight_decay": 0.0, "quantize": False, "apollo_rank": apollo_rank},
        # 2D Weights: 8-bit, WD, APOLLO
        {"params": group_2d_weights, "weight_decay": weight_decay, "quantize": True, "apollo_rank": apollo_rank},
        # >2D Weights (e.g., Conv2d): 8-bit, WD, Adafactor
        {"params": group_nd_weights, "weight_decay": weight_decay, "quantize": True, "apollo_rank": 0},
    ]

optimizer = Adafactor8Bit(
    get_hybrid_param_groups(model, weight_decay=1e-2, apollo_rank=256),
    lr=1e-3,
    relative_step=False,
    beta2=0.999,
    enable_fira_for_adafactor=True, # Enable Fira Limiter for the Adafactor paths
    fira_margin=0.01,               # Allow 1% norm growth before limiting
)
```

For a complete example, please refer to [basic_usage.py](https://github.com/yanfeiwong/adafactor-8bit/blob/main/examples/basic_usage.py) and [advanced_usage.py](https://github.com/yanfeiwong/adafactor-8bit/blob/main/examples/advanced_usage.py).


## Advanced Configuration

### Continual Learning (`beta2` & `relative_step`)
By default, Adafactor's second-moment decay rate dynamically decays with the training step, and the internal learning rate schedule (`relative_step`) scales the learning rate accordingly. 

For endless fine-tuning or lifelong learning, this often leads to overly small learning rates and "blunted" second-moment estimates. To avoid these issues and keep the optimizer responsive:
- Set `relative_step=False` to disable the built-in LR schedule (allowing you to use an external scheduler).
- Set `beta2=0.999` to lock the EMA window (similar to Adam).

### Decoupled Weight Decay (`scale_weight_decay=False`)
By default, Adafactor's weight decay is coupled with the parameter's RMS scale. 
- If you prefer the AdamW-style decoupled weight decay, set `scale_weight_decay=False`.

### Fira Limiter (`enable_fira_for_adafactor` & `fira_margin`)
The Norm-Growth Limiter (introduced in the Fira paper) smooths gradient updates by limiting the relative increase of update norms, effectively suppressing destructive loss spikes.
- **`enable_fira_for_adafactor`**: Defaults to `False`. Set to `True` to enable the limiter for the standard Adafactor path. *(Note: It is inherently active in the APOLLO path)*. When enabled, external gradient clipping (e.g., `torch.nn.utils.clip_grad_norm_`) can generally be safely removed to simplify the training pipeline.
- **`fira_margin`**: Defaults to `0.01`. The tolerance margin for norm growth. The limiter activates only when the current update norm grows by more than this margin (e.g., `0.01` means a 1% growth) compared to the previous step.

### No-Compiler Environments (`use_cuda_kernel=False`)
If you are in an environment without a CUDA compiler and want to bypass JIT compilation entirely:
- Set `use_cuda_kernel=False` to fall back to the pure PyTorch implementation.

## APOLLO Low-Rank Subspace Projection
Enable the APOLLO path to compute gradient scaling factors in a memory-efficient low-rank subspace. Compared to Adafactor's standard row/column factorization (which assumes spatial independence), APOLLO uses random subspace projection to capture cross-dimensional covariance information, potentially leading to better generalization while keeping memory overhead extremely low.

- **`apollo_rank`**: The target rank for the projection subspace. The default is `0` (disabled). Setting it to `256` might work well for most 1B to 7B models.  
  *Note: Setting this to `1` (APOLLO-Mini style) pushes VRAM savings to the limit (saves even more VRAM than the Adafactor path). However, the original APOLLO-Mini relies on Adam's first-moment (beta1) to smooth out noise. Since our implementation uses a pure second-moment architecture, rank=1 may lead to distorted scaling factors and training instability.*
- **`apollo_scale_type`**: Determines how the scaling factor is applied. `'channel'` applies it per channel (Standard APOLLO), while `'tensor'` applies it globally (APOLLO-Mini).
- **`apollo_update_proj_gap`**: Steps between projection matrix refreshes. Defaults to `200`. Setting this too small may cause frequent oscillations due to abrupt basis mutations, while setting it too large might cause the projection space to become stale and fail to track the drift of the gradient manifold.
- **`apollo_factorize` (Experimental)**: Applies Adafactor's row/column factorization within the low-rank subspace. Mathematically, this leverages the norm-preserving property of random projections to approximate the variance of the primary dimension, while the secondary dimension's variance is estimated across random bases, introducing inherent noise. This dual-compression mechanism drastically reduces optimizer state overhead. Note that for smaller models, the actual VRAM savings might be marginal, and the introduced noise could impact convergence stability. Use with caution.
- **Fira Limiter Integration**: The APOLLO path automatically applies the Fira Norm-Growth Limiter to the scaled gradients to prevent sudden gradient rises from causing loss spikes. You can adjust its sensitivity using the global `fira_margin` parameter.



## Learning Rate Guide for Beginners

If you are migrating from optimizers like AdamW, Adafactor's learning rate behavior might feel a bit different. This is mainly due to the `scale_parameter` option.

- **`scale_parameter=True` (default)**
  Because of RMS scaling, a very small `lr` (e.g., `1e-5`) often leads to extremely slow progress. Start with `lr=1e-3` and adjust in the range `1e-4`–`5e-3` if needed.

- **`scale_parameter=False`**
  Disables RMS scaling, making the update scale more similar to AdamW. Use the learning rates you're familiar with for AdamW and tune as usual. (Note: the second moment is still factorized, so behavior is not identical.)

*These are safe starting points. Always validate on your own task and batch size.*





## Acknowledgements

Thanks to **Noam Shazeer** and **Mitchell Stern** for proposing the original Adafactor algorithm in the paper [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235).

Thanks to **Tim Dettmers** for the inspiration from the paper [8-BIT OPTIMIZERS VIA BLOCK-WISE QUANTIZATION](https://arxiv.org/abs/2110.02861) and the [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) library.

Thanks to **Hanqing Zhu**, **Zhenyu Zhang**, and the team for proposing the approximated gradient scaling method in the paper [APOLLO: SGD-Like Memory, AdamW-level Performance](https://arxiv.org/abs/2412.05270).

Thanks to **Xi Chen**, **Kaituo Feng**, and the team for the Norm-Growth Limiter mechanism introduced in [Fira: Can We Achieve Full-rank Training of LLMs Under Low-rank Constraint?](https://arxiv.org/abs/2410.01623).

Thanks to the **PyTorch team** for providing the foundational Optimizer implementation and the C++ Extension toolchain.

Thanks to the large language models **Qwen** and **DeepSeek** for valuable technical discussions and code reviews on CUDA low-level optimization, memory safety mechanisms, and cross-platform compilation pipeline design.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yanfeiwong/adafactor-8bit&type=Date&theme=dark)](https://star-history.com/#yanfeiwong/adafactor-8bit&Date)

## License

[The project is released under the MIT License.](https://github.com/yanfeiwong/adafactor-8bit/blob/main/LICENSE)
