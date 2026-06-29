<p align="center">
  <a href="https://github.com/yanfeiwong/adafactor-8bit">
    <img src="https://github.com/yanfeiwong/adafactor-8bit/raw/main/assets/banner.png"
         alt="Adafactor8Bit"
         width="80%">
  </a>
</p>
<div align="center">

# 8-bit Adafactor with Fused CUDA Kernels

**English** | [中文](https://github.com/yanfeiwong/adafactor-8bit/blob/main/README_ZH.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/adafactor8bit.svg)](https://pypi.org/project/adafactor8bit/)
[![Total Downloads](https://static.pepy.tech/badge/adafactor8bit)](https://pepy.tech/project/adafactor8bit)
[![GitHub Stars](https://img.shields.io/github/stars/yanfeiwong/adafactor-8bit?style=social)](https://github.com/yanfeiwong/adafactor-8bit/stargazers)

</div>

An enhanced 8-bit Adafactor optimizer featuring fused CUDA kernels, log-space block-wise quantization, and optional add-ons including 4-bit packed first moments, APOLLO low-rank updates, and CAME confidence-guided optimization. It delivers substantially lower optimizer memory while preserving the low-overhead and numerical stability that make Adafactor attractive for training LLMs and diffusion models.


## ⚡ Key Features

- **Log-Space Quantization**: Maps the second moment (variance) to the log2 space before 8-bit quantization. This approach accommodates the long-tail distribution of variances, reducing the risk of small second-moment estimates being truncated to zero and improving overall training stability.
- **Fused CUDA Kernels**: Combines dequantization, EMA updates, Warp-Shuffle reductions, and requantization into single kernels. It utilizes `float4` vectorization to optimize memory bandwidth usage.
- **Optional 4-bit Packed First Moment**: Stores the first moment (`beta1`) in a physically packed 4-bit format when enabled, providing momentum with minimal additional memory overhead.
- **CAME Confidence Guidance**: Optional Confidence-guided Adaptive Memory Efficient Optimization (CAME) that estimates update confidence from historical momentum and adaptively suppresses unstable update directions, improving training stability and reducing loss spikes.
- **APOLLO Subspace Projection**: Opt-in random subspace projection that estimates adaptive gradient scaling in a low-rank space, preventing stale second-moment statistics and potentially improving convergence and generalization.
- **Fira Norm-Growth Limiter**: Suppresses destructive gradient spikes by regulating the relative increase of update norms. Originally used for the APOLLO path, it is now available for the standard Adafactor path as well. It improves training stability and often allows the safe removal of external gradient clipping.
- **Zero CPU-GPU Sync**: Eliminates implicit synchronizations (e.g., D2H copies) in the control flow, ensuring the GPU computation pipeline runs without blocking.
- **Cross-Platform JIT**: Uses Just-In-Time (JIT) compilation for straightforward setup across both Windows and Linux environments.

## 📊 Performance

- **Memory Footprint**: Due to Adafactor's factorized second-moment estimation, 8-bit quantization, and optional 4-bit packed first moments, the optimizer typically consumes substantially less memory than `AdamW8Bit`.
- **Training Speed**: The fused kernel design and reduced synchronization overhead allow it to achieve step times comparable to other mainstream 8-bit optimizers.
- **Quantization Precision**: The second moment (variance) in Adafactor is strictly non-negative and spans multiple orders of magnitude. By mapping it to `UINT8` in log2 space rather than linear space, the optimizer preserves relative precision for small variances, mitigating the instability often caused by outlier gradients in standard 8-bit quantization.

## 📦 Installation

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

> [!IMPORTANT]
> **First-Time Compilation**: The first time you instantiate the optimizer (or run the example script), it will automatically trigger the JIT compilation of the CUDA source code in the background. This may take anywhere from a few seconds to a couple of minutes depending on your system, and the terminal might appear unresponsive. Once compiled, the binary will be cached, and all subsequent runs will be instantaneous.



## 🚀 Quick Start

Using it is as simple as using a standard PyTorch optimizer.

```python
from adafactor8bit import Adafactor8Bit

optimizer = Adafactor8Bit(model.parameters(), lr=1e-3)
```

> [!TIP]
> Passing `model.parameters()` directly works for a quick test. In production, `param_groups` are recommended to protect sensitive layers (Norms, Biases) from quantization and weight decay. For **sparse token embeddings** (large vocabularies + small batch sizes), please refer to the [Advanced Example](#-advanced-example) to avoid cold-start variance explosion.


```python
from adafactor8bit import Adafactor8Bit

def get_param_groups(model, weight_decay=1e-2):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        
        # Protect 1D tensors, biases, norms, and embeddings
        if param.ndim <= 1 or "bias" in name or "norm" in name or "embed" in name:
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
    # For continual learning or when using an external LR scheduler
    relative_step=False,     # Disable internal LR scheduling
    beta2=0.999,             # Lock EMA window to prevent "blunting" over steps
)

# Training loop...
```

## 🛠️ Advanced Example

Here we demonstrate a **hybrid grouping** strategy for complex hybrid architectures (e.g., Vision-Language Models, Diffusion UNets) to achieve stable and efficient training.

📌 **The following strategies are applied:**
| Layer Type | Strategy |
|------------|----------|
| **1D / Sensitive Parameters** (Norms, Biases) | No quantization, no weight decay |
| **Embedding Layers** | `factored=False`, `scale_parameter=False`, `d=1e9` → Momentum-free Adam. Paired with an Adam-style learning rate, this allows for fine-grained, per-token updates while avoiding cold-token interference. |
| **2D Weights** (Linear Layers) | 8-bit quantization, weight decay, **APOLLO** path. Continuously switching random subspace projection captures comprehensive gradient information and acts as a regularizer. |
| **>2D Weights** (Conv2d, etc.) | 8-bit quantization, weight decay, **Full-Rank** (`factored=False`). Trades some VRAM to preserve complete spatial structures. |
| **Momentum (`beta1`)** | Enabled only for dense weight matrices, where the optimization benefit typically outweighs the small memory overhead of the packed 4-bit first moment. Sensitive parameters (Norms/Biases) and sparse Embeddings remain momentum-free. |

**Implementation:**

```python
from adafactor8bit import Adafactor8Bit

# Define learning rates
lr = 1e-3
lr_emb = 1e-4 # For Embedding layers, we use an Adam-style learning rate

def get_param_groups(model, lr_emb, weight_decay, apollo_rank=256):
    group_1d, group_embed, group_2d, group_nd = [], [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        
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
            "beta1": 0.9,              # Remove if minimizing optimizer memory is the priority.
        },

        # 4. >2D Weights: 8-bit quantization, Weight Decay, Full-Rank
        {
            "params": group_nd, 
            "weight_decay": weight_decay, 
            "quantize": True, 
            "apollo_rank": 0,
            "beta1": 0.9,              # Remove if minimizing optimizer memory is the priority.
            "factored": False          # Disables factorization to preserve spatial structures, enabling finer gradient scaling.
                                       # Note: This increases state memory for >2D weights, depending on your model architecture.
                                       # If VRAM is constrained, reverting to factored=True is a safe alternative.
        },
    ]

model = MyModel().cuda()
optimizer = Adafactor8Bit(
    get_param_groups(model, lr_emb = lr_emb, weight_decay=1e-2, apollo_rank=256), 
    lr=lr, 
    # For continual learning or when using an external LR scheduler
    relative_step=False,              # Disable internal LR scheduling
    beta2=0.999,                      # Lock EMA window to prevent "blunting" over steps
    enable_fira_for_adafactor=True    # Enable Fira Limiter globally; external grad clipping can be safely removed
)

# Training loop...
```

> [!NOTE]
> For more complete examples, please refer to the [examples folder](https://github.com/yanfeiwong/adafactor-8bit/tree/main/examples).


## ⚙️ Advanced Configuration

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

### Full-Rank and Factorized Variance (`factored`)

By default, Adafactor factorizes the second moment of $\ge$ 2D tensors into row and column statistics (`factored=True`) to minimize state memory. Setting `factored=False` switches to full element-wise variance (similar to RMSProp), while still retaining Adafactor's global RMS clipping mechanism (controlled by the `d` parameter). This configuration can be useful in the following scenarios:
- **Convolution Weights (>2D)**: Setting `factored=False` maintains independent variance for each spatial position in the convolution kernel, enabling finer per-element gradient scaling.
- **Sparse Embeddings**: Combining `factored=False`, `scale_parameter=False`, `d=1e9` and a lower learning rate creates a momentum-free adaptive optimizer. This allows for fine-grained, per-token updates while avoiding cold-token interference or cold-start explosions.

### No-Compiler Environments (`use_cuda_kernel=False`)
If you are in an environment without a CUDA compiler and want to bypass JIT compilation entirely:
- Set `use_cuda_kernel=False` to fall back to the pure PyTorch implementation.

## 🌌 APOLLO Low-Rank Subspace Projection
Enable the APOLLO path to compute gradient scaling factors in a memory-efficient low-rank subspace. Compared to Adafactor's standard row/column factorization (which assumes spatial independence), APOLLO uses random subspace projection to capture cross-dimensional covariance information, potentially leading to better generalization while keeping memory overhead extremely low.

- **`apollo_rank`**: The target rank for the projection subspace. The default is `0` (disabled).  
  
  - The official APOLLO GitHub repository recommends a rank of `256` for 1B and 7B models. 
  - The [LLaMA-Factory](https://llamafactory.readthedocs.io/en/latest/advanced/arguments.html#apollo) default is `16`.
  - Setting this to `1` (APOLLO-Mini style) pushes VRAM savings to the limit (saves even more VRAM than the Adafactor path). The original APOLLO-Mini relies on the first-moment (beta1) to smooth out projection noise. To replicate this, set `beta1=0.9` alongside `apollo_rank=1`. Without beta1, rank=1 may still work but can exhibit noisier scaling factors, especially at small batch sizes.


- **`apollo_scale_type`**: Determines how the scaling factor is applied. `'channel'` applies it per channel (Standard APOLLO), while `'tensor'` applies it globally (APOLLO-Mini).
- **`apollo_update_proj_gap`**: Steps between projection matrix refreshes. Defaults to `200`. Setting this too small may cause frequent oscillations due to abrupt basis mutations, while setting it too large might cause the projection space to become stale and fail to track the drift of the gradient manifold.
- **`apollo_factorize` (Experimental)**: Applies Adafactor's row/column factorization within the low-rank subspace. Mathematically, this leverages the norm-preserving property of random projections to approximate the variance of the primary dimension, while the secondary dimension's variance is estimated across random bases, introducing inherent noise. This dual-compression mechanism drastically reduces optimizer state overhead. Note that for smaller models, the actual VRAM savings might be marginal, and the introduced noise could impact convergence stability. Use with caution.
- **Fira Limiter Integration**: The APOLLO path automatically applies the Fira Norm-Growth Limiter to the scaled gradients to prevent sudden gradient rises from causing loss spikes. You can adjust its sensitivity using the global `fira_margin` parameter.

## 🧊 CAME Confidence-Guided Updates

Enable the CAME (Confidence-guided Adaptive Memory Efficient Optimization) path to add a confidence estimation stage after momentum accumulation:

**Adaptive Scaling ($V$) → Momentum Accumulation ($M$) → Confidence Weighting ($C$)**

### Key Parameters & Tuning

The confidence stage measures the consistency between the current update direction and historical momentum, adaptively suppressing highly oscillatory updates.

- **`beta3`**: EMA decay coefficient for the confidence matrix. Requires `beta1` (momentum) and `factored=True`. Mutually exclusive with `apollo_rank`. Defaults to `None` (disabled).
- **Learning Rate**: The official CAME implementation recommends **0.5–0.9×** the AdamW learning rate (see [official tuning guide](https://github.com/yangluo7/CAME/tree/master#hyper-parameter-tuning)). To use this learning rate in this library, you need to disable Adafactor's scaling and clipping (`scale_parameter=False`, `d=1e9`) to align with the original CAME behavior.
- **Warmup**: Since the confidence matrix is zero-initialized without bias correction, a learning rate warmup is recommended to safely establish the confidence baseline.
- **Choosing `beta3`**: `beta3` should generally be larger than `beta2` so the confidence estimate evolves more slowly than the variance estimate. A practical starting range is **0.9995–0.99995** when `beta2=0.999`.


### Configuration Example

To replicate "vanilla" CAME (stripping Adafactor's native modifications), replace the standard 2D APOLLO group in your `param_groups` with the following configuration:

```python
{
    "params": param_group,
    "lr": lr,                           # Original CAME recommends 0.5-0.9x AdamW LR
    "weight_decay": weight_decay,
    "quantize": True,
    "beta1": 0.9,
    "beta3": 0.9999,                    # Enable CAME confidence guidance
    "apollo_rank": 0,                   # Mutually exclusive with CAME
    "scale_parameter": False,           # Disable Adafactor RMS scaling to align with vanilla CAME
    "d": 1e9,                           # Disable Adafactor global RMS clipping
    "enable_fira_for_adafactor": False, # Disable Fira Limiter to prevent interference with CAME's scaling
},
```

## 📈 Learning Rate Guide for Beginners

If you are migrating from optimizers like AdamW, Adafactor's learning rate behavior might feel a bit different. This is mainly due to the `scale_parameter` option.

- **`scale_parameter=True` (default)**
  Because of RMS scaling, a very small `lr` (e.g., `1e-5`) often leads to extremely slow progress. Start with `lr=1e-3` and adjust in the range `1e-4`–`5e-3` if needed.

- **`scale_parameter=False`**
  Disables RMS scaling, making the update scale more similar to AdamW. Use the learning rates you're familiar with for AdamW and tune as usual. (Note: the second moment is still factorized, so behavior is not identical.)

*These are safe starting points. Always validate on your own task and batch size.*





## 🎓 Acknowledgements

Thanks to **Noam Shazeer** and **Mitchell Stern** for proposing the original Adafactor algorithm in the paper [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235).

Thanks to **Tim Dettmers** for the inspiration from the paper [8-BIT OPTIMIZERS VIA BLOCK-WISE QUANTIZATION](https://arxiv.org/abs/2110.02861) and the [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) library.

Thanks to **Hanqing Zhu**, **Zhenyu Zhang**, and the team for proposing the approximated gradient scaling method in the paper [APOLLO: SGD-Like Memory, AdamW-level Performance](https://arxiv.org/abs/2412.05270).

Thanks to **Xi Chen**, **Kaituo Feng**, and the team for the Norm-Growth Limiter mechanism introduced in [Fira: Can We Achieve Full-rank Training of LLMs Under Low-rank Constraint?](https://arxiv.org/abs/2410.01623).

Thanks to **Yang Luo** and the team for proposing the confidence-guided strategy in the paper [CAME: Confidence-guided Adaptive Memory Efficient Optimization](https://arxiv.org/abs/2307.02047).

Thanks to the **PyTorch team** for providing the foundational Optimizer implementation and the C++ Extension toolchain.

Thanks to the large language models **Qwen**, **ChatGLM** and **DeepSeek** for valuable technical discussions and code reviews on CUDA low-level optimization and memory safety mechanisms.

## 🏛️ License

[The project is released under the MIT License.](https://github.com/yanfeiwong/adafactor-8bit/blob/main/LICENSE)

## ⭐ Star the Project

If this optimizer has been useful in your work, consider giving the repository a star. It helps others discover the project and supports future development.

[![Star History Chart](https://api.star-history.com/svg?repos=yanfeiwong/adafactor-8bit&type=Date&theme=dark)](https://star-history.com/#yanfeiwong/adafactor-8bit&Date)
