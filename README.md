<p align="center">
  <a href="https://github.com/yanfeiwong/adafactor-8bit">
    <img src="https://github.com/yanfeiwong/adafactor-8bit/raw/main/assets/banner.png"
         alt="Adafactor8Bit"
         width="80%">
  </a>
</p>
<div align="center">

# Adafactor8Bit: Memory-Efficient Optimizer with Adaptive Log-Space Quantization

**English** | [中文](https://github.com/yanfeiwong/adafactor-8bit/blob/main/README_ZH.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/adafactor8bit.svg)](https://pypi.org/project/adafactor8bit/)
[![Total Downloads](https://static.pepy.tech/badge/adafactor8bit)](https://pepy.tech/project/adafactor8bit)
[![arXiv](https://img.shields.io/badge/arXiv-2608.22322-b31b1b.svg)](https://arxiv.org/abs/2608.22322)
[![GitHub Stars](https://img.shields.io/github/stars/yanfeiwong/adafactor-8bit?style=social)](https://github.com/yanfeiwong/adafactor8bit/stargazers)

</div>

A configurable memory-efficient optimizer for PyTorch. It combines fused CUDA kernels with state-specific low-precision storage across Adam-style, factored Adafactor, CAME, and APOLLO paths. Non-negative states can use Adaptive Log-Space (`al8` / `al16`) quantization, while signed momentum and CAME confidence precision are configured independently.

## 📄 Paper

**[Beyond Dense Adam States: Adaptive Log-Space Quantization for Memory-Efficient Optimizers](https://arxiv.org/abs/2608.22322)** · [PDF](https://arxiv.org/pdf/2608.22322)

<!-- Append artifact and experimental-data repository links to the resource line once public. -->

## 📐 Adaptive Log-Space (AL)

> **Adaptive nonzero range** — each block fits its own `[ℓ_min, ℓ_max]` to the observed positive values instead of sharing one fixed global log range.  
> **Exact-zero preservation** — code zero is reserved exclusively for exact zero, giving `q = 0 ⇔ x = 0`.  
> **AL8 / AL16** — `al8` provides 255 nonzero codes; `al16` provides 65,535. The markers in the animation are schematic.  
> **State-specific storage** — signed momentum uses an independent encoding, and CAME confidence states can use a different precision from the second moment.

<p align="center">
  <img src="https://github.com/yanfeiwong/adafactor-8bit/raw/main/assets/fig02_al_encoding_readme.gif"
       alt="Adaptive Log-Space encoding: exact-zero reservation and adaptive block-local nonzero range"
       width="94%">
</p>

<p align="center">
  <img src="https://github.com/yanfeiwong/adafactor-8bit/raw/main/assets/al_update_error_general_readme.png"
       alt="Controlled single-step Adam-style update error across diverse V distributions"
       width="90%">
</p>

## ⚡ Key Features

- **Adaptive Log-Space Quantization**: `al8` / `al16` storage for non-negative optimizer states, with per-block adaptive logarithmic ranges and an exact-zero code.
- **Fused CUDA Kernels**: Combines dequantization, EMA updates, Warp-Shuffle reductions, parameter updates, and requantization in fused CUDA paths where supported, reducing memory traffic and optimizer-step overhead.
- **Configurable First-Moment**: Stores the optional first moment (`beta1`) using configurable 4-bit / 8-bit formats or full FP32 precision.
- **CAME Confidence Guidance**: Optional confidence-guided updates with confidence-state precision configurable independently from the second-moment state.
- **APOLLO Subspace Projection**: Opt-in low-rank projected-gradient path that maintains adaptive statistics in projected space while reducing optimizer-state storage.
- **Fira Norm-Growth Limiter**: Regulates the relative growth of update norms to suppress destructive gradient spikes, available as an optional stabilizer across update paths.
- **Zero CPU-GPU Sync**: Avoids implicit CPU-GPU synchronization in the optimizer control flow, keeping the GPU pipeline from blocking.
- **Cross-Platform JIT**: JIT-compiles the CUDA extension locally on Windows and Linux, with a pure PyTorch fallback when CUDA compilation is unavailable.

## 📊 Performance

**TinyLlama-1.1B · WikiText-103 · 20K steps · `batch_size=4` · `seq_len=512` · BF16 compute / FP32 model parameters · single NVIDIA RTX 3090 Ti · seed 921.**

| Path | Group | Variant | M | V | C | B<sub>V</sub> | PPL ↓ | Peak ↓ | State ↓ | tok/s ↑ |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| AdamW | G0 | PyTorch | FP32 | FP32 | – | – | 72.48 | 21046.0 | 8392.7 | 2502 |
| AdamW | G0 | bnb8 | bnb8 | bnb8 | – | 256 | 73.54 | 10608.6 | 2131.8 | 2640 |
| **AdamW** | **G0** | **Ours** | **UF8** | **AL8** | **–** | **2048** | **72.90** | **10596.8** | **2119.2** | **2960** |
| CAME | G0 | Official | FP32 | FP32 | FP32 | – | 86.68 | 13428.5 | 4203.2 | 1866 |
| CAME | G0 | Ours | UF8 | AL8 | AL8 | 2048 | 90.19 | 9509.0 | 1068.0 | 2432 |
| CAME | G0 | Ours | UF8 | AL8 | FP32 | 2048 | 88.41 | 9510.0 | 1070.3 | 2423 |
| **CAME** | **G0** | **Ours** | **UF8** | **AL16** | **AL16** | **2048** | **86.16** | **9510.8** | **1069.8** | **2428** |
| Adafactor | G0 | HF | – | FP32 | – | – | 77.56 | 8934.7 | 3.6 | 2206 |
| Adafactor | G0 | Ours | – | AL8 | – | 256 | 78.72 | 8442.6 | 1.2 | 2740 |
| Adafactor | G0 | Ours | – | AL8 | – | 2048 | 79.36 | 8442.7 | 1.3 | 2734 |
| **Adafactor** | **G1** | **Ours** | **–** | **AL8*** | **–** | **256** | **78.15** | **8999.2** | **1.4** | **2708** |
| APOLLO | G0 | Official | FP32 | FP32 | – | – | 74.68 | 11560.2 | 2078.7 | 2331 |
| APOLLO | G0 | Ours | UF8 | AL8 | – | 2048 | 75.24 | 10290.9 | 1694.7 | 2077 |

> **Benchmark notes:** Memory is MiB. `M`, `V`, and `C` denote first-moment, second-moment, and CAME confidence-state storage; `B<sub>V</sub>` is the non-negative-state block size (momentum blocks use 256).  
> `Peak` is PyTorch maximum allocated CUDA memory; `State` is live CUDA tensor storage owned by the optimizer; throughput is Hugging Face Trainer input tokens/s. `bnb8` is the evaluated bitsandbytes 8-bit state encoding; `AL8*` uses G1 protection for selected sensitive tensors while the main factored state remains AL8.

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
> **First-Time Compilation**: The first time you instantiate the optimizer (or run the example script), it will synchronously trigger the JIT compilation of the CUDA source code. This may take anywhere from a few seconds to a couple of minutes depending on your system, and the terminal might appear unresponsive. Once compiled, the extension is cached for reuse.



## 🚀 Quick Start

Using it is as simple as using a standard PyTorch optimizer.

```python
from adafactor8bit import Adafactor8Bit

optimizer = Adafactor8Bit(model.parameters(), lr=1e-3)
```

> [!TIP]
> Passing `model.parameters()` directly works for a quick test. In production, `param_groups` are recommended to protect sensitive layers (Norms, Biases) from quantization and weight decay. For **sparse token embeddings** (large vocabularies + small batch sizes), please refer to the [Advanced Example](#-advanced-example) to avoid cold-start instability.


```python
from adafactor8bit import Adafactor8Bit

def get_param_groups(model, weight_decay=1e-2):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        
        # Protect 1D tensors, biases, norms, embeddings, and LM head
        if param.ndim <= 1 or "bias" in name or "norm" in name or "embed" in name or "lm_head" in name:
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

<details>
<summary><strong>Show topology-aware grouping example</strong></summary>

Here we demonstrate a **hybrid grouping** strategy for complex hybrid architectures (e.g., Vision-Language Models, Diffusion UNets) to achieve stable and efficient training.

📌 **The following strategies are applied:**
| Layer Type | Strategy |
|------------|----------|
| **1D / Sensitive Parameters** (Norms, Biases) | No quantization, no weight decay. |
| **Embedding Layers** | `factored=False`, `scale_parameter=False`, `d=0.0`, `beta1=None` → Momentum-free Adam-style scaling. Paired with an Adam-style learning rate, this allows for fine-grained, per-token updates while avoiding cold-token interference. |
| **LM Head** | 8-bit quantization, `factored=False`, `scale_parameter=False`, `d=0.0`, `beta1=0.9` → Adam-style with momentum. Avoids factored distortion for the final output projection. |
| **2D Weights (APOLLO targets)** (e.g., Attn, MLP) | 8-bit quantization, weight decay, **APOLLO** path. Continuously switching random subspace projection captures comprehensive gradient information. |
| **2D Weights (Others)** | Default Adafactor path (factored), 8-bit quantization, weight decay. |
| **>2D Weights** (Conv2d, etc.) | 8-bit quantization, weight decay, **Full-Rank & No RMS Scaling** (`factored=False`, `scale_parameter=False`). Trades some VRAM to preserve spatial structures for finer optimization. |

**Implementation:**

```python
from adafactor8bit import Adafactor8Bit

# Define learning rates
lr = 1e-3
lr_adam = 1e-4 # For Embedding and N-D layers, we use an Adam-style learning rate

apollo_targets = ["attn", "mlp"]

def get_param_groups(model, lr_adam, weight_decay, apollo_rank=256):
    group_1d, group_embed, group_lm_head, group_apollo, group_2d, group_nd = [], [], [], [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        
        is_1d = param.ndim <= 1 or "bias" in name or "norm" in name
        # Match true Token Embeddings, excluding Position and Time Embeddings
        is_embedding = ("embed" in name.lower() 
                        and "position" not in name.lower() 
                        and "pos_embed" not in name.lower()
                        and "time" not in name.lower())
        is_lm_head = param.ndim == 2 and "lm_head" in name.lower()
        is_apollo_target = param.ndim == 2 and any(t in name for t in apollo_targets)
        
        if is_1d:
            group_1d.append(param)
        elif is_embedding:
            group_embed.append(param)
        elif is_lm_head:
            group_lm_head.append(param)
        elif is_apollo_target:
            group_apollo.append(param)
        elif param.ndim == 2:
            group_2d.append(param)
        else:
            group_nd.append(param)

    # Common configuration for Adam-style groups (element-wise, no RMS scaling, no clipping)
    adam_style = {
        "factored": False,         # full-rank V
        "scale_parameter": False,  # no parameter RMS scaling
        "d": 0.0,                  # no RMS clipping
        "beta3": None,             # CAME requires factored=True
        "apollo_rank": 0,
        "lr": lr_adam,  
    }

    return [
        # 1. 1D / Sensitive: FP32, No Weight Decay
        {"params": group_1d, "weight_decay": 0.0, "quantize": False, "apollo_rank": 0},
        
        # 2. Embeddings: Momentum-free Adam-style scaling
        {
            "params": group_embed, 
            "weight_decay": 0.0, 
            "quantize": False,
            "beta1": None,             # Momentum-free
            **adam_style,
        },

        # 3. LM Head: Adam-style with momentum (avoid factored distortion)
        {
            "params": group_lm_head, 
            "weight_decay": 0.0, 
            "quantize": True,
            "beta1": 0.9,
            **adam_style,
        },
        
        # 4. 2D Weights (APOLLO targets): 8-bit quantization, APOLLO low-rank projection
        {
            "params": group_apollo, 
            "weight_decay": weight_decay, 
            "quantize": True, 
            "apollo_rank": apollo_rank,
            "beta1": 0.9,              # Remove if minimizing optimizer memory is the priority.
        },

        # 5. 2D Weights (Others): Default Adafactor path (factored), 8-bit quantization
        {
            "params": group_2d, 
            "weight_decay": weight_decay, 
            "quantize": True, 
        },
        
        # 6. >2D Weights: 8-bit quantization, Weight Decay, Full-Rank
        {
            "params": group_nd, 
            "weight_decay": weight_decay, 
            "quantize": True, 
            "beta1": 0.9,
            **adam_style,
        },
    ]

model = MyModel().cuda()
optimizer = Adafactor8Bit(
    get_param_groups(model, lr_adam=lr_adam, weight_decay=1e-2, apollo_rank=256),
    lr=lr,
    # For continual learning or when using an external LR scheduler
    relative_step=False,              # Disable internal LR scheduling
    beta2=0.999,                      # Lock EMA window to prevent "blunting" over steps
    enable_fira_for_adafactor=True    # Enable Fira Limiter for all non-APOLLO paths
)

# Training loop...
```

> [!NOTE]
> For more complete examples, please refer to the [examples folder](https://github.com/yanfeiwong/adafactor-8bit/tree/main/examples).

</details>

## ⚙️ Advanced Configuration

<details>
<summary><strong>Show advanced configuration options</strong></summary>

### Continual Learning (`beta2` & `relative_step`)
By default, Adafactor's second-moment decay rate dynamically decays with the training step, and the internal learning rate schedule (`relative_step`) scales the learning rate accordingly. 

For endless fine-tuning or lifelong learning, this often leads to overly small learning rates and "blunted" second-moment estimates. To avoid these issues and keep the optimizer responsive:
- Set `relative_step=False` to disable the built-in LR schedule (allowing you to use an external scheduler).
- Set `beta2=0.999` to lock the EMA window (similar to Adam).

### Decoupled Weight Decay (`scale_weight_decay=False`)
By default, Adafactor's weight decay is coupled with the parameter's RMS scale. 
- If you prefer the AdamW-style decoupled weight decay, set `scale_weight_decay=False`.

### Fira Limiter
The Norm-Growth Limiter limits the relative increase of update norms to suppress destructive gradient spikes.
- **`enable_fira_for_apollo`**: Defaults to `True`. The APOLLO path applies the limiter by default to guard against sudden gradient rises in the low-rank subspace.
- **`enable_fira_for_adafactor`**: Defaults to `False`. Enables the limiter for the non-APOLLO paths.
- **`fira_margin`**: Defaults to `0.01`. The tolerance margin for norm growth, shared by both switches. The limiter activates only when the current update norm grows by more than this margin (e.g., `0.01` = 1%) compared to the previous step.

### Full-Rank and Factorized Variance (`factored`)

By default, Adafactor factorizes the second moment of $\ge$ 2D tensors into row and column statistics (`factored=True`) to minimize state memory. Setting `factored=False` switches to full element-wise variance (similar to RMSProp), while still retaining Adafactor's update RMS clipping mechanism (controlled by the `d` parameter). This configuration can be useful in the following scenarios:
- **Convolution Weights (>2D)**: Setting `factored=False` maintains independent variance for each spatial position in the convolution kernel, enabling finer per-element gradient scaling.
- **Sparse Embeddings**: Combining `factored=False`, `scale_parameter=False`, `d=0.0` and a lower learning rate creates a momentum-free adaptive optimizer. This allows for fine-grained, per-token updates while avoiding cold-token interference.

### No-Compiler Environments (`use_cuda_kernel=False`)
If you are in an environment without a CUDA compiler and want to bypass JIT compilation entirely:
- Set `use_cuda_kernel=False` to fall back to the pure PyTorch implementation.

</details>

## 🌌 APOLLO Low-Rank Subspace Projection

<details>
<summary><strong>Show APOLLO configuration and notes</strong></summary>

Enable the APOLLO path to compute gradient scaling factors in a memory-efficient low-rank subspace. Compared to Adafactor's standard row/column factorization (which assumes spatial independence), APOLLO uses random subspace projection to capture cross-dimensional covariance information while keeping memory overhead low.

- **`apollo_rank`**: The target rank for the projection subspace. The default is `0` (disabled).  
  
  - The official APOLLO GitHub repository recommends a rank of `256` for 1B and 7B models. 
  - The [LLaMA-Factory](https://llamafactory.readthedocs.io/en/latest/advanced/arguments.html#apollo) default is `16`.
  - Setting this to `1` (APOLLO-Mini style) minimizes the low-rank state (saves even more VRAM than the Adafactor path). The original APOLLO-Mini relies on the first-moment (`beta1`) to smooth out projection noise. To replicate this, set `beta1=0.9` alongside `apollo_rank=1`.

- **`apollo_scale`**: Heuristic scale parameter used to compensate for approximation error introduced by low-rank gradient scaling. The implementation applies `sqrt(apollo_scale)` to the resulting scaling factor. Defaults to 1.0.
- **`apollo_scale_type`**: Determines how the scaling factor is applied. `'channel'` applies it per channel (Standard APOLLO), while `'tensor'` applies it globally (APOLLO-Mini).
- **`apollo_update_proj_gap`**: Steps between projection matrix refreshes. Defaults to `200`.
- **`apollo_factorize` (Experimental)**: Applies Adafactor-style row/column factorization within the low-rank subspace to further reduce state memory overhead.

</details>

## 🧊 CAME Confidence-Guided Updates

<details>
<summary><strong>Show CAME configuration and tuning</strong></summary>

Enable the CAME (Confidence-guided Adaptive Memory Efficient Optimization) path to add a confidence estimation stage after momentum accumulation:

**Adaptive Scaling ($V$) → Momentum Accumulation ($M$) → Confidence Weighting ($C$)**

### Key Parameters & Tuning

The confidence stage measures the consistency between the current update direction and historical momentum, adaptively suppressing highly oscillatory updates.

- **`beta3`**: EMA decay coefficient for the confidence matrix. Requires `beta1` (momentum) and `factored=True`. Defaults to `None` (disabled).
- **Learning Rate**: The official CAME implementation recommends **0.5–0.9×** the AdamW learning rate (see [official tuning guide](https://github.com/yangluo7/CAME/tree/master#hyper-parameter-tuning)). To align with the original CAME behavior in this library, disable Adafactor's RMS scaling (`scale_parameter=False`) and set `d=1.0` (which corresponds to CAME's `clip_threshold`).
- **Warmup**: Since the confidence matrix is zero-initialized without bias correction, a learning rate warmup is recommended to safely establish the confidence baseline.
- **Choosing `beta3`**: `beta3` should generally be larger than `beta2` so the confidence estimate evolves more slowly than the variance estimate. A practical starting range is **0.9995–0.99995** when `beta2=0.999`.


### Configuration Example

To replicate "vanilla" CAME (stripping Adafactor's native modifications), you can use the following configuration:

```python
{
    "params": param_group,
    "lr": lr,                           # Original CAME recommends 0.5-0.9x AdamW LR
    "beta1": 0.9,
    "beta2": 0.999,
    "beta3": 0.9999,                    # Enable CAME confidence guidance
    "apollo_rank": 0,                   # Set to 0 for vanilla CAME. (Set >0 to enable APOLLO+CAME fusion)
    "weight_decay": weight_decay,
    "scale_weight_decay": False,
    "scale_parameter": False,           # Disable Adafactor RMS scaling to align with vanilla CAME
    "d": 1.0,
    "relative_step": False,
},
```

</details>

## 📈 Learning Rate Guide for Beginners

<details>
<summary><strong>Show learning-rate guidance</strong></summary>

If you are migrating from optimizers like AdamW, Adafactor's learning rate behavior might feel a bit different. This is mainly due to the `scale_parameter` option.

- **`scale_parameter=True` (default)**
  Because of RMS scaling, a very small `lr` (e.g., `1e-5`) often leads to extremely slow progress. Start with `lr=1e-3` and adjust in the range `1e-4`–`5e-3` if needed.

- **`scale_parameter=False`**
  Disables RMS scaling, making the update scale more similar to AdamW. Use the learning rates you're familiar with for AdamW and tune as usual. (Note: the second moment is still factorized, so behavior is not identical.)

*These are safe starting points. Always validate on your own task and batch size.*

</details>

## 📌 Implementation Differences from Reference Optimizers

To maintain a unified and robust framework, some algorithmic paths have implementation-level differences compared to their original standalone repositories:
- **APOLLO**: Uses deterministic per-state seed allocation and an optimizer-level seed counter for projection matrix management.
- **Adafactor**: When `beta1` is enabled, the momentum update ordering differs from the Hugging Face implementation.

## 🎓 Acknowledgements

This project builds upon the foundational work of several researchers and open-source communities. Sincere thanks to the following for their invaluable contributions:

### Core Algorithm & Optimizer Design
- **Noam Shazeer & Mitchell Stern** for proposing the original **Adafactor** algorithm ([Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235)).
- **Tim Dettmers** for the inspiration from **8-bit block-wise quantization** ([8-BIT OPTIMIZERS VIA BLOCK-WISE QUANTIZATION](https://arxiv.org/abs/2110.02861)) and the [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) library.
- **Hanqing Zhu, Zhenyu Zhang, et al.** for the **APOLLO** algorithm ([APOLLO: SGD-Like Memory, AdamW-level Performance](https://arxiv.org/abs/2412.05270)).
- **Xi Chen, Kaituo Feng, et al.** for the **Norm-Growth Limiter** mechanism in **Fira** ([Fira: Can We Achieve Full-rank Training of LLMs Under Low-rank Constraint?](https://arxiv.org/abs/2410.01623)).
- **Yang Luo, et al.** for the **confidence-guided strategy** in **CAME** ([CAME: Confidence-guided Adaptive Memory Efficient Optimization](https://arxiv.org/abs/2307.02047)).

### Quantization & Implementation
- **The QLoRA Team** for their work on memory-efficient fine-tuning and quantization techniques ([QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)).
- **The PyTorch AO Team** for their work on [4-bit optimizer states](https://github.com/pytorch/ao/tree/main/torchao/optim), validating distribution-aware quantization for optimizer moments.
- **The PyTorch Team** for providing the foundational optimizer implementation and the C++ Extension toolchain.

### Technical Review & Discussion
- **Qwen, DeepSeek, ChatGPT, and ChatGLM** for technical discussions and code reviews on CUDA optimization, memory safety, cross-platform compilation, and implementation design.

## 📝 Citation

```bibtex
@misc{wang2026beyond,
  title         = {Beyond Dense Adam States: Adaptive Log-Space Quantization for Memory-Efficient Optimizers},
  author        = {Yan Wang},
  year          = {2026},
  eprint        = {2608.22322},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2608.22322}
}
```

## 🏛️ License

[The project is released under the MIT License.](https://github.com/yanfeiwong/adafactor-8bit/blob/main/LICENSE)

## ⭐ Star the Project

If this optimizer has been useful in your work, consider giving the repository a star. It helps others discover the project and supports future development.