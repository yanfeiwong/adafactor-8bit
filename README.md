<div align="center">

# 8-bit Adafactor with Fused CUDA Kernels

**English** | [中文](https://github.com/yanfeiwong/adafactor-8bit/blob/main/README_ZH.md)

[![PyPI version](https://badge.fury.io/py/adafactor8bit.svg)](https://badge.fury.io/py/adafactor8bit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Stars](https://img.shields.io/github/stars/yanfeiwong/adafactor-8bit?style=social)](https://github.com/yanfeiwong/adafactor-8bit/stargazers)

</div>

An 8-bit Adafactor optimizer featuring fused CUDA kernels and log-space block-wise quantization, designed to further reduce optimizer state memory while maintaining low step overhead and stability — suitable for large models such as LLMs and diffusion models.


## Key Features

- **Log-Space Quantization**: Maps the second moment (variance) to the log2 space before 8-bit quantization. This approach accommodates the long-tail distribution of variances, reducing the risk of small second-moment estimates being truncated to zero and improving overall training stability.
- **Fused CUDA Kernels**: Combines dequantization, EMA updates, Warp-Shuffle reductions, and requantization into single kernels. It utilizes `float4` vectorization to optimize memory bandwidth usage.
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

It is recommended to use `param_groups` to keep sensitive layers (Embedding, Norm, Bias) in FP32, enabling 8-bit quantization only for large 2D weight matrices.

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
)

# Training loop...
```

For a complete example, please refer to [basic_usage.py](https://github.com/yanfeiwong/adafactor-8bit/blob/main/examples/basic_usage.py).


## Advanced Configuration

### Continual Learning (`beta2` & `relative_step`)
By default, Adafactor's second-moment decay rate dynamically decays with the training step, and the internal learning rate schedule (`relative_step`) scales the learning rate accordingly. 

For endless fine-tuning or lifelong learning, this often leads to overly small learning rates and "blunted" second-moment estimates. To avoid these issues and keep the optimizer responsive:
- Set `relative_step=False` to disable the built-in LR schedule (allowing you to use an external scheduler).
- Set `beta2=0.999` to lock the EMA window (similar to Adam).

### Decoupled Weight Decay (`scale_weight_decay=False`)
By default, Adafactor's weight decay is coupled with the parameter's RMS scale. 
- If you prefer the AdamW-style decoupled weight decay, set `scale_weight_decay=False`.

### No-Compiler Environments (`use_cuda_kernel=False`)
If you are in an environment without a CUDA compiler and want to bypass JIT compilation entirely:
- Set `use_cuda_kernel=False` to fall back to the pure PyTorch implementation.




## Learning Rate Guide for Beginners

If you are migrating from optimizers like AdamW, Adafactor's learning rate behavior might feel a bit different. This is mainly due to the `scale_parameter` option.

- **`scale_parameter=True` (default)**
  Because of RMS scaling, a very small `lr` (e.g., `1e-5`) often leads to extremely slow progress. Start with `lr=1e-3` and adjust in the range `1e-4`–`5e-3` if needed.

- **`scale_parameter=False`**
  Disables RMS scaling, making the update scale more similar to AdamW. Use the learning rates you're familiar with for AdamW and tune as usual. (Note: the second moment is still factorized, so behavior is not identical.)

*These are safe starting points; Always validate on your own task and batch size.*





## Acknowledgements

Thanks to **Noam Shazeer** and **Mitchell Stern** for proposing the original Adafactor algorithm in the paper [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235).

Thanks to **Tim Dettmers** for the inspiration from the paper [8-BIT OPTIMIZERS VIA BLOCK-WISE QUANTIZATION](https://arxiv.org/abs/2110.02861) and the [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) library.

Thanks to the **PyTorch team** for providing the foundational Optimizer implementation and the C++ Extension toolchain.

Thanks to the large language models **Qwen** and **DeepSeek** for valuable technical discussions and code reviews on CUDA low-level optimization, memory safety mechanisms, and cross-platform compilation pipeline design.


## License

[The project is released under the MIT License.](https://github.com/yanfeiwong/adafactor-8bit/blob/main/LICENSE)
