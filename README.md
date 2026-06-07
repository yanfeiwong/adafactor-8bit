**English** | [中文](./README_ZH.md)

# Adafactor 8-bit with Fused CUDA Kernels

An 8-bit Adafactor optimizer designed for memory-efficient large-scale model training.

It uses fused CUDA kernels and block-wise quantization to reduce optimizer state memory while maintaining training stability, making it suitable for training large models such as LLMs and diffusion models.

## Key Features

- **Fused CUDA Kernel**: Integrates dequantization, EMA updates, Warp-Shuffle reduction, and requantization into a single kernel, utilizing `float4` vectorization to maximize memory bandwidth utilization.
- **Zero CPU-GPU Sync**: Refactored the control flow to eliminate implicit synchronizations, ensuring the GPU computation pipeline runs asynchronously at high speed.
- **Cross-Platform JIT**: Utilizes JIT (Just-In-Time) automatic compilation for seamless setup across Windows and Linux environments.

## Algorithm Details

Rebuilt upon the official PyTorch Adafactor, the mathematical logic **aligns more closely with the original paper and `HuggingFace transformers`**. Key differences include:

1. **Safe Injection of `eps1`**: The official PyTorch implementation defaults to `eps1=None` and relies on `clamp`, which can lead to NaNs when encountering zero or extremely small gradients. This project adopts the original `grad_squared + eps1` approach, fundamentally guaranteeing the strict positive definiteness of the second moment and preventing training crashes caused by `rsqrt(0)`.
2. **Coupled Weight Decay**: Unlike the official PyTorch implementation which decouples Weight Decay from RMS, this project retains the Coupled mechanism from the original paper (Weight Decay multiplied by the effective learning rate that includes RMS scaling).
3. **Standard Parameter Support**: Fully retains core Adafactor switches such as `relative_step` and `scale_parameter`, ensuring compatibility with existing learning rate scheduling strategies.

## Performance

- **Memory Footprint**: The memory usage of optimizer states is **significantly lower than `AdamW8Bit`** (bitsandbytes), making it an ideal choice for training massive models or when memory-constrained.
- **Training Speed**: The Fused Kernel and Zero-Sync design enable it to achieve step speeds comparable to mainstream 8-bit optimizers.
- **Quantization Precision & Stability**: The second moment (variance) in Adafactor is always non-negative, so we map it to `UINT8 (0~255)`. Compared to traditional 8-bit optimizers that map to `INT8 (-127~127)`, providing higher effective quantization precision within the non-negative variance domain.

## Installation

This project uses JIT (Just-In-Time) compilation.

Please ensure torch and ninja are installed, and a CUDA compiler (such as MSVC or GCC) is available in your environment.

If CUDA compilation fails, the optimizer will automatically fall back to the pure PyTorch implementation.


```bash
pip install git+https://github.com/yanfeiwong/adafactor-8bit.git
```

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
    relative_step=False,
    block_size=2048,
    min_8bit_size=4096
)

# Training loop...
```

For a complete example, please refer to [basic_usage.py](./examples/basic_usage.py).

## Acknowledgements

Thanks to the large language models Qwen and DeepSeek for valuable technical discussions and code reviews on CUDA low-level optimization, memory safety mechanisms, and cross-platform compilation pipeline design.

Thanks to Tim Dettmers for the inspiration from the paper [8-BIT OPTIMIZERS VIA BLOCK-WISE QUANTIZATION](https://arxiv.org/pdf/2110.02861) and the [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) library.

Thanks to the PyTorch team for providing the foundational Optimizer implementation and the C++ Extension toolchain.

## License

[The project is released under the MIT License.](./LICENSE)