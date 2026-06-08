[English](./README.md) | **中文**

# Adafactor 8-bit with Fused CUDA Kernels

[![PyPI version](https://badge.fury.io/py/adafactor8bit.svg)](https://badge.fury.io/py/adafactor8bit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Stars](https://img.shields.io/github/stars/yanfeiwong/adafactor-8bit?style=social)](https://github.com/yanfeiwong/adafactor-8bit/stargazers)

一个面向大规模模型训练的高显存效率 8-bit Adafactor 优化器。

通过 CUDA 融合算子与分块量化（Block-wise Quantization），在保持训练稳定性的同时降低优化器状态的显存占用，适用于 LLM 与 Diffusion 等大模型训练。

## 核心特性

- **对数空间量化**：在 8-bit 量化前，将二阶矩（方差）映射到 log2 空间。这种方式适应了方差的长尾分布，降低了极小梯度值被截断为零的风险，提升了训练稳定性。
- **CUDA 融合算子**：将反量化、EMA 更新、Warp-Shuffle 归约与重新量化整合到单一 Kernel 中，并利用 `float4` 向量化优化显存带宽使用。
- **零同步开销**：重构了控制流，减少了隐式的 CPU-GPU 同步，使 GPU 计算流水线能够异步运行。
- **Transformers 兼容**：对齐 `transformers` 中 Adafactor 的行为（如耦合权重衰减、稳健的 epsilon 处理），确保大规模训练的稳定性，同时完整支持 `relative_step` 等标准调度开关。
- **跨平台 JIT 编译**：使用即时编译（JIT），在 Windows 和 Linux 环境下均可便捷配置。

## 性能表现

- **显存占用**：得益于 Adafactor 的分解二阶矩估计与 8-bit 量化，优化器状态的显存占用通常低于 `AdamW8Bit`，有助于缓解显存受限环境下的压力。
- **训练速度**：融合算子设计与减少的同步开销，使其能够实现与主流 8-bit 优化器相当的单步（step）耗时。
- **量化精度**：Adafactor 的二阶矩（方差）严格非负且跨越多个数量级。通过将其映射到 log2 空间的 `UINT8` 而非线性空间，优化器为极小方差保留了相对精度，缓解了标准 8-bit 量化中异常梯度引起的不稳定性。

## 安装

这个项目采用 JIT (Just-In-Time) 编译，无预编译二进制文件。

请确保你的环境中已安装 `torch` 和 `ninja`，并配置好了 CUDA 编译器（如 MSVC 或 GCC）。

如果 CUDA 编译失败，优化器会自动回退到纯 PyTorch 实现。

### From PyPI

```bash
pip install -U adafactor8bit
```

### From Source

```bash
pip install git+https://github.com/yanfeiwong/adafactor-8bit.git
```

**注意**：首次实例化优化器（或运行示例代码）时，会自动触发 CUDA 源码的 JIT 编译。这可能需要几十秒到几分钟的时间（取决于您的硬件与编译器），期间终端可能无明显输出，请耐心等待。编译完成后结果会被自动缓存，后续无需等待。

## 使用示例

建议通过 `param_groups` 将敏感层（Embedding, Norm, Bias）保持在 FP32，仅对大型 2D 权重矩阵启用 8-bit 量化。

```python
import torch
import torch.nn as nn
from adafactor8bit import Adafactor8Bit

def get_param_groups(model, weight_decay=1e-2):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        # 保护 1D 张量、bias、norm 和 embedding
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

更多完整示例请参考 [basic_usage.py](./examples/basic_usage.py)

## 致谢

感谢来自 Qwen 与 DeepSeek 的大语言模型在 CUDA 底层优化、内存安全防御机制以及跨平台编译链路设计上提供的深度技术探讨与代码审查。

感谢 Tim Dettmers 的 [8-BIT OPTIMIZERS VIA BLOCK-WISE QUANTIZATION](https://arxiv.org/pdf/2110.02861) 论文及 [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) 库带来的启发。

感谢 PyTorch 团队提供的基础 Optimizer 实现与 C++ Extension 工具链。

## License

[MIT License](./LICENSE)