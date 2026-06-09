<div align="center">

# 8-bit Adafactor with Fused CUDA Kernels

[English](./README.md) | **中文**

[![PyPI version](https://badge.fury.io/py/adafactor8bit.svg)](https://badge.fury.io/py/adafactor8bit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Stars](https://img.shields.io/github/stars/yanfeiwong/adafactor-8bit?style=social)](https://github.com/yanfeiwong/adafactor-8bit/stargazers)

</div>

一个专为显存高效的大规模模型训练而设计的 8-bit Adafactor 优化器。结合了融合 CUDA 算子与对数空间分块量化技术，旨在进一步降低优化器状态显存占用的同时，保持极低的单步更新开销与训练稳定性，适合训练 LLMs 和 Diffusion 模型。

## 核心特性

- **对数空间量化**：在 8-bit 量化前，将二阶矩（方差）映射到 log2 空间。这种方式适应了方差的长尾分布，降低了极小的二阶矩估计值被截断为零的风险，提升了训练稳定性。
- **CUDA 融合算子**：将反量化、EMA 更新、Warp-Shuffle 归约与重新量化整合到单一 Kernel 中，并利用 `float4` 向量化优化显存带宽使用。
- **零同步开销**：重构了控制流，消除了隐式的 CPU-GPU 同步（如 D2H 拷贝），确保 GPU 计算流水线能够无阻塞地异步运行。
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
    # 针对长期连续训练和搭配外部学习率调度的情况
    relative_step=False,     # 禁用内部LR的调度
    beta2=0.999,             # 如果设置了 beta2，那么优化器将会恒定保持对新梯度分布的自适应能力，不会随着训练步骤的推进而钝化，适合长期连续的训练
)

# Training loop...
```

更多完整示例请参考 [basic_usage.py](./examples/basic_usage.py)



## 高级配置

### 长期连续训练 (`beta2` 与 `relative_step`)
默认情况下，Adafactor 的二阶矩衰减率会随着训练步数动态衰减，内部学习率调度 (`relative_step`) 也会相应地缩放学习率。

对于无休止的微调或持续学习场景，这通常会导致 `后期学习率过小` 和 `二阶矩估计“钝化”` 。为了避免这些问题并保持优化器对新梯度分布的自适应能力：
- 设置 `relative_step=False` 以禁用内置的学习率调度（从而允许您使用外部的学习率调度器）。
- 设置 `beta2=0.999` 以锁定 EMA 窗口（类似于 Adam）。

### 解耦权重衰减 (`scale_weight_decay=False`)
默认情况下，Adafactor 的权重衰减与参数的 RMS 缩放相耦合。
- 如果您倾向于 AdamW 风格的解耦权重衰减，请设置 `scale_weight_decay=False`。

### 无编译器环境 (`use_cuda_kernel=False`)
如果您处于没有 CUDA 编译器的环境中，并希望完全绕过 JIT 编译：
- 设置 `use_cuda_kernel=False` 即可回退到纯 PyTorch 实现。


## 新手学习率指南

如果你是从 AdamW 等优化器迁移过来的，可能会发现 Adafactor 的学习率表现有些不同。这主要与 `scale_parameter` 选项有关。

- **`scale_parameter=True`（默认）**
  由于 RMS 缩放效应，设置过小的 `lr`（例如 `1e-5`）通常会导致训练进展极其缓慢。建议从 `lr=1e-3` 开始，并根据需要在 `1e-4` 到 `5e-3` 的范围内进行微调。

- **`scale_parameter=False`**
  关闭 RMS 缩放后，更新步长的量级会更接近 AdamW。此时可以使用你熟悉的 AdamW 学习率，并按常规方式进行调参。（注：由于二阶矩依然采用分解估计，其实际行为与 AdamW 并不完全相同。）

*以上仅为安全的初始配置参考；请务必在你自己的任务和 batch size 下进行验证。*


## 致谢

感谢 **Noam Shazeer** 与 **Mitchell Stern** 在论文 [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235) 中提出了原版的 Adafactor 算法。

感谢 **Tim Dettmers** 的 [8-BIT OPTIMIZERS VIA BLOCK-WISE QUANTIZATION](https://arxiv.org/abs/2110.02861) 论文及 [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) 库带来的启发。

感谢 **PyTorch 团队**提供的基础 Optimizer 实现与 C++ Extension 工具链。

感谢来自 **Qwen** 与 **DeepSeek** 的大语言模型在 CUDA 底层优化、内存安全防御机制以及跨平台编译链路设计上提供的深度技术探讨与代码审查。

## License

[MIT License](./LICENSE)