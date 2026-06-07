[English](./README.md) | **中文**

# Adafactor 8-bit with Fused CUDA Kernels

一个高显存效率、极速的 8-bit Adafactor 优化器。

通过 CUDA 融合算子与 Block-wise 量化，在保持训练稳定性的同时，进一步降低优化器状态的显存占用，为 LLM 与 Diffusion 等大模型训练提供更优的显存方案。

## 核心特性

- **Fused CUDA Kernel**: 将反量化、EMA 更新、Warp-Shuffle 归约与重新量化融合为单一 Kernel，并使用 `float4` 向量化以最大化显存带宽利用率。
- **Zero CPU-GPU Sync**: 重构了控制流，消除了一些隐式同步，确保 GPU 计算流水线高速异步运行。
- **Cross-Platform JIT**: 使用 Windows/Linux 环境下的 JIT 自动编译。

## 算法细节改动

项目基于 PyTorch 官方 Adafactor 的基础上进行了重构，数学逻辑**更贴近原版论文代码与 `HuggingFace transformers` 的实现**，主要区别如下：

1. **`eps1` 的安全注入**：PyTorch 官方默认 `eps1=None` 并依赖 `clamp`，这在遇到全零或极小梯度时容易引发 NaN。本项目采用原版的 `grad_squared + eps1` 方式，从根本上保证了二阶矩的严格正定性，解决了 `rsqrt(0)` 导致的训练崩溃。
2. **Coupled Weight Decay**：与 PyTorch 官方将 Weight Decay 与 RMS 解耦（Decoupled）的做法不同，本项目保留了原版论文中的 Coupled 机制（Weight Decay 乘以包含 RMS 缩放的有效学习率）。
3. **标准参数支持**：完整保留了 `relative_step` 与 `scale_parameter` 等原版 Adafactor 的核心开关，确保与现有学习率调度策略的兼容。

## 性能表现

- **显存占用**：优化器状态显存占用**显著低于 `AdamW8Bit`** (bitsandbytes)，是训练超大模型或受限于显存时的理想选择。
- **训练速度**：Fused Kernel 与 Zero-Sync 设计使其具备与主流 8-bit 优化器几乎相当的 Step 速度。
- **量化精度与稳定性**：Adafactor 的二阶矩（方差）始终非负，所以我们使用 `UINT8 (0~255)` 进行映射。相比传统的 8bit 优化器映射 `INT8 (-127~127)`，有效精度更集中，可以不严谨的说是提升了一倍，在训练中有更优的数值稳定性。

## 安装

这个项目采用 JIT (Just-In-Time) 编译，无预编译二进制文件。

请确保你的环境中已安装 `torch` 和 `ninja`，并配置好了 CUDA 编译器（如 MSVC 或 GCC）。

如果 CUDA 编译失败，优化器会自动回退到纯 PyTorch 实现。


```bash
pip install git+https://github.com/yanfeiwong/adafactor-8bit.git
```

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

感谢 Tim Dettmers 的 [8-BIT OPTIMIZERS VIA BLOCK-WISE QUANTIZATION](https://arxiv.org/pdf/2110.02861) 论文  及 [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) 库带来的启发。

感谢 PyTorch 团队提供的基础 Optimizer 实现与 C++ Extension 工具链。
## License
[MIT License](./LICENSE) 