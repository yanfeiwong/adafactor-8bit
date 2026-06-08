# Copyright (c) 2026 WANG YAN
# Licensed under the MIT License.

import os
import sys
import math
import logging
from typing import Tuple, Optional, Union, List, Dict, Any, Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.cpp_extension import load

__all__ = ["Adafactor8Bit"]

logger = logging.getLogger(__name__)

_FP32_TINY = 1.17549435e-38 
_FP32_MIN_LOG = -126.0

# ==========================================
# 1. CUDA Kernel JIT Loading
# ==========================================
_CUDA_MODULE = None
_CUDA_AVAILABLE = False

def _load_cuda_module() -> bool:
    global _CUDA_MODULE, _CUDA_AVAILABLE
    if _CUDA_MODULE is not None or _CUDA_AVAILABLE:
        return _CUDA_AVAILABLE

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] < 7:
            logger.warning(f"Adafactor8Bit: GPU Compute Capability {cap[0]}.{cap[1]} < 7.0. Falling back to PyTorch.")
            return False
        
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cu_path = os.path.join(current_dir, "kernels.cu")

    if not os.path.exists(cu_path):
        logger.warning("kernels.cu not found. Falling back to pure PyTorch implementation.")
        return False

    try:
        is_windows = sys.platform == "win32"
        extra_cflags = ["/Zc:preprocessor"] if is_windows else []
        extra_cuda_cflags = ["-O3", "--use_fast_math"]
        if is_windows:
            extra_cuda_cflags.extend(["-Xcompiler", "/Zc:preprocessor"])

        _CUDA_MODULE = load(
            name="adafactor8bit_cuda",
            sources=[cu_path],
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=False
        )
        _CUDA_AVAILABLE = True
        logger.info("Adafactor8Bit: CUDA Kernel loaded successfully!")
    except Exception as e:
        logger.warning(f"Adafactor8Bit: Failed to load CUDA Kernel. Falling back to PyTorch. Error: {e}")

    return _CUDA_AVAILABLE

# ==========================================
# 2. Log-Quantization Utilities
# ==========================================
def _log_quantize_nonneg(tensor: Tensor, block_size: int = 2048) -> Tuple[Tensor, Tensor, torch.Size, int]:
    """将非负 FP32 张量映射到对数空间后分块量化为 UINT8 (0-255)"""
    shape = tensor.shape
    flat = tensor.flatten()
    pad = (block_size - flat.numel() % block_size) % block_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))

    blocks = flat.view(-1, block_size)
    log_blocks = torch.log2(blocks.clamp(min=_FP32_TINY))
    max_log = log_blocks.amax(dim=1, keepdim=True)
    
    scale = ((max_log - _FP32_MIN_LOG) / 255.0).clamp(min=1e-12)
    q = torch.round((log_blocks - _FP32_MIN_LOG) / scale * 255.0).clamp(0, 255).to(torch.uint8)
    return q, scale.squeeze(-1), shape, pad

def _log_dequantize_nonneg(q: Tensor, scale: Tensor, shape: torch.Size, pad: int) -> Tensor:
    """从对数空间反量化回线性空间"""
    log_blocks = q.float() * scale.unsqueeze(-1) / 255.0 + _FP32_MIN_LOG
    blocks = torch.pow(2.0, log_blocks)
    flat = blocks.flatten()
    if pad:
        flat = flat[:-pad]
    return flat.view(shape)

def _pad_to_block_size(tensor: Tensor, block_size: int) -> Tensor:
    flat = tensor.contiguous().flatten()
    pad_len = (block_size - flat.numel() % block_size) % block_size
    if pad_len > 0:
        return torch.nn.functional.pad(flat, (0, pad_len))
    return flat

# ==========================================
# 3. Optimizer Core
# ==========================================
class Adafactor8Bit(Optimizer):
    def __init__(
        self,
        params: Iterable[Union[Tensor, Dict[str, Any]]],
        lr: float = 1e-2,
        beta2_decay: float = -0.8,
        eps: Tuple[Optional[float], float] = (1e-30, 1e-3),
        d: float = 1.0,
        weight_decay: float = 0.0,
        maximize: bool = False,
        relative_step: bool = False,
        scale_parameter: bool = True,
        quantize: bool = True,
        block_size: int = 2048,
        min_8bit_size: int = 4096,
    ):
        if not 0.0 <= lr: raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 >= beta2_decay: raise ValueError(f"Invalid beta2_decay: {beta2_decay}")
        eps1, eps2 = eps
        if eps1 is not None and not 0.0 <= eps1: raise ValueError(f"Invalid eps1: {eps1}")
        if not 0.0 <= eps2: raise ValueError(f"Invalid eps2: {eps2}")
        if not 1.0 <= d: raise ValueError(f"Invalid d: {d}")
        if not 0.0 <= weight_decay: raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        if quantize and block_size % 1024 != 0:
            raise ValueError(f"block_size must be a multiple of 1024, but got {block_size}.")

        defaults = dict(
            lr=lr, beta2_decay=beta2_decay, eps=eps, d=d, weight_decay=weight_decay,
            maximize=maximize, relative_step=relative_step, scale_parameter=scale_parameter,
            quantize=quantize, block_size=block_size, min_8bit_size=min_8bit_size,
        )
        super().__init__(params, defaults)

    def _init_group(self, group, params_with_grad, grads, states, state_steps):
        group_quantize = group.get("quantize", True)
        block_size = group.get("block_size", 2048)
        min_8bit_size = group.get("min_8bit_size", 4096)

        for p in group["params"]:
            if p.grad is None: continue

            force_fp32 = p.numel() < min_8bit_size
            use_quant = group_quantize and not force_fp32

            params_with_grad.append(p)
            grads.append(p.grad)
            state = self.state[p]

            needs_init = False
            if len(state) == 0:
                needs_init = True
            else:
                is_nd = (p.grad.dim() >= 2)
                has_row_col = ("row_var" in state or "row_var_q" in state)
                has_var = ("variance" in state or "variance_q" in state)
                
                if (is_nd and has_var) or (not is_nd and has_row_col):
                    logger.warning(f"Adafactor8Bit: State structure mismatch for param shape {p.shape}. Re-initializing state.")
                    step_backup = state.get("step", 0)
                    state.clear()
                    state["step"] = step_backup
                    needs_init = True

            if needs_init:
                state["step"] = 0 
                state["is_quantized"] = use_quant
                state["block_size"] = block_size

                if p.grad.dim() >= 2:
                    shape = p.grad.shape
                    R = shape[-2]
                    C = shape[-1]
                    batch_shape = shape[:-2]
                    
                    r_shape = list(batch_shape) + [R, 1]
                    c_shape = list(batch_shape) + [1, C]
                    
                    if use_quant:
                        r_tmp = torch.full(r_shape, _FP32_TINY, device=p.device)
                        q, s, sh, pad = _log_quantize_nonneg(r_tmp, block_size)
                        state["row_var_q"], state["row_var_scale"], state["row_var_shape"], state["row_var_pad"] = q, s, sh, pad
                        
                        c_tmp = torch.full(c_shape, _FP32_TINY, device=p.device)
                        q, s, sh, pad = _log_quantize_nonneg(c_tmp, block_size)
                        state["col_var_q"], state["col_var_scale"], state["col_var_shape"], state["col_var_pad"] = q, s, sh, pad
                    else:
                        state["row_var"] = torch.zeros(r_shape, device=p.device)
                        state["col_var"] = torch.zeros(c_shape, device=p.device)
                else:
                    if use_quant:
                        v_tmp = torch.full_like(p.grad, _FP32_TINY, memory_format=torch.preserve_format)
                        q, s, sh, pad = _log_quantize_nonneg(v_tmp, block_size)
                        state["variance_q"], state["variance_scale"], state["variance_shape"], state["variance_pad"] = q, s, sh, pad
                    else:
                        state["variance"] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
            else:
                if torch.is_tensor(state["step"]):
                    state["step"] = int(state["step"].cpu().item())
                elif not isinstance(state["step"], int):
                    state["step"] = int(state["step"])

                for k in list(state.keys()):
                    if isinstance(state[k], torch.Tensor):
                        if state[k].device != p.device:
                            state[k] = state[k].to(p.device)
                        if k.endswith("_q") and state[k].dtype != torch.uint8:
                            state[k] = state[k].to(torch.uint8)
                            scale_key = k.replace("_q", "_scale")
                            if scale_key in state and state[scale_key].dtype != torch.float32:
                                state[scale_key] = state[scale_key].to(torch.float32)

                curr_block_size = state.get("block_size", block_size)

                if use_quant and not state.get("is_quantized", False):
                    if p.grad.dim() >= 2:
                        if "row_var" in state and "row_var_q" not in state:
                            state["row_var"].clamp_(min=_FP32_TINY)
                            q, s, sh, pad = _log_quantize_nonneg(state["row_var"], curr_block_size)
                            state["row_var_q"], state["row_var_scale"], state["row_var_shape"], state["row_var_pad"] = q, s, sh, pad
                            del state["row_var"]
                        if "col_var" in state and "col_var_q" not in state:
                            state["col_var"].clamp_(min=_FP32_TINY)
                            q, s, sh, pad = _log_quantize_nonneg(state["col_var"], curr_block_size)
                            state["col_var_q"], state["col_var_scale"], state["col_var_shape"], state["col_var_pad"] = q, s, sh, pad
                            del state["col_var"]
                    else:
                        if "variance" in state and "variance_q" not in state:
                            state["variance"].clamp_(min=_FP32_TINY)
                            q, s, sh, pad = _log_quantize_nonneg(state["variance"], curr_block_size)
                            state["variance_q"], state["variance_scale"], state["variance_shape"], state["variance_pad"] = q, s, sh, pad
                            del state["variance"]
                    state["is_quantized"] = True
                
                elif not use_quant and state.get("is_quantized", False):
                    if p.grad.dim() >= 2:
                        if "row_var_q" in state:
                            state["row_var"] = _log_dequantize_nonneg(state.pop("row_var_q"), state.pop("row_var_scale"), state.pop("row_var_shape"), state.pop("row_var_pad"))
                        if "col_var_q" in state:
                            state["col_var"] = _log_dequantize_nonneg(state.pop("col_var_q"), state.pop("col_var_scale"), state.pop("col_var_shape"), state.pop("col_var_pad"))
                    else:
                        if "variance_q" in state:
                            state["variance"] = _log_dequantize_nonneg(state.pop("variance_q"), state.pop("variance_scale"), state.pop("variance_shape"), state.pop("variance_pad"))
                    state["is_quantized"] = False

                if "is_quantized" not in state:
                    state["is_quantized"] = ("row_var_q" in state or "col_var_q" in state or "variance_q" in state)
                if "block_size" not in state:
                    state["block_size"] = block_size

            states.append(state)
            state_steps.append(state["step"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            params_with_grad, grads, states, state_steps = [], [], [], []
            eps1, eps2 = group["eps"]
            self._init_group(group, params_with_grad, grads, states, state_steps)

            for i in range(len(params_with_grad)):
                _update_param_8bit(
                    params_with_grad[i], grads[i], states[i],
                    d=group["d"], lr=group["lr"], beta2_decay=group["beta2_decay"],
                    weight_decay=group["weight_decay"], eps1=eps1, eps2=eps2,
                    maximize=group["maximize"], relative_step=group["relative_step"],
                    scale_parameter=group["scale_parameter"], block_size=group.get("block_size", 2048),
                )
        return loss

# ==========================================
# 4. Parameter Update Logic
# ==========================================
def _update_param_8bit(
    param: Tensor, grad: Tensor, 
    state: Dict[str, Any],
    d: float, lr: Union[float, Tensor], 
    beta2_decay: float, weight_decay: float,
    eps1: Optional[float], 
    eps2: float, 
    maximize: bool, 
    relative_step: bool,
    scale_parameter: bool, 
    block_size: int
):
    if eps1 is None:
        eps1 = torch.finfo(param.dtype).eps

    original_dtype = param.dtype
    
    grad_fp32 = grad.float()
    if not grad_fp32.is_contiguous():
        grad_fp32 = grad_fp32.contiguous()
    if maximize:
        grad_fp32 = grad_fp32.neg()

    if not param.is_contiguous():
        param_work = param.contiguous()
        needs_copy_back = True
    else:
        param_work = param
        needs_copy_back = False

    quantize = state.get("is_quantized", False)
    curr_block_size = state.get("block_size", block_size)

    step = state["step"] + 1
    state["step"] = step 
    beta_val = math.pow(step, beta2_decay)

    if isinstance(lr, float):
        rho = min(lr, 1.0 / math.sqrt(step)) if relative_step else lr
        rho_t = torch.tensor(rho, device=param_work.device, dtype=torch.float32)
    else:
        if relative_step:
            step_t = torch.tensor(step, device=param_work.device, dtype=torch.float32)
            rho_t = torch.minimum(step_t.rsqrt(), lr)
        else:
            rho_t = lr

    if scale_parameter:
        param_rms = param_work.float().norm(2) / math.sqrt(param_work.numel())
        alpha = torch.clamp(param_rms, min=eps2) * rho_t
    else:
        alpha = rho_t

    if weight_decay != 0:
        param_work.mul_(1.0 - alpha * weight_decay)

    is_1d = grad_fp32.dim() < 2 

    if is_1d:
        # 1D 路径：绝对无偏，底层 _FP32_TINY 兜底
        grad_sq = grad_fp32.square() 
        
        if quantize:
            if _load_cuda_module():
                grad_sq_padded = _pad_to_block_size(grad_sq, curr_block_size)
                _CUDA_MODULE.fused_log_quantize_lerp(state["variance_q"], state["variance_scale"], grad_sq_padded, beta_val, curr_block_size)
                
                numel = param_work.numel()
                grad_fp32_flat = grad_fp32.view(-1)
                variance_q_flat = state["variance_q"].view(-1)
                variance_scale_flat = state["variance_scale"].view(-1)
                
                total_sum_sq = torch.zeros(1, device=param_work.device, dtype=torch.float32)
                
                _CUDA_MODULE.compute_update_norm_1d(
                    variance_q_flat, variance_scale_flat,
                    grad_fp32_flat, total_sum_sq, eps1, numel, curr_block_size
                )
                
                _CUDA_MODULE.apply_update_1d(
                    param_work.view(-1), grad_fp32_flat,
                    variance_q_flat, variance_scale_flat,
                    total_sum_sq, alpha, d, eps1, numel, curr_block_size
                )
            else:
                variance = _log_dequantize_nonneg(state["variance_q"], state["variance_scale"], state["variance_shape"], state["variance_pad"])
                variance.lerp_(grad_sq, beta_val)
                q, s, sh, pad = _log_quantize_nonneg(variance, curr_block_size)
                state["variance_q"], state["variance_scale"], state["variance_shape"], state["variance_pad"] = q, s, sh, pad
                
                update = variance.clamp_(min=eps1).rsqrt_().mul_(grad_fp32)
                denom = torch.clamp(update.norm(2) / (math.sqrt(update.numel()) * d), min=1.0)
                param_work.add_(update, alpha=-alpha / denom)
        else:
            variance = state["variance"]
            variance.lerp_(grad_sq, beta_val)
            update = variance.clamp_(min=eps1).rsqrt_().mul_(grad_fp32)
            denom = torch.clamp(update.norm(2) / (math.sqrt(update.numel()) * d), min=1.0)
            param_work.add_(update, alpha=-alpha / denom)

    else:
        shape = grad_fp32.shape
        R = shape[-2]
        C = shape[-1]
        numel = grad_fp32.numel()
        
        # 2D 路径：绝对无偏 EMA
        row_mean = torch.norm(grad_fp32, dim=-1, keepdim=True).square_().div_(C)
        col_mean = torch.norm(grad_fp32, dim=-2, keepdim=True).square_().div_(R)

        if quantize:
            if _load_cuda_module():
                row_mean_padded = _pad_to_block_size(row_mean, curr_block_size)
                col_mean_padded = _pad_to_block_size(col_mean, curr_block_size)

                _CUDA_MODULE.fused_log_quantize_lerp(state["row_var_q"], state["row_var_scale"], row_mean_padded, beta_val, curr_block_size)
                _CUDA_MODULE.fused_log_quantize_lerp(state["col_var_q"], state["col_var_scale"], col_mean_padded, beta_val, curr_block_size)

                row_var = _log_dequantize_nonneg(state["row_var_q"], state["row_var_scale"], state["row_var_shape"], state["row_var_pad"])
                row_mean_val_flat = row_var.mean(dim=-2, keepdim=True).clamp_(min=eps1).flatten().contiguous()
                
                grad_flat = grad_fp32.reshape(-1)
                row_var_q_flat = state["row_var_q"].reshape(-1)
                col_var_q_flat = state["col_var_q"].reshape(-1)
                
                total_sum_sq = torch.zeros(1, device=param_work.device, dtype=torch.float32)
                
                _CUDA_MODULE.compute_update_norm_2d(
                    row_var_q_flat, state["row_var_scale"],
                    col_var_q_flat, state["col_var_scale"],
                    grad_flat, total_sum_sq, row_mean_val_flat, eps1, R, C, numel, curr_block_size
                )
                
                param_flat = param_work.reshape(-1)
                _CUDA_MODULE.apply_update_2d(
                    param_flat, grad_flat,
                    row_var_q_flat, state["row_var_scale"],
                    col_var_q_flat, state["col_var_scale"],
                    total_sum_sq, alpha, row_mean_val_flat, d, eps1, R, C, numel, curr_block_size
                )
            else:
                row_var = _log_dequantize_nonneg(state["row_var_q"], state["row_var_scale"], state["row_var_shape"], state["row_var_pad"])
                col_var = _log_dequantize_nonneg(state["col_var_q"], state["col_var_scale"], state["col_var_shape"], state["col_var_pad"])
                row_var.lerp_(row_mean, beta_val)
                col_var.lerp_(col_mean, beta_val)
                q, s, sh, pad = _log_quantize_nonneg(row_var, curr_block_size)
                state["row_var_q"], state["row_var_scale"], state["row_var_shape"], state["row_var_pad"] = q, s, sh, pad
                q, s, sh, pad = _log_quantize_nonneg(col_var, curr_block_size)
                state["col_var_q"], state["col_var_scale"], state["col_var_shape"], state["col_var_pad"] = q, s, sh, pad
                
                var_estimate = row_var * col_var 
                row_mean_val = row_var.mean(dim=-2, keepdim=True).clamp_(min=eps1)
                var_estimate.div_(row_mean_val)
                
                update = var_estimate.clamp_(min=eps1).rsqrt_().mul_(grad_fp32)
                denom = torch.clamp(update.norm(2) / (math.sqrt(update.numel()) * d), min=1.0)
                param_work.add_(update, alpha=-alpha / denom)
        else:
            row_var = state["row_var"]
            col_var = state["col_var"]
            row_var.lerp_(row_mean, beta_val)
            col_var.lerp_(col_mean, beta_val)
            
            var_estimate = row_var * col_var
            row_mean_val = row_var.mean(dim=-2, keepdim=True).clamp_(min=eps1)
            var_estimate.div_(row_mean_val)
            
            update = var_estimate.clamp_(min=eps1).rsqrt_().mul_(grad_fp32)
            denom = torch.clamp(update.norm(2) / (math.sqrt(update.numel()) * d), min=1.0)
            param_work.add_(update, alpha=-alpha / denom)

    if needs_copy_back:
        param.copy_(param_work)