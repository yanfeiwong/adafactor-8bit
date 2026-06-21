# Copyright (c) 2026 WANG YAN
# Licensed under the MIT License.

import os
import sys
import math
import logging
from typing import Tuple, Optional, Union, Dict, Any, Iterable, Sequence

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
_CUDA_LOAD_ATTEMPTED = False

def _load_cuda_module(enable: bool = True) -> bool:
    global _CUDA_MODULE, _CUDA_AVAILABLE, _CUDA_LOAD_ATTEMPTED

    if not enable:
        return False
    
    if _CUDA_LOAD_ATTEMPTED:
        return _CUDA_AVAILABLE
    
    _CUDA_LOAD_ATTEMPTED = True
    
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
# 2. APOLLO Random Seed Utilities
# ==========================================
_ADV_DEFAULT = 0xF

def stable_randn(
    shape: Sequence[int],
    seed: int,
    device: Union[str, torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a reproducible random tensor from a normal distribution."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)

def next_seed(seed: int, adv: int = _ADV_DEFAULT) -> int:
    """Deterministically advance a seed by consuming `adv` random integers."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, torch.iinfo(torch.int64).max, (adv,), generator=generator).tolist()[-1]

# ==========================================
# 3. Log-Quantization Utilities
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
# 4. Optimizer Core
# ==========================================
class Adafactor8Bit(Optimizer):
    """
    8-bit Adafactor optimizer with fused CUDA kernels for memory-efficient large-scale training.
    
    Args:
        params (Iterable): Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (float, optional): External learning rate. Defaults to 1e-2.
        beta2 (float, optional): Fixed second-moment decay rate (e.g., 0.999 like Adam). 
            Locks the EMA window size, preventing "blunting" in long-term continual learning. 
            Mutually exclusive with `beta2_decay`. Defaults to None.
        beta2_decay (float): Dynamic decay rate coefficient. 
            The EMA weight is computed as `step ** beta2_decay`. Ignored if `beta2` is specified. 
            Defaults to -0.8.
        eps (Tuple[Optional[float], float]): Regularization constants (eps1, eps2).
            - `eps1`: Added to the squared gradient. If `None`, defaults to the machine epsilon 
              of the parameter's dtype (e.g., ~1.19e-7 for FP32), aligning with PyTorch official 
              behavior and preventing underflow.
            - `eps2`: Lower threshold for parameter RMS scaling. Defaults to (None, 1e-3).
        d (float): Clipping threshold for the final gradient update RMS. Defaults to 1.0.
        weight_decay (float): Weight decay (L2 penalty). Defaults to 0.0.
        scale_weight_decay (bool): If `True` (default), weight decay is coupled with the 
            parameter's RMS scale. If `False`, weight decay is decoupled and only scaled by the 
            base learning rate (AdamW-style).
        maximize (bool): Maximize the params based on the objective. Defaults to False.
        relative_step (bool): If `True`, uses time-dependent learning rate. Defaults to True.
        scale_parameter (bool): If `True`, scales learning rate by parameter RMS. Defaults to True.
        quantize (bool): Enable 8-bit log-space quantization for optimizer states. Defaults to True.
        block_size (int): Block size for quantization. Must be a multiple of 1024. Defaults to 2048.
        min_8bit_size (int): Minimum number of elements to apply 8-bit quantization. Defaults to 4096.
        use_cuda_kernel (bool): Whether to use custom CUDA kernels. Defaults to True.
        apollo_rank (int): If > 0, enables APOLLO-style random projection to low-rank space
            before applying Adafactor (no momentum). Defaults to 0 (disabled).
        apollo_beta1 (float): Momentum coefficient for first moment in low-rank space.
            Currently unused in momentum-free variant. Reserved for future extension.
        apollo_update_proj_gap (int): Steps between projection matrix updates. Defaults to 200.
        apollo_scale_type (str): How to compute the gradient scaling factor: 'channel' or 'tensor'.
            Defaults to 'channel'.
        apollo_eps (float): Epsilon for low-rank variance normalization. Defaults to 1e-8.
        apollo_factorize (bool): If True, uses Adafactor-style row/col factorization in the 
            low-rank space (FP32, ~16KB state) instead of full matrix variance (8-bit, ~100KB+ state). 
            For large models to drastically reduce optimizer state memory. Defaults to False.
        enable_fira_for_adafactor (bool): If `True`, enables Fira Limiter for the standard Adafactor path 
            to prevent gradient explosion by smoothing update norms. Defaults to False.
        fira_margin (float): The tolerance margin for Fira Limiter. The limiter activates when the 
            update norm grows by more than `fira_margin` (e.g., 0.01 for 1%). Shared with Apollo path. Defaults to 0.01.
    """
    def __init__(
        self,
        params: Iterable[Union[Tensor, Dict[str, Any]]],
        lr: float = 1e-2,
        beta2: Optional[float] = None,
        beta2_decay: float = -0.8,
        eps: Tuple[Optional[float], float] = (None, 1e-3),
        d: float = 1.0,
        weight_decay: float = 0.0,
        maximize: bool = False,
        relative_step: bool = True,
        scale_parameter: bool = True,
        scale_weight_decay: bool = True,
        quantize: bool = True,
        block_size: int = 2048,
        min_8bit_size: int = 4096,
        use_cuda_kernel: bool = True,
        apollo_rank: int = 0,
        apollo_beta1: float = 0.9,
        apollo_update_proj_gap: int = 200,
        apollo_scale_type: str = 'channel',
        apollo_eps: float = 1e-8,
        apollo_factorize: bool = False,
        enable_fira_for_adafactor: bool = False,
        fira_margin: float = 0.01,
    ):
        if not 0.0 <= lr: raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 >= beta2_decay: raise ValueError(f"Invalid beta2_decay: {beta2_decay}")
        eps1, eps2 = eps
        if eps1 is not None and not 0.0 <= eps1: raise ValueError(f"Invalid eps1: {eps1}")
        if not 0.0 <= eps2: raise ValueError(f"Invalid eps2: {eps2}")
        if not 1.0 <= d: raise ValueError(f"Invalid d: {d}")
        if not 0.0 <= weight_decay: raise ValueError(f"Invalid weight_decay: {weight_decay}")

        if beta2 is not None and not (0.0 <= beta2 < 1.0):
            raise ValueError(f"Invalid beta2: {beta2}, must be in [0.0, 1.0)")
        
        if quantize and block_size % 1024 != 0:
            raise ValueError(f"block_size must be a multiple of 1024, but got {block_size}.")

        if apollo_rank > 0 and apollo_scale_type not in ('channel', 'tensor'):
            raise ValueError(f"apollo_scale_type must be 'channel' or 'tensor', got {apollo_scale_type}.")
            
        if not 0.0 <= fira_margin: raise ValueError(f"Invalid fira_margin: {fira_margin}")

        defaults = dict(
            lr=lr, beta2_decay=beta2_decay, beta2=beta2, eps=eps, d=d, weight_decay=weight_decay,
            maximize=maximize, relative_step=relative_step, scale_parameter=scale_parameter,
            scale_weight_decay=scale_weight_decay,
            quantize=quantize, block_size=block_size, min_8bit_size=min_8bit_size,
            use_cuda_kernel=use_cuda_kernel,
            apollo_rank=apollo_rank, apollo_beta1=apollo_beta1,
            apollo_update_proj_gap=apollo_update_proj_gap,
            apollo_scale_type=apollo_scale_type, apollo_eps=apollo_eps,
            apollo_factorize=apollo_factorize,
            enable_fira_for_adafactor=enable_fira_for_adafactor,
            fira_margin=fira_margin,
        )
        super().__init__(params, defaults)

        self._apollo_seed_counter = 0

    def _init_group(self, group, params_with_grad, grads, states, state_steps):
        group_quantize = group.get("quantize", True)
        block_size = group.get("block_size", 2048)
        min_8bit_size = group.get("min_8bit_size", 4096)
        apollo_rank = group.get("apollo_rank", 0)

        for p in group["params"]:
            if p.grad is None: continue

            force_fp32 = p.numel() < min_8bit_size
            use_quant = group_quantize and not force_fp32

            params_with_grad.append(p)
            grads.append(p.grad)
            state = self.state[p]

            use_apollo = apollo_rank > 0 and p.grad.dim() >= 2

            needs_init = False
            if len(state) == 0:
                needs_init = True
            else:
                is_apollo_state = ('apollo_seed' in state)
                if (use_apollo and not is_apollo_state) or (not use_apollo and is_apollo_state):
                    logger.warning(f"Adafactor8Bit: Apollo/Adafactor mode changed for param shape {p.shape}. Re-initializing state.")
                    step_backup = state.get("step", 0)
                    state.clear()
                    state["step"] = step_backup
                    needs_init = True
                elif use_apollo and is_apollo_state:
                    if state.get("apollo_rank") != apollo_rank:
                        logger.warning(f"Adafactor8Bit: Apollo rank changed for param shape {p.shape}. Re-initializing state.")
                        step_backup = state.get("step", 0)
                        state.clear()
                        state["step"] = step_backup
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

                if use_apollo:
                    seed = self._apollo_seed_counter
                    self._apollo_seed_counter += 1
                    update_proj_gap = group.get("apollo_update_proj_gap", 200)
                    
                    state["apollo_seed"] = seed
                    state["apollo_rank"] = apollo_rank
                    state["apollo_update_proj_gap"] = update_proj_gap
                    state["last_proj_step"] = -update_proj_gap
                    state["v_low_q"] = None
                    state["v_low_scale"] = None
                    state["v_low_shape"] = None
                    state["v_low_pad"] = None
                    state["v_low"] = None

                else:
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
                    if use_apollo:
                        state["is_quantized"] = use_quant
                    else:
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

            apollo_rank = group.get("apollo_rank", 0)

            for i in range(len(params_with_grad)):
                if apollo_rank > 0 and 'apollo_seed' in states[i]:
                    _update_param_apollo_nomomentum(
                        params_with_grad[i], grads[i], states[i],
                        d=group["d"], lr=group["lr"], beta2=group["beta2"],
                        beta2_decay=group["beta2_decay"],
                        weight_decay=group["weight_decay"], eps1=eps1, eps2=eps2,
                        maximize=group["maximize"], relative_step=group["relative_step"],
                        scale_parameter=group["scale_parameter"],
                        scale_weight_decay=group.get("scale_weight_decay", True),
                        block_size=group.get("block_size", 2048),
                        use_cuda_kernel=group.get("use_cuda_kernel", True),
                        apollo_scale_type=group.get("apollo_scale_type", "channel"),
                        apollo_eps=group.get("apollo_eps", 1e-8),
                        apollo_factorize=group.get("apollo_factorize", False),
                        fira_margin=group.get("fira_margin", 0.01),
                    )
                else:
                    _update_param_8bit(
                        params_with_grad[i], grads[i], states[i],
                        d=group["d"], lr=group["lr"], beta2=group["beta2"], beta2_decay=group["beta2_decay"],
                        weight_decay=group["weight_decay"], eps1=eps1, eps2=eps2,
                        maximize=group["maximize"], relative_step=group["relative_step"],
                        scale_parameter=group["scale_parameter"], scale_weight_decay=group.get("scale_weight_decay", True),
                        block_size=group.get("block_size", 2048), use_cuda_kernel=group.get("use_cuda_kernel", True),
                        enable_fira_for_adafactor=group.get("enable_fira_for_adafactor", False),
                        fira_margin=group.get("fira_margin", 0.01),
                    )
        return loss

# ==========================================
# 5. Parameter Update Logic
# ==========================================
def _apply_fira_cuda(state: Dict[str, Any], total_sum_sq: Tensor, alpha: Tensor, fira_margin: float) -> Tuple[Tensor, Tensor]:
    current_norm = total_sum_sq.sqrt().view([]) 
    fira_threshold = 1.0 + fira_margin
    
    prev_norm = state.get("fira_prev_norm", None)
    if prev_norm is not None:
        if not isinstance(prev_norm, Tensor):
            prev_norm = torch.tensor(prev_norm, device=total_sum_sq.device, dtype=torch.float32)
        
        ratio = current_norm / (prev_norm + 1e-8)
        limiter = torch.clamp_min(ratio, fira_threshold) / fira_threshold
        final_scale = 1.0 / limiter
    else:
        final_scale = torch.tensor(1.0, device=total_sum_sq.device, dtype=torch.float32)
        
    state["fira_prev_norm"] = current_norm * final_scale
    
    alpha_scaled = alpha * final_scale
    total_sum_sq.mul_(final_scale.square()) 
    
    return alpha_scaled, total_sum_sq

def _apply_fira_pytorch(state: Dict[str, Any], update: Tensor, fira_margin: float, numel: int, d: float) -> Tuple[Tensor, Tensor]:
    current_norm = update.norm(2)
    fira_threshold = 1.0 + fira_margin
    
    prev_norm = state.get("fira_prev_norm", None)
    if prev_norm is not None:
        if not isinstance(prev_norm, Tensor):
            prev_norm = torch.tensor(prev_norm, device=update.device, dtype=torch.float32)
        ratio = current_norm / (prev_norm + 1e-8)
        limiter = torch.clamp_min(ratio, fira_threshold) / fira_threshold
        final_scale = 1.0 / limiter
    else:
        final_scale = torch.tensor(1.0, device=update.device, dtype=torch.float32)
        
    state["fira_prev_norm"] = current_norm * final_scale
    
    update_scaled = update * final_scale
    norm_final = current_norm * final_scale
    denom = torch.clamp(norm_final / (math.sqrt(numel) * d), min=1.0)
    
    return update_scaled, denom

def _update_param_8bit(
    param: Tensor, grad: Tensor, 
    state: Dict[str, Any],
    d: float, lr: Union[float, Tensor], 
    beta2: Optional[float],
    beta2_decay: float, 
    weight_decay: float,
    eps1: Optional[float], 
    eps2: float, 
    maximize: bool, 
    relative_step: bool,
    scale_parameter: bool, 
    scale_weight_decay: bool,
    block_size: int,
    use_cuda_kernel: bool,
    enable_fira_for_adafactor: bool = False,
    fira_margin: float = 0.01,
):
    if eps1 is None:
        eps1 = torch.finfo(param.dtype).eps
        
    eps_sq = max(eps1 * eps1, torch.finfo(torch.float32).tiny)
    log_eps_sq = math.log2(eps_sq)

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

    if beta2 is not None:
        beta_val = 1.0 - beta2
    else:
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
        wd_multiplier = alpha if scale_weight_decay else rho_t
        param_work.mul_(1.0 - (wd_multiplier * weight_decay))

    is_1d = grad_fp32.dim() < 2 

    if is_1d:
        grad_sq = grad_fp32.square() 
        
        if quantize:
            if _load_cuda_module(use_cuda_kernel):
                grad_sq_padded = _pad_to_block_size(grad_sq, curr_block_size)
                _CUDA_MODULE.fused_log_quantize_lerp(state["variance_q"], state["variance_scale"], grad_sq_padded, beta_val, curr_block_size)
                
                numel = param_work.numel()
                grad_fp32_flat = grad_fp32.view(-1)
                variance_q_flat = state["variance_q"].view(-1)
                variance_scale_flat = state["variance_scale"].view(-1)
                
                total_sum_sq = torch.zeros(1, device=param_work.device, dtype=torch.float32)
                
                _CUDA_MODULE.compute_update_norm_1d(
                    variance_q_flat, variance_scale_flat,
                    grad_fp32_flat, total_sum_sq, log_eps_sq, numel, curr_block_size
                )
                
                if enable_fira_for_adafactor:
                    alpha, total_sum_sq = _apply_fira_cuda(state, total_sum_sq, alpha, fira_margin)
                
                _CUDA_MODULE.apply_update_1d(
                    param_work.view(-1), grad_fp32_flat,
                    variance_q_flat, variance_scale_flat,
                    total_sum_sq, alpha, d, log_eps_sq, numel, curr_block_size
                )
            else:
                variance = _log_dequantize_nonneg(state["variance_q"], state["variance_scale"], state["variance_shape"], state["variance_pad"])
                variance.lerp_(grad_sq, beta_val)
                q, s, sh, pad = _log_quantize_nonneg(variance, curr_block_size)
                state["variance_q"], state["variance_scale"], state["variance_shape"], state["variance_pad"] = q, s, sh, pad
                
                update = variance.clamp_(min=eps_sq).rsqrt_().mul_(grad_fp32)
                if enable_fira_for_adafactor:
                    update, denom = _apply_fira_pytorch(state, update, fira_margin, update.numel(), d)
                else:
                    denom = torch.clamp(update.norm(2) / (math.sqrt(update.numel()) * d), min=1.0)
                param_work.add_(update, alpha=-alpha / denom)
        else:
            variance = state["variance"]
            variance.lerp_(grad_sq, beta_val)
            update = variance.clamp(min=eps_sq).rsqrt().mul_(grad_fp32)
            if enable_fira_for_adafactor:
                update, denom = _apply_fira_pytorch(state, update, fira_margin, update.numel(), d)
            else:
                denom = torch.clamp(update.norm(2) / (math.sqrt(update.numel()) * d), min=1.0)
            param_work.add_(update, alpha=-alpha / denom)

    else:
        shape = grad_fp32.shape
        R = shape[-2]
        C = shape[-1]
        numel = grad_fp32.numel()
        
        row_mean = torch.norm(grad_fp32, dim=-1, keepdim=True).square_().div_(C)
        col_mean = torch.norm(grad_fp32, dim=-2, keepdim=True).square_().div_(R)

        if quantize:
            if _load_cuda_module(use_cuda_kernel):
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
                    grad_flat, total_sum_sq, row_mean_val_flat, log_eps_sq, R, C, numel, curr_block_size
                )
                
                if enable_fira_for_adafactor:
                    alpha, total_sum_sq = _apply_fira_cuda(state, total_sum_sq, alpha, fira_margin)
                
                param_flat = param_work.reshape(-1)
                _CUDA_MODULE.apply_update_2d(
                    param_flat, grad_flat,
                    row_var_q_flat, state["row_var_scale"],
                    col_var_q_flat, state["col_var_scale"],
                    total_sum_sq, alpha, row_mean_val_flat, d, log_eps_sq, R, C, numel, curr_block_size
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
                
                update = var_estimate.clamp_(min=eps_sq).rsqrt_().mul_(grad_fp32)
                if enable_fira_for_adafactor:
                    update, denom = _apply_fira_pytorch(state, update, fira_margin, update.numel(), d)
                else:
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
            
            update = var_estimate.clamp_(min=eps_sq).rsqrt_().mul_(grad_fp32)
            if enable_fira_for_adafactor:
                update, denom = _apply_fira_pytorch(state, update, fira_margin, update.numel(), d)
            else:
                denom = torch.clamp(update.norm(2) / (math.sqrt(update.numel()) * d), min=1.0)
            param_work.add_(update, alpha=-alpha / denom)

    if needs_copy_back:
        param.copy_(param_work)


def _get_apollo_proj_matrix(state: Dict[str, Any], shape: torch.Size, step: int,
                            dtype: torch.dtype, device: torch.device) -> Tuple[Tensor, bool]:
    rank = int(state["apollo_rank"])
    update_proj_gap = int(state["apollo_update_proj_gap"])
    last_proj_step = int(state.get("last_proj_step", -update_proj_gap))
    
    needs_refresh = (step - last_proj_step) >= update_proj_gap
    
    if needs_refresh:
        state["current_proj_seed"] = int(state["apollo_seed"])
        state["apollo_seed"] = next_seed(state["current_proj_seed"])
        state["last_proj_step"] = step
        
    R, C = shape[-2], shape[-1]
    side = "right" if R >= C else "left"
    m_shape = (rank, C) if side == "right" else (R, rank)

    proj = stable_randn(m_shape, seed=state["current_proj_seed"], device=device, dtype=dtype) / math.sqrt(rank)
    
    return proj, needs_refresh 


def _update_param_apollo_nomomentum(
    param: Tensor, grad: Tensor, state: Dict[str, Any],
    d: float, lr: Union[float, Tensor],
    beta2: Optional[float], beta2_decay: float, weight_decay: float,
    eps1: Optional[float], eps2: float, maximize: bool, relative_step: bool,
    scale_parameter: bool, scale_weight_decay: bool,
    block_size: int, use_cuda_kernel: bool,
    apollo_scale_type: str, apollo_eps: float,
    apollo_factorize: bool,
    fira_margin: float = 0.01,
):
    grad_work = grad.neg().float() if maximize else grad.float()

    if apollo_factorize:
        state.pop("v_low_q", None)
        state.pop("v_low_scale", None)
        state.pop("v_low_shape", None)
        state.pop("v_low_pad", None)
        state.pop("v_low", None)
    else:
        state.pop("row_var_low", None)
        state.pop("col_var_low", None)

    if not param.is_contiguous():
        param_work = param.contiguous()
        needs_copy_back = True
    else:
        param_work = param
        needs_copy_back = False

    original_shape = param_work.shape
    if param_work.dim() > 2:
        param_work = param_work.reshape(param_work.shape[0], -1)
        grad_work = grad_work.reshape(grad_work.shape[0], -1)

    step = state["step"] + 1
    state["step"] = step
    beta_val = 1.0 - beta2 if beta2 is not None else math.pow(step, beta2_decay)

    if isinstance(lr, float):
        rho_val = min(lr, 1.0 / math.sqrt(step)) if relative_step else lr
        rho_t = torch.tensor(rho_val, device=param_work.device, dtype=torch.float32)
    else:
        if relative_step:
            step_t = torch.tensor(step, device=param_work.device, dtype=torch.float32)
            rho_t = torch.minimum(step_t.rsqrt(), lr)
        else:
            rho_t = lr

    if scale_parameter:
        param_rms_t = torch.linalg.vector_norm(param_work, ord=2, dtype=torch.float32) / math.sqrt(param_work.numel())
        alpha_t = torch.clamp(param_rms_t, min=eps2) * rho_t
    else:
        alpha_t = rho_t

    shape = grad_work.shape
    proj_matrix, is_proj_refreshed = _get_apollo_proj_matrix(
        state, shape, step, dtype=torch.float32, device=grad_work.device
    )

    if is_proj_refreshed:
        state["v_low_q"] = None
        state["v_low_scale"] = None
        state["v_low_shape"] = None
        state["v_low_pad"] = None
        state["v_low"] = None
        state.pop("row_var_low", None)
        state.pop("col_var_low", None)

    R, C = shape[-2], shape[-1]
    side = "right" if R >= C else "left"
    
    proj_matrix_T = proj_matrix.T.to(grad_work.dtype)
    if side == "right":
        grad_low = torch.matmul(grad_work, proj_matrix_T).float()
    else:
        grad_low = torch.matmul(proj_matrix_T, grad_work).float()

    if apollo_factorize:
        row_mean_low = grad_low.square().mean(dim=-1, keepdim=True)
        col_mean_low = grad_low.square().mean(dim=-2, keepdim=True)
        
        if "row_var_low" not in state:
            state["row_var_low"] = row_mean_low.clone().clamp(min=_FP32_TINY)
            state["col_var_low"] = col_mean_low.clone().clamp(min=_FP32_TINY)
        else:
            state["row_var_low"].mul_(1.0 - beta_val).add_(row_mean_low, alpha=beta_val)
            state["col_var_low"].mul_(1.0 - beta_val).add_(col_mean_low, alpha=beta_val)
            
        row_var = state["row_var_low"]
        col_var = state["col_var_low"]
        
        eps1_val = eps1 if eps1 is not None else apollo_eps
        v_low_est = row_var * col_var / row_var.mean(dim=-2, keepdim=True).clamp(min=eps1_val)
        update_low = grad_low / torch.sqrt(v_low_est + apollo_eps)
        
    else:
        is_first_step = (state.get("v_low_q") is None and state.get("v_low") is None)
        
        if is_first_step:
            v_init = grad_low.flatten().square().clamp(min=_FP32_TINY)
            if state.get("is_quantized", True):
                q, s, sh, pad = _log_quantize_nonneg(v_init, block_size)
                state["v_low_q"], state["v_low_scale"], state["v_low_shape"], state["v_low_pad"] = q, s, sh, pad
                v_low_deq = v_init
            else:
                state["v_low"] = v_init
                v_low_deq = v_init
        else:
            grad_low_sq_flat = grad_low.flatten().square()
            quantize = state.get("is_quantized", True)
            if quantize:
                if _load_cuda_module(use_cuda_kernel):
                    sq_padded = _pad_to_block_size(grad_low_sq_flat, block_size)
                    _CUDA_MODULE.fused_log_quantize_lerp(
                        state["v_low_q"], state["v_low_scale"], sq_padded, beta_val, block_size
                    )
                else:
                    v_deq = _log_dequantize_nonneg(state["v_low_q"], state["v_low_scale"], state["v_low_shape"], state["v_low_pad"])
                    v_deq.lerp_(grad_low_sq_flat, beta_val)
                    q, s, sh, pad = _log_quantize_nonneg(v_deq, block_size)
                    state["v_low_q"], state["v_low_scale"], state["v_low_shape"], state["v_low_pad"] = q, s, sh, pad
            else:
                state["v_low"].lerp_(grad_low_sq_flat, beta_val)

            v_low_deq = _log_dequantize_nonneg(state["v_low_q"], state["v_low_scale"], state["v_low_shape"], state["v_low_pad"]) if quantize else state["v_low"]
            
        update_low = grad_low.flatten() / torch.sqrt(v_low_deq + apollo_eps)
        update_low = update_low.view_as(grad_low)

    if apollo_scale_type == "channel":
        norm_dim = -1 if side == "right" else -2
        scaling_factor = update_low.norm(dim=norm_dim, keepdim=True) / (grad_low.norm(dim=norm_dim, keepdim=True) + 1e-8)
    else:
        scaling_factor = update_low.norm() / (grad_low.norm() + 1e-8)

    if apollo_scale_type == "channel":
        if side == "right":
            norm_G = torch.linalg.vector_norm(grad_work, ord=2, dim=-1, keepdim=True, dtype=torch.float32)
            norm_G_sq = norm_G.square()
        else:
            norm_G = torch.linalg.vector_norm(grad_work, ord=2, dim=-2, keepdim=True, dtype=torch.float32)
            norm_G_sq = norm_G.square()
        current_norm_t = torch.sqrt((norm_G_sq * scaling_factor.square()).sum())
    else:
        current_norm_t = torch.linalg.vector_norm(grad_work, ord=2, dtype=torch.float32) * scaling_factor

    fira_threshold = 1.0 + fira_margin
    if "scaled_grad_norm_prev" in state:
        prev_norm_t = state["scaled_grad_norm_prev"]
        if not isinstance(prev_norm_t, Tensor):
            prev_norm_t = torch.tensor(prev_norm_t, device=param_work.device, dtype=torch.float32)
            
        ratio = current_norm_t / (prev_norm_t + 1e-8)
        limiter = torch.clamp_min(ratio, fira_threshold) / fira_threshold
        final_scale = scaling_factor / limiter
        state["scaled_grad_norm_prev"] = current_norm_t / limiter
    else:
        final_scale = scaling_factor
        state["scaled_grad_norm_prev"] = current_norm_t

    numel = grad_work.numel()
    if apollo_scale_type == "channel":
        norm_final_t = torch.sqrt((norm_G_sq * final_scale.square()).sum())
    else:
        norm_final_t = torch.linalg.vector_norm(grad_work, ord=2, dtype=torch.float32) * final_scale
        
    denom_t = torch.clamp_min(norm_final_t / (math.sqrt(numel) * d), 1.0)

    if weight_decay != 0:
        wd_multiplier_t = alpha_t if scale_weight_decay else rho_t
        decay_factor = 1.0 - (wd_multiplier_t * weight_decay)
        param_work.mul_(decay_factor)

    update_scale_t = -alpha_t / denom_t
        
    if apollo_scale_type == "channel":
        final_scale_cast = final_scale.to(grad_work.dtype)
        param_work.addcmul_(grad_work, final_scale_cast, value=update_scale_t)
    else:
        param_work.add_(grad_work, alpha=update_scale_t * final_scale)

    if apollo_scale_type == "channel":
        del norm_G_sq, norm_G
    del update_low, grad_low, scaling_factor, final_scale, proj_matrix_T, proj_matrix
    
    if needs_copy_back:
        param.copy_(param_work.view(original_shape))