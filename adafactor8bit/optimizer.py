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
_INV_255 = 1.0 / 255.0
_UNIFORM_SCALE = 0.125


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
    """Quantize a non-negative FP32 tensor to UINT8 (0-255) in log-space with block-wise scaling."""
    shape = tensor.shape
    flat = tensor.flatten()
    pad = (block_size - flat.numel() % block_size) % block_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))

    blocks = flat.view(-1, block_size)
    log_blocks = torch.log2(blocks.clamp(min=_FP32_TINY))
    max_log = log_blocks.amax(dim=1, keepdim=True)
    
    scale = (max_log - _FP32_MIN_LOG).clamp(min=1e-12)
    q = torch.round((log_blocks - _FP32_MIN_LOG) / scale * 255.0).clamp(0, 255).to(torch.uint8)
    return q, scale.squeeze(-1), shape, pad

def _log_dequantize_nonneg(q: Tensor, scale: Tensor, shape: torch.Size, pad: int) -> Tensor:
    """Dequantize from log-space back to linear-space FP32."""
    if q.dim() == 1:
        block_size = q.numel() // scale.numel()
        q = q.view(-1, block_size)
    log_blocks = q.float() * scale.unsqueeze(-1) * _INV_255 + _FP32_MIN_LOG
    blocks = torch.pow(2.0, log_blocks)
    flat = blocks.flatten()
    if pad:
        flat = flat[:-pad]
    return flat.view(shape)

def _quantize_4bit_pytorch(m: Tensor, block_size: int) -> Tuple[Tensor, Tensor]:
    """4-bit Uniform symmetric quantization with physical packing into uint8."""
    flat = m.flatten()
    pad = (block_size - flat.numel() % block_size) % block_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.view(-1, block_size)
    abs_max = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = abs_max
    q = (torch.round(blocks / scale * 8.0).clamp(-8, 7) + 8).to(torch.uint8)
    q_even = q[:, 0::2]
    q_odd = q[:, 1::2]
    packed = (q_even << 4) | q_odd
    return packed.view(-1), scale.squeeze(-1)

def _dequantize_4bit(m_q: Tensor, m_scale: Tensor, numel: int, shape: torch.Size, block_size: int, device: torch.device) -> Tensor:
    """Dequantize 4-bit physically packed tensor to FP32."""
    if _CUDA_MODULE is not None and _CUDA_AVAILABLE:
        output = torch.empty(numel, device=device, dtype=torch.float32)
        _CUDA_MODULE.dequantize_4bit(output, m_q, m_scale, numel, block_size)
        return output.view(shape)
    else:
        high = ((m_q >> 4) & 0x0F).to(torch.float32) - 8.0
        low = (m_q & 0x0F).to(torch.float32) - 8.0
        m_flat = torch.stack((high, low), dim=-1).view(-1)
        m_blocks = m_flat.view(-1, block_size)
        result = (m_blocks * (m_scale.unsqueeze(-1) * _UNIFORM_SCALE)).view(-1)[:numel]
        return result.view(shape)

# ==========================================
# 4. Optimizer Core
# ==========================================
class Adafactor8Bit(Optimizer):
    """
    8-bit Adafactor optimizer with fused CUDA kernels for memory-efficient large-scale training.
    
    Args:
        params (Iterable): Iterable of parameters to optimize or dictionaries defining parameter groups.
        
        --- Core Optimization ---
        lr (float, optional): External learning rate. Defaults to 1e-2.
        beta1 (float, optional): Momentum coefficient for first moment (4-bit packed). 
            If None, disables first moment (pure Adafactor/RMSProp). Defaults to None.
        beta2 (float, optional): Fixed second-moment decay rate (e.g., 0.999 like Adam). 
            Locks the EMA window size, preventing "blunting" in long-term continual learning. 
            Mutually exclusive with `beta2_decay`. Defaults to None.
        beta2_decay (float): Dynamic decay rate coefficient. 
            The EMA weight is computed as `step ** beta2_decay`. Ignored if `beta2` is specified. 
            Defaults to -0.8.
        beta3 (float, optional): Confidence-guided decay coefficient for CAME 
            (Confidence-guided Adaptive Memory Efficient Optimization). 
            Computes the instability of the update direction and scales the update accordingly. 
            Strictly requires `beta1` and `factored=True`. 
            Defaults to None (disabled).
        eps (Tuple[Optional[float], float]): Regularization constants (eps1, eps2).
            - `eps1`: Added to the squared gradient. If `None`, defaults to the machine epsilon 
              of the parameter's dtype (e.g., ~1.19e-7 for FP32), preventing underflow.
            - `eps2`: Lower threshold for parameter RMS scaling. Defaults to (None, 1e-3).
        eps_came (float): Lower bound for the residual term in CAME's confidence estimation.
            Under 4-bit UF4 quantization, momentum discretization can yield artificially 
            tiny residuals, potentially causing CAME to excessively amplify updates. 
            Setting `eps_came` (default 1e-8) anchors the confidence state and mitigates 
            anomalous scaling from these quantization artifacts. Independent of the 
            variance `eps`. Defaults to 1e-8.
        weight_decay (float): Weight decay (L2 penalty). Defaults to 0.0.
        d (float): Clipping threshold for the final gradient update RMS. 
            Setting to an extremely large value (e.g., ``1e9``) effectively disables the global 
            clipping constraint, useful for decoupling updates in sparse layers like Embeddings. 
            Defaults to 1.0.
        maximize (bool): Maximize the params based on the objective. Defaults to False.
            
        --- Factorization & Scaling ---
        relative_step (bool): If `True`, uses time-dependent learning rate. Defaults to True.
        scale_parameter (bool): If `True`, scales learning rate by parameter RMS. 
            Setting to False decouples the step size from parameter magnitude, which can be useful 
            for sparse layers like Embeddings to ensure sufficient update strength. Defaults to True.
        factored (bool): Whether to use row/col factorization for >=2D tensors. 
            Setting to False uses element-wise variance (like RMSProp, but still applies Adafactor's 
            global RMS clipping). This can be useful for preserving spatial structure in >2D tensors 
            such as CNN convolutions, or enabling per-element updates in Embeddings. Defaults to True.
            
        --- Quantization Control ---
        quantize (bool): Enable 8-bit log-space quantization for optimizer states. Defaults to True.
        block_size (int): Block size for variance quantization. Must be a multiple of 1024. Defaults to 2048.
        m_block_size (int): Block size for 4-bit momentum quantization. 
            Must be a multiple of 4 and >= 32. Defaults to 128.
        min_8bit_size (int): Minimum number of elements to apply 8-bit quantization. Defaults to 4096.
        use_cuda_kernel (bool): Whether to use custom CUDA kernels. Defaults to True.
            
        --- APOLLO Low-Rank Projection ---
        apollo_rank (int): Rank for APOLLO (An Optimizer for Memory-Efficient Large-Scale Training) 
            style random projection to low-rank space. If > 0, enables APOLLO. 
            Defaults to 0 (disabled).
        apollo_update_proj_gap (int): Steps between random projection matrix refreshes. 
            Defaults to 200.
        apollo_scale_type (str): Strategy to map low-rank updates back to full-rank: 
            'channel' (row-wise norm matching) or 'tensor' (global norm matching). 
            Defaults to 'channel'.
        apollo_scale (float): Heuristic multiplier to compensate for norm attenuation 
            caused by low-rank projection. Defaults to 1.0.
        apollo_scale_front (bool): If ``True``, applies ``apollo_scale`` before the Fira 
            Norm-Growth Limiter. If ``False``, applies it after. Defaults to False.
        apollo_eps (float): Epsilon for low-rank variance normalization to prevent division by zero. 
            Defaults to 1e-8.
        apollo_factorize (bool): If True, applies Adafactor-style row/col factorization 
            within the low-rank space (FP32, ~16KB state) instead of full matrix variance 
            (8-bit, ~100KB+ state) to drastically reduce optimizer state memory. Defaults to False.
            
        --- Stabilizers & Regularization ---
        scale_weight_decay (bool): If `True` (default), weight decay is coupled with the 
            parameter's RMS scale. If `False`, decoupled (AdamW-style).
        enable_fira_for_adafactor (bool): If `True`, enables Fira Limiter to prevent gradient 
            explosion by smoothing update norms. Defaults to False.
        fira_margin (float): The tolerance margin for Fira Limiter (e.g., 0.01 for 1%). 
            Shared with Apollo path. Defaults to 0.01.
    """
    
    def __init__(
        self,
        params: Iterable[Union[Tensor, Dict[str, Any]]],
        # --- Core Optimization ---
        lr: float = 1e-2,
        beta1: Optional[float] = None,
        beta2: Optional[float] = None,
        beta2_decay: float = -0.8,
        beta3: Optional[float] = None,
        eps_came: float = 1e-8,
        eps: Tuple[Optional[float], float] = (None, 1e-3),
        weight_decay: float = 0.0,
        d: float = 1.0,
        maximize: bool = False,
        # --- Factorization & Scaling ---
        relative_step: bool = True,
        scale_parameter: bool = True,
        factored: bool = True,
        # --- Quantization Control ---
        quantize: bool = True,
        block_size: int = 2048,
        m_block_size: int = 128,
        min_8bit_size: int = 4096,
        use_cuda_kernel: bool = True,
        # --- APOLLO Low-Rank Projection ---
        apollo_rank: int = 0,
        apollo_update_proj_gap: int = 200,
        apollo_scale_type: str = 'channel',
        apollo_scale: float = 1.0,
        apollo_scale_front: bool = False,
        apollo_eps: float = 1e-8,
        apollo_factorize: bool = False,
        # --- Stabilizers & Regularization ---
        scale_weight_decay: bool = True,
        enable_fira_for_adafactor: bool = False,
        fira_margin: float = 0.01,
    ):
        
        if lr < 0.0: raise ValueError(f"Invalid lr: {lr}, must be >= 0.0")
        if beta1 is not None and (beta1 < 0.0 or beta1 >= 1.0):
            raise ValueError(f"Invalid beta1: {beta1}, must be in [0.0, 1.0)")
        if beta2_decay > 0.0: raise ValueError(f"Invalid beta2_decay: {beta2_decay}, must be <= 0.0")
        eps1, eps2 = eps
        if eps1 is not None and eps1 < 0.0: raise ValueError(f"Invalid eps1: {eps1}, must be >= 0.0")
        if eps2 < 0.0: raise ValueError(f"Invalid eps2: {eps2}, must be >= 0.0")
        if d < 1.0: raise ValueError(f"Invalid d: {d}, must be >= 1.0")
        if weight_decay < 0.0: raise ValueError(f"Invalid weight_decay: {weight_decay}, must be >= 0.0")

        if beta2 is not None and (beta2 < 0.0 or beta2 >= 1.0):
            raise ValueError(f"Invalid beta2: {beta2}, must be in [0.0, 1.0)")
        
        if quantize and block_size % 1024 != 0:
            raise ValueError(f"block_size must be a multiple of 1024, but got {block_size}.")
            
        if m_block_size < 32 or m_block_size % 4 != 0:
            raise ValueError(f"m_block_size must be a multiple of 4 and >= 32, but got {m_block_size}.")

        if apollo_rank > 0 and apollo_scale_type not in ('channel', 'tensor'):
            raise ValueError(f"apollo_scale_type must be 'channel' or 'tensor', got {apollo_scale_type}.")
            
        if apollo_scale < 0.0:
            raise ValueError(f"Invalid apollo_scale: {apollo_scale}, must be >= 0.0")
            
        if fira_margin < 0.0: raise ValueError(f"Invalid fira_margin: {fira_margin}, must be >= 0.0")

        if beta3 is not None:
            if beta3 < 0.0 or beta3 >= 1.0:
                raise ValueError(f"Invalid beta3: {beta3}, must be in [0.0, 1.0)")
            if beta1 is None:
                raise ValueError("CAME (beta3) strictly requires momentum (beta1) to compute update instability.")
            if apollo_rank == 0 and not factored:
                raise ValueError("CAME (beta3) requires factored=True (2D row/col factorization). It is not supported for 1D full-rank paths.")

        defaults = dict(
            # Core Optimization
            lr=lr, beta1=beta1, beta2=beta2, beta2_decay=beta2_decay, beta3=beta3, eps_came=eps_came,
            eps=eps, weight_decay=weight_decay, d=d, maximize=maximize,
            # Factorization & Scaling
            relative_step=relative_step, scale_parameter=scale_parameter, factored=factored,
            # Quantization Control
            quantize=quantize, block_size=block_size, m_block_size=m_block_size, 
            min_8bit_size=min_8bit_size, use_cuda_kernel=use_cuda_kernel,
            # APOLLO Low-Rank Projection
            apollo_rank=apollo_rank, apollo_update_proj_gap=apollo_update_proj_gap,
            apollo_scale_type=apollo_scale_type, apollo_scale=apollo_scale, 
            apollo_scale_front=apollo_scale_front, apollo_eps=apollo_eps, 
            apollo_factorize=apollo_factorize,
            # Stabilizers & Regularization
            scale_weight_decay=scale_weight_decay, 
            enable_fira_for_adafactor=enable_fira_for_adafactor, fira_margin=fira_margin,
        )
        super().__init__(params, defaults)

        self._apollo_seed_counter = 0

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['_apollo_seed_counter'] = self._apollo_seed_counter
        return state_dict

    def load_state_dict(self, state_dict):
        self._apollo_seed_counter = state_dict.get('_apollo_seed_counter', 0)
        super().load_state_dict(state_dict)
        self._compact_state()

    def _compact_state(self):
        """
        Compact the optimizer state tensors in GPU memory.
        Moves all CUDA tensors to CPU, clears the GPU cache to eliminate external fragmentation,
        and then reallocates them contiguously on the GPU. This prevents OOM errors caused by
        fragmented memory layouts after loading a checkpoint.
        """
        saved = []
        for p, state in self.state.items():
            for k in list(state.keys()):
                v = state[k]
                if isinstance(v, torch.Tensor) and v.is_cuda:
                    saved.append((p, k, v.cpu()))
                    state[k] = None
        
        if not saved:
            return
        
        torch.cuda.empty_cache()
        
        for p, k, cpu_tensor in saved:
            if k.endswith("_q"):
                target_dtype = torch.uint8
            elif k.endswith("_scale"):
                target_dtype = torch.float32
            else:
                target_dtype = cpu_tensor.dtype
            self.state[p][k] = cpu_tensor.to(
                device=p.device, dtype=target_dtype
            ).contiguous()

    def _init_group(self, group, params_with_grad, grads, states, state_steps):

        group_quantize = group.get("quantize", True)
        block_size = group.get("block_size", 2048)
        m_block_size = group.get("m_block_size", 128)
        min_8bit_size = group.get("min_8bit_size", 4096)
        apollo_rank = group.get("apollo_rank", 0)
        apollo_factorize = group.get("apollo_factorize", False)
        factored = group.get("factored", True)
        beta1 = group.get("beta1")
        beta3 = group.get("beta3")

        for p in group["params"]:
            if p.grad is None: continue

            force_fp32 = p.numel() < min_8bit_size
            use_quant = group_quantize and not force_fp32

            params_with_grad.append(p)
            grads.append(p.grad)
            state = self.state[p]

            use_apollo = apollo_rank > 0 and p.grad.dim() >= 2
            expected_is_factored = (p.grad.dim() >= 2) and factored

            needs_init = False
            if len(state) == 0:
                needs_init = True
            else:
                is_apollo_state = ('apollo_seed' in state)
                if (use_apollo and not is_apollo_state) or (not use_apollo and is_apollo_state):
                    logger.warning(f"Adafactor8Bit: Apollo/Adafactor mode changed for param shape {p.shape}. Re-initializing state.")
                    step_backup = state.get("step", 0)
                    if torch.is_tensor(step_backup):
                        step_backup = int(step_backup.cpu().item())
                    elif not isinstance(step_backup, int):
                        step_backup = int(step_backup)
                    state.clear()
                    state["step"] = step_backup
                    needs_init = True
                elif use_apollo and is_apollo_state:
                    if (state.get("apollo_rank") != apollo_rank or 
                        state.get("apollo_factorize", False) != apollo_factorize or
                        state.get("beta3") != beta3):
                        logger.warning(f"Adafactor8Bit: Apollo/CAME config changed for param shape {p.shape}. Re-initializing state.")
                        step_backup = state.get("step", 0)
                        if torch.is_tensor(step_backup):
                            step_backup = int(step_backup.cpu().item())
                        elif not isinstance(step_backup, int):
                            step_backup = int(step_backup)
                        state.clear()
                        state["step"] = step_backup
                        needs_init = True
                else:
                    state_is_factored = ("row_var" in state or "row_var_q" in state)
                    
                    if state_is_factored != expected_is_factored:
                        logger.warning(f"Adafactor8Bit: factored mode mismatch for param shape {p.shape}. Re-initializing state.")
                        step_backup = state.get("step", 0)
                        if torch.is_tensor(step_backup):
                            step_backup = int(step_backup.cpu().item())
                        elif not isinstance(step_backup, int):
                            step_backup = int(step_backup)
                        state.clear()
                        state["step"] = step_backup
                        needs_init = True

            if needs_init:
                if "step" not in state:
                    state["step"] = 0
                state["is_quantized"] = use_quant
                state["block_size"] = block_size

                if use_apollo:
                    seed = self._apollo_seed_counter
                    self._apollo_seed_counter += 1
                    update_proj_gap = group.get("apollo_update_proj_gap", 200)
                    
                    state["apollo_seed"] = seed
                    state["apollo_rank"] = apollo_rank
                    state["apollo_factorize"] = apollo_factorize
                    state["beta3"] = beta3
                    state["apollo_update_proj_gap"] = update_proj_gap
                    state["last_proj_step"] = -update_proj_gap
                    state["v_low_q"] = None
                    state["v_low_scale"] = None
                    state["v_low_shape"] = None
                    state["v_low_pad"] = None
                    state["v_low"] = None
                    
                    if beta3 is not None:
                        if apollo_factorize:
                            state["conf_row_low_q"] = None
                            state["conf_row_low_scale"] = None
                            state["conf_row_low_shape"] = None
                            state["conf_row_low_pad"] = None
                            state["conf_col_low_q"] = None
                            state["conf_col_low_scale"] = None
                            state["conf_col_low_shape"] = None
                            state["conf_col_low_pad"] = None
                            state["conf_row_low"] = None
                            state["conf_col_low"] = None
                        else:
                            state["res_low_q"] = None
                            state["res_low_scale"] = None
                            state["res_low_shape"] = None
                            state["res_low_pad"] = None
                            state["res_low"] = None

                else:
                    if expected_is_factored:
                        shape = p.grad.shape
                        R = shape[-2]
                        C = shape[-1]
                        batch_shape = shape[:-2]
                        
                        r_shape = list(batch_shape) + [R, 1]
                        c_shape = list(batch_shape) + [1, C]
                        
                        if use_quant:
                            r_numel = math.prod(r_shape)
                            r_pad = (block_size - r_numel % block_size) % block_size
                            state["row_var_q"] = torch.zeros(r_numel + r_pad, dtype=torch.uint8, device=p.device)
                            state["row_var_scale"] = torch.ones((r_numel + r_pad) // block_size, dtype=torch.float32, device=p.device)
                            state["row_var_shape"] = r_shape
                            state["row_var_pad"] = r_pad
                            
                            c_numel = math.prod(c_shape)
                            c_pad = (block_size - c_numel % block_size) % block_size
                            state["col_var_q"] = torch.zeros(c_numel + c_pad, dtype=torch.uint8, device=p.device)
                            state["col_var_scale"] = torch.ones((c_numel + c_pad) // block_size, dtype=torch.float32, device=p.device)
                            state["col_var_shape"] = c_shape
                            state["col_var_pad"] = c_pad
                            
                            if beta1 is not None:
                                m_padded_numel = ((p.numel() + m_block_size - 1) // m_block_size) * m_block_size
                                state["m_q"] = torch.full((m_padded_numel // 2,), 0x88, dtype=torch.uint8, device=p.device)
                                state["m_scale"] = torch.ones(m_padded_numel // m_block_size, dtype=torch.float32, device=p.device)
                                state["m_block_size"] = m_block_size
                                
                            if beta3 is not None:
                                state["conf_row_q"] = torch.zeros_like(state["row_var_q"])
                                state["conf_row_scale"] = torch.ones_like(state["row_var_scale"])
                                state["conf_row_shape"] = state["row_var_shape"]
                                state["conf_row_pad"] = state["row_var_pad"]
                                
                                state["conf_col_q"] = torch.zeros_like(state["col_var_q"])
                                state["conf_col_scale"] = torch.ones_like(state["col_var_scale"])
                                state["conf_col_shape"] = state["col_var_shape"]
                                state["conf_col_pad"] = state["col_var_pad"]

                        else:
                            state["row_var"] = torch.zeros(r_shape, dtype=torch.float32, device=p.device)
                            state["col_var"] = torch.zeros(c_shape, dtype=torch.float32, device=p.device)
                            if beta1 is not None:
                                state["m"] = torch.zeros_like(p.grad, dtype=torch.float32, device=p.device, memory_format=torch.preserve_format)
                            if beta3 is not None:
                                state["conf_row"] = torch.zeros(r_shape, device=p.device)
                                state["conf_col"] = torch.zeros(c_shape, device=p.device)
                    else:
                        if use_quant:
                            v_numel = p.grad.numel()
                            v_pad = (block_size - v_numel % block_size) % block_size
                            state["variance_q"] = torch.zeros(v_numel + v_pad, dtype=torch.uint8, device=p.device)
                            state["variance_scale"] = torch.ones((v_numel + v_pad) // block_size, dtype=torch.float32, device=p.device)
                            state["variance_shape"] = p.grad.shape
                            state["variance_pad"] = v_pad
                            
                            if beta1 is not None:
                                m_padded_numel = ((p.numel() + m_block_size - 1) // m_block_size) * m_block_size
                                state["m_q"] = torch.full((m_padded_numel // 2,), 0x88, dtype=torch.uint8, device=p.device)
                                state["m_scale"] = torch.ones(m_padded_numel // m_block_size, dtype=torch.float32, device=p.device)
                                state["m_block_size"] = m_block_size
                        else:
                            state["variance"] = torch.zeros_like(p.grad, dtype=torch.float32, memory_format=torch.preserve_format)
                            if beta1 is not None:
                                state["m"] = torch.zeros_like(p.grad, dtype=torch.float32, device=p.device, memory_format=torch.preserve_format)
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
                        if k.endswith("_scale") and state[k].dtype != torch.float32:
                            state[k] = state[k].to(torch.float32)

                curr_block_size = state.get("block_size", block_size)
                state_is_factored = ("row_var" in state or "row_var_q" in state)

                if use_quant and not state.get("is_quantized", False):
                    if isinstance(state.get("v_low"), Tensor) and state.get("v_low_q") is None:
                        state["v_low"].clamp_(min=_FP32_TINY)
                        q, s, sh, pad = _log_quantize_nonneg(state["v_low"], curr_block_size)
                        state["v_low_q"], state["v_low_scale"], state["v_low_shape"], state["v_low_pad"] = q, s, sh, pad
                        state["v_low"] = None
                    
                    if "m_low" in state and state.get("m_low_q") is None:
                        local_m_block_size = state.get("m_block_size", m_block_size)
                        state["m_low_q"], state["m_low_scale"] = _quantize_4bit_pytorch(state["m_low"], local_m_block_size)
                        state["m_block_size"] = local_m_block_size
                        state.pop("m_low", None)

                    if state_is_factored:
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
                        if beta3 is not None:
                            if "conf_row" in state and "conf_row_q" not in state:
                                state["conf_row"].clamp_(min=_FP32_TINY)
                                q, s, sh, pad = _log_quantize_nonneg(state["conf_row"], curr_block_size)
                                state["conf_row_q"], state["conf_row_scale"], state["conf_row_shape"], state["conf_row_pad"] = q, s, sh, pad
                                del state["conf_row"]
                            if "conf_col" in state and "conf_col_q" not in state:
                                state["conf_col"].clamp_(min=_FP32_TINY)
                                q, s, sh, pad = _log_quantize_nonneg(state["conf_col"], curr_block_size)
                                state["conf_col_q"], state["conf_col_scale"], state["conf_col_shape"], state["conf_col_pad"] = q, s, sh, pad
                                del state["conf_col"]
                    else:
                        if "variance" in state and "variance_q" not in state:
                            state["variance"].clamp_(min=_FP32_TINY)
                            q, s, sh, pad = _log_quantize_nonneg(state["variance"], curr_block_size)
                            state["variance_q"], state["variance_scale"], state["variance_shape"], state["variance_pad"] = q, s, sh, pad
                            del state["variance"]
                            
                    if beta1 is not None and "m_q" not in state:
                        m_curr_block_size = state.get("m_block_size", m_block_size)
                        m_padded_numel = ((p.numel() + m_curr_block_size - 1) // m_curr_block_size) * m_curr_block_size
                        if "m" in state:
                            state["m_q"], state["m_scale"] = _quantize_4bit_pytorch(state["m"], m_curr_block_size)
                            state.pop("m")
                        else:
                            state["m_q"] = torch.full((m_padded_numel // 2,), 0x88, dtype=torch.uint8, device=p.device)
                            state["m_scale"] = torch.ones(m_padded_numel // m_curr_block_size, dtype=torch.float32, device=p.device)
                        state["m_block_size"] = m_curr_block_size
                        
                    state["is_quantized"] = True
                
                elif not use_quant and state.get("is_quantized", False):
                    if isinstance(state.get("v_low_q"), Tensor):
                        state["v_low"] = _log_dequantize_nonneg(
                            state.pop("v_low_q"), state.pop("v_low_scale"),
                            state.pop("v_low_shape"), state.pop("v_low_pad")
                        )
                    
                    if "m_low_q" in state:
                        logger.warning("Adafactor8Bit: Apollo m_low_q discarded due to quantize flag change. Momentum state will be reset.")
                        state.pop("m_low_q", None)
                        state.pop("m_low_scale", None)
                        state.pop("m_block_size", None)

                    if state_is_factored:
                        if "row_var_q" in state:
                            state["row_var"] = _log_dequantize_nonneg(state.pop("row_var_q"), state.pop("row_var_scale"), state.pop("row_var_shape"), state.pop("row_var_pad"))
                        if "col_var_q" in state:
                            state["col_var"] = _log_dequantize_nonneg(state.pop("col_var_q"), state.pop("col_var_scale"), state.pop("col_var_shape"), state.pop("col_var_pad"))
                        if beta3 is not None:
                            if "conf_row_q" in state:
                                state["conf_row"] = _log_dequantize_nonneg(state.pop("conf_row_q"), state.pop("conf_row_scale"), state.pop("conf_row_shape"), state.pop("conf_row_pad"))
                            if "conf_col_q" in state:
                                state["conf_col"] = _log_dequantize_nonneg(state.pop("conf_col_q"), state.pop("conf_col_scale"), state.pop("conf_col_shape"), state.pop("conf_col_pad"))
                    else:
                        if "variance_q" in state:
                            state["variance"] = _log_dequantize_nonneg(state.pop("variance_q"), state.pop("variance_scale"), state.pop("variance_shape"), state.pop("variance_pad"))
                    
                    # Dequantize 4-bit M_t back to FP32, preserving historical momentum
                    if "m_q" in state:
                        m_curr_block_size = state.get("m_block_size", m_block_size)
                        state["m"] = _dequantize_4bit(
                            state["m_q"], state["m_scale"], 
                            p.numel(), p.shape, m_curr_block_size, p.device
                        )

                        state.pop("m_q", None)
                        state.pop("m_scale", None)
                        state.pop("m_block_size", None)
                        
                    state["is_quantized"] = False


                if beta1 is not None and "m_q" not in state and use_quant:
                    m_curr_block_size = state.get("m_block_size", m_block_size)
                    m_padded_numel = ((p.numel() + m_curr_block_size - 1) // m_curr_block_size) * m_curr_block_size
                    if "m" in state:
                        state["m_q"], state["m_scale"] = _quantize_4bit_pytorch(state["m"], m_curr_block_size)
                        state.pop("m")
                    else:
                        state["m_q"] = torch.full((m_padded_numel // 2,), 0x88, dtype=torch.uint8, device=p.device)
                        state["m_scale"] = torch.ones(m_padded_numel // m_curr_block_size, dtype=torch.float32, device=p.device)
                    state["m_block_size"] = m_curr_block_size

                if "is_quantized" not in state:
                    if use_apollo:
                        state["is_quantized"] = use_quant
                    else:
                        state["is_quantized"] = ("row_var_q" in state or "col_var_q" in state or "variance_q" in state)
                if "block_size" not in state:
                    state["block_size"] = block_size
                if beta1 is None:
                    state.pop("m_q", None)
                    state.pop("m_scale", None)
                    state.pop("m_block_size", None)
                    state.pop("m", None)

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
                    _update_param_apollo(
                        params_with_grad[i], grads[i], states[i],
                        d=group["d"], lr=group["lr"], 
                        beta1=group.get("beta1"),
                        beta2=group["beta2"],
                        beta2_decay=group["beta2_decay"],
                        weight_decay=group["weight_decay"], eps1=eps1, eps2=eps2,
                        maximize=group["maximize"], relative_step=group["relative_step"],
                        scale_parameter=group["scale_parameter"],
                        scale_weight_decay=group.get("scale_weight_decay", True),
                        block_size=group.get("block_size", 2048),
                        m_block_size=group.get("m_block_size", 128),
                        use_cuda_kernel=group.get("use_cuda_kernel", True),
                        apollo_scale_type=group.get("apollo_scale_type", "channel"),
                        apollo_scale=group.get("apollo_scale", 1.0),
                        apollo_scale_front=group.get("apollo_scale_front", False),
                        apollo_eps=group.get("apollo_eps", 1e-8),
                        apollo_factorize=group.get("apollo_factorize", False),
                        fira_margin=group.get("fira_margin", 0.01),
                        eps_came=group.get("eps_came", 1e-8),
                        beta3=group.get("beta3"),
                    )
                else:
                    _update_param_8bit(
                        params_with_grad[i], grads[i], states[i],
                        d=group["d"], lr=group["lr"], 
                        beta1=group.get("beta1"),
                        beta2=group["beta2"], beta2_decay=group["beta2_decay"],
                        weight_decay=group["weight_decay"], eps1=eps1, eps2=eps2,
                        maximize=group["maximize"], relative_step=group["relative_step"],
                        scale_parameter=group["scale_parameter"], scale_weight_decay=group.get("scale_weight_decay", True),
                        block_size=group.get("block_size", 2048), 
                        m_block_size=group.get("m_block_size", 128),
                        use_cuda_kernel=group.get("use_cuda_kernel", True),
                        enable_fira_for_adafactor=group.get("enable_fira_for_adafactor", False),
                        fira_margin=group.get("fira_margin", 0.01),
                        factored=group.get("factored", True),
                        beta3=group.get("beta3"),
                        eps_came=group.get("eps_came", 1e-8),
                    )
        return loss

# ==========================================
# 5. Parameter Update Logic
# ==========================================
def _apply_fira_cuda(state: Dict[str, Any], total_sum_sq: Tensor, alpha: Tensor, fira_margin: float) -> Tuple[Tensor, Tensor]:
    current_norm = total_sum_sq.sqrt().view([]) 
    
    is_finite = torch.isfinite(current_norm)
    current_norm = torch.where(is_finite, current_norm, torch.zeros_like(current_norm))
    total_sum_sq = torch.where(is_finite, total_sum_sq, torch.zeros_like(total_sum_sq))
    
    fira_threshold = 1.0 + fira_margin
    
    prev_norm = state.get("fira_prev_norm", None)
    if prev_norm is not None:
        if not isinstance(prev_norm, Tensor):
            prev_norm = torch.tensor(prev_norm, device=total_sum_sq.device, dtype=torch.float32)
        
        is_reset = prev_norm < 1e-6
        ratio = current_norm / (prev_norm + 1e-8)
        limiter = torch.clamp_min(ratio, fira_threshold) / fira_threshold
        final_scale = torch.where(is_reset, torch.ones_like(current_norm), 1.0 / limiter)
        state["fira_prev_norm"] = torch.where(is_reset, current_norm, current_norm * final_scale)
    else:
        final_scale = torch.tensor(1.0, device=total_sum_sq.device, dtype=torch.float32)
        state["fira_prev_norm"] = current_norm
    
    alpha_scaled = alpha * final_scale

    return alpha_scaled, total_sum_sq

def _apply_fira_pytorch(state: Dict[str, Any], update: Tensor, fira_margin: float, numel: int, d: float) -> Tuple[Tensor, Tensor]:
    current_norm = torch.linalg.vector_norm(update)
    
    is_finite = torch.isfinite(current_norm)
    current_norm = torch.where(is_finite, current_norm, torch.zeros_like(current_norm))
    update = torch.where(is_finite, update, torch.zeros_like(update))
    
    fira_threshold = 1.0 + fira_margin
    
    prev_norm = state.get("fira_prev_norm", None)
    if prev_norm is not None:
        if not isinstance(prev_norm, Tensor):
            prev_norm = torch.tensor(prev_norm, device=update.device, dtype=torch.float32)
        
        is_reset = prev_norm < 1e-6
        ratio = current_norm / (prev_norm + 1e-8)
        limiter = torch.clamp_min(ratio, fira_threshold) / fira_threshold
        final_scale = torch.where(is_reset, torch.ones_like(current_norm), 1.0 / limiter)
        state["fira_prev_norm"] = torch.where(is_reset, current_norm, current_norm * final_scale)
    else:
        final_scale = torch.tensor(1.0, device=update.device, dtype=torch.float32)
        state["fira_prev_norm"] = current_norm
    
    update_scaled = update * final_scale
    denom = torch.clamp(current_norm / (math.sqrt(numel) * d), min=1.0)
    
    return update_scaled, denom

def _update_param_8bit(
    param: Tensor, grad: Tensor, 
    state: Dict[str, Any],
    d: float, lr: Union[float, Tensor], 
    beta1: Optional[float],
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
    m_block_size: int,
    use_cuda_kernel: bool,
    enable_fira_for_adafactor: bool = False,
    fira_margin: float = 0.01,
    factored: bool = True,
    beta3: Optional[float] = None,
    eps_came: float = 1e-8,
):
    if eps1 is None:
        eps1 = torch.finfo(torch.float32).eps
        
    eps_sq = max(eps1 * eps1, torch.finfo(torch.float32).tiny)
    log_eps_sq = math.log2(eps_sq)

    grad_contig = grad.contiguous()
    if maximize:
        grad_contig = grad_contig.neg()
    
    quantize = state.get("is_quantized", False)
    curr_block_size = state.get("block_size", block_size)
    m_curr_block_size = state.get("m_block_size", m_block_size) if beta1 is not None else 0
    
    grad_fp32 = grad_contig.float()
    N = grad_fp32.numel()

    if not param.is_contiguous():
        param_work = param.contiguous()
        needs_copy_back = True
    else:
        param_work = param
        needs_copy_back = False

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
        param_rms = torch.linalg.vector_norm(param_work, ord=2, dtype=torch.float32) / math.sqrt(param_work.numel())
        alpha = torch.clamp(param_rms, min=eps2) * rho_t
    else:
        alpha = rho_t

    if beta1 is not None and not relative_step:
        bc1 = 1.0 - beta1 ** step
        if beta2 is not None:
            bc2 = 1.0 - beta2 ** step
            alpha = alpha * (math.sqrt(bc2) / bc1)
        else:
            alpha = alpha / bc1

    if weight_decay != 0:
        wd_multiplier = alpha if scale_weight_decay else rho_t
        param_work.mul_(1.0 - (wd_multiplier * weight_decay))

    is_full_rank = (grad_fp32.dim() < 2) or (not factored)

    if is_full_rank:
        if quantize:
            if _load_cuda_module(use_cuda_kernel):
                _CUDA_MODULE.fused_log_quantize_lerp(state["variance_q"], state["variance_scale"], grad_fp32.view(-1), beta_val, curr_block_size, True, eps1, N)
                
                numel = param_work.numel()
                grad_fp32_flat = grad_fp32.view(-1)
                variance_q_flat = state["variance_q"].view(-1)
                variance_scale_flat = state["variance_scale"].view(-1)
                
                total_sum_sq = torch.zeros(1, device=param_work.device, dtype=torch.float32)
                
                if beta1 is not None:
                    _CUDA_MODULE.fused_4bit_quantize_lerp(
                        state["m_q"], state["m_scale"], grad_fp32.view(-1), beta1, m_curr_block_size, N
                    )
                    del grad_fp32

                    m_q_flat = state["m_q"].view(-1)
                    m_scale_flat = state["m_scale"].view(-1)
                    
                    _CUDA_MODULE.compute_update_norm_m_1d(
                        m_q_flat, m_scale_flat,
                        variance_q_flat, variance_scale_flat,
                        total_sum_sq, log_eps_sq, numel, m_curr_block_size, curr_block_size
                    )
                    
                    if enable_fira_for_adafactor:
                        alpha, total_sum_sq = _apply_fira_cuda(state, total_sum_sq, alpha, fira_margin)
                    
                    _CUDA_MODULE.apply_update_m_1d(
                        param_work.view(-1),
                        m_q_flat, m_scale_flat,
                        variance_q_flat, variance_scale_flat,
                        total_sum_sq, alpha, d, log_eps_sq, numel, m_curr_block_size, curr_block_size
                    )
                else:
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
                grad_sq = grad_fp32.square().add_(eps1)
                variance = _log_dequantize_nonneg(state["variance_q"], state["variance_scale"], state["variance_shape"], state["variance_pad"])
                variance.lerp_(grad_sq, beta_val)
                del grad_sq
                
                q, s, sh, pad = _log_quantize_nonneg(variance, curr_block_size)
                state["variance_q"], state["variance_scale"], state["variance_shape"], state["variance_pad"] = q, s, sh, pad
                
                if beta1 is not None:
                    if "m_q" in state:
                        m_temp = _dequantize_4bit(state["m_q"], state["m_scale"], grad_fp32.numel(), grad_fp32.shape, m_curr_block_size, grad_fp32.device)
                    else:
                        m_temp = torch.zeros_like(grad_fp32)
                    m_temp.lerp_(grad_fp32, 1.0 - beta1)
                    del grad_fp32 
                    
                    update = m_temp * variance.clamp_(min=eps_sq).rsqrt_()
                    state["m_q"], state["m_scale"] = _quantize_4bit_pytorch(m_temp, m_curr_block_size)
                    del m_temp, variance
                    
                    if enable_fira_for_adafactor:
                        update, denom = _apply_fira_pytorch(state, update, fira_margin, update.numel(), d)
                    else:
                        denom = torch.clamp(torch.linalg.vector_norm(update) / (math.sqrt(update.numel()) * d), min=1.0)
                    param_work.add_(update, alpha=-alpha / denom)
                else:
                    update = variance.clamp_(min=eps_sq).rsqrt_().mul_(grad_fp32)
                    del variance
                    
                    if enable_fira_for_adafactor:
                        update, denom = _apply_fira_pytorch(state, update, fira_margin, update.numel(), d)
                    else:
                        denom = torch.clamp(torch.linalg.vector_norm(update) / (math.sqrt(update.numel()) * d), min=1.0)
                    param_work.add_(update, alpha=-alpha / denom)
        else:
            if _load_cuda_module(use_cuda_kernel):
                total_sum_sq = torch.zeros(1, device=param_work.device, dtype=torch.float32)
                
                var_flat = state["variance"]
                if not var_flat.is_contiguous():
                    var_flat = var_flat.contiguous()
                    state["variance"] = var_flat
                    
                if beta1 is not None:
                    if "m" not in state:
                        state["m"] = torch.zeros_like(grad_fp32)
                    m_flat = state["m"]
                    if not m_flat.is_contiguous():
                        m_flat = m_flat.contiguous()
                        state["m"] = m_flat
                        
                    _CUDA_MODULE.compute_update_norm_1d_full_m(
                        var_flat.view(-1), m_flat.view(-1), grad_fp32.view(-1), total_sum_sq,
                        beta1, beta_val, eps_sq, N
                    )
                    if enable_fira_for_adafactor:
                        alpha, total_sum_sq = _apply_fira_cuda(state, total_sum_sq, alpha, fira_margin)
                        
                    _CUDA_MODULE.apply_update_1d_full_m(
                        param_work.view(-1), var_flat.view(-1), m_flat.view(-1),
                        total_sum_sq, alpha, d, eps_sq, N
                    )
                else:
                    _CUDA_MODULE.compute_update_norm_1d_full(
                        var_flat.view(-1), grad_fp32.view(-1), total_sum_sq,
                        beta_val, eps_sq, N
                    )
                    if enable_fira_for_adafactor:
                        alpha, total_sum_sq = _apply_fira_cuda(state, total_sum_sq, alpha, fira_margin)
                        
                    _CUDA_MODULE.apply_update_1d_full(
                        param_work.view(-1), var_flat.view(-1), grad_fp32.view(-1),
                        total_sum_sq, alpha, d, eps_sq, N
                    )
            else:
                variance = state["variance"]
                variance.mul_(1.0 - beta_val).addcmul_(grad_fp32, grad_fp32, value=beta_val)
                
                if beta1 is not None:
                    if "m" not in state:
                        state["m"] = torch.zeros_like(grad_fp32)
                    state["m"].lerp_(grad_fp32, 1.0 - beta1)
                    del grad_fp32 
                    
                    update = torch.empty_like(variance)
                    torch.clamp(variance, min=eps_sq, out=update)
                    torch.rsqrt(update, out=update)
                    update.mul_(state["m"])
                    
                    if enable_fira_for_adafactor:
                        update, denom = _apply_fira_pytorch(state, update, fira_margin, update.numel(), d)
                    else:
                        denom = torch.clamp(torch.linalg.vector_norm(update) / (math.sqrt(update.numel()) * d), min=1.0)
                    param_work.add_(update, alpha=-alpha / denom)
                else:
                    update = torch.empty_like(variance)
                    torch.clamp(variance, min=eps_sq, out=update)
                    torch.rsqrt(update, out=update)
                    update.mul_(grad_fp32)
                    
                    if enable_fira_for_adafactor:
                        update, denom = _apply_fira_pytorch(state, update, fira_margin, update.numel(), d)
                    else:
                        denom = torch.clamp(torch.linalg.vector_norm(update) / (math.sqrt(update.numel()) * d), min=1.0)
                    param_work.add_(update, alpha=-alpha / denom)

    else:
        shape = grad_fp32.shape
        R = shape[-2]
        C = shape[-1]
        numel = grad_fp32.numel()
        
        g_sq = grad_fp32.square()
        row_mean = g_sq.mean(dim=-1, keepdim=True).add_(eps1)
        col_mean = g_sq.mean(dim=-2, keepdim=True).add_(eps1)

        if quantize:
            if _load_cuda_module(use_cuda_kernel):
                del g_sq
                _CUDA_MODULE.fused_log_quantize_lerp(state["row_var_q"], state["row_var_scale"], row_mean.reshape(-1), beta_val, curr_block_size, False, 0.0, row_mean.numel())
                _CUDA_MODULE.fused_log_quantize_lerp(state["col_var_q"], state["col_var_scale"], col_mean.reshape(-1), beta_val, curr_block_size, False, 0.0, col_mean.numel())

                row_var = _log_dequantize_nonneg(state["row_var_q"], state["row_var_scale"], state["row_var_shape"], state["row_var_pad"])
                row_mean_val_flat = row_var.mean(dim=-2, keepdim=True).clamp_(min=eps1).flatten().contiguous()
                del row_var 
                
                if beta3 is not None and beta1 is not None:
                    v_row = _log_dequantize_nonneg(state["row_var_q"], state["row_var_scale"], state["row_var_shape"], state["row_var_pad"])
                    v_col = _log_dequantize_nonneg(state["col_var_q"], state["col_var_scale"], state["col_var_shape"], state["col_var_pad"])
                    row_mean_val = v_row.mean(dim=-2, keepdim=True).clamp(min=eps1)
                    inv_row = v_row.clamp(min=eps_sq).rsqrt() * row_mean_val.sqrt()
                    inv_col = v_col.clamp(min=eps_sq).rsqrt()
                    del v_row, v_col
                    
                    U_t = grad_fp32 * inv_row
                    U_t.mul_(inv_col)
                    del inv_row, inv_col, grad_fp32
                    
                    rms_u = torch.linalg.vector_norm(U_t) / math.sqrt(U_t.numel())
                    U_t.div_(torch.clamp(rms_u / d, min=1.0))
                    
                    if _load_cuda_module(use_cuda_kernel):
                        _CUDA_MODULE.fused_4bit_quantize_lerp(
                            state["m_q"], state["m_scale"], U_t.view(-1), beta1, m_curr_block_size, numel
                        )
                    else:
                        m_temp = _dequantize_4bit(state["m_q"], state["m_scale"], numel, U_t.shape, m_curr_block_size, U_t.device)
                        m_temp.lerp_(U_t, 1.0 - beta1)
                        state["m_q"], state["m_scale"] = _quantize_4bit_pytorch(m_temp, m_curr_block_size)
                        del m_temp
                        
                    if _load_cuda_module(use_cuda_kernel):
                        batch_size = math.prod(shape[:-2]) if len(shape) > 2 else 1
                        res_row_sum = torch.zeros(batch_size * R, device=param_work.device, dtype=torch.float32)
                        res_col_sum = torch.zeros(batch_size * C, device=param_work.device, dtype=torch.float32)
                        
                        _CUDA_MODULE.came_compute_residual_2d(
                            state["m_q"].view(-1), state["m_scale"].view(-1),
                            U_t.reshape(-1), 
                            res_row_sum, res_col_sum,
                            eps_came, R, C, numel, m_curr_block_size
                        )
                        
                        beta3_val = 1.0 - beta3
                        u_row_mean = (res_row_sum / C).contiguous().view(-1)
                        u_col_mean = (res_col_sum / R).contiguous().view(-1)
                        del res_row_sum, res_col_sum
                        
                        _CUDA_MODULE.fused_log_quantize_lerp(state["conf_row_q"], state["conf_row_scale"], u_row_mean, beta3_val, curr_block_size, False, 0.0, u_row_mean.numel())
                        _CUDA_MODULE.fused_log_quantize_lerp(state["conf_col_q"], state["conf_col_scale"], u_col_mean, beta3_val, curr_block_size, False, 0.0, u_col_mean.numel())
                    else:
                        m_temp = _dequantize_4bit(state["m_q"], state["m_scale"], numel, U_t.shape, m_curr_block_size, U_t.device)
                        res = (U_t - m_temp).square() + eps_came
                        del m_temp
                        res_row = res.mean(dim=-1, keepdim=True)
                        res_col = res.mean(dim=-2, keepdim=True)
                        c_row = _log_dequantize_nonneg(state["conf_row_q"], state["conf_row_scale"], state["conf_row_shape"], state["conf_row_pad"])
                        c_col = _log_dequantize_nonneg(state["conf_col_q"], state["conf_col_scale"], state["conf_col_shape"], state["conf_col_pad"])
                        c_row.lerp_(res_row, 1.0 - beta3)
                        c_col.lerp_(res_col, 1.0 - beta3)
                        q_cr, s_cr, sh_cr, pad_cr = _log_quantize_nonneg(c_row, curr_block_size)
                        state["conf_row_q"], state["conf_row_scale"], state["conf_row_shape"], state["conf_row_pad"] = q_cr, s_cr, sh_cr, pad_cr
                        q_cc, s_cc, sh_cc, pad_cc = _log_quantize_nonneg(c_col, curr_block_size)
                        state["conf_col_q"], state["conf_col_scale"], state["conf_col_shape"], state["conf_col_pad"] = q_cc, s_cc, sh_cc, pad_cc
                        
                    if _load_cuda_module(use_cuda_kernel):
                        c_row = _log_dequantize_nonneg(state["conf_row_q"], state["conf_row_scale"], state["conf_row_shape"], state["conf_row_pad"])
                        kernel_row_mean = c_row.mean(dim=-2, keepdim=True).clamp_(min=eps1).flatten().contiguous()
                        del c_row, U_t
                        
                        kernel_row_q_flat = state["conf_row_q"].reshape(-1)
                        kernel_row_scale = state["conf_row_scale"]
                        kernel_col_q_flat = state["conf_col_q"].reshape(-1)
                        kernel_col_scale = state["conf_col_scale"]
                        
                        total_sum_sq = torch.zeros(1, device=param_work.device, dtype=torch.float32)
                        m_q_flat = state["m_q"].view(-1)
                        m_scale_flat = state["m_scale"].view(-1)
                        
                        effective_d = 1e9 if beta3 is not None else d
                        
                        _CUDA_MODULE.compute_update_norm_m_2d(
                            m_q_flat, m_scale_flat,
                            kernel_row_q_flat, kernel_row_scale,
                            kernel_col_q_flat, kernel_col_scale,
                            total_sum_sq, kernel_row_mean, log_eps_sq, R, C, numel, m_curr_block_size, curr_block_size
                        )
                        
                        if enable_fira_for_adafactor:
                            alpha, total_sum_sq = _apply_fira_cuda(state, total_sum_sq, alpha, fira_margin)
                        
                        param_flat = param_work.reshape(-1)
                        _CUDA_MODULE.apply_update_m_2d(
                            param_flat,
                            m_q_flat, m_scale_flat,
                            kernel_row_q_flat, kernel_row_scale,
                            kernel_col_q_flat, kernel_col_scale,
                            total_sum_sq, alpha, kernel_row_mean, effective_d, log_eps_sq, R, C, numel, m_curr_block_size, curr_block_size
                        )
                    else:
                        c_row = _log_dequantize_nonneg(state["conf_row_q"], state["conf_row_scale"], state["conf_row_shape"], state["conf_row_pad"])
                        c_col = _log_dequantize_nonneg(state["conf_col_q"], state["conf_col_scale"], state["conf_col_shape"], state["conf_col_pad"])
                        
                        c_row_mean = c_row.mean(dim=-2, keepdim=True).clamp(min=eps1)
                        inv_row_conf = c_row.clamp(min=eps_sq).rsqrt() * c_row_mean.sqrt()
                        inv_col_conf = c_col.clamp(min=eps_sq).rsqrt()
                        
                        m_temp = _dequantize_4bit(state["m_q"], state["m_scale"], numel, U_t.shape, m_curr_block_size, U_t.device)
                        update = m_temp * inv_row_conf
                        update.mul_(inv_col_conf)
                        del m_temp, U_t, c_row, c_col
                        
                        effective_d = 1e9 if beta3 is not None else d
                        if enable_fira_for_adafactor:
                            update, denom = _apply_fira_pytorch(state, update, fira_margin, update.numel(), effective_d)
                        else:
                            denom = torch.clamp(torch.linalg.vector_norm(update) / (math.sqrt(update.numel()) * effective_d), min=1.0)
                        param_work.add_(update, alpha=-alpha / denom)
                    
                else:
                    if beta1 is not None:
                        _CUDA_MODULE.fused_4bit_quantize_lerp(
                            state["m_q"], state["m_scale"], grad_fp32.view(-1), beta1, m_curr_block_size, numel
                        )
                    kernel_row_mean = row_mean_val_flat
                    kernel_row_q_flat = state["row_var_q"].reshape(-1)
                    kernel_row_scale = state["row_var_scale"]
                    kernel_col_q_flat = state["col_var_q"].reshape(-1)
                    kernel_col_scale = state["col_var_scale"]

                    if beta1 is not None:
                        grad_flat = None
                        del grad_fp32
                    else:
                        grad_flat = grad_fp32.reshape(-1)
                        del grad_fp32
                    
                    total_sum_sq = torch.zeros(1, device=param_work.device, dtype=torch.float32)
                    
                    if beta1 is not None:
                        m_q_flat = state["m_q"].view(-1)
                        m_scale_flat = state["m_scale"].view(-1)
                        
                        _CUDA_MODULE.compute_update_norm_m_2d(
                            m_q_flat, m_scale_flat,
                            kernel_row_q_flat, kernel_row_scale,
                            kernel_col_q_flat, kernel_col_scale,
                            total_sum_sq, kernel_row_mean, log_eps_sq, R, C, numel, m_curr_block_size, curr_block_size
                        )
                        
                        if enable_fira_for_adafactor:
                            alpha, total_sum_sq = _apply_fira_cuda(state, total_sum_sq, alpha, fira_margin)
                        
                        param_flat = param_work.reshape(-1)
                        _CUDA_MODULE.apply_update_m_2d(
                            param_flat,
                            m_q_flat, m_scale_flat,
                            kernel_row_q_flat, kernel_row_scale,
                            kernel_col_q_flat, kernel_col_scale,
                            total_sum_sq, alpha, kernel_row_mean, d, log_eps_sq, R, C, numel, m_curr_block_size, curr_block_size
                        )
                    else:
                        _CUDA_MODULE.compute_update_norm_2d(
                            kernel_row_q_flat, kernel_row_scale,
                            kernel_col_q_flat, kernel_col_scale,
                            grad_flat, total_sum_sq, kernel_row_mean, log_eps_sq, R, C, numel, curr_block_size
                        )
                        
                        if enable_fira_for_adafactor:
                            alpha, total_sum_sq = _apply_fira_cuda(state, total_sum_sq, alpha, fira_margin)
                        
                        param_flat = param_work.reshape(-1)
                        _CUDA_MODULE.apply_update_2d(
                            param_flat, grad_flat,
                            kernel_row_q_flat, kernel_row_scale,
                            kernel_col_q_flat, kernel_col_scale,
                            total_sum_sq, alpha, kernel_row_mean, d, log_eps_sq, R, C, numel, curr_block_size
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
                
                row_mean_val = row_var.mean(dim=-2, keepdim=True).clamp(min=eps1)
                inv_row = row_var.clamp(min=eps_sq).rsqrt() * row_mean_val.sqrt()
                inv_col = col_var.clamp(min=eps_sq).rsqrt()
                
                if beta3 is not None and beta1 is not None:
                    U_t = grad_fp32 * inv_row
                    U_t.mul_(inv_col)
                    del g_sq, grad_fp32
                    
                    rms_u = torch.linalg.vector_norm(U_t) / math.sqrt(U_t.numel())
                    U_t.div_(torch.clamp(rms_u / d, min=1.0))
                    
                    if "m_q" in state:
                        m_temp = _dequantize_4bit(state["m_q"], state["m_scale"], U_t.numel(), U_t.shape, m_curr_block_size, U_t.device)
                    else:
                        m_temp = torch.zeros_like(U_t)
                    m_temp.lerp_(U_t, 1.0 - beta1)
                    state["m_q"], state["m_scale"] = _quantize_4bit_pytorch(m_temp, m_curr_block_size)
                    
                    res = (U_t - m_temp).square() + eps_came
                    res_row = res.mean(dim=-1, keepdim=True)
                    res_col = res.mean(dim=-2, keepdim=True)
                    
                    c_row = _log_dequantize_nonneg(state["conf_row_q"], state["conf_row_scale"], state["conf_row_shape"], state["conf_row_pad"])
                    c_col = _log_dequantize_nonneg(state["conf_col_q"], state["conf_col_scale"], state["conf_col_shape"], state["conf_col_pad"])
                    c_row.lerp_(res_row, 1.0 - beta3)
                    c_col.lerp_(res_col, 1.0 - beta3)
                    q_cr, s_cr, sh_cr, pad_cr = _log_quantize_nonneg(c_row, curr_block_size)
                    state["conf_row_q"], state["conf_row_scale"], state["conf_row_shape"], state["conf_row_pad"] = q_cr, s_cr, sh_cr, pad_cr
                    q_cc, s_cc, sh_cc, pad_cc = _log_quantize_nonneg(c_col, curr_block_size)
                    state["conf_col_q"], state["conf_col_scale"], state["conf_col_shape"], state["conf_col_pad"] = q_cc, s_cc, sh_cc, pad_cc
                    
                    c_row_mean = c_row.mean(dim=-2, keepdim=True).clamp(min=eps1)
                    inv_row_conf = c_row.clamp(min=eps_sq).rsqrt() * c_row_mean.sqrt()
                    inv_col_conf = c_col.clamp(min=eps_sq).rsqrt()
                    
                    update = m_temp * inv_row_conf
                    update.mul_(inv_col_conf)
                    del m_temp, U_t, res, c_row, c_col
                    
                    if enable_fira_for_adafactor:
                        update, denom = _apply_fira_pytorch(state, update, fira_margin, update.numel(), d)
                    else:
                        denom = torch.clamp(torch.linalg.vector_norm(update) / (math.sqrt(update.numel()) * d), min=1.0)
                    param_work.add_(update, alpha=-alpha / denom)
                elif beta1 is not None:
                    if "m_q" in state:
                        m_temp = _dequantize_4bit(state["m_q"], state["m_scale"], grad_fp32.numel(), grad_fp32.shape, m_curr_block_size, grad_fp32.device)
                    else:
                        m_temp = torch.zeros_like(grad_fp32)
                    m_temp.lerp_(grad_fp32, 1.0 - beta1)
                    del g_sq, grad_fp32
                    
                    update = m_temp * inv_row
                    update.mul_(inv_col)
                    state["m_q"], state["m_scale"] = _quantize_4bit_pytorch(m_temp, m_curr_block_size)
                    del m_temp
                    
                    if enable_fira_for_adafactor:
                        update, denom = _apply_fira_pytorch(state, update, fira_margin, update.numel(), d)
                    else:
                        denom = torch.clamp(torch.linalg.vector_norm(update) / (math.sqrt(update.numel()) * d), min=1.0)
                    param_work.add_(update, alpha=-alpha / denom)
                else:
                    del g_sq
                    update = grad_fp32 * inv_row
                    update.mul_(inv_col)
                    
                    if enable_fira_for_adafactor:
                        update, denom = _apply_fira_pytorch(state, update, fira_margin, update.numel(), d)
                    else:
                        denom = torch.clamp(torch.linalg.vector_norm(update) / (math.sqrt(update.numel()) * d), min=1.0)
                    param_work.add_(update, alpha=-alpha / denom)
        else:
            row_var = state["row_var"]
            col_var = state["col_var"]
            row_var.lerp_(row_mean, beta_val)
            col_var.lerp_(col_mean, beta_val)
            
            row_mean_val = row_var.mean(dim=-2, keepdim=True).clamp(min=eps1)
            inv_row = row_var.clamp(min=eps_sq).rsqrt() * row_mean_val.sqrt()
            inv_col = col_var.clamp(min=eps_sq).rsqrt()
            
            if beta3 is not None and beta1 is not None:
                U_t = grad_fp32 * inv_row
                U_t.mul_(inv_col)
                del g_sq, grad_fp32
                
                rms_u = torch.linalg.vector_norm(U_t) / math.sqrt(U_t.numel())
                U_t.div_(torch.clamp(rms_u / d, min=1.0))
                
                if "m" not in state:
                    state["m"] = torch.zeros_like(U_t)
                state["m"].lerp_(U_t, 1.0 - beta1)
                M_t = state["m"]
                
                res = (U_t - M_t).square() + eps_came
                res_row = res.mean(dim=-1, keepdim=True)
                res_col = res.mean(dim=-2, keepdim=True)
                
                state["conf_row"].lerp_(res_row, 1.0 - beta3)
                state["conf_col"].lerp_(res_col, 1.0 - beta3)
                
                combined_row = state["conf_row"].clamp(min=eps_sq)
                combined_col = state["conf_col"].clamp(min=eps_sq)
                combined_row_mean_val = combined_row.mean(dim=-2, keepdim=True).clamp(min=eps1)
                inv_row_conf = combined_row.rsqrt() * combined_row_mean_val.sqrt()
                inv_col_conf = combined_col.rsqrt()
                
                update = M_t * inv_row_conf
                update.mul_(inv_col_conf)
                del U_t, res
                
                effective_d = 1e9 if beta3 is not None else d
                if enable_fira_for_adafactor:
                    update, denom = _apply_fira_pytorch(state, update, fira_margin, update.numel(), effective_d)
                else:
                    denom = torch.clamp(torch.linalg.vector_norm(update) / (math.sqrt(update.numel()) * effective_d), min=1.0)
                param_work.add_(update, alpha=-alpha / denom)

            elif beta1 is not None:
                if "m" not in state:
                    state["m"] = torch.zeros_like(grad_fp32)
                state["m"].lerp_(grad_fp32, 1.0 - beta1)
                del g_sq, grad_fp32
                
                update = state["m"] * inv_row
                update.mul_(inv_col)
                
                if enable_fira_for_adafactor:
                    update, denom = _apply_fira_pytorch(state, update, fira_margin, update.numel(), d)
                else:
                    denom = torch.clamp(torch.linalg.vector_norm(update) / (math.sqrt(update.numel()) * d), min=1.0)
                param_work.add_(update, alpha=-alpha / denom)

            else:
                del g_sq
                update = grad_fp32 * inv_row
                update.mul_(inv_col)
                
                if enable_fira_for_adafactor:
                    update, denom = _apply_fira_pytorch(state, update, fira_margin, update.numel(), d)
                else:
                    denom = torch.clamp(torch.linalg.vector_norm(update) / (math.sqrt(update.numel()) * d), min=1.0)
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

    current_seed = state.get("current_proj_seed", state["apollo_seed"])
    proj = stable_randn(m_shape, seed=current_seed, device=device, dtype=dtype) / math.sqrt(rank)
    
    return proj, needs_refresh 


def _update_param_apollo(
    param: Tensor, grad: Tensor, state: Dict[str, Any],
    d: float, lr: Union[float, Tensor],
    beta1: Optional[float],
    beta2: Optional[float], beta2_decay: float, weight_decay: float,
    eps1: Optional[float], eps2: float, maximize: bool, relative_step: bool,
    scale_parameter: bool, scale_weight_decay: bool,
    block_size: int, m_block_size: int, use_cuda_kernel: bool,
    apollo_scale_type: str, apollo_scale: float, apollo_scale_front: bool, apollo_eps: float,
    apollo_factorize: bool,
    fira_margin: float = 0.01,
    eps_came: float = 1e-8,
    beta3: Optional[float] = None,
):
    grad_work = grad.neg().float() if maximize else grad.float()
    grad_work = torch.where(torch.isfinite(grad_work), grad_work, torch.zeros_like(grad_work))
    update_low = None 

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

    R, C = shape[-2], shape[-1]
    side = "right" if R >= C else "left"
    
    proj_matrix_T = proj_matrix.T.to(grad_work.dtype)
    if side == "right":
        grad_low = torch.matmul(grad_work, proj_matrix_T).float()
    else:
        grad_low = torch.matmul(proj_matrix_T, grad_work).float()

    norm_G = None
    norm_G_sq = None
    scaling_factor_computed = False

    if apollo_factorize:
        quantize = state.get("is_quantized", True)
        eps1_val = eps1 if eps1 is not None else apollo_eps
        row_mean_low = grad_low.square().mean(dim=-1, keepdim=True).add_(eps1_val)
        col_mean_low = grad_low.square().mean(dim=-2, keepdim=True).add_(eps1_val)
        
        if "row_var_low" not in state:
            state["row_var_low"] = (row_mean_low * beta_val).clamp(min=_FP32_TINY)
            state["col_var_low"] = (col_mean_low * beta_val).clamp(min=_FP32_TINY)
        else:
            state["row_var_low"].mul_(1.0 - beta_val).add_(row_mean_low, alpha=beta_val)
            state["col_var_low"].mul_(1.0 - beta_val).add_(col_mean_low, alpha=beta_val)
            
        row_var = state["row_var_low"]
        col_var = state["col_var_low"]
        
        eps1_val = eps1 if eps1 is not None else apollo_eps
        v_low_est = row_var * col_var / row_var.mean(dim=-2, keepdim=True).clamp(min=eps1_val)
        
        if beta1 is not None:
            if beta3 is not None:
                eps_sq = max(eps1_val * eps1_val, torch.finfo(torch.float32).tiny)
                row_mean_val = row_var.mean(dim=-2, keepdim=True).clamp(min=eps1_val)
                inv_row = row_var.clamp(min=eps_sq).rsqrt() * row_mean_val.sqrt()
                inv_col = col_var.clamp(min=eps_sq).rsqrt()
                
                U_t = grad_low * inv_row
                U_t.mul_(inv_col)
                rms_u = torch.linalg.vector_norm(U_t) / math.sqrt(U_t.numel())
                U_t.div_(torch.clamp(rms_u / d, min=1.0))

                if quantize:
                    grad_low_numel = grad_low.numel()
                    m_curr_block_size = state.get("m_block_size", m_block_size)
                    if state.get("m_low_q") is None:
                        m_padded_numel = ((grad_low_numel + m_curr_block_size - 1) // m_curr_block_size) * m_curr_block_size
                        state["m_low_q"] = torch.full((m_padded_numel // 2,), 0x88, dtype=torch.uint8, device=grad_low.device)
                        state["m_low_scale"] = torch.ones(m_padded_numel // m_curr_block_size, dtype=torch.float32, device=grad_low.device)
                        state["m_block_size"] = m_curr_block_size

                    if _load_cuda_module(use_cuda_kernel):
                        _CUDA_MODULE.fused_4bit_quantize_lerp(
                            state["m_low_q"], state["m_low_scale"], U_t.flatten(), beta1, m_curr_block_size, grad_low_numel
                        )
                        m_low_deq = _dequantize_4bit(
                            state["m_low_q"], state["m_low_scale"],
                            grad_low_numel, grad_low.shape, m_curr_block_size, grad_low.device
                        )
                    else:
                        if "m_low_q" in state:
                            m_temp = _dequantize_4bit(state["m_low_q"], state["m_low_scale"], grad_low.numel(), grad_low.shape, m_curr_block_size, grad_low.device)
                        else:
                            m_temp = torch.zeros_like(grad_low)
                        m_temp.lerp_(U_t, 1.0 - beta1)
                        state["m_low_q"], state["m_low_scale"] = _quantize_4bit_pytorch(m_temp, m_curr_block_size)
                        m_low_deq = m_temp
                        del m_temp
                else:
                    if "m_low" not in state:
                        state["m_low"] = torch.zeros_like(grad_low)
                    state["m_low"].lerp_(U_t, 1.0 - beta1)
                    m_low_deq = state["m_low"]
                
                res = (U_t - m_low_deq).square() + eps_came
                del U_t
                res_row = res.mean(dim=-1, keepdim=True)
                res_col = res.mean(dim=-2, keepdim=True)
                
                curr_block_size = state.get("block_size", block_size)
                if quantize:
                    if state.get("conf_row_low_q") is None:
                        r_numel = res_row.numel()
                        r_pad = (curr_block_size - r_numel % curr_block_size) % curr_block_size
                        state["conf_row_low_q"] = torch.zeros(r_numel + r_pad, dtype=torch.uint8, device=grad_low.device)
                        state["conf_row_low_scale"] = torch.ones((r_numel + r_pad) // curr_block_size, dtype=torch.float32, device=grad_low.device)
                        state["conf_row_low_shape"] = res_row.shape
                        state["conf_row_low_pad"] = r_pad
                        c_numel = res_col.numel()
                        c_pad = (curr_block_size - c_numel % curr_block_size) % curr_block_size
                        state["conf_col_low_q"] = torch.zeros(c_numel + c_pad, dtype=torch.uint8, device=grad_low.device)
                        state["conf_col_low_scale"] = torch.ones((c_numel + c_pad) // curr_block_size, dtype=torch.float32, device=grad_low.device)
                        state["conf_col_low_shape"] = res_col.shape
                        state["conf_col_low_pad"] = c_pad
                    c_row = _log_dequantize_nonneg(state["conf_row_low_q"], state["conf_row_low_scale"], state["conf_row_low_shape"], state["conf_row_low_pad"])
                    c_col = _log_dequantize_nonneg(state["conf_col_low_q"], state["conf_col_low_scale"], state["conf_col_low_shape"], state["conf_col_low_pad"])
                    c_row.lerp_(res_row, 1.0 - beta3)
                    c_col.lerp_(res_col, 1.0 - beta3)
                    q_cr, s_cr, sh_cr, pad_cr = _log_quantize_nonneg(c_row, curr_block_size)
                    state["conf_row_low_q"], state["conf_row_low_scale"], state["conf_row_low_shape"], state["conf_row_low_pad"] = q_cr, s_cr, sh_cr, pad_cr
                    q_cc, s_cc, sh_cc, pad_cc = _log_quantize_nonneg(c_col, curr_block_size)
                    state["conf_col_low_q"], state["conf_col_low_scale"], state["conf_col_low_shape"], state["conf_col_low_pad"] = q_cc, s_cc, sh_cc, pad_cc
                else:
                    if "conf_row_low" not in state:
                        state["conf_row_low"] = torch.zeros_like(res_row)
                        state["conf_col_low"] = torch.zeros_like(res_col)
                    state["conf_row_low"].lerp_(res_row, 1.0 - beta3)
                    state["conf_col_low"].lerp_(res_col, 1.0 - beta3)
                    c_row = state["conf_row_low"]
                    c_col = state["conf_col_low"]
                
                conf_row_mean = c_row.mean(dim=-2, keepdim=True).clamp(min=eps1_val)
                inv_row_conf = c_row.clamp(min=eps_sq).rsqrt() * conf_row_mean.sqrt()
                inv_col_conf = c_col.clamp(min=eps_sq).rsqrt()
                
                update_low = m_low_deq * inv_row_conf
                update_low.mul_(inv_col_conf)
                
            else:
                if quantize:
                    grad_low_flat = grad_low.flatten()
                    grad_low_numel = grad_low_flat.numel()
                    m_curr_block_size = state.get("m_block_size", m_block_size)
                    
                    if state.get("m_low_q") is None:
                        m_padded_numel = ((grad_low_numel + m_curr_block_size - 1) // m_curr_block_size) * m_curr_block_size
                        state["m_low_q"] = torch.full((m_padded_numel // 2,), 0x88, dtype=torch.uint8, device=grad_low.device)
                        state["m_low_scale"] = torch.ones(m_padded_numel // m_curr_block_size, dtype=torch.float32, device=grad_low.device)
                        state["m_block_size"] = m_curr_block_size

                    if _load_cuda_module(use_cuda_kernel):
                        _CUDA_MODULE.fused_4bit_quantize_lerp(
                            state["m_low_q"], state["m_low_scale"], grad_low_flat, beta1, m_curr_block_size, grad_low_numel
                        )
                        m_low_deq = _dequantize_4bit(
                            state["m_low_q"], state["m_low_scale"],
                            grad_low_numel, grad_low.shape, m_curr_block_size, grad_low.device
                        )
                    else:
                        if "m_low_q" in state:
                            m_temp = _dequantize_4bit(state["m_low_q"], state["m_low_scale"], grad_low.numel(), grad_low.shape, m_curr_block_size, grad_low.device)
                        else:
                            m_temp = torch.zeros_like(grad_low)
                        m_temp.lerp_(grad_low, 1.0 - beta1)
                        state["m_low_q"], state["m_low_scale"] = _quantize_4bit_pytorch(m_temp, m_curr_block_size)
                        m_low_deq = m_temp
                        del m_temp
                else:
                    if "m_low" not in state:
                        state["m_low"] = torch.zeros_like(grad_low)
                    state["m_low"].lerp_(grad_low, 1.0 - beta1)
                    m_low_deq = state["m_low"]
                update_low = m_low_deq / (torch.sqrt(v_low_est) + apollo_eps)
        else:
            update_low = grad_low / (torch.sqrt(v_low_est) + apollo_eps)
        
    else:
        is_first_step = (state.get("v_low_q") is None and state.get("v_low") is None)
        quantize = state.get("is_quantized", True)
        curr_block_size = state.get("block_size", block_size)
        eps1_val = eps1 if eps1 is not None else apollo_eps
        eps_sq = max(eps1_val * eps1_val, torch.finfo(torch.float32).tiny)

        if is_first_step:
            v_init = (grad_low.flatten().square() * beta_val).clamp(min=_FP32_TINY)
            if quantize:
                q, s, sh, pad = _log_quantize_nonneg(v_init, curr_block_size)
                state["v_low_q"], state["v_low_scale"], state["v_low_shape"], state["v_low_pad"] = q, s, sh, pad
            else:
                state["v_low"] = v_init
        else:
            if quantize:
                if _load_cuda_module(use_cuda_kernel):
                    _CUDA_MODULE.fused_log_quantize_lerp(
                        state["v_low_q"], state["v_low_scale"], grad_low.flatten(), beta_val, curr_block_size, True, eps1_val, grad_low.numel()
                    )
                else:
                    grad_low_sq_flat = grad_low.flatten().square().add_(eps1_val)
                    v_deq = _log_dequantize_nonneg(state["v_low_q"], state["v_low_scale"], state["v_low_shape"], state["v_low_pad"])
                    v_deq.lerp_(grad_low_sq_flat, beta_val)
                    q, s, sh, pad = _log_quantize_nonneg(v_deq, curr_block_size)
                    state["v_low_q"], state["v_low_scale"], state["v_low_shape"], state["v_low_pad"] = q, s, sh, pad
            else:
                grad_low_sq_flat = grad_low.flatten().square().add_(eps1_val)
                state["v_low"].lerp_(grad_low_sq_flat, beta_val)

        if beta3 is not None:
            eps1_val = eps1 if eps1 is not None else apollo_eps
            eps_sq = max(eps1_val * eps1_val, torch.finfo(torch.float32).tiny)
            curr_block_size = state.get("block_size", block_size)

            if beta1 is not None:
                grad_low_numel = grad_low.numel()
                m_curr_block_size = state.get("m_block_size", m_block_size)
                
                if state.get("m_low_q") is None and quantize:
                    m_padded_numel = ((grad_low_numel + m_curr_block_size - 1) // m_curr_block_size) * m_curr_block_size
                    state["m_low_q"] = torch.full((m_padded_numel // 2,), 0x88, dtype=torch.uint8, device=grad_low.device)
                    state["m_low_scale"] = torch.ones(m_padded_numel // m_curr_block_size, dtype=torch.float32, device=grad_low.device)
                    state["m_block_size"] = m_curr_block_size
                
                if state.get("res_low_q") is None and quantize:
                    r_numel = grad_low_numel
                    r_pad = (curr_block_size - r_numel % curr_block_size) % curr_block_size
                    state["res_low_q"] = torch.zeros(r_numel + r_pad, dtype=torch.uint8, device=grad_low.device)
                    state["res_low_scale"] = torch.ones((r_numel + r_pad) // curr_block_size, dtype=torch.float32, device=grad_low.device)
                    state["res_low_shape"] = grad_low.shape
                    state["res_low_pad"] = r_pad

                if quantize and _load_cuda_module(use_cuda_kernel):
                    total_sum_sq = torch.zeros(1, device=grad_low.device, dtype=torch.float32)
                    _CUDA_MODULE.compute_apollo_came_rms(
                        grad_low.contiguous(), state["v_low_q"].view(-1), state["v_low_scale"],
                        total_sum_sq, eps_sq, grad_low_numel, curr_block_size
                    )
                    
                    clip_factor_t = torch.clamp(torch.sqrt(total_sum_sq / grad_low_numel) / d, min=1.0)
                    
                    m_temp = torch.empty(grad_low_numel, device=grad_low.device, dtype=torch.float32)
                    res_temp = torch.empty(grad_low_numel, device=grad_low.device, dtype=torch.float32)
                    
                    _CUDA_MODULE.apollo_came_compute_m_res(
                        grad_low.contiguous(), 
                        state["v_low_q"].view(-1), state["v_low_scale"],
                        state["m_low_q"].view(-1), state["m_low_scale"],
                        state["res_low_q"].view(-1), state["res_low_scale"],
                        m_temp, res_temp,
                        clip_factor_t, beta1, beta3, eps_came, eps_sq,
                        curr_block_size, m_curr_block_size, curr_block_size, grad_low_numel
                    )
                    
                    _CUDA_MODULE.fused_4bit_quantize_lerp(
                        state["m_low_q"].view(-1), state["m_low_scale"], m_temp, 0.0, m_curr_block_size, grad_low_numel
                    )
                    _CUDA_MODULE.fused_log_quantize_lerp(
                        state["res_low_q"].view(-1), state["res_low_scale"], res_temp, 1.0, curr_block_size, False, 0.0, grad_low_numel
                    )
                    
                    del m_temp, res_temp
                    
                    if apollo_scale_type == "channel":
                        R_low, C_low = grad_low.shape[-2], grad_low.shape[-1]
                        if side == "right":
                            N, D, stride_N, stride_D = R_low, C_low, C_low, 1
                        else:
                            N, D, stride_N, stride_D = C_low, R_low, 1, C_low
                    else:
                        N, D, stride_N, stride_D = 1, grad_low_numel, grad_low_numel, 1

                    norm_update = torch.zeros(N, device=grad_low.device, dtype=torch.float32)
                    norm_grad = torch.zeros(N, device=grad_low.device, dtype=torch.float32)
                    
                    _CUDA_MODULE.apollo_came_compute_update_norms(
                        state["m_low_q"].view(-1), state["m_low_scale"],
                        state["res_low_q"].view(-1), state["res_low_scale"],
                        grad_low.contiguous(),
                        norm_update, norm_grad,
                        eps_sq, N, D, stride_N, stride_D, grad_low_numel,
                        m_curr_block_size, curr_block_size
                    )
                    
                    scaling_factor = norm_update / (norm_grad + 1e-8)
                    if apollo_scale_type == "channel":
                        scaling_factor = scaling_factor.unsqueeze(1 if side == "right" else 0)
                    scaling_factor_computed = True
                else:
                    v_low_deq = _log_dequantize_nonneg(state["v_low_q"], state["v_low_scale"], state["v_low_shape"], state["v_low_pad"]) if quantize else state["v_low"]
                    v_low_reshaped = v_low_deq.view_as(grad_low)
                    
                    U_t = grad_low * v_low_reshaped.clamp(min=eps_sq).rsqrt()
                    rms_u = torch.linalg.vector_norm(U_t) / math.sqrt(U_t.numel())
                    U_t.div_(torch.clamp(rms_u / d, min=1.0))
                    
                    if "m_low_q" in state:
                        m_temp = _dequantize_4bit(state["m_low_q"], state["m_low_scale"], grad_low.numel(), grad_low.shape, m_curr_block_size, grad_low.device)
                    else:
                        m_temp = torch.zeros_like(grad_low)
                    m_temp.lerp_(U_t, 1.0 - beta1)
                    state["m_low_q"], state["m_low_scale"] = _quantize_4bit_pytorch(m_temp, m_curr_block_size)
                    m_low_deq = m_temp
                    del m_temp
                    
                    res = (U_t - m_low_deq).square() + eps_came
                    del U_t
                    
                    res_deq = _log_dequantize_nonneg(state["res_low_q"], state["res_low_scale"], state["res_low_shape"], state["res_low_pad"])
                    res_deq.lerp_(res.flatten(), 1.0 - beta3)
                    q, s, sh, pad = _log_quantize_nonneg(res_deq, curr_block_size)
                    state["res_low_q"], state["res_low_scale"], state["res_low_shape"], state["res_low_pad"] = q, s, sh, pad
                    
                    res_low_deq = _log_dequantize_nonneg(state["res_low_q"], state["res_low_scale"], state["res_low_shape"], state["res_low_pad"])
                    res_low_reshaped = res_low_deq.view_as(grad_low)
                    update_low = m_low_deq * res_low_reshaped.clamp(min=eps_sq).rsqrt()
            else:
                v_low_deq = _log_dequantize_nonneg(state["v_low_q"], state["v_low_scale"], state["v_low_shape"], state["v_low_pad"]) if quantize else state["v_low"]
                v_low_reshaped = v_low_deq.view_as(grad_low)
                
                U_t = grad_low * v_low_reshaped.clamp(min=eps_sq).rsqrt()
                rms_u = torch.linalg.vector_norm(U_t) / math.sqrt(U_t.numel())
                U_t.div_(torch.clamp(rms_u / d, min=1.0))
                update_low = U_t

        else:
            if beta1 is not None:
                if quantize and _load_cuda_module(use_cuda_kernel) and apollo_scale_type == "channel":
                    v_low_deq = None
                else:
                    v_low_deq = _log_dequantize_nonneg(state["v_low_q"], state["v_low_scale"], state["v_low_shape"], state["v_low_pad"]) if quantize else state["v_low"]
                
                grad_low_flat = grad_low.flatten()
                grad_low_numel = grad_low_flat.numel()
                m_curr_block_size = state.get("m_block_size", m_block_size)
                if state.get("m_low_q") is None:
                    m_padded_numel = ((grad_low_numel + m_curr_block_size - 1) // m_curr_block_size) * m_curr_block_size
                    state["m_low_q"] = torch.full((m_padded_numel // 2,), 0x88, dtype=torch.uint8, device=grad_low.device)
                    state["m_low_scale"] = torch.ones(m_padded_numel // m_curr_block_size, dtype=torch.float32, device=grad_low.device)
                    state["m_block_size"] = m_curr_block_size
                
                if _load_cuda_module(use_cuda_kernel):
                    _CUDA_MODULE.fused_4bit_quantize_lerp(
                        state["m_low_q"], state["m_low_scale"], grad_low_flat, beta1, m_curr_block_size, grad_low_numel
                    )
                    if apollo_scale_type == "channel":
                        R_low, C_low = grad_low.shape[-2], grad_low.shape[-1]
                        if side == "right":
                            N, D = R_low, C_low
                            stride_N, stride_D = C_low, 1
                        else:
                            N, D = C_low, R_low
                            stride_N, stride_D = 1, C_low
                        
                        norm_update = torch.empty(N, device=grad_low.device, dtype=torch.float32)
                        norm_grad = torch.empty(N, device=grad_low.device, dtype=torch.float32)
                        _CUDA_MODULE.compute_apollo_norms(
                            state["m_low_q"].view(-1), state["m_low_scale"], state["v_low_q"].view(-1), state["v_low_scale"],
                            grad_low.contiguous(), norm_update, norm_grad, N, D, stride_N, stride_D, m_curr_block_size, block_size, apollo_eps
                        )
                        scaling_factor = norm_update / (norm_grad + 1e-8)
                        if side == "right":
                            scaling_factor = scaling_factor.unsqueeze(1)
                        else:
                            scaling_factor = scaling_factor.unsqueeze(0)
                        scaling_factor_computed = True
                    else:
                        m_low_deq = _dequantize_4bit(
                            state["m_low_q"], state["m_low_scale"],
                            grad_low_numel, grad_low.shape, m_curr_block_size, grad_low.device
                        )
                else:
                    if "m_low_q" in state:
                        m_temp = _dequantize_4bit(state["m_low_q"], state["m_low_scale"], grad_low.numel(), grad_low.shape, m_curr_block_size, grad_low.device)
                    else:
                        m_temp = torch.zeros_like(grad_low)
                    m_temp.lerp_(grad_low, 1.0 - beta1)
                    state["m_low_q"], state["m_low_scale"] = _quantize_4bit_pytorch(m_temp, m_curr_block_size)
                    m_low_deq = m_temp
                    del m_temp
            else:
                v_low_deq = _log_dequantize_nonneg(state["v_low_q"], state["v_low_scale"], state["v_low_shape"], state["v_low_pad"]) if quantize else state["v_low"]
                update_low = grad_low / (torch.sqrt(v_low_deq) + apollo_eps)
                update_low = update_low.view_as(grad_low)
            
            if beta1 is not None and not scaling_factor_computed:
                v_low_reshaped = v_low_deq.view_as(grad_low)
                update_low = m_low_deq / (torch.sqrt(v_low_reshaped) + apollo_eps)

    if not scaling_factor_computed:
        if apollo_scale_type == "channel":
            norm_dim = -1 if side == "right" else -2
            scaling_factor = update_low.norm(dim=norm_dim, keepdim=True) / (grad_low.norm(dim=norm_dim, keepdim=True) + 1e-8)
        else:
            scaling_factor = update_low.norm() / (grad_low.norm() + 1e-8)
            
    if beta1 is not None and not relative_step:
        bc1 = 1.0 - beta1 ** step
        if beta2 is not None:
            bc2 = 1.0 - beta2 ** step
            scaling_factor = scaling_factor * (math.sqrt(bc2) / bc1)
        else:
            scaling_factor = scaling_factor / bc1

    # --- APOLLO Scale Front ---
    apollo_scale_val = math.sqrt(apollo_scale)
    if apollo_scale_front:
        scaling_factor = scaling_factor * apollo_scale_val

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

    is_finite = torch.isfinite(current_norm_t)
    current_norm_t = torch.where(is_finite, current_norm_t, torch.zeros_like(current_norm_t))

    fira_threshold = 1.0 + fira_margin
    if "scaled_grad_norm_prev" in state:
        prev_norm_t = state["scaled_grad_norm_prev"]
        if not isinstance(prev_norm_t, Tensor):
            prev_norm_t = torch.tensor(prev_norm_t, device=param_work.device, dtype=torch.float32)
            
        is_reset = prev_norm_t < 1e-6
        ratio = current_norm_t / (prev_norm_t + 1e-8)
        limiter = torch.clamp_min(ratio, fira_threshold) / fira_threshold
        final_scale = torch.where(is_reset, scaling_factor, scaling_factor / limiter)
        state["scaled_grad_norm_prev"] = torch.where(is_reset, current_norm_t, current_norm_t / limiter)
    else:
        final_scale = scaling_factor
        state["scaled_grad_norm_prev"] = current_norm_t

    # --- APOLLO Scale Back ---
    if not apollo_scale_front:
        final_scale = final_scale * apollo_scale_val

    numel = grad_work.numel()
    if apollo_scale_type == "channel":
        norm_unscaled_t = torch.sqrt((norm_G_sq * scaling_factor.square()).sum())
    else:
        norm_unscaled_t = torch.linalg.vector_norm(grad_work, ord=2, dtype=torch.float32) * scaling_factor
        
    denom_t = torch.clamp_min(norm_unscaled_t / (math.sqrt(numel) * d), 1.0)

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
        
    del grad_low, scaling_factor, final_scale, proj_matrix_T, proj_matrix
    del update_low
    
    if needs_copy_back:
        param.copy_(param_work.view(original_shape))
