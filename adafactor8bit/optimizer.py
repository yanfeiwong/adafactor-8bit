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


# ==========================================
# 0. Global Constants
# ==========================================
_LOG_QUANT_FLOOR = 1.17549435e-38
_VALID_M_QUANT_TYPES = ('uf4', 'uf8', 'd4', 'd8', 'fp32')
_VALID_V_QUANT_TYPES = ('al8', 'al16', 'fp32')
_ADV_DEFAULT = 0xF


# ==========================================
# 1. Global Caches & Module State
# ==========================================
# --- Workspace Buffer Caches ---

_V_BUF_CACHE = {}
_M_BUF_CACHE = {}
_NORM_BUF_CACHE = {}
_ONE_CACHE = {}
_RC_BUF_CACHE = {}
_RHO_CACHE = {}


# --- CUDA Module State ---

_CUDA_MODULE = None
_CUDA_AVAILABLE = False
_CUDA_LOAD_ATTEMPTED = False


# --- Dynamic Quantization Map (QMap) State ---

_QMAP_8BIT_CPU = None
_QMAP_4BIT_CPU = None
_QMAP_INITIALIZED_DEVICES = set()
_QMAP_CACHE = {}


# ==========================================
# 2. CUDA Kernel JIT Loading
# ==========================================
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
        if torch.cuda.is_available():
            _ensure_qmap(torch.device("cuda", torch.cuda.current_device()))
    except Exception as e:
        logger.warning(f"Adafactor8Bit: Failed to load CUDA Kernel. Falling back to PyTorch. Error: {e}")

    return _CUDA_AVAILABLE


def _ensure_qmap(device):
    global _CUDA_AVAILABLE, _QMAP_8BIT_CPU, _QMAP_4BIT_CPU
    if not _CUDA_AVAILABLE or _CUDA_MODULE is None:
        return
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type != 'cuda':
        return
    dev_key = str(device)
    if dev_key in _QMAP_INITIALIZED_DEVICES:
        return
    try:
        if _QMAP_8BIT_CPU is None:
            _QMAP_8BIT_CPU = _create_dynamic_map(signed=True, total_bits=8)
            _QMAP_4BIT_CPU = _create_dynamic_map(signed=True, max_exponent_bits=3, total_bits=4)
        target = device if isinstance(device, torch.device) else torch.device(device)
        with torch.cuda.device(target):
            _CUDA_MODULE.set_qmap(
                _QMAP_8BIT_CPU.to(target),
                _QMAP_4BIT_CPU.to(target)
            )
        _QMAP_INITIALIZED_DEVICES.add(dev_key)
    except Exception as e:
        _CUDA_AVAILABLE = False
        logger.warning(f"Adafactor8Bit: Failed to initialize dynamic quantization maps on {device}, falling back to PyTorch: {e}")


# ==========================================
# 3. APOLLO Random Seed Utilities
# ==========================================
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
# 4. Adaptive Log-Space V Quantization
#    q=0 -> V=0, q=1..255 -> V = 2^((q-1)*scale/254 + min_log)
# ==========================================
def _log_quantize_nonneg(tensor: Tensor, block_size: int = 2048, min_log_floor: float = -126.0, n_levels: int = 256) -> Tuple[Tensor, Tensor, Tensor, torch.Size, int]:
    """Quantize a non-negative FP32 tensor in adaptive log-space with zero point."""
    shape = tensor.shape
    flat = tensor.flatten()
    pad = (block_size - flat.numel() % block_size) % block_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))

    blocks = flat.view(-1, block_size)
    is_zero = (blocks == 0)
    v_safe = blocks.clamp(min=1e-38)
    log_blocks = torch.log2(v_safe)

    min_log = log_blocks.masked_fill(is_zero, float('inf')).min(dim=1, keepdim=True).values
    max_log = log_blocks.amax(dim=1, keepdim=True)

    min_log = min_log.clamp(min=min_log_floor)
    min_log = torch.where(min_log >= max_log, max_log - 1.0, min_log)

    scale = (max_log - min_log).clamp(min=1e-12)
    q_max = n_levels - 1
    q = (torch.round((log_blocks - min_log) / scale * (q_max - 1.0)) + 1.0).clamp(1, q_max)
    q[is_zero] = 0.0
    
    target_dtype = torch.uint8 if n_levels <= 256 else torch.int16 if n_levels <= 65536 else torch.int32
    q = q.to(target_dtype)
    return q, scale.squeeze(-1), min_log.squeeze(-1), shape, pad


def _log_dequantize_nonneg(q: Tensor, scale: Tensor, min_log: Tensor, shape: torch.Size, pad: int, n_levels: int = 256) -> Tensor:
    """Dequantize from adaptive log-space back to linear-space FP32."""
    q_flat = q.view(-1)
    block_size = q_flat.numel() // scale.numel()
    
    if _CUDA_MODULE is not None and _CUDA_AVAILABLE and q_flat.is_cuda and q_flat.dtype in (torch.uint8, torch.int16):
        output = torch.empty(q_flat.numel(), device=q_flat.device, dtype=torch.float32)
        v_bits = 16 if q_flat.dtype == torch.int16 else 8
        _CUDA_MODULE.dequantize_log_nonneg(output, q_flat, scale, min_log, q_flat.numel(), block_size, v_bits)
        if pad:
            output = output[:-pad]
        return output.view(shape)
        
    q_2d = q_flat.view(-1, block_size)
    is_zero = (q_2d == 0)
    q_max = n_levels - 1
    q_float = q_2d.float()
    if q_2d.dtype == torch.int16:
        q_float = torch.where(q_float < 0, q_float + 65536.0, q_float)
    log_blocks = (q_float - 1.0) * scale.unsqueeze(-1) / (q_max - 1.0) + min_log.unsqueeze(-1)
    blocks = torch.pow(2.0, log_blocks)
    blocks[is_zero] = 0.0
    flat = blocks.flatten()
    if pad:
        flat = flat[:-pad]
    return flat.view(shape)


# --- V-state helpers ---

def _deq_v(state: Dict[str, Any], prefix: str, pop: bool = False) -> Tensor:
    if pop:
        q = state.pop(f"{prefix}_q")
        return _log_dequantize_nonneg(
            q, state.pop(f"{prefix}_scale"),
            state.pop(f"{prefix}_min_log"),
            state.pop(f"{prefix}_shape"), state.pop(f"{prefix}_pad"),
            n_levels=256 if q.dtype == torch.uint8 else 65536)
    q = state[f"{prefix}_q"]
    return _log_dequantize_nonneg(
        q, state[f"{prefix}_scale"], state[f"{prefix}_min_log"],
        state[f"{prefix}_shape"], state[f"{prefix}_pad"],
        n_levels=256 if q.dtype == torch.uint8 else 65536)


def _quant_v(state: Dict[str, Any], prefix: str, tensor: Tensor, block_size: int, v_quant_type: str = 'al8'):
    n_levels = 256 if v_quant_type == 'al8' else 65536
    q, s, ml, sh, pad = _log_quantize_nonneg(tensor, block_size, n_levels=n_levels)
    state[f"{prefix}_q"], state[f"{prefix}_scale"], state[f"{prefix}_min_log"] = q, s, ml
    state[f"{prefix}_shape"], state[f"{prefix}_pad"] = sh, pad


def _reset_keep_step(state: Dict[str, Any]):
    step = state.get("step", 0)
    state.clear()
    state["step"] = step


def _init_v_state(state: Dict[str, Any], prefix: str, shape, block_size: int, device, v_quant_type: str = 'al8'):
    numel = math.prod(shape) if not isinstance(shape, int) else shape
    if v_quant_type == 'fp32':
        state[prefix] = torch.zeros(shape, dtype=torch.float32, device=device)
        return
    pad = (block_size - numel % block_size) % block_size
    nblocks = (numel + pad) // block_size
    target_dtype = torch.uint8 if v_quant_type == 'al8' else torch.int16
    state[f"{prefix}_q"] = torch.zeros(numel + pad, dtype=target_dtype, device=device)
    state[f"{prefix}_scale"] = torch.ones(nblocks, dtype=torch.float32, device=device)
    state[f"{prefix}_min_log"] = torch.zeros(nblocks, dtype=torch.float32, device=device)
    state[f"{prefix}_shape"] = shape
    state[f"{prefix}_pad"] = pad


def _init_v_or_fp32(state, prefix, shape, block_size, device, v_quant_type, use_quant):
    if use_quant and v_quant_type != 'fp32':
        _init_v_state(state, prefix, shape, block_size, device, v_quant_type)
    else:
        state[prefix] = torch.zeros(shape, dtype=torch.float32, device=device)


# ==========================================
# 5. M Quantization
# ==========================================
# --- Dynamic Map (Dettmers et al., 2022) ---

def _create_dynamic_map(signed: bool = True, max_exponent_bits: int = 7, total_bits: int = 8) -> Tensor:
    data = []
    non_sign_bits = total_bits - 1
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    for i in range(max_exponent_bits):
        fraction_items = int(
            2 ** (i + non_sign_bits - max_exponent_bits) + 1
            if signed
            else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1
        )
        boundaries = torch.linspace(0.1, 1, fraction_items, dtype=torch.float32)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
    if additional_items > 0:
        boundaries = torch.linspace(0.1, 1, additional_items + 1, dtype=torch.float32)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += means.tolist()
        if signed:
            data += (-means).tolist()
    data.append(0)
    data.append(1.0)
    assert len(data) == 2 ** total_bits, f"Expected {2 ** total_bits} items, got {len(data)}"
    data.sort()
    return torch.tensor(data, dtype=torch.float32)


def _get_qmap(total_bits: int, device: torch.device) -> Tensor:
    key = (total_bits, str(device))
    if key not in _QMAP_CACHE:
        if total_bits == 8:
            _QMAP_CACHE[key] = _create_dynamic_map(signed=True, total_bits=8).to(device)
        else:
            _QMAP_CACHE[key] = _create_dynamic_map(signed=True, max_exponent_bits=3, total_bits=4).to(device)
    return _QMAP_CACHE[key]


def _quantize_dynamic_pytorch(m: Tensor, block_size: int, m_quant_type: str) -> Tuple[Tensor, Tensor]:
    total_bits = 8 if m_quant_type == 'd8' else 4
    qmap = _get_qmap(total_bits, m.device)
    flat = m.flatten()
    pad = (block_size - flat.numel() % block_size) % block_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.view(-1, block_size)
    abs_max = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    normalized = (blocks / abs_max).flatten()
    indices = torch.searchsorted(qmap, normalized).clamp(0, len(qmap) - 1)
    prev_indices = (indices - 1).clamp(0)
    curr_dist = (normalized - qmap[indices]).abs()
    prev_dist = (normalized - qmap[prev_indices]).abs()
    indices = torch.where(prev_dist < curr_dist, prev_indices, indices)
    orig_sign = normalized.sign()
    map_sign = qmap[indices].sign()
    flip = (orig_sign != map_sign) & (orig_sign != 0)
    indices = torch.where(flip & (orig_sign > 0) & (indices < len(qmap) - 1), indices + 1, indices)
    indices = torch.where(flip & (orig_sign < 0) & (indices > 0), indices - 1, indices)
    if m_quant_type == 'd8':
        return indices.to(torch.uint8).view(-1), abs_max.squeeze(-1)
    indices = indices.view(-1, block_size)
    q_even = indices[:, 0::2].to(torch.uint8)
    q_odd = indices[:, 1::2].to(torch.uint8)
    packed = (q_even << 4) | q_odd
    return packed.view(-1), abs_max.squeeze(-1)


def _dequantize_dynamic_pytorch(m_q: Tensor, m_scale: Tensor, numel: int, shape: torch.Size, block_size: int, m_quant_type: str, use_cuda: bool = True) -> Tensor:
    m_bits = 8 if m_quant_type == 'd8' else 4
    if use_cuda and _CUDA_MODULE is not None and _CUDA_AVAILABLE:
        output = torch.empty(numel, device=m_q.device, dtype=torch.float32)
        _CUDA_MODULE.dequantize_dynamic(output, m_q, m_scale, numel, block_size, m_bits)
        return output.view(shape)
    qmap = _get_qmap(m_bits, m_q.device)
    if m_quant_type == 'd8':
        indices = m_q.long()
    else:
        high = ((m_q >> 4) & 0x0F).long()
        low = (m_q & 0x0F).long()
        indices = torch.stack((high, low), dim=-1).view(-1)
    values = qmap[indices]
    blocks = values.view(-1, block_size)
    result = (blocks * m_scale.unsqueeze(-1)).view(-1)[:numel]
    return result.view(shape)


# --- Uniform (UF4/UF8) ---

def _quantize_uniform_pytorch(m: Tensor, block_size: int, m_bits: int) -> Tuple[Tensor, Tensor]:
    flat = m.flatten()
    pad = (block_size - flat.numel() % block_size) % block_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.view(-1, block_size)
    abs_max = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    if m_bits == 4:
        q = (torch.round(blocks / abs_max * 8.0).clamp(-8, 7) + 8).to(torch.uint8)
        q_even = q[:, 0::2]
        q_odd = q[:, 1::2]
        packed = (q_even << 4) | q_odd
        return packed.view(-1), abs_max.squeeze(-1)
    else:
        q = (torch.round(blocks / abs_max * 128.0).clamp(-128, 127) + 128).to(torch.uint8)
        return q.view(-1), abs_max.squeeze(-1)


def _dequantize_uniform_pytorch(m_q: Tensor, m_scale: Tensor, numel: int, shape: torch.Size, block_size: int, m_bits: int) -> Tensor:
    if m_bits == 4:
        high = ((m_q >> 4) & 0x0F).to(torch.float32) - 8.0
        low = (m_q & 0x0F).to(torch.float32) - 8.0
        m_flat = torch.stack((high, low), dim=-1).view(-1)
    else:
        m_flat = m_q.to(torch.float32) - 128.0
    m_blocks = m_flat.view(-1, block_size)
    divisor = 8.0 if m_bits == 4 else 128.0
    result = (m_blocks * (m_scale.unsqueeze(-1) / divisor)).view(-1)[:numel]
    return result.view(shape)


# --- Dispatch ---

def _get_m_mode(m_quant_type: str) -> int:
    return 0 if m_quant_type in ('uf4', 'uf8') else 1


def _get_m_bits(m_quant_type: str) -> int:
    if m_quant_type in ('uf4', 'd4'):
        return 4
    return 8


def _m_quantize(m: Tensor, m_quant_type: str, block_size: int) -> Tuple[Tensor, Tensor]:
    if m_quant_type == 'fp32':
        raise ValueError("_m_quantize should not be called with m_quant_type='fp32'")
    if m_quant_type in ('uf4', 'uf8'):
        return _quantize_uniform_pytorch(m, block_size, _get_m_bits(m_quant_type))
    return _quantize_dynamic_pytorch(m, block_size, m_quant_type)


def _m_dequantize(m_q: Tensor, m_scale: Tensor, numel: int, shape: torch.Size, block_size: int, device: torch.device, m_quant_type: str, use_cuda: bool = True) -> Tensor:
    if m_quant_type == 'fp32':
        raise ValueError("_m_dequantize should not be called with m_quant_type='fp32'")
    if m_quant_type in ('uf4', 'uf8'):
        return _dequantize_uniform_pytorch(m_q, m_scale, numel, shape, block_size, _get_m_bits(m_quant_type))
    return _dequantize_dynamic_pytorch(m_q, m_scale, numel, shape, block_size, m_quant_type, use_cuda=use_cuda)


def _m_fused_quantize_lerp(state: Dict[str, Any], grad_flat: Tensor, beta1: float, m_curr_block_size: int, numel: int, m_quant_type: str, prefix: str = "m", fp32_out: Optional[Tensor] = None):
    if _CUDA_MODULE is None or not _CUDA_AVAILABLE:
        raise RuntimeError("_m_fused_quantize_lerp requires CUDA kernels.")
    if m_quant_type == 'fp32':
        raise ValueError("_m_fused_quantize_lerp should not be called with m_quant_type='fp32'")
    _CUDA_MODULE.fused_m_quantize_lerp(
        state[f"{prefix}_q"], state[f"{prefix}_scale"], grad_flat, fp32_out, beta1, m_curr_block_size, numel,
        _get_m_bits(m_quant_type), _get_m_mode(m_quant_type)
    )


def _m_init_state(state: Dict[str, Any], numel: int, m_block_size: int, m_quant_type: str, device: torch.device, shape: torch.Size = None, prefix: str = "m"):
    qk, sk = f"{prefix}_q", f"{prefix}_scale"
    if m_quant_type == 'fp32':
        assert shape is not None, "_m_init_state: shape is required for fp32 momentum"
        state[prefix] = torch.zeros(shape, dtype=torch.float32, device=device)
    else:
        m_padded_numel = ((numel + m_block_size - 1) // m_block_size) * m_block_size
        m_bits = _get_m_bits(m_quant_type)
        if m_bits == 4:
            fill = 0x88 if m_quant_type == 'uf4' else 0x77
            state[qk] = torch.full((m_padded_numel // 2,), fill, dtype=torch.uint8, device=device)
        else:
            fill = 128 if m_quant_type == 'uf8' else 127
            state[qk] = torch.full((m_padded_numel,), fill, dtype=torch.uint8, device=device)
        state[sk] = torch.ones(m_padded_numel // m_block_size, dtype=torch.float32, device=device)
        state["m_block_size"] = m_block_size
    state["m_quant_type"] = m_quant_type


def _init_m_or_fp32(state, numel, m_block_size, m_quant_type, device, shape, use_quant):
    if use_quant and m_quant_type != 'fp32':
        _m_init_state(state, numel, m_block_size, m_quant_type, device, shape=shape)
    else:
        state["m"] = torch.zeros(shape, dtype=torch.float32, device=device)
        state["m_quant_type"] = 'fp32'


# ==========================================
# 6. Workspace Buffer Pool
# ==========================================
def _get_v_buf(device: torch.device, numel: int) -> Tensor:
    key = str(device)
    buf = _V_BUF_CACHE.get(key)
    if buf is None or buf.numel() < numel:
        _V_BUF_CACHE[key] = torch.empty(numel, device=device, dtype=torch.float32)
    return _V_BUF_CACHE[key][:numel]


def _get_m_buf(device: torch.device, numel: int) -> Tensor:
    key = str(device)
    buf = _M_BUF_CACHE.get(key)
    if buf is None or buf.numel() < numel:
        _M_BUF_CACHE[key] = torch.empty(numel, device=device, dtype=torch.float32)
    return _M_BUF_CACHE[key][:numel]


def _get_norm_buf(device: torch.device) -> Tensor:
    key = str(device)
    buf = _NORM_BUF_CACHE.get(key)
    if buf is None:
        buf = torch.zeros(1, device=device, dtype=torch.float32)
        _NORM_BUF_CACHE[key] = buf
    else:
        buf.zero_()
    return buf


def _get_one(device: torch.device) -> Tensor:
    key = str(device)
    t = _ONE_CACHE.get(key)
    if t is None:
        t = torch.tensor(1.0, device=device)
        _ONE_CACHE[key] = t
    return t


def _get_rc_buf(device: torch.device, total: int) -> Tensor:
    key = str(device)
    buf = _RC_BUF_CACHE.get(key)
    if buf is None or buf.numel() < total:
        _RC_BUF_CACHE[key] = torch.empty(total, device=device, dtype=torch.float32)
    return _RC_BUF_CACHE[key][:total]


def _prepare_rc_workspace(device: torch.device, batch_size: int, R: int, C: int, has_came: bool):
    """Lay out the shared rc_buf into row/col sum + fp32 slots (+ CAME slots)."""
    mult = 4 if has_came else 2
    buf = _get_rc_buf(device, batch_size * (R + C) * mult)
    rs, cs = batch_size * R, batch_size * C
    base = rs + cs
    row_sum  = buf[:rs].zero_()
    col_sum  = buf[rs:base].zero_()
    row_fp32 = buf[base:base + rs]
    col_fp32 = buf[base + rs:base + rs + cs]
    came_base = base + rs + cs
    return buf, row_sum, col_sum, row_fp32, col_fp32, came_base, rs, cs


# ==========================================
# 7. EMA Subroutines (V & M)
# ==========================================
# --- V-state EMA (CUDA fused) ---

def _fused_v_ema(state: Dict[str, Any], prefix: str, new_val: Tensor, fp32_out: Tensor,
                 beta: float, block_size: int,
                 square_input: bool = False, eps: float = 0.0, log_floor: float = 0.0):
    """Thin wrapper: fused EMA + log-quantize for a V-state slot."""
    q_tensor = state[f"{prefix}_q"]
    v_bits = 16 if q_tensor.dtype == torch.int16 else 8
    _CUDA_MODULE.fused_log_quantize_lerp(
        q_tensor, state[f"{prefix}_scale"], state[f"{prefix}_min_log"],
        new_val, fp32_out, beta, block_size, square_input, eps, new_val.numel(), log_floor, v_bits)


# --- M-state EMA ---

def _m_ema_pytorch(state: Dict[str, Any], grad: Tensor, beta1: float,
                   m_quant_type: str, m_curr_block_size: int, prefix: str = "m") -> Tensor:
    qk, sk = f"{prefix}_q", f"{prefix}_scale"
    if qk in state:
        m_temp = _m_dequantize(state[qk], state[sk], grad.numel(), grad.shape,
                               m_curr_block_size, grad.device, m_quant_type, use_cuda=False)
    else:
        m_temp = torch.zeros_like(grad)
    m_temp.lerp_(grad, 1.0 - beta1)
    state[qk], state[sk] = _m_quantize(m_temp, m_quant_type, m_curr_block_size)
    return m_temp


def _get_updated_m(state: Dict[str, Any], grad_or_ut: Tensor, beta1: float,
                   m_quant_type: str, m_curr_block_size: int, quantize: bool, prefix: str = "m") -> Tensor:
    """Unified M-state update: handles both quantized and fp32 paths safely."""
    if quantize and m_quant_type != 'fp32':
        return _m_ema_pytorch(state, grad_or_ut, beta1, m_quant_type, m_curr_block_size, prefix=prefix)
    if prefix not in state:
        state[prefix] = torch.zeros_like(grad_or_ut)
    state[prefix].lerp_(grad_or_ut, 1.0 - beta1)
    return state[prefix]


def _apollo_m_ema(state, U_t_or_grad, beta1, m_quant_type, m_curr_block_size, numel, shape, device, cuda_ready, quantize, prefix="m_low"):
    if not quantize or m_quant_type == 'fp32':
        if prefix not in state:
            state[prefix] = torch.zeros_like(U_t_or_grad)
        state[prefix].lerp_(U_t_or_grad, 1.0 - beta1)
        return state[prefix]
    
    if state.get(f"{prefix}_q") is None:
        _m_init_state(state, numel, m_curr_block_size, m_quant_type, device, prefix=prefix)
        
    if cuda_ready:
        _m_fp32 = _get_m_buf(device, numel)
        _m_fused_quantize_lerp(state, U_t_or_grad.flatten(), beta1, m_curr_block_size, numel, m_quant_type, prefix=prefix, fp32_out=_m_fp32)
        return _m_fp32.view(shape)
    else:
        return _m_ema_pytorch(state, U_t_or_grad, beta1, m_quant_type, m_curr_block_size, prefix=prefix)


# ==========================================
# 8. FiRA Update Scaling
# ==========================================
def _compute_fira_scale(state: Dict[str, Any], curr_norm: Tensor,
                        fira_margin: float, device: torch.device, state_key: str = "fira_prev_norm") -> Tensor:
    is_fin = torch.isfinite(curr_norm)
    curr_norm = torch.where(is_fin, curr_norm, torch.zeros_like(curr_norm))
    fp = state.get(state_key, None)
    if fp is not None:
        if not isinstance(fp, Tensor):
            fp = torch.tensor(fp, device=device, dtype=torch.float32)
        is_reset = fp < 1e-6
        ratio = curr_norm / (fp + 1e-8)
        limiter = torch.clamp_min(ratio, 1.0 + fira_margin) / (1.0 + fira_margin)
        fs = torch.where(is_reset, torch.ones_like(curr_norm), 1.0 / limiter)
        state[state_key] = torch.where(is_reset, curr_norm, curr_norm * fs)
    else:
        fs = torch.tensor(1.0, device=device, dtype=torch.float32)
        state[state_key] = curr_norm
    return fs


def _apply_fira_cuda(state: Dict[str, Any], total_sum_sq: Tensor, alpha: Tensor, fira_margin: float) -> Tuple[Tensor, Tensor]:
    current_norm = total_sum_sq.sqrt().view([])
    is_finite = torch.isfinite(current_norm)
    total_sum_sq = torch.where(is_finite, total_sum_sq, torch.zeros_like(total_sum_sq))
    final_scale = _compute_fira_scale(state, current_norm, fira_margin, total_sum_sq.device)
    return alpha * final_scale, total_sum_sq


def _apply_fira_pytorch(state: Dict[str, Any], update: Tensor, fira_margin: float, d: float) -> Tuple[Tensor, Tensor]:
    current_norm = torch.linalg.vector_norm(update)
    if not torch.isfinite(current_norm):
        update.zero_()
        current_norm.zero_()
        
    final_scale = _compute_fira_scale(state, current_norm, fira_margin, update.device)
    update_scaled = update * final_scale
    
    if d > 0:
        denom = torch.clamp(current_norm / (math.sqrt(update.numel()) * d), min=1.0)
    else:
        denom = _get_one(update.device)
        
    return update_scaled, denom


# ==========================================
# 9. Update Assembly & Fallbacks
# ==========================================
# --- Factored second-moment helpers ---

def _factored_inv_std(row_var: Tensor, col_var: Tensor, row_mean: Tensor) -> Tuple[Tensor, Tensor]:
    inv_row = (row_var / row_mean).rsqrt_()
    inv_col = col_var.rsqrt()
    return inv_row, inv_col


def _compute_ut_clipped(grad_fp32: Tensor, inv_row: Tensor, inv_col: Tensor, d: float) -> Tensor:
    grad_fp32.mul_(inv_row).mul_(inv_col)
    if d > 0:
        rms_u = torch.linalg.vector_norm(grad_fp32) / math.sqrt(grad_fp32.numel())
        grad_fp32.div_(torch.clamp(rms_u / d, min=1.0))
    return grad_fp32


def _compute_alpha(param_work: Tensor, lr: Union[float, Tensor],
                   step: int, relative_step: bool, scale_parameter: bool, eps2: float) -> Tuple[Tensor, Tensor]:
    if isinstance(lr, float):
        if relative_step:
            rho = min(lr, 1.0 / math.sqrt(step))
            rho_t = torch.tensor(rho, device=param_work.device, dtype=torch.float32)
        else:
            cache_key = (lr, str(param_work.device))
            rho_t = _RHO_CACHE.get(cache_key)
            if rho_t is None:
                rho_t = torch.tensor(lr, device=param_work.device, dtype=torch.float32)
                _RHO_CACHE[cache_key] = rho_t
    else:
        if relative_step:
            step_t = torch.tensor(step, device=param_work.device, dtype=torch.float32)
            rho_t = torch.minimum(step_t.rsqrt(), lr)
        else:
            rho_t = lr
    if scale_parameter:
        param_rms = torch.linalg.vector_norm(
            param_work, ord=2, dtype=torch.float32) / math.sqrt(param_work.numel())
        alpha = torch.clamp(param_rms, min=eps2) * rho_t
    else:
        alpha = rho_t
    return alpha, rho_t


def _safe_rms_denom(update: Tensor, d: float, device: torch.device) -> Tensor:
    if d > 0:
        return torch.clamp(
            torch.linalg.vector_norm(update) / (math.sqrt(update.numel()) * d), min=1.0)
    return _get_one(device)


def _apply_update_pytorch(param_work: Tensor, update: Tensor, alpha: Tensor,
                          state: Dict[str, Any], d: float, enable_fira: bool, fira_margin: float):
    if enable_fira:
        update, denom = _apply_fira_pytorch(state, update, fira_margin, d)
    else:
        denom = _safe_rms_denom(update, d, param_work.device)
    param_work.add_(update, alpha=-alpha / denom)


# --- Fallback paths (pure PyTorch) ---

def _fallback_1d_update(
    param_work, grad_fp32, variance, state, alpha, d, enable_fira, fira_margin,
    beta1, m_quant_type, m_curr_block_size, quantize, momentum_only_1d, 
    use_adam_denom, eps_for_denom, eps_sq
):
    is_temp_m = quantize and m_quant_type != 'fp32'

    if beta1 is not None:
        if momentum_only_1d:
            inv_std = variance.clamp(min=eps_sq).rsqrt()
            u_t = grad_fp32 * inv_std
            if d > 0:
                rms_u = torch.linalg.vector_norm(u_t) / math.sqrt(u_t.numel())
                u_t.div_(torch.clamp(rms_u / d, min=1.0))
            m_temp = _get_updated_m(state, u_t, beta1, m_quant_type, m_curr_block_size, quantize)
            _apply_update_pytorch(param_work, m_temp, alpha, state, 0, enable_fira, fira_margin)
        else:
            m_temp = _get_updated_m(state, grad_fp32, beta1, m_quant_type, m_curr_block_size, quantize)
            if use_adam_denom:
                inv_std = 1.0 / (variance.sqrt() + eps_for_denom)
            else:
                inv_std = variance.clamp(min=eps_sq).rsqrt()
            
            update = m_temp.mul_(inv_std) if is_temp_m else m_temp * inv_std
            _apply_update_pytorch(param_work, update, alpha, state, d, enable_fira, fira_margin)
    else:
        if use_adam_denom:
            update = grad_fp32 / (variance.sqrt() + eps_for_denom)
        else:
            inv_std = variance.clamp(min=eps_sq).rsqrt()
            update = grad_fp32 * inv_std
        _apply_update_pytorch(param_work, update, alpha, state, d, enable_fira, fira_margin)


def _fallback_2d_update(
    param_work, grad_fp32, state, alpha, d, enable_fira, fira_margin,
    beta1, beta3, eps_came, eps1, m_quant_type, m_curr_block_size, quantize,
    inv_row, inv_col, curr_block_size, v_quant_type
):
    is_temp_m = quantize and m_quant_type != 'fp32'

    if beta3 is not None and beta1 is not None:
        U_t = _compute_ut_clipped(grad_fp32, inv_row, inv_col, d)
        M_t = _get_updated_m(state, U_t, beta1, m_quant_type, m_curr_block_size, quantize)
        
        res = U_t.sub_(M_t).square_().add_(eps_came)
        res_row = res.mean(dim=-1, keepdim=True)
        res_col = res.mean(dim=-2, keepdim=True)

        if quantize and v_quant_type != 'fp32':
            c_row = _deq_v(state, "conf_row")
            c_col = _deq_v(state, "conf_col")
        else:
            c_row = state["conf_row"]
            c_col = state["conf_col"]
        c_row.lerp_(res_row, 1.0 - beta3)
        c_col.lerp_(res_col, 1.0 - beta3)
        if quantize and v_quant_type != 'fp32':
            _quant_v(state, "conf_row", c_row, curr_block_size, v_quant_type)
            _quant_v(state, "conf_col", c_col, curr_block_size, v_quant_type)

        c_row_mean = c_row.mean(dim=-2, keepdim=True).clamp(min=eps1)
        inv_row_conf, inv_col_conf = _factored_inv_std(c_row, c_col, c_row_mean)
        
        if is_temp_m:
            M_t.mul_(inv_row_conf).mul_(inv_col_conf)
            _apply_update_pytorch(param_work, M_t, alpha, state, 0, enable_fira, fira_margin)
        else:
            update = M_t.mul(inv_row_conf).mul_(inv_col_conf)
            _apply_update_pytorch(param_work, update, alpha, state, 0, enable_fira, fira_margin)

    elif beta1 is not None:
        m_temp = _get_updated_m(state, grad_fp32, beta1, m_quant_type, m_curr_block_size, quantize)
        if is_temp_m:
            m_temp.mul_(inv_row).mul_(inv_col)
            _apply_update_pytorch(param_work, m_temp, alpha, state, d, enable_fira, fira_margin)
        else:
            update = m_temp.mul(inv_row).mul_(inv_col)
            _apply_update_pytorch(param_work, update, alpha, state, d, enable_fira, fira_margin)

    else:
        update = grad_fp32 * inv_row
        update.mul_(inv_col)
        _apply_update_pytorch(param_work, update, alpha, state, d, enable_fira, fira_margin)


# ==========================================
# 10. State Migration & CUDA Dispatch
# ==========================================
def _migrate_quantize_flag(state, use_quant, curr_block_size, m_curr_block_size, m_quant_type, v_quant_type, beta1, beta3, p):
    if use_quant and not state.get("is_quantized", False):
        if isinstance(state.get("v_low"), Tensor) and state.get("v_low_q") is None and v_quant_type != 'fp32':
            _quant_v(state, "v_low", state["v_low"], curr_block_size, v_quant_type)
            state["v_low"] = None
        
        if "m_low" in state and state.get("m_low_q") is None and m_quant_type != 'fp32':
            local_m_block_size = state.get("m_block_size", m_curr_block_size)
            state["m_low_q"], state["m_low_scale"] = _m_quantize(state["m_low"], m_quant_type, local_m_block_size)
            state["m_block_size"] = local_m_block_size
            state["m_quant_type"] = m_quant_type
            state.pop("m_low", None)

        if v_quant_type != 'fp32':
            _v_keys = ("row_var", "col_var") + (("conf_row", "conf_col") if beta3 is not None else ())
            for _k in _v_keys:
                if _k in state and f"{_k}_q" not in state:
                    _quant_v(state, _k, state.pop(_k), curr_block_size, v_quant_type)
            if "variance" in state and "variance_q" not in state:
                _quant_v(state, "variance", state.pop("variance"), curr_block_size, v_quant_type)
                
        if beta1 is not None and "m_q" not in state and m_quant_type != 'fp32':
            m_block_size_local = state.get("m_block_size", m_curr_block_size)
            if "m" in state:
                state["m_q"], state["m_scale"] = _m_quantize(state["m"], m_quant_type, m_block_size_local)
                state.pop("m")
                state["m_quant_type"] = m_quant_type
            else:
                _m_init_state(state, p.numel(), m_block_size_local, m_quant_type, p.device, shape=p.shape)
            state["m_block_size"] = m_block_size_local
            
        state["is_quantized"] = True
    
    elif not use_quant and state.get("is_quantized", False):
        if isinstance(state.get("v_low_q"), Tensor):
            state["v_low"] = _deq_v(state, "v_low", pop=True)

        if "m_low_q" in state:
            logger.warning("Adafactor8Bit: Apollo m_low_q discarded due to quantize flag change. Momentum state will be reset.")
            state.pop("m_low_q", None)
            state.pop("m_low_scale", None)
            state.pop("m_block_size", None)

        for _k in ("row_var", "col_var", "conf_row", "conf_col", "variance"):
            if f"{_k}_q" in state:
                state[_k] = _deq_v(state, _k, pop=True)
        
        if "m_q" in state:
            m_block_size_local = state.get("m_block_size", m_curr_block_size)
            old_mqt = state.get("m_quant_type", 'uf8')
            state["m"] = _m_dequantize(
                state["m_q"], state["m_scale"],
                p.numel(), p.shape, m_block_size_local, p.device, old_mqt
            )
            state.pop("m_q", None)
            state.pop("m_scale", None)
            state.pop("m_block_size", None)
            state.pop("m_quant_type", None)
            
        state["is_quantized"] = False


def _dispatch_cuda_clip(state, device, alpha, d, enable_fira, fira_margin, lag_norm,
                        launch_noclip, launch_lag, launch_norm, launch_apply):
    """Unified dispatcher for noclip / lag / norm+apply CUDA kernel patterns."""
    need_norm = (d > 0.0) or enable_fira
    use_lag = lag_norm and need_norm

    if not need_norm:
        launch_noclip(alpha)
    elif use_lag:
        if "prev_update_norm" not in state:
            state["prev_update_norm"] = torch.zeros(1, device=device, dtype=torch.float32)
        alpha_eff = alpha
        if enable_fira:
            lag_fs = state.get("lag_fira_scale")
            if lag_fs is not None:
                alpha_eff = alpha * lag_fs
        norm_buf = _get_norm_buf(device)
        launch_lag(norm_buf, alpha_eff, state["prev_update_norm"])
        
        curr_norm = norm_buf.sqrt()
        state["prev_update_norm"] = curr_norm
        if enable_fira:
            state["lag_fira_scale"] = _compute_fira_scale(state, curr_norm.view([]), fira_margin, device)
    else:
        norm_buf = _get_norm_buf(device)
        launch_norm(norm_buf)
        if enable_fira:
            alpha, norm_buf = _apply_fira_cuda(state, norm_buf, alpha, fira_margin)
        launch_apply(norm_buf, alpha)


def _dispatch_1d_cuda(state, param_flat, grad_flat, m_q_flat, m_scale_flat, 
                      v_q_flat, v_scale_flat, v_min_log_flat, alpha, 
                      beta1, beta_val, eps_for_denom, use_adam_denom, 
                      log_eps_sq, eps_for_grad_sq, numel, curr_block_size, 
                      m_curr_block_size, m_bits, m_mode, d, v_bits, 
                      enable_fira, fira_margin, lag_norm, has_m, momentum_only_1d):
    device = param_flat.device
    
    def l_noclip(a):
        if has_m:
            _CUDA_MODULE.fused_update_1d_noclip(
                param_flat, grad_flat, m_q_flat, m_scale_flat, v_q_flat, v_scale_flat, v_min_log_flat, a,
                beta1, beta_val, eps_for_denom, use_adam_denom, log_eps_sq, eps_for_grad_sq,
                numel, curr_block_size, m_curr_block_size, m_bits, m_mode, v_bits, momentum_only_1d)
        else:
            _CUDA_MODULE.fused_update_1d_vonly_noclip(
                param_flat, grad_flat, v_q_flat, v_scale_flat, v_min_log_flat, a,
                beta_val, eps_for_denom, use_adam_denom, log_eps_sq, eps_for_grad_sq, numel, curr_block_size, v_bits)

    def l_lag(nb, a, pn):
        if has_m:
            _CUDA_MODULE.fused_update_1d_lag(
                param_flat, grad_flat, m_q_flat, m_scale_flat, v_q_flat, v_scale_flat, v_min_log_flat, nb, a, pn,
                beta1, beta_val, eps_for_denom, use_adam_denom, log_eps_sq, eps_for_grad_sq,
                numel, curr_block_size, m_curr_block_size, m_bits, m_mode, d, v_bits, momentum_only_1d)
        else:
            _CUDA_MODULE.fused_update_1d_vonly_lag(
                param_flat, grad_flat, v_q_flat, v_scale_flat, v_min_log_flat, nb, a, pn,
                beta_val, eps_for_denom, use_adam_denom, log_eps_sq, eps_for_grad_sq, numel, curr_block_size, d, v_bits)

    def l_norm(nb):
        v_buf = _get_v_buf(device, numel)
        if has_m:
            _CUDA_MODULE.fused_update_1d_norm(
                grad_flat, m_q_flat, m_scale_flat, v_q_flat, v_scale_flat, v_min_log_flat, v_buf, nb,
                beta1, beta_val, eps_for_denom, use_adam_denom, log_eps_sq, eps_for_grad_sq,
                numel, curr_block_size, m_curr_block_size, m_bits, m_mode, v_bits, momentum_only_1d)
        else:
            _CUDA_MODULE.fused_update_1d_vonly_norm(
                grad_flat, v_q_flat, v_scale_flat, v_min_log_flat, v_buf, nb,
                beta_val, eps_for_denom, use_adam_denom, log_eps_sq, eps_for_grad_sq, numel, curr_block_size, v_bits)

    def l_apply(nb, a):
        v_buf = _get_v_buf(device, numel)
        if has_m:
            _CUDA_MODULE.fused_update_1d_apply(
                param_flat, grad_flat, m_q_flat, m_scale_flat, v_q_flat, v_scale_flat, v_min_log_flat, v_buf, nb, a,
                beta1, beta_val, eps_for_denom, use_adam_denom, log_eps_sq, eps_for_grad_sq,
                numel, curr_block_size, m_curr_block_size, m_bits, m_mode, d, v_bits, momentum_only_1d)
        else:
            _CUDA_MODULE.fused_update_1d_vonly_apply(
                param_flat, grad_flat, v_q_flat, v_scale_flat, v_min_log_flat, v_buf, nb, a,
                beta_val, eps_for_denom, use_adam_denom, log_eps_sq, eps_for_grad_sq, numel, curr_block_size, d, v_bits)

    _dispatch_cuda_clip(state, device, alpha, d, enable_fira, fira_margin, lag_norm,
                        l_noclip, l_lag, l_norm, l_apply)


# ==========================================
# 11. Optimizer Core
# ==========================================
class Adafactor8Bit(Optimizer):
    """
    8-bit Adafactor optimizer with fused CUDA kernels for memory-efficient large-scale training.
    
    Args:
        params (Iterable): Iterable of parameters to optimize or dictionaries defining parameter groups.
        
        **Core Optimization**

        lr (float, optional): External learning rate. Defaults to 1e-2.
        beta1 (float, optional): Momentum coefficient for first moment. 
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
            - `eps1`: Added to the squared gradient. Defaults to 1e-30.
            - `eps2`: Lower threshold for parameter RMS scaling. Defaults to 1e-3.
        eps_came (float): Lower bound for the residual term in CAME's confidence estimation.
            Matches official CAME's ``eps[1]``. Independent of the variance ``eps``.
            Defaults to 1e-16.
        weight_decay (float): Weight decay (L2 penalty). Defaults to 0.0.
        d (float): Clipping threshold for the final gradient update RMS. 
            Setting to 0 disables RMS clipping entirely. Defaults to 1.0.
        maximize (bool): Maximize the params based on the objective. Defaults to False.
            
        **Factorization & Scaling**

        relative_step (bool): If `True`, uses time-dependent learning rate. Defaults to True.
        scale_parameter (bool): If `True`, scales learning rate by parameter RMS. 
            Setting to False decouples the step size from parameter magnitude, which can be useful 
            for sparse layers like Embeddings to ensure sufficient update strength. Defaults to True.
        factored (bool): Whether to use row/col factorization for >=2D tensors. 
            Setting to False uses element-wise variance (like RMSProp, but still applies Adafactor's 
            global RMS clipping). This can be useful for preserving spatial structure in >2D tensors 
            such as CNN convolutions, or enabling per-element updates in Embeddings. Defaults to True.
            
        **Quantization Control**

        quantize (bool): Enable 8-bit log-space quantization for optimizer states. Defaults to True.
        block_size (int): Block size for variance quantization. 
            Must be >= 128 and a multiple of 128. Defaults to 2048.
        m_block_size (int): Block size for momentum quantization. 
            Must be a multiple of 4 and >= 32. When quantize=True with beta1 
            and m_quant_type != 'fp32', block_size must be a multiple of 
            m_block_size (required by the fused 1D kernel). Defaults to 256.
        m_quant_type (str): Quantization scheme for momentum state.
            'uf4': Uniform 4-bit, 16 equally-spaced levels, packed 2 per byte.
            'uf8': Uniform 8-bit, 256 equally-spaced levels, 1 byte per element.
            'd4': Dynamic map 4-bit, non-uniform, packed 2 per byte.
            'd8': Dynamic map 8-bit (Dettmers et al., 2022), non-uniform, 1 byte per element.
            'fp32': Full precision float32 momentum.
            Defaults to 'uf8'.
        v_quant_type (str): Quantization scheme for the second moment (variance).
            'al8': Adaptive log-space 8-bit. 255 non-zero levels + 1 reserved zero.
            'al16': Adaptive log-space 16-bit. 65535 non-zero levels + 1 reserved zero.
            'fp32': Full precision float32 variance.
            Defaults to 'al8'.
        min_8bit_size (int): Minimum number of elements to apply quantization. Defaults to 4096.
        use_cuda_kernel (bool): Whether to use custom CUDA kernels. Defaults to True.
            
        **APOLLO Low-Rank Projection**

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
        apollo_eps (float): Epsilon for APOLLO low-rank variance normalization to prevent division by zero. 
            Independent of `eps1`. Defaults to 1e-6.
        apollo_factorize (bool): If True, applies Adafactor-style row/col factorization 
            within the low-rank space (FP32, ~16KB state) instead of full matrix variance 
            (8-bit, ~100KB+ state) to drastically reduce optimizer state memory. Defaults to False.
        apollo_cache_proj (bool): If ``True``, caches the APOLLO random projection matrix in the 
            optimizer state to avoid the computational overhead of regenerating it at every step. 
            Defaults to False.
        enable_fira_for_apollo (bool): If ``True`` (default), enables the Fira Norm-Growth Limiter 
            in the APOLLO path to prevent destructive gradient updates. Set to ``False`` to match 
            the official APOLLO ``disable_nl=True`` behavior. Defaults to True.

        **Stabilizers & Regularization**

        scale_weight_decay (bool): If `True` (default), weight decay is coupled with the 
            parameter's RMS scale. If `False`, decoupled (AdamW-style).
        enable_fira_for_adafactor (bool): If `True`, enables Fira Limiter to prevent gradient 
            explosion by smoothing update norms. Defaults to False.
        fira_margin (float): The tolerance margin for Fira Limiter (e.g., 0.01 for 1%). 
            Shared with Apollo path. Defaults to 0.01.
        lag_norm (bool): If True and d > 0, uses the previous step's update norm for RMS 
            clipping instead of computing the exact norm (single kernel pass, faster). 
            Defaults to False.
        momentum_only_1d (bool, optional): Controls the 1D update rule when momentum (beta1) is enabled.
            - If True (Adafactor/CAME style): Computes normalized gradient U_t = grad / sqrt(V), 
              updates momentum M = b1*M + (1-b1)*U_t, and uses M directly as the final update.
            - If False (Adam style): Updates momentum M = b1*M + (1-b1)*grad, and uses 
              M / sqrt(V) as the final update.
            - If None (default): Automatically selects True for Adafactor/CAME configurations 
              and False for Adam configurations.
            Note: This flag has no effect when beta1 is None (no momentum).
        bias_correction (bool, optional): Override the default bias correction behavior.
            - If `None` (default), uses algorithm-specific defaults (Enabled for Adam/Apollo, 
              Disabled for Adafactor/CAME to match official implementations).
            - If `True`, forces bias correction on all momentum-based paths.
            - If `False`, forces bias correction off.

        **Memory Management**

        warmup_multiplier (float): Multiplier for the maximum parameter size to 
            pre-allocate and free on the first step, priming the CUDA caching 
            allocator to prevent external fragmentation (Segment Proliferation). 
            Set to 0.0 (default) to disable. A value of 2.0 is recommended as a 
            workaround for PyTorch versions lacking `expandable_segments` support 
            on Windows. Defaults to 0.0.
    """
    
    def __init__(
        self,
        params: Iterable[Union[Tensor, Dict[str, Any]]],
        # --- Core Optimization ---
        lr: float = 1e-2,
        beta1: Optional[float] = None,
        beta2: Optional[float] = None,
        beta2_decay: Optional[float] = -0.8,
        beta3: Optional[float] = None,
        eps_came: float = 1e-16,
        eps: Tuple[Optional[float], float] = (1e-30, 1e-3),
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
        m_block_size: int = 256,
        m_quant_type: str = 'uf8',
        v_quant_type: str = 'al8',
        min_8bit_size: int = 4096,
        use_cuda_kernel: bool = True,
        # --- APOLLO Low-Rank Projection ---
        apollo_rank: int = 0,
        apollo_update_proj_gap: int = 200,
        apollo_scale_type: str = 'channel',
        apollo_scale: float = 1.0,
        apollo_scale_front: bool = False,
        apollo_eps: float = 1e-6,
        apollo_factorize: bool = False,
        apollo_cache_proj: bool = False,
        enable_fira_for_apollo: bool = True,
        # --- Stabilizers & Regularization ---
        scale_weight_decay: bool = True,
        enable_fira_for_adafactor: bool = False,
        fira_margin: float = 0.01,
        lag_norm: bool = False,
        momentum_only_1d: Optional[bool] = None,
        bias_correction: Optional[bool] = None,
        # --- Memory Management ---
        warmup_multiplier: float = 0.0,
    ):
        
        if lr < 0.0: raise ValueError(f"Invalid lr: {lr}, must be >= 0.0")
        if beta1 is not None and (beta1 < 0.0 or beta1 >= 1.0):
            raise ValueError(f"Invalid beta1: {beta1}, must be in [0.0, 1.0)")
        
        if beta2_decay is not None and beta2_decay > 0.0: raise ValueError(f"Invalid beta2_decay: {beta2_decay}, must be <= 0.0")

        is_sgd_mode = (beta2 is None) and (beta2_decay is None)
        if is_sgd_mode:
            if relative_step:
                raise ValueError(
                    "When `beta2` and `beta2_decay` are both None, `relative_step` must be False. "
                    "Please set `relative_step=False` for standard SGD, or specify `beta2`/`beta2_decay` to use adaptive optimizers."
                )
            if apollo_rank > 0:
                raise ValueError(
                    "APOLLO requires a second-moment estimate (`beta2` or `beta2_decay`), but both are currently None."
                )
            if beta3 is not None:
                raise ValueError(
                    "CAME (`beta3`) requires both `beta1` and `beta2`, but `beta2` is currently None."
                )

        eps1, eps2 = eps
        if eps1 is None:
            eps1 = 1e-30
        if eps1 < 0.0: raise ValueError(f"Invalid eps1: {eps1}, must be >= 0.0")
        if eps2 < 0.0: raise ValueError(f"Invalid eps2: {eps2}, must be >= 0.0")
        if d < 0.0: raise ValueError(f"Invalid d: {d}, must be >= 0.0. d=0 disables RMS clipping.")
        if weight_decay < 0.0: raise ValueError(f"Invalid weight_decay: {weight_decay}, must be >= 0.0")

        if beta2 is not None and (beta2 < 0.0 or beta2 >= 1.0):
            raise ValueError(f"Invalid beta2: {beta2}, must be in [0.0, 1.0)")
        
        if quantize:
            if block_size < 128 or block_size % 128 != 0:
                raise ValueError(f"block_size must be >= 128 and a multiple of 128, but got {block_size}.")
            
        if m_block_size < 32 or m_block_size % 4 != 0:
            raise ValueError(f"m_block_size must be a multiple of 4 and >= 32, but got {m_block_size}.")

        if m_quant_type not in _VALID_M_QUANT_TYPES:
            raise ValueError(f"m_quant_type must be one of {_VALID_M_QUANT_TYPES}, got '{m_quant_type}'.")

        if v_quant_type not in _VALID_V_QUANT_TYPES:
            raise ValueError(f"v_quant_type must be one of {_VALID_V_QUANT_TYPES}, got '{v_quant_type}'.")

        if quantize and beta1 is not None and m_quant_type != 'fp32':
            if block_size % m_block_size != 0:
                raise ValueError(
                    f"For the fused 1D kernel, block_size must be a multiple of m_block_size "
                    f"(got block_size={block_size}, m_block_size={m_block_size})."
                )

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

        if momentum_only_1d is None:
            is_adam_mode = (beta2 is not None) and (not relative_step) and (beta3 is None)
            momentum_only_1d = not is_adam_mode

        defaults = dict(
            # Core Optimization
            lr=lr, beta1=beta1, beta2=beta2, beta2_decay=beta2_decay, beta3=beta3, eps_came=eps_came,
            eps=eps, weight_decay=weight_decay, d=d, maximize=maximize,
            # Factorization & Scaling
            relative_step=relative_step, scale_parameter=scale_parameter, factored=factored,
            # Quantization Control
            quantize=quantize, block_size=block_size, m_block_size=m_block_size, 
            m_quant_type=m_quant_type, v_quant_type=v_quant_type,
            min_8bit_size=min_8bit_size, use_cuda_kernel=use_cuda_kernel,
            # APOLLO Low-Rank Projection
            apollo_rank=apollo_rank, apollo_update_proj_gap=apollo_update_proj_gap,
            apollo_scale_type=apollo_scale_type, apollo_scale=apollo_scale, 
            apollo_scale_front=apollo_scale_front, apollo_eps=apollo_eps, 
            apollo_factorize=apollo_factorize, apollo_cache_proj=apollo_cache_proj,
            enable_fira_for_apollo=enable_fira_for_apollo,
            # Stabilizers & Regularization
            scale_weight_decay=scale_weight_decay, 
            enable_fira_for_adafactor=enable_fira_for_adafactor, fira_margin=fira_margin,
            lag_norm=lag_norm, momentum_only_1d=momentum_only_1d, bias_correction=bias_correction,
            # Memory Management
            warmup_multiplier=warmup_multiplier,
        )
        super().__init__(params, defaults)

        self._apollo_seed_counter = 1

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['_apollo_seed_counter'] = self._apollo_seed_counter
        return state_dict

    def load_state_dict(self, state_dict):
        old_vqt, old_mqt = {}, {}
        for idx, st in state_dict.get('state', {}).items():
            if isinstance(st, dict):
                old_vqt[idx] = st.get('v_quant_type', 'al8')
                old_mqt[idx] = st.get('m_quant_type', 'uf8')

        self._apollo_seed_counter = state_dict.get('_apollo_seed_counter', 0)
        super().load_state_dict(state_dict)

        param_idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state:
                    st = self.state[p]
                    st["v_quant_type"] = old_vqt.get(param_idx, 'al8')
                    st["m_quant_type"] = old_mqt.get(param_idx, 'uf8')
                    s = st.get("step")
                    if torch.is_tensor(s):
                        st["step"] = int(s)
                param_idx += 1
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
            if k.endswith("_scale") or k.endswith("_min_log"):
                target_dtype = torch.float32
            elif k.endswith("_q"):
                st = self.state[p]
                if "m_q" in k or "m_low_q" in k:
                    target_dtype = torch.uint8
                else:
                    v_qt = st.get("v_quant_type", 'al8')
                    target_dtype = torch.int16 if v_qt == 'al16' else torch.uint8
            else:
                target_dtype = cpu_tensor.dtype
            self.state[p][k] = cpu_tensor.to(
                device=p.device, dtype=target_dtype
            ).contiguous()

    def _warmup_allocator(self):
        """
        Primes the CUDA caching allocator to maintain healthy memory segments.
        
        Pre-allocates and immediately frees a contiguous block sized to the largest
        parameter on each active device multiplied by `warmup_multiplier`. This 
        ensures the allocator retains large reusable segments, preventing external 
        fragmentation during training (especially useful on older PyTorch versions 
        lacking `expandable_segments`).
        """
        multiplier = self.param_groups[0].get("warmup_multiplier", 0.0)
        if multiplier <= 0.0 or not torch.cuda.is_available():
            return
        
        device_max_numel = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.is_cuda:
                    dev = p.device
                    if dev not in device_max_numel or p.numel() > device_max_numel[dev]:
                        device_max_numel[dev] = p.numel()
        
        for dev, max_numel in device_max_numel.items():
            dummy = torch.empty(int(max_numel * multiplier), device=dev, dtype=torch.float32)
            del dummy

    def _init_group(self, group, params_with_grad, grads, states):

        group_quantize = group.get("quantize", True)
        block_size = group.get("block_size", 2048)
        m_block_size = group.get("m_block_size", 256)
        m_quant_type = group.get("m_quant_type", 'uf8')
        v_quant_type = group.get("v_quant_type", 'al8')
        min_8bit_size = group.get("min_8bit_size", 4096)
        apollo_rank = group.get("apollo_rank", 0)
        apollo_factorize = group.get("apollo_factorize", False)
        factored = group.get("factored", True)
        beta1 = group.get("beta1")
        beta2 = group.get("beta2")
        beta2_decay = group.get("beta2_decay")
        beta3 = group.get("beta3")
        
        is_sgd_mode = (beta2 is None) and (beta2_decay is None)

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
                old_mqt = state.get("m_quant_type", 'uf8')
                if old_mqt not in _VALID_M_QUANT_TYPES:
                    state["m_quant_type"] = 'uf8'
                    old_mqt = 'uf8'
                old_mbs = state.get("m_block_size", m_block_size)
                if not isinstance(old_mbs, int):
                    state["m_block_size"] = m_block_size
                    old_mbs = m_block_size
                if (old_mqt != m_quant_type or old_mbs != m_block_size) and use_quant:
                    logger.warning(f"Adafactor8Bit: M config changed (type: '{old_mqt}'->'{m_quant_type}', block: {old_mbs}->{m_block_size}) for param shape {p.shape}. Momentum history will be reset to zero.")
                    state.pop("m_q", None)
                    state.pop("m_scale", None)
                    state.pop("m_block_size", None)
                    state.pop("m", None)
                    state.pop("m_low_q", None)
                    state.pop("m_low_scale", None)
                    state.pop("m_low", None)
                    state["m_quant_type"] = m_quant_type
                for mq_key in ("m_q", "m_low_q"):
                    if mq_key in state and state[mq_key].dtype != torch.uint8:
                        logger.warning(f"Adafactor8Bit: Momentum dtype mismatch (Current: {state[mq_key].dtype}, Target: torch.uint8) for {mq_key}. Re-initializing.")
                        state.pop(mq_key, None)
                        state.pop(mq_key.replace("_q", "_scale"), None)
                        state.pop("m_block_size", None)
                        state.pop("m_quant_type", None)
                old_vqt = state.get("v_quant_type", 'al8')
                if old_vqt != v_quant_type and use_quant:
                    logger.warning(f"Adafactor8Bit: V config changed ('{old_vqt}'->'{v_quant_type}') for param shape {p.shape}. Re-initializing state.")
                    _reset_keep_step(state)
                    needs_init = True

                is_apollo_state = ('apollo_seed' in state)
                if (use_apollo and not is_apollo_state) or (not use_apollo and is_apollo_state):
                    logger.warning(f"Adafactor8Bit: Apollo/Adafactor mode changed for param shape {p.shape}. Re-initializing state.")
                    _reset_keep_step(state)
                    needs_init = True
                elif use_apollo and is_apollo_state:
                    if (state.get("apollo_rank") != apollo_rank or 
                        state.get("apollo_factorize", False) != apollo_factorize or
                        state.get("beta3") != beta3):
                        logger.warning(f"Adafactor8Bit: Apollo/CAME config changed for param shape {p.shape}. Re-initializing state.")
                        _reset_keep_step(state)
                        needs_init = True
                elif not is_sgd_mode:
                    state_is_factored = ("row_var" in state or "row_var_q" in state)
                    
                    if state_is_factored != expected_is_factored:
                        logger.warning(f"Adafactor8Bit: factored mode mismatch for param shape {p.shape}. Re-initializing state.")
                        _reset_keep_step(state)
                        needs_init = True

            if needs_init:
                if "step" not in state:
                    state["step"] = 0
                state["is_quantized"] = use_quant
                state["block_size"] = block_size
                state["v_quant_type"] = v_quant_type

                if is_sgd_mode:
                    if beta1 is not None:
                        if use_quant and m_quant_type != 'fp32':
                            _m_init_state(state, p.numel(), m_block_size, m_quant_type, p.device)
                        else:
                            state["m"] = torch.zeros_like(p.grad, dtype=torch.float32, device=p.device, memory_format=torch.preserve_format)
                            state["m_quant_type"] = 'fp32'
                elif use_apollo:
                    seed = self._apollo_seed_counter
                    self._apollo_seed_counter += 1
                    update_proj_gap = group.get("apollo_update_proj_gap", 200)
                    state["apollo_seed"] = seed
                    state["apollo_rank"] = apollo_rank
                    state["apollo_factorize"] = apollo_factorize
                    state["beta3"] = beta3
                    state["apollo_update_proj_gap"] = update_proj_gap
                    state["last_proj_step"] = -update_proj_gap
                    state["apollo_proj_matrix_T"] = None
                    state["m_quant_type"] = m_quant_type
                    for k in ("v_low_q", "v_low_scale", "v_low_min_log", "v_low_shape", "v_low_pad", "v_low"):
                        state[k] = None
                    if beta3 is not None:
                        if apollo_factorize:
                            for k in ("conf_row_low_q", "conf_row_low_scale", "conf_row_low_min_log", "conf_row_low_shape", "conf_row_low_pad",
                                      "conf_col_low_q", "conf_col_low_scale", "conf_col_low_min_log", "conf_col_low_shape", "conf_col_low_pad",
                                      "conf_row_low", "conf_col_low"):
                                state[k] = None
                        else:
                            for k in ("res_low_q", "res_low_scale", "res_low_min_log", "res_low_shape", "res_low_pad", "res_low"):
                                state[k] = None

                else:
                    if expected_is_factored:
                        shape = p.grad.shape
                        R = shape[-2]
                        C = shape[-1]
                        batch_shape = shape[:-2]
                        
                        r_shape = list(batch_shape) + [R, 1]
                        c_shape = list(batch_shape) + [1, C]
                        
                        _init_v_or_fp32(state, "row_var", r_shape, block_size, p.device, v_quant_type, use_quant)
                        _init_v_or_fp32(state, "col_var", c_shape, block_size, p.device, v_quant_type, use_quant)

                        if beta1 is not None:
                            _init_m_or_fp32(state, p.numel(), m_block_size, m_quant_type, p.device, p.shape, use_quant)
                                
                        if beta3 is not None:
                            _init_v_or_fp32(state, "conf_row", r_shape, block_size, p.device, v_quant_type, use_quant)
                            _init_v_or_fp32(state, "conf_col", c_shape, block_size, p.device, v_quant_type, use_quant)

                    else:
                        _init_v_or_fp32(state, "variance", p.grad.shape, block_size, p.device, v_quant_type, use_quant)
                        if beta1 is not None:
                            _init_m_or_fp32(state, p.numel(), m_block_size, m_quant_type, p.device, p.shape, use_quant)
            else:
                for k in list(state.keys()):
                    if isinstance(state[k], torch.Tensor):
                        if state[k].device != p.device:
                            state[k] = state[k].to(p.device)
                        if k.endswith("_q"):
                            if "m_q" in k or "m_low_q" in k:
                                if state[k].dtype != torch.uint8:
                                    state[k] = state[k].to(torch.uint8)
                            else:
                                v_qt = state.get("v_quant_type", 'al8')
                                expected = torch.int16 if v_qt == 'al16' else torch.uint8
                                if state[k].dtype != expected:
                                    state[k] = state[k].to(expected)
                        if (k.endswith("_scale") or k.endswith("_min_log")) and state[k].dtype != torch.float32:
                            state[k] = state[k].to(torch.float32)

                curr_block_size = state.get("block_size", block_size)
                m_curr_block_size = state.get("m_block_size", m_block_size) if beta1 is not None else 0

                _migrate_quantize_flag(state, use_quant, curr_block_size, m_curr_block_size, m_quant_type, v_quant_type, beta1, beta3, p)

                if beta1 is not None and "m_q" not in state:
                    m_curr_block_size = state.get("m_block_size", m_block_size)
                    if use_quant and m_quant_type != 'fp32':
                        if "m" in state:
                            state["m_q"], state["m_scale"] = _m_quantize(state["m"], m_quant_type, m_curr_block_size)
                            state.pop("m")
                            state["m_quant_type"] = m_quant_type
                        else:
                            _m_init_state(state, p.numel(), m_curr_block_size, m_quant_type, p.device, shape=p.shape)
                        state["m_block_size"] = m_curr_block_size
                    elif "m" not in state and m_quant_type == 'fp32':
                        state["m"] = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
                        state["m_quant_type"] = 'fp32'

                if "is_quantized" not in state:
                    if use_apollo:
                        state["is_quantized"] = use_quant
                    else:
                        state["is_quantized"] = ("variance_q" in state or "row_var_q" in state or "m_q" in state)
                if "block_size" not in state:
                    state["block_size"] = block_size
                if beta1 is None:
                    state.pop("m_q", None)
                    state.pop("m_scale", None)
                    state.pop("m_block_size", None)
                    state.pop("m_quant_type", None)
                    state.pop("m", None)

            states.append(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        if not hasattr(self, '_allocator_warmed_up'):
            self._warmup_allocator()
            self._allocator_warmed_up = True

        for group in self.param_groups:
            params_with_grad, grads, states = [], [], []
            eps1, eps2 = group["eps"]
            self._init_group(group, params_with_grad, grads, states)

            if _CUDA_AVAILABLE and params_with_grad:
                _ensure_qmap(params_with_grad[0].device)

            apollo_rank = group.get("apollo_rank", 0)

            for i in range(len(params_with_grad)):
                if apollo_rank > 0 and 'apollo_seed' in states[i]:
                    _update_param_apollo(
                        params_with_grad[i], grads[i], states[i],
                        d=group["d"], lr=group["lr"], beta1=group.get("beta1"),
                        beta2=group["beta2"], beta2_decay=group["beta2_decay"],
                        weight_decay=group["weight_decay"], eps2=eps2,
                        maximize=group["maximize"], relative_step=group["relative_step"],
                        scale_parameter=group["scale_parameter"],
                        scale_weight_decay=group.get("scale_weight_decay", True),
                        block_size=group.get("block_size", 2048),
                        m_block_size=group.get("m_block_size", 256),
                        use_cuda_kernel=group.get("use_cuda_kernel", True),
                        apollo_scale_type=group.get("apollo_scale_type", "channel"),
                        apollo_scale=group.get("apollo_scale", 1.0),
                        apollo_scale_front=group.get("apollo_scale_front", False),
                        apollo_eps=group.get("apollo_eps", 1e-6),
                        apollo_factorize=group.get("apollo_factorize", False),
                        apollo_cache_proj=group.get("apollo_cache_proj", False),
                        enable_fira_for_apollo=group.get("enable_fira_for_apollo", True),
                        fira_margin=group.get("fira_margin", 0.01),
                        eps_came=group.get("eps_came", 1e-16),
                        beta3=group.get("beta3"), bias_correction=group.get("bias_correction"),
                    )
                else:
                    _update_param_8bit(
                        params_with_grad[i], grads[i], states[i],
                        d=group["d"], lr=group["lr"], beta1=group.get("beta1"),
                        beta2=group["beta2"], beta2_decay=group["beta2_decay"],
                        weight_decay=group["weight_decay"], eps1=eps1, eps2=eps2,
                        maximize=group["maximize"], relative_step=group["relative_step"],
                        scale_parameter=group["scale_parameter"],
                        scale_weight_decay=group.get("scale_weight_decay", True),
                        block_size=group.get("block_size", 2048),
                        m_block_size=group.get("m_block_size", 256),
                        use_cuda_kernel=group.get("use_cuda_kernel", True),
                        enable_fira_for_adafactor=group.get("enable_fira_for_adafactor", False),
                        fira_margin=group.get("fira_margin", 0.01),
                        lag_norm=group.get("lag_norm", False), momentum_only_1d=group.get("momentum_only_1d", False), 
                        factored=group.get("factored", True),
                        beta3=group.get("beta3"), eps_came=group.get("eps_came", 1e-16),
                        bias_correction=group.get("bias_correction"),
                    )
        return loss


# ==========================================
# 12. Parameter Update Entry Points
# ==========================================
def _update_param_8bit(
    param: Tensor, grad: Tensor, state: Dict[str, Any],
    d: float, lr: Union[float, Tensor],
    beta1: Optional[float], beta2: Optional[float], beta2_decay: float,
    weight_decay: float, eps1: Optional[float], eps2: float,
    maximize: bool, relative_step: bool, scale_parameter: bool, scale_weight_decay: bool,
    block_size: int, m_block_size: int, use_cuda_kernel: bool,
    enable_fira_for_adafactor: bool = False, fira_margin: float = 0.01,
    lag_norm: bool = False, momentum_only_1d: bool = False, factored: bool = True,
    beta3: Optional[float] = None, eps_came: float = 1e-16,
    bias_correction: Optional[bool] = None,
):
    if eps1 is None:
        eps1 = 1e-30
    eps_sq = max(eps1 * eps1, _LOG_QUANT_FLOOR)
    log_eps_sq = math.log2(eps_sq)
    _cuda_ready = _load_cuda_module(use_cuda_kernel)

    use_adam_denom = (not factored) and (beta2 is not None) and (not relative_step) and (beta3 is None)
    eps_for_denom = eps1 if use_adam_denom else 0.0
    eps_for_grad_sq = eps1 if not use_adam_denom else 0.0

    grad_contig = grad.contiguous()
    if maximize:
        grad_contig = grad_contig.neg()

    quantize = state.get("is_quantized", False)
    v_quant_type = state.get("v_quant_type", 'al8')
    v_bits = 16 if v_quant_type == 'al16' else 8
    curr_block_size = state.get("block_size", block_size)
    m_curr_block_size = state.get("m_block_size", m_block_size) if beta1 is not None else 0
    if beta1 is not None:
        m_quant_type = state.get("m_quant_type", 'uf8')
        m_bits = _get_m_bits(m_quant_type)
        m_mode = _get_m_mode(m_quant_type)
    else:
        m_quant_type = 'uf8'
        m_bits = 0
        m_mode = 0

    N = grad_contig.numel()
    grad_fp32 = None
    
    def _get_safe_grad_fp32():
        nonlocal grad_fp32
        if grad_fp32 is None:
            grad_fp32 = grad_contig.float()
            if grad_fp32.data_ptr() == grad_contig.data_ptr():
                grad_fp32 = grad_fp32.clone()
        return grad_fp32

    if not param.is_contiguous():
        param_work = param.contiguous()
        needs_copy_back = True
    else:
        param_work = param
        needs_copy_back = False

    step = state["step"] + 1
    state["step"] = step
    is_sgd = (beta2 is None) and (beta2_decay is None)

    if is_sgd:
        beta_val = 0.0
    elif beta2 is not None:
        beta_val = 1.0 - beta2
    else:
        beta_val = math.pow(step, beta2_decay)
    alpha, rho_t = _compute_alpha(param_work, lr, step, relative_step, scale_parameter, eps2)

    if beta1 is not None and not relative_step and not is_sgd:
        apply_bc = use_adam_denom if bias_correction is None else bias_correction
        if apply_bc:
            bc1 = 1.0 - beta1 ** step
            if beta2 is not None:
                bc2 = 1.0 - beta2 ** step
                alpha = alpha * (math.sqrt(bc2) / bc1)
                if use_adam_denom:
                    eps_for_denom = eps1 * math.sqrt(bc2)
            else:
                alpha = alpha / bc1

    if weight_decay != 0:
        wd_multiplier = alpha if scale_weight_decay else rho_t
        param_work.mul_(1.0 - (wd_multiplier * weight_decay))

    # ================================================================
    # SGD path
    # ================================================================
    is_full_rank = (grad_contig.dim() < 2) or (not factored) or is_sgd

    if is_sgd:
        grad_fp32 = _get_safe_grad_fp32()
        if beta1 is not None:
            if quantize and m_quant_type != 'fp32':
                if _cuda_ready:
                    _m_fused_quantize_lerp(state, grad_fp32.view(-1), beta1, m_curr_block_size, N, m_quant_type)
                    update = _m_dequantize(state["m_q"], state["m_scale"], N, grad_fp32.shape, m_curr_block_size, param_work.device, m_quant_type, use_cuda=use_cuda_kernel)
                else:
                    if "m_q" in state:
                        update = _m_dequantize(state["m_q"], state["m_scale"], N, grad_fp32.shape, m_curr_block_size, param_work.device, m_quant_type, use_cuda=False)
                    else:
                        update = torch.zeros_like(grad_fp32)
                    update.lerp_(grad_fp32, 1.0 - beta1)
                    state["m_q"], state["m_scale"] = _m_quantize(update, m_quant_type, m_curr_block_size)
            else:
                if "m" not in state:
                    state["m"] = torch.zeros_like(grad_fp32)
                state["m"].lerp_(grad_fp32, 1.0 - beta1)
                update = state["m"]
            update = update / (1.0 - beta1)
        else:
            update = grad_fp32

        _apply_update_pytorch(param_work, update, alpha, state, d, enable_fira_for_adafactor, fira_margin)
        if needs_copy_back:
            param.copy_(param_work)
        return

    # ================================================================
    # 1D full-rank path
    # ================================================================
    if is_full_rank:
        if _cuda_ready and quantize and v_quant_type != 'fp32' and m_quant_type == 'fp32' and beta1 is not None:
            grad_fp32 = _get_safe_grad_fp32()
            _v_fp32 = _get_v_buf(param_work.device, N)
            _fused_v_ema(state, "variance", grad_fp32.view(-1), _v_fp32, beta_val, curr_block_size,
                         square_input=True, eps=eps_for_grad_sq, log_floor=log_eps_sq)
            variance = _v_fp32.view(grad_fp32.shape)
            _fallback_1d_update(param_work, grad_fp32, variance, state, alpha, d, enable_fira_for_adafactor, fira_margin,
                                beta1, m_quant_type, m_curr_block_size, quantize, momentum_only_1d, use_adam_denom, eps_for_denom, eps_sq)
            if needs_copy_back:
                param.copy_(param_work)
            return

        if quantize and v_quant_type != 'fp32':
            if _cuda_ready and v_quant_type in ('al8', 'al16'):
                numel = param_work.numel()
                grad_flat = grad_contig.view(-1)
                variance_q_flat = state["variance_q"].view(-1)
                variance_scale_flat = state["variance_scale"].view(-1)
                variance_min_log_flat = state["variance_min_log"].view(-1)

                if beta1 is not None:
                    m_q_flat = state["m_q"].view(-1)
                    m_scale_flat = state["m_scale"].view(-1)
                    _dispatch_1d_cuda(
                        state, param_work.view(-1), grad_flat, m_q_flat, m_scale_flat,
                        variance_q_flat, variance_scale_flat, variance_min_log_flat,
                        alpha, beta1, beta_val, eps_for_denom, use_adam_denom,
                        log_eps_sq, eps_for_grad_sq, numel, curr_block_size,
                        m_curr_block_size, m_bits, m_mode, d, v_bits,
                        enable_fira_for_adafactor, fira_margin, lag_norm, has_m=True, momentum_only_1d=momentum_only_1d)
                else:
                    _dispatch_1d_cuda(
                        state, param_work.view(-1), grad_flat, None, None,
                        variance_q_flat, variance_scale_flat, variance_min_log_flat,
                        alpha, None, beta_val, eps_for_denom, use_adam_denom,
                        log_eps_sq, eps_for_grad_sq, numel, curr_block_size,
                        0, 0, 0, d, v_bits,
                        enable_fira_for_adafactor, fira_margin, lag_norm, has_m=False, momentum_only_1d=momentum_only_1d)
            else:
                grad_fp32 = _get_safe_grad_fp32()
                grad_sq = grad_fp32.square().add_(eps_for_grad_sq)
                variance = _deq_v(state, "variance")
                variance.lerp_(grad_sq, beta_val)
                del grad_sq
                _quant_v(state, "variance", variance, curr_block_size, v_quant_type)
                _fallback_1d_update(param_work, grad_fp32, variance, state, alpha, d, enable_fira_for_adafactor, fira_margin,
                                    beta1, m_quant_type, m_curr_block_size, quantize, momentum_only_1d, use_adam_denom, eps_for_denom, eps_sq)

        # --- Full precision path (quantize=False or v_quant_type='fp32') ---
        else:
            grad_fp32 = _get_safe_grad_fp32()
            if _cuda_ready and not (quantize and m_quant_type != 'fp32'):
                var_flat = state["variance"]
                if not var_flat.is_contiguous():
                    var_flat = var_flat.contiguous()
                    state["variance"] = var_flat

                m_flat = None
                if beta1 is not None:
                    if "m" not in state:
                        state["m"] = torch.zeros_like(grad_fp32)
                    m_flat = state["m"]
                    if not m_flat.is_contiguous():
                        m_flat = m_flat.contiguous()
                        state["m"] = m_flat
                    m_flat = m_flat.view(-1)

                b1_val = beta1 if beta1 is not None else 0.0
                p_flat, g_flat, v_flat = param_work.view(-1), grad_fp32.view(-1), var_flat.view(-1)

                def l_noclip(a):
                    _CUDA_MODULE.fused_update_1d_full_noclip(
                        p_flat, g_flat, v_flat, m_flat, a,
                        b1_val, beta_val, eps_sq, eps_for_denom, use_adam_denom, eps_for_grad_sq, N, momentum_only_1d)
                def l_lag(nb, a, pn):
                    _CUDA_MODULE.fused_update_1d_full_lag(
                        p_flat, g_flat, v_flat, m_flat, nb, a, pn,
                        b1_val, beta_val, eps_sq, eps_for_denom, use_adam_denom, eps_for_grad_sq, N, d, momentum_only_1d)
                def l_norm(nb):
                    _CUDA_MODULE.fused_update_1d_full_norm(
                        g_flat, v_flat, m_flat, nb,
                        b1_val, beta_val, eps_sq, eps_for_denom, use_adam_denom, eps_for_grad_sq, N, momentum_only_1d)
                def l_apply(nb, a):
                    _CUDA_MODULE.fused_update_1d_full_apply(
                        p_flat, g_flat, v_flat, m_flat, nb, a,
                        b1_val, beta_val, eps_sq, eps_for_denom, use_adam_denom, eps_for_grad_sq, N, d, momentum_only_1d)

                _dispatch_cuda_clip(state, param_work.device, alpha, d, enable_fira_for_adafactor, fira_margin, lag_norm,
                                    l_noclip, l_lag, l_norm, l_apply)
            else:
                variance = state["variance"]
                variance.mul_(1.0 - beta_val).addcmul_(grad_fp32, grad_fp32, value=beta_val)
                if eps_for_grad_sq > 0.0:
                    variance.add_(beta_val * eps_for_grad_sq)
                _fallback_1d_update(param_work, grad_fp32, variance, state, alpha, d, enable_fira_for_adafactor, fira_margin,
                                    beta1, m_quant_type, m_curr_block_size, quantize, momentum_only_1d, use_adam_denom, eps_for_denom, eps_sq)

        if needs_copy_back:
            param.copy_(param_work)

    # ================================================================
    # 2D factored path
    # NOTE: When beta1 is enabled for pure Adafactor (no CAME), 
    # this implementation uses Adam-style momentum (EMA on raw grad, then multiply by inv_std).
    # Official Adafactor uses momentum_only style (EMA on normalized U_t).
    # ================================================================
    else:
        shape = grad_contig.shape
        R = shape[-2]
        C = shape[-1]
        numel = grad_contig.numel()
        batch_size = math.prod(shape[:-2]) if len(shape) > 2 else 1
        if _cuda_ready and quantize and v_quant_type != 'fp32':
            _rc_buf, _row_sum, _col_sum, _row_fp32, _col_fp32, _came_base, _rs, _cs = \
                _prepare_rc_workspace(param_work.device, batch_size, R, C, beta3 is not None)
            _CUDA_MODULE.compute_factored_sums(
                grad_contig.view(-1), _row_sum, _col_sum, R, C, numel)
            row_mean = (_row_sum.view(batch_size, R, 1) / C).add_(eps1)
            col_mean = (_col_sum.view(batch_size, 1, C) / R).add_(eps1)
        else:
            grad_fp32 = _get_safe_grad_fp32()
            g_sq = grad_fp32.square()
            row_mean = g_sq.mean(dim=-1, keepdim=True).add_(eps1)
            col_mean = g_sq.mean(dim=-2, keepdim=True).add_(eps1)
            del g_sq

        if quantize and v_quant_type != 'fp32':
            if _cuda_ready:
                _fused_v_ema(state, "row_var", row_mean.reshape(-1), _row_fp32, beta_val, curr_block_size, log_floor=log_eps_sq)
                _fused_v_ema(state, "col_var", col_mean.reshape(-1), _col_fp32, beta_val, curr_block_size, log_floor=log_eps_sq)
            else:
                _row_fp32 = _col_fp32 = None
                row_var = _deq_v(state, "row_var")
                col_var = _deq_v(state, "col_var")
                row_var.lerp_(row_mean, beta_val)
                col_var.lerp_(col_mean, beta_val)
                row_mean_val = row_var.mean(dim=-2, keepdim=True).clamp(min=eps1)
                _quant_v(state, "row_var", row_var, curr_block_size, v_quant_type)
                _quant_v(state, "col_var", col_var, curr_block_size, v_quant_type)
            if _cuda_ready and beta1 is not None and m_quant_type == 'fp32':
                row_var = _row_fp32.view(shape[:-2] + (R, 1))
                col_var = _col_fp32.view(shape[:-2] + (1, C))
                row_mean_val = _row_fp32.view(batch_size, R).mean(dim=-1).clamp_(min=eps1).view(shape[:-2] + (1, 1))
            elif _cuda_ready:
                row_var = col_var = None
                row_mean_val = _row_fp32.view(batch_size, R).mean(dim=-1).clamp_(min=eps1)
        else:
            _row_fp32 = _col_fp32 = None
            row_var = state["row_var"]
            col_var = state["col_var"]
            row_var.lerp_(row_mean, beta_val)
            col_var.lerp_(col_mean, beta_val)
            row_mean_val = row_var.mean(dim=-2, keepdim=True).clamp(min=eps1)
        del row_mean, col_mean

        if quantize and beta1 is not None and m_quant_type != 'fp32':
            if _cuda_ready and v_quant_type != 'fp32':
                grad_flat_2d = grad_contig.reshape(-1)
                m_q_flat = state["m_q"].view(-1)
                m_scale_flat = state["m_scale"].view(-1)
                param_flat = param_work.reshape(-1)
                row_mean_val_flat = row_mean_val

                if beta3 is not None:
                    ut_sq_sum = _get_norm_buf(param_work.device)
                    _CUDA_MODULE.compute_ut_rms(
                        grad_flat_2d,
                        _row_fp32, _col_fp32,
                        row_mean_val_flat, ut_sq_sum,
                        R, C, numel)
                    clip_factor = torch.clamp(torch.sqrt(ut_sq_sum / numel) / d, min=1.0) if d > 0 else _get_one(param_work.device)

                    res_row_sum = _rc_buf[_came_base:_came_base + _rs].zero_()
                    res_col_sum = _rc_buf[_came_base + _rs:_came_base + _rs + _cs].zero_()
                    _CUDA_MODULE.fused_came_pass1(
                        grad_flat_2d, m_q_flat, m_scale_flat,
                        _row_fp32, _col_fp32,
                        row_mean_val_flat,
                        res_row_sum, res_col_sum, clip_factor,
                        beta1, eps_came,
                        R, C, numel, m_curr_block_size, m_bits, m_mode)

                    beta3_val = 1.0 - beta3
                    u_row_mean = res_row_sum / C
                    u_col_mean = res_col_sum / R
                    del res_row_sum, res_col_sum

                    _conf_row_fp32 = _rc_buf[_came_base + _rs + _cs:_came_base + _rs + _cs + _rs]
                    _conf_col_fp32 = _rc_buf[_came_base + _rs + _cs + _rs:]
                    _fused_v_ema(state, "conf_row", u_row_mean, _conf_row_fp32, beta3_val, curr_block_size, log_floor=log_eps_sq)
                    _fused_v_ema(state, "conf_col", u_col_mean, _conf_col_fp32, beta3_val, curr_block_size, log_floor=log_eps_sq)
                    del u_row_mean, u_col_mean
                    conf_row_mean = _conf_row_fp32.view(batch_size, R).mean(dim=-1).clamp_(min=eps1)

                    if enable_fira_for_adafactor:
                        norm_buf = _get_norm_buf(param_work.device)
                        _CUDA_MODULE.fused_came_pass2_norm(
                            grad_flat_2d, m_q_flat, m_scale_flat,
                            _row_fp32, _col_fp32,
                            row_mean_val_flat,
                            _conf_row_fp32, _conf_col_fp32,
                            conf_row_mean,
                            clip_factor, norm_buf,
                            beta1,
                            R, C, numel, m_curr_block_size, m_bits, m_mode)
                        alpha, norm_buf = _apply_fira_cuda(state, norm_buf, alpha, fira_margin)

                    _CUDA_MODULE.fused_came_pass2(
                        param_flat, grad_flat_2d,
                        m_q_flat, m_scale_flat,
                        _row_fp32, _col_fp32,
                        row_mean_val_flat,
                        _conf_row_fp32, _conf_col_fp32,
                        conf_row_mean,
                        clip_factor, alpha,
                        beta1,
                        R, C, numel, m_curr_block_size, m_bits, m_mode)

                else:
                    def l_noclip(a):
                        _CUDA_MODULE.fused_update_2d_noclip(
                            param_flat, grad_flat_2d, m_q_flat, m_scale_flat, _row_fp32, _col_fp32, row_mean_val_flat, a,
                            beta1, R, C, numel, m_curr_block_size, m_bits, m_mode)
                    def l_lag(nb, a, pn):
                        _CUDA_MODULE.fused_update_2d_lag(
                            param_flat, grad_flat_2d, m_q_flat, m_scale_flat, _row_fp32, _col_fp32, row_mean_val_flat, nb, a, pn,
                            beta1, R, C, numel, m_curr_block_size, m_bits, m_mode, d)
                    def l_norm(nb):
                        _CUDA_MODULE.fused_update_2d_norm(
                            grad_flat_2d, m_q_flat, m_scale_flat, _row_fp32, _col_fp32, row_mean_val_flat, nb,
                            beta1, R, C, numel, m_curr_block_size, m_bits, m_mode)
                    def l_apply(nb, a):
                        _CUDA_MODULE.fused_update_2d_apply(
                            param_flat, grad_flat_2d, m_q_flat, m_scale_flat, _row_fp32, _col_fp32, row_mean_val_flat, nb, a,
                            beta1, R, C, numel, m_curr_block_size, m_bits, m_mode, d)

                    _dispatch_cuda_clip(state, param_work.device, alpha, d, enable_fira_for_adafactor, fira_margin, lag_norm,
                                        l_noclip, l_lag, l_norm, l_apply)

            else:
                grad_fp32 = _get_safe_grad_fp32()
                inv_row, inv_col = _factored_inv_std(row_var, col_var, row_mean_val)
                _fallback_2d_update(param_work, grad_fp32, state, alpha, d, enable_fira_for_adafactor, fira_margin,
                                    beta1, beta3, eps_came, eps1, m_quant_type, m_curr_block_size, quantize,
                                    inv_row, inv_col, curr_block_size, v_quant_type)

        elif _cuda_ready and beta3 is None and (not quantize or v_quant_type == 'fp32' or beta1 is None):
            # Unified 2D Full Precision Path (Adafactor/AdamW)
            m_flat = None
            b1_val = 0.0
            if beta1 is not None:
                m_flat = state.get("m")
                if m_flat is not None and not m_flat.is_contiguous():
                    m_flat = m_flat.contiguous()
                    state["m"] = m_flat
                b1_val = beta1
                
            row_var_flat = row_var.view(-1) if row_var is not None else _row_fp32
            col_var_flat = col_var.view(-1) if col_var is not None else _col_fp32

            p_flat, g_flat = param_work.reshape(-1), grad_contig.reshape(-1)
            rm_flat = row_mean_val.view(-1)

            def l_noclip(a):
                _CUDA_MODULE.fused_update_2d_full_noclip(
                    p_flat, g_flat, m_flat, row_var_flat, col_var_flat, rm_flat, a,
                    b1_val, R, C, numel, m_block_size)
            def l_lag(nb, a, pn):
                _CUDA_MODULE.fused_update_2d_full_lag(
                    p_flat, g_flat, m_flat, row_var_flat, col_var_flat, rm_flat, nb, a, pn,
                    b1_val, R, C, numel, m_block_size, d)
            def l_norm(nb):
                _CUDA_MODULE.fused_update_2d_full_norm(
                    g_flat, m_flat, row_var_flat, col_var_flat, rm_flat, nb,
                    b1_val, R, C, numel, m_block_size)
            def l_apply(nb, a):
                _CUDA_MODULE.fused_update_2d_full_apply(
                    p_flat, g_flat, m_flat, row_var_flat, col_var_flat, rm_flat, nb, a,
                    b1_val, R, C, numel, m_block_size, d)

            _dispatch_cuda_clip(state, param_work.device, alpha, d, enable_fira_for_adafactor, fira_margin, lag_norm,
                                l_noclip, l_lag, l_norm, l_apply)

        elif _cuda_ready and beta3 is not None and beta1 is not None and (not quantize or v_quant_type == 'fp32'):
            # Unified 2D CAME Full Precision Path
            need_norm = enable_fira_for_adafactor

            m_flat = state.get("m")
            if m_flat is not None and not m_flat.is_contiguous():
                m_flat = m_flat.contiguous()
                state["m"] = m_flat

            ut_sq_sum = _get_norm_buf(param_work.device)
            _CUDA_MODULE.compute_ut_rms(
                grad_contig.reshape(-1),
                row_var.view(-1), col_var.view(-1),
                row_mean_val.view(-1), ut_sq_sum,
                R, C, numel)
            clip_factor = torch.clamp(torch.sqrt(ut_sq_sum / numel) / d, min=1.0) if d > 0 else _get_one(param_work.device)

            res_row_sum = torch.zeros(batch_size * R, device=param_work.device, dtype=torch.float32)
            res_col_sum = torch.zeros(batch_size * C, device=param_work.device, dtype=torch.float32)
            _CUDA_MODULE.fused_came_full_pass1(
                grad_contig.reshape(-1), m_flat.view(-1),
                row_var.view(-1), col_var.view(-1),
                row_mean_val.view(-1),
                res_row_sum, res_col_sum,
                clip_factor,
                beta1, eps_came, R, C, numel, m_block_size)

            c_row = state["conf_row"]
            c_col = state["conf_col"]
            res_row_mean = (res_row_sum.view(c_row.shape) / C).add_(eps_came)
            res_col_mean = (res_col_sum.view(c_col.shape) / R).add_(eps_came)
            c_row.lerp_(res_row_mean, 1.0 - beta3)
            c_col.lerp_(res_col_mean, 1.0 - beta3)
            c_row_mean = c_row.mean(dim=-2, keepdim=True).clamp(min=eps1)

            if need_norm:
                norm_buf = _get_norm_buf(param_work.device)
                _CUDA_MODULE.fused_came_full_pass2(
                    param_work.reshape(-1), m_flat.view(-1),
                    c_row.view(-1), c_col.view(-1),
                    c_row_mean.view(-1),
                    norm_buf, alpha,
                    R, C, numel, m_block_size, True)
                alpha, norm_buf = _apply_fira_cuda(state, norm_buf, alpha, fira_margin)

            _CUDA_MODULE.fused_came_full_pass2(
                param_work.reshape(-1), m_flat.view(-1),
                c_row.view(-1), c_col.view(-1),
                c_row_mean.view(-1),
                _get_norm_buf(param_work.device), alpha,
                R, C, numel, m_block_size, False)

        else:
            # 2D Fallback Path
            grad_fp32 = _get_safe_grad_fp32()
            inv_row, inv_col = _factored_inv_std(row_var, col_var, row_mean_val)
            _fallback_2d_update(param_work, grad_fp32, state, alpha, d, enable_fira_for_adafactor, fira_margin,
                                beta1, beta3, eps_came, eps1, m_quant_type, m_curr_block_size, quantize,
                                inv_row, inv_col, curr_block_size, v_quant_type)

        if needs_copy_back:
            param.copy_(param_work)


# --- APOLLO ---

def _get_apollo_proj_matrix(state: Dict[str, Any], shape: torch.Size, step: int,
                            dtype: torch.dtype, device: torch.device, cache_proj: bool = False) -> Tuple[Tensor, bool]:
    rank = int(state["apollo_rank"])
    update_proj_gap = int(state["apollo_update_proj_gap"])
    last_proj_step = int(state.get("last_proj_step", -update_proj_gap))
    
    needs_refresh = (step - last_proj_step) >= update_proj_gap

    if not needs_refresh and cache_proj and state.get("apollo_proj_matrix_T") is not None:
        return state["apollo_proj_matrix_T"], False
        
    if needs_refresh:
        state["current_proj_seed"] = int(state["apollo_seed"])
        state["apollo_seed"] = next_seed(state["current_proj_seed"])
        state["last_proj_step"] = step
        
    R, C = shape[-2], shape[-1]
    side = "right" if R >= C else "left"
    m_shape = (rank, C) if side == "right" else (R, rank)

    current_seed = state.get("current_proj_seed", state["apollo_seed"])
    proj = stable_randn(m_shape, seed=current_seed, device=device, dtype=dtype) / math.sqrt(rank)
    proj_T = proj.T.to(dtype).contiguous()
    
    if cache_proj:
        state["apollo_proj_matrix_T"] = proj_T
        
    return proj_T, needs_refresh 


def _update_param_apollo(
    param: Tensor, grad: Tensor, state: Dict[str, Any],
    d: float, lr: Union[float, Tensor],
    beta1: Optional[float], beta2: Optional[float], beta2_decay: float,
    weight_decay: float, eps2: float, maximize: bool, relative_step: bool,
    scale_parameter: bool, scale_weight_decay: bool,
    block_size: int, m_block_size: int, use_cuda_kernel: bool,
    apollo_scale_type: str, apollo_scale: float, apollo_scale_front: bool,
    apollo_eps: float, apollo_factorize: bool, apollo_cache_proj: bool = False,
    enable_fira_for_apollo: bool = True, fira_margin: float = 0.01,
    eps_came: float = 1e-16, beta3: Optional[float] = None,
    bias_correction: Optional[bool] = None,
):
    if maximize:
        grad_work = grad.neg().float()
    else:
        grad_work = grad.float()
        if grad_work.data_ptr() == grad.data_ptr():
            grad_work = grad_work.clone()
            
    grad_work[~torch.isfinite(grad_work)] = 0.0
    update_low = None
    _cuda_ready = _load_cuda_module(use_cuda_kernel)
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
    v_quant_type = state.get("v_quant_type", 'al8')
    if beta1 is not None:
        m_quant_type = state.get("m_quant_type", 'uf8')
        m_bits = _get_m_bits(m_quant_type)
        m_mode = _get_m_mode(m_quant_type)
    else:
        m_quant_type = 'uf8'
        m_bits = 0
        m_mode = 0
    beta_val = 1.0 - beta2 if beta2 is not None else math.pow(step, beta2_decay)
    alpha_t, rho_t = _compute_alpha(param_work, lr, step, relative_step, scale_parameter, eps2)
    shape = grad_work.shape
    proj_matrix_T, _ = _get_apollo_proj_matrix(
        state, shape, step, dtype=grad_work.dtype, device=grad_work.device, cache_proj=apollo_cache_proj
    )

    R, C = shape[-2], shape[-1]
    side = "right" if R >= C else "left"
    if side == "right":
        grad_low = torch.matmul(grad_work, proj_matrix_T).float()
    else:
        grad_low = torch.matmul(proj_matrix_T, grad_work).float()

    norm_G = None
    norm_G_sq = None
    scaling_factor_computed = False

    if apollo_factorize:
        quantize = state.get("is_quantized", False)
        grad_low_sq = grad_low.square()
        row_mean_low = grad_low_sq.mean(dim=-1, keepdim=True).add_(_LOG_QUANT_FLOOR)
        col_mean_low = grad_low_sq.mean(dim=-2, keepdim=True).add_(_LOG_QUANT_FLOOR)
        del grad_low_sq
        
        if "row_var_low" not in state:
            state["row_var_low"] = row_mean_low * beta_val
            state["col_var_low"] = col_mean_low * beta_val
        else:
            state["row_var_low"].mul_(1.0 - beta_val).add_(row_mean_low, alpha=beta_val)
            state["col_var_low"].mul_(1.0 - beta_val).add_(col_mean_low, alpha=beta_val)
            
        row_var = state["row_var_low"]
        col_var = state["col_var_low"]
        
        v_low_est = row_var.mul(col_var).div_(row_var.mean(dim=-2, keepdim=True).clamp_(min=apollo_eps))
        
        if beta1 is not None:
            if beta3 is not None:
                eps_sq = max(apollo_eps * apollo_eps, _LOG_QUANT_FLOOR)
                row_mean_val = row_var.mean(dim=-2, keepdim=True).clamp(min=_LOG_QUANT_FLOOR)
                inv_row, inv_col = _factored_inv_std(row_var, col_var, row_mean_val)
                
                U_t = grad_low.mul_(inv_row).mul_(inv_col)
                if d > 0:
                    rms_u = torch.linalg.vector_norm(U_t) / math.sqrt(U_t.numel())
                    U_t.div_(torch.clamp(rms_u / d, min=1.0))

                m_low_deq = _apollo_m_ema(
                    state, U_t, beta1, m_quant_type,
                    state.get("m_block_size", m_block_size), grad_low.numel(),
                    grad_low.shape, grad_low.device, _cuda_ready, quantize, prefix="m_low")
                
                res = U_t.sub(m_low_deq).square_().add_(eps_came)
                del U_t
                res_row = res.mean(dim=-1, keepdim=True)
                res_col = res.mean(dim=-2, keepdim=True)
                
                curr_block_size = state.get("block_size", block_size)
                if quantize and v_quant_type != 'fp32':
                    if state.get("conf_row_low_q") is None:
                        _init_v_state(state, "conf_row_low", res_row.shape, curr_block_size, grad_low.device, v_quant_type)
                        _init_v_state(state, "conf_col_low", res_col.shape, curr_block_size, grad_low.device, v_quant_type)
                    c_row = _deq_v(state, "conf_row_low")
                    c_col = _deq_v(state, "conf_col_low")
                    c_row.lerp_(res_row, 1.0 - beta3)
                    c_col.lerp_(res_col, 1.0 - beta3)
                    _quant_v(state, "conf_row_low", c_row, curr_block_size, v_quant_type)
                    _quant_v(state, "conf_col_low", c_col, curr_block_size, v_quant_type)
                else:
                    if "conf_row_low" not in state:
                        state["conf_row_low"] = torch.zeros_like(res_row)
                        state["conf_col_low"] = torch.zeros_like(res_col)
                    state["conf_row_low"].lerp_(res_row, 1.0 - beta3)
                    state["conf_col_low"].lerp_(res_col, 1.0 - beta3)
                    c_row = state["conf_row_low"]
                    c_col = state["conf_col_low"]
                
                conf_row_mean = c_row.mean(dim=-2, keepdim=True).clamp(min=_LOG_QUANT_FLOOR)
                inv_row_conf, inv_col_conf = _factored_inv_std(c_row, c_col, conf_row_mean)
                
                update_low = m_low_deq * inv_row_conf * inv_col_conf
                
            else:
                m_low_deq = _apollo_m_ema(
                    state, grad_low, beta1, m_quant_type,
                    state.get("m_block_size", m_block_size), grad_low.numel(),
                    grad_low.shape, grad_low.device, _cuda_ready, quantize, prefix="m_low")
                update_low = m_low_deq / (torch.sqrt(v_low_est) + apollo_eps)
        else:
            update_low = grad_low / (torch.sqrt(v_low_est) + apollo_eps)
        
    else:
        is_first_step = (state.get("v_low_q") is None and state.get("v_low") is None)
        quantize = state.get("is_quantized", False)
        curr_block_size = state.get("block_size", block_size)
        eps_sq = max(apollo_eps * apollo_eps, _LOG_QUANT_FLOOR)
        log_floor_apollo = math.log2(eps_sq)
        v_low_deq = _deq_v(state, "v_low") if (quantize and v_quant_type != 'fp32' and not is_first_step) else state.get("v_low")

        if is_first_step:
            v_init = grad_low.flatten().square() * beta_val
            if quantize and v_quant_type != 'fp32':
                _quant_v(state, "v_low", v_init, curr_block_size, v_quant_type)
                _v_fp32 = v_init
                v_low_deq = _v_fp32
            else:
                state["v_low"] = v_init
                v_low_deq = v_init
                _v_fp32 = v_init
        else:
            if quantize and v_quant_type != 'fp32':
                if _cuda_ready:
                    _v_fp32 = _get_v_buf(grad_low.device, grad_low.numel())
                    _fused_v_ema(state, "v_low", grad_low.flatten(), _v_fp32, beta_val, curr_block_size,
                                 square_input=True, log_floor=log_floor_apollo)
                    v_low_deq = _v_fp32
                else:
                    grad_low_sq_flat = grad_low.flatten().square()
                    v_deq = _deq_v(state, "v_low")
                    v_deq.lerp_(grad_low_sq_flat, beta_val)
                    _v_fp32 = v_deq
                    n_levels = 256 if v_quant_type == 'al8' else 65536
                    q, s, ml, sh, pad = _log_quantize_nonneg(v_deq, curr_block_size, min_log_floor=log_floor_apollo, n_levels=n_levels)
                    state["v_low_q"], state["v_low_scale"], state["v_low_min_log"], state["v_low_shape"], state["v_low_pad"] = q, s, ml, sh, pad
                    v_low_deq = _v_fp32
            else:
                grad_low_sq_flat = grad_low.flatten().square()
                state["v_low"].lerp_(grad_low_sq_flat, beta_val)
                _v_fp32 = state["v_low"]

        if beta3 is not None:
            if beta1 is not None:
                grad_low_numel = grad_low.numel()
                m_curr_block_size = state.get("m_block_size", m_block_size)

                if state.get("res_low_q") is None and quantize and v_quant_type != 'fp32':
                    _init_v_state(state, "res_low", grad_low.shape, curr_block_size, grad_low.device, v_quant_type)

                if quantize and _cuda_ready and m_quant_type != 'fp32' and v_quant_type != 'fp32':
                    if state.get("m_low_q") is None:
                        _m_init_state(state, grad_low_numel, m_curr_block_size, m_quant_type, grad_low.device, prefix="m_low")
                    _res_fp32 = _deq_v(state, "res_low")
                    total_sum_sq = _get_norm_buf(grad_low.device)
                    _v_for_rms = _v_fp32 if (quantize and v_quant_type != 'fp32') else v_low_deq.flatten()
                    _CUDA_MODULE.compute_apollo_came_rms(
                        grad_low, _v_for_rms,
                        total_sum_sq, eps_sq, grad_low_numel
                    )
                    
                    clip_factor_t = torch.clamp(torch.sqrt(total_sum_sq / grad_low_numel) / d, min=1.0) if d > 0 else _get_one(grad_low.device)
                    
                    m_temp = torch.empty(grad_low_numel, device=grad_low.device, dtype=torch.float32)
                    res_temp = torch.empty(grad_low_numel, device=grad_low.device, dtype=torch.float32)
                    
                    _CUDA_MODULE.apollo_came_compute_m_res(
                        grad_low, _v_fp32,
                        state["m_low_q"].view(-1), state["m_low_scale"],
                        _res_fp32,
                        m_temp, res_temp,
                        clip_factor_t, beta1, beta3, eps_came, eps_sq,
                        m_curr_block_size, grad_low_numel, m_bits, m_mode
                    )
                    
                    _m_fused_quantize_lerp(state, m_temp, 0.0, m_curr_block_size, grad_low_numel, m_quant_type, prefix="m_low", fp32_out=m_temp)
                    _res_fp32_new = torch.empty(grad_low_numel, device=grad_low.device, dtype=torch.float32)
                    _fused_v_ema(state, "res_low", res_temp, _res_fp32_new, 1.0, curr_block_size,
                                 log_floor=math.log2(_LOG_QUANT_FLOOR))
                    
                    if apollo_scale_type == "channel":
                        R_low, C_low = grad_low.shape[-2], grad_low.shape[-1]
                        if side == "right":
                            N, D, stride_N, stride_D = R_low, C_low, C_low, 1
                        else:
                            N, D, stride_N, stride_D = C_low, R_low, 1, C_low
                    else:
                        N, D, stride_N, stride_D = 1, grad_low_numel, grad_low_numel, 1

                    norm_update = torch.empty(N, device=grad_low.device, dtype=torch.float32)
                    norm_grad = torch.empty(N, device=grad_low.device, dtype=torch.float32)
                    
                    _CUDA_MODULE.apollo_came_compute_update_norms(
                        m_temp, 
                        _res_fp32_new,
                        grad_low,
                        norm_update, norm_grad,
                        eps_sq, N, D, stride_N, stride_D, grad_low_numel
                    )
                    del m_temp, res_temp
                    
                    scaling_factor = norm_update / (norm_grad + 1e-8)
                    if apollo_scale_type == "channel":
                        scaling_factor = scaling_factor.unsqueeze(1 if side == "right" else 0)
                    scaling_factor_computed = True
                else:
                    v_low_reshaped = v_low_deq.view_as(grad_low)
                    
                    U_t = grad_low.mul_(v_low_reshaped.clamp(min=eps_sq).rsqrt())
                    if d > 0:
                        rms_u = torch.linalg.vector_norm(U_t) / math.sqrt(U_t.numel())
                        U_t.div_(torch.clamp(rms_u / d, min=1.0))
                    
                    m_low_deq = _apollo_m_ema(
                        state, U_t, beta1, m_quant_type, m_curr_block_size,
                        grad_low.numel(), grad_low.shape, grad_low.device,
                        _cuda_ready, quantize, prefix="m_low")
                    
                    res = U_t.sub(m_low_deq).square_().add_(eps_came)
                    del U_t
                    
                    if quantize and v_quant_type != 'fp32':
                        res_deq = _deq_v(state, "res_low")
                        res_deq.lerp_(res, 1.0 - beta3)
                        _quant_v(state, "res_low", res_deq, curr_block_size, v_quant_type)
                        
                        res_low_deq = _deq_v(state, "res_low")
                    else:
                        if "res_low" not in state:
                            state["res_low"] = torch.zeros_like(res)
                        state["res_low"].lerp_(res, 1.0 - beta3)
                        res_low_deq = state["res_low"]
                        
                    res_low_reshaped = res_low_deq.view_as(grad_low)
                    update_low = m_low_deq * res_low_reshaped.clamp(min=eps_sq).rsqrt()
            else:
                v_low_reshaped = v_low_deq.view_as(grad_low)
                
                U_t = grad_low * v_low_reshaped.clamp(min=eps_sq).rsqrt()
                if d > 0:
                    rms_u = torch.linalg.vector_norm(U_t) / math.sqrt(U_t.numel())
                    U_t.div_(torch.clamp(rms_u / d, min=1.0))
                update_low = U_t

        else:
            if beta1 is not None:
                if not quantize or m_quant_type == 'fp32':
                    if "m_low" not in state:
                        state["m_low"] = torch.zeros_like(grad_low)
                    state["m_low"].lerp_(grad_low, 1.0 - beta1)
                    m_low_deq = state["m_low"]
                elif quantize:
                    grad_low_flat = grad_low.flatten()
                    grad_low_numel = grad_low_flat.numel()
                    m_curr_block_size = state.get("m_block_size", m_block_size)
                    if state.get("m_low_q") is None:
                        _m_init_state(state, grad_low_numel, m_curr_block_size, m_quant_type, grad_low.device, prefix="m_low")
                    
                    if _cuda_ready:
                        _m_fp32 = _get_m_buf(grad_low.device, grad_low_numel)
                        _m_fused_quantize_lerp(state, grad_low_flat, beta1, m_curr_block_size,
                                               grad_low_numel, m_quant_type, prefix="m_low", fp32_out=_m_fp32)
                        m_low_deq = _m_fp32.view(grad_low.shape)
                        if apollo_scale_type == "channel":
                            R_low, C_low = grad_low.shape[-2], grad_low.shape[-1]
                            if side == "right":
                                N, D = R_low, C_low
                                stride_N, stride_D = C_low, 1
                            else:
                                N, D = C_low, R_low
                                stride_N, stride_D = 1, C_low
                        else:
                            N, D = 1, grad_low_numel
                            stride_N, stride_D = grad_low_numel, 1
                        
                        norm_update = torch.empty(N, device=grad_low.device, dtype=torch.float32)
                        norm_grad = torch.empty(N, device=grad_low.device, dtype=torch.float32)
                        _v_for_norm = _v_fp32 if (quantize and v_quant_type != 'fp32') else v_low_deq.flatten()
                        _CUDA_MODULE.compute_apollo_norms(
                            _m_fp32,
                            _v_for_norm,
                            grad_low, norm_update, norm_grad, N, D, stride_N, stride_D, apollo_eps
                        )
                        scaling_factor = norm_update / (norm_grad + 1e-8)
                        if apollo_scale_type == "channel":
                            if side == "right":
                                scaling_factor = scaling_factor.unsqueeze(1)
                            else:
                                scaling_factor = scaling_factor.unsqueeze(0)
                        else:
                            scaling_factor = scaling_factor.view([])
                        scaling_factor_computed = True
                    else:
                        m_low_deq = _m_ema_pytorch(state, grad_low, beta1, m_quant_type, m_curr_block_size, prefix="m_low")
            else:
                update_low = grad_low / (torch.sqrt(v_low_deq).view_as(grad_low) + apollo_eps)
            
            if beta1 is not None and not scaling_factor_computed:
                v_low_reshaped = v_low_deq.view_as(grad_low)
                update_low = m_low_deq / (torch.sqrt(v_low_reshaped) + apollo_eps)

    if not scaling_factor_computed:
        if apollo_scale_type == "channel":
            norm_dim = -1 if side == "right" else -2
            scaling_factor = update_low.norm(dim=norm_dim, keepdim=True) / (grad_low.norm(dim=norm_dim, keepdim=True) + 1e-8)
        else:
            scaling_factor = update_low.norm() / (grad_low.norm() + 1e-8)

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

    if not enable_fira_for_apollo:
        final_scale = scaling_factor
    else:
        fira_fs = _compute_fira_scale(
            state, current_norm_t, fira_margin, param_work.device,
            state_key="scaled_grad_norm_prev")
        final_scale = scaling_factor * fira_fs

    # --- APOLLO Scale Back ---
    if not apollo_scale_front:
        final_scale = final_scale * apollo_scale_val

    numel = grad_work.numel()
    if apollo_scale_type == "channel":
        norm_unscaled_t = torch.sqrt((norm_G_sq * scaling_factor.square()).sum())
    else:
        norm_unscaled_t = torch.linalg.vector_norm(grad_work, ord=2, dtype=torch.float32) * scaling_factor
        
    denom_t = torch.clamp_min(norm_unscaled_t / (math.sqrt(numel) * d), 1.0) if d > 0 else _get_one(param_work.device)

    if weight_decay != 0:
        wd_multiplier_t = alpha_t if scale_weight_decay else rho_t
        decay_factor = 1.0 - (wd_multiplier_t * weight_decay)
        param_work.mul_(decay_factor)

    update_scale_t = -alpha_t / denom_t
    
    if beta1 is not None and not relative_step:
        apply_bc = True if bias_correction is None else bias_correction
        if apply_bc:
            bc1 = 1.0 - beta1 ** step
            if beta2 is not None:
                bc2 = 1.0 - beta2 ** step
                update_scale_t = update_scale_t * (math.sqrt(bc2) / bc1)
            else:
                update_scale_t = update_scale_t / bc1
        
    if apollo_scale_type == "channel":
        final_scale_cast = final_scale.to(grad_work.dtype)
        param_work.addcmul_(grad_work, final_scale_cast, value=update_scale_t)
    else:
        param_work.add_(grad_work, alpha=update_scale_t * final_scale)

    if apollo_scale_type == "channel":
        del norm_G_sq, norm_G
    del grad_low, scaling_factor, final_scale, proj_matrix_T, update_low
    
    if needs_copy_back:
        param.copy_(param_work.view(original_shape))
