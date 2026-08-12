// Copyright (c) 2026 WANG YAN
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ constexpr float INV_254 = 1.0f / 254.0f;
__device__ constexpr float INV_65534 = 1.0f / 65534.0f;
__device__ constexpr float MIN_VAL = 1.17549435e-38f;  // FP32_TINY
__device__ constexpr float LOG_ZERO  = -1e30f;   // sentinel: V=0 in log space
__device__ constexpr float LOG_MAX   = 126.0f;    // practical upper bound for log2(V)
__device__ constexpr float MIN_SCALE = 1e-12f;    // floor for quantization scale

// Kernel mode constants
// NOCLIP: d<=0, single pass, no norm clipping
// NORM:   d>0 pass 1, accumulate norm only (read-only, no state writes)
// APPLY:  d>0 pass 2, apply update with exact norm scaling, quantize M
// LAG:    d>0, single pass with previous step's norm (approximate clipping)
enum { MODE_NOCLIP = 0, MODE_NORM = 1, MODE_APPLY = 2, MODE_LAG = 3 };

// ==========================================
// 0. Constant Memory + Initialization
// ==========================================
__constant__ float d_qmap_8[256];
__constant__ float d_qmap_4[16];

void set_qmap_cuda(torch::Tensor qmap_8, torch::Tensor qmap_4) {
    TORCH_CHECK(qmap_8.numel() == 256 && qmap_8.scalar_type() == at::kFloat && qmap_8.is_cuda());
    TORCH_CHECK(qmap_4.numel() == 16 && qmap_4.scalar_type() == at::kFloat && qmap_4.is_cuda());
    cudaMemcpyToSymbol(d_qmap_8, qmap_8.data_ptr<float>(), 256 * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(d_qmap_4, qmap_4.data_ptr<float>(), 16 * sizeof(float), 0, cudaMemcpyDeviceToDevice);
}

// ==========================================
// 1. Layer 0: Shared Device Functions
// ==========================================
template <typename T>
__device__ __forceinline__ float load_grad(const T* g, int idx) {
    float v = static_cast<float>(g[idx]);
    return (isnan(v) || isinf(v)) ? 0.0f : v;
}

__device__ __forceinline__ void load_codebook_smem(float* s_qmap8, float* s_qmap4, int tid, int nthreads) {
    for (int i = tid; i < 256; i += nthreads) s_qmap8[i] = d_qmap_8[i];
    if (tid < 16) s_qmap4[tid] = d_qmap_4[tid];
}

// m_mode: 0 = uniform (uf4/uf8), 1 = dynamic map (d4/d8)
// s_qmap8/s_qmap4: shared-memory codebook (replaces constant-memory reads)
__device__ __forceinline__ float dequant_m(const void* m_q, int idx, const float* m_scale,
                                           int m_block_size, int m_bits, int m_mode,
                                           const float* s_qmap8, const float* s_qmap4) {
    float scale = m_scale[idx / m_block_size];
    if (m_mode == 0) {
        if (m_bits == 4) {
            unsigned char packed = ((const unsigned char*)m_q)[idx / 2];
            int q_int = (idx & 1) ? (packed & 0x0F) : (packed >> 4);
            return ((float)q_int - 8.0f) * (scale * 0.125f);
        } else {
            return ((float)((const unsigned char*)m_q)[idx] - 128.0f) * (scale * (1.0f / 128.0f));
        }
    } else {
        if (m_bits == 8) return s_qmap8[((const unsigned char*)m_q)[idx]] * scale;
        unsigned char packed = ((const unsigned char*)m_q)[idx / 2];
        int q_idx = (idx & 1) ? (packed & 0x0F) : (packed >> 4);
        return s_qmap4[q_idx] * scale;
    }
}

// Half-range search: restrict to correct-sign half (128 entries, 7 iterations).
// Sign correction retained as safety net.
__device__ __forceinline__ unsigned char quant_dynamic_8(float x_norm, float x_orig, const float* s_qmap8) {
    int lo = (x_orig >= 0.0f) ? 127 : 0;
    int hi = (x_orig >= 0.0f) ? 255 : 127;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (s_qmap8[mid] < x_norm) lo = mid + 1; else hi = mid; }
    if (lo > 0 && fabsf(x_norm - s_qmap8[lo - 1]) < fabsf(x_norm - s_qmap8[lo])) lo--;
    if (x_orig != 0.0f && signbit(s_qmap8[lo]) != signbit(x_orig)) {
        if (x_orig > 0.0f && lo < 255) lo++; else if (lo > 0) lo--;
    }
    return (unsigned char)lo;
}

// Half-range search: restrict to correct-sign half (8 entries, 3 iterations).
// Sign correction retained as safety net.
__device__ __forceinline__ unsigned char quant_dynamic_4(float x_norm, float x_orig, const float* s_qmap4) {
    int lo = (x_orig >= 0.0f) ? 7 : 0;
    int hi = (x_orig >= 0.0f) ? 15 : 7;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (s_qmap4[mid] < x_norm) lo = mid + 1; else hi = mid; }
    if (lo > 0 && fabsf(x_norm - s_qmap4[lo - 1]) < fabsf(x_norm - s_qmap4[lo])) lo--;
    if (x_orig != 0.0f && signbit(s_qmap4[lo]) != signbit(x_orig)) {
        if (x_orig > 0.0f && lo < 15) lo++; else if (lo > 0) lo--;
    }
    return (unsigned char)lo;
}

__device__ __forceinline__ void quant_m_store(void* m_q, int idx, float m0, float m1,
                                              float inv_abs, int m_bits, int m_mode, int numel,
                                              const float* s_qmap8, const float* s_qmap4) {
    if (m_mode == 0) {
        if (m_bits == 4) {
            float inv_s = inv_abs * 8.0f;
            int q0 = max(0, min(15, __float2int_rn(m0 * inv_s) + 8));
            int q1 = 8;
            if (idx + 1 < numel) q1 = max(0, min(15, __float2int_rn(m1 * inv_s) + 8));
            ((unsigned char*)m_q)[idx / 2] = (unsigned char)((q0 << 4) | q1);
        } else {
            ((unsigned char*)m_q)[idx] = (unsigned char)max(0, min(255, __float2int_rn(m0 * inv_abs * 128.0f) + 128));
        }
    } else {
        if (m_bits == 8) {
            ((unsigned char*)m_q)[idx] = quant_dynamic_8(m0 * inv_abs, m0, s_qmap8);
        } else {
            unsigned char q0 = quant_dynamic_4(m0 * inv_abs, m0, s_qmap4);
            unsigned char q1 = 7;
            if (idx + 1 < numel) q1 = quant_dynamic_4(m1 * inv_abs, m1, s_qmap4);
            ((unsigned char*)m_q)[idx / 2] = (unsigned char)((q0 << 4) | q1);
        }
    }
}

__device__ __forceinline__ float block_reduce_max(float val, float* s, int tid, int num_warps, float identity = 0.0f) {
    for (int o = 16; o > 0; o /= 2) val = fmaxf(val, __shfl_down_sync(0xffffffff, val, o));
    int lane = tid % 32, wid = tid / 32;
    if (lane == 0) s[wid] = val;
    __syncthreads();
    if (wid == 0) {
        val = (lane < num_warps) ? s[lane] : identity;
        for (int o = 16; o > 0; o /= 2) { float t = __shfl_down_sync(0xffffffff, val, o); if (lane + o < num_warps) val = fmaxf(val, t); }
        if (lane == 0) s[0] = val;
    }
    __syncthreads();
    return s[0];
}

__device__ __forceinline__ void block_reduce_minmax(
    float val_max, float val_min,
    float* s_max, float* s_min,
    int tid, int num_warps,
    float id_max, float id_min,
    float* out_max, float* out_min) {
    for (int o = 16; o > 0; o /= 2) {
        val_max = fmaxf(val_max, __shfl_down_sync(0xffffffff, val_max, o));
        val_min = fminf(val_min, __shfl_down_sync(0xffffffff, val_min, o));
    }
    int lane = tid % 32, wid = tid / 32;
    if (lane == 0) { s_max[wid] = val_max; s_min[wid] = val_min; }
    __syncthreads();
    if (wid == 0) {
        val_max = (lane < num_warps) ? s_max[lane] : id_max;
        val_min = (lane < num_warps) ? s_min[lane] : id_min;
        for (int o = 16; o > 0; o /= 2) {
            float t_max = __shfl_down_sync(0xffffffff, val_max, o);
            float t_min = __shfl_down_sync(0xffffffff, val_min, o);
            if (lane + o < num_warps) { val_max = fmaxf(val_max, t_max); val_min = fminf(val_min, t_min); }
        }
        if (lane == 0) { s_max[0] = val_max; s_min[0] = val_min; }
    }
    __syncthreads();
    *out_max = s_max[0];
    *out_min = s_min[0];
}

__device__ __forceinline__ float block_reduce_sum(float val, float* s, int tid, int num_warps) {
    for (int o = 16; o > 0; o /= 2) val += __shfl_down_sync(0xffffffff, val, o);
    int lane = tid % 32, wid = tid / 32;
    if (lane == 0) s[wid] = val;
    __syncthreads();
    if (wid == 0) {
        val = (lane < num_warps) ? s[lane] : 0.0f;
        for (int o = 16; o > 0; o /= 2) { float t = __shfl_down_sync(0xffffffff, val, o); if (lane + o < num_warps) val += t; }
        if (lane == 0) s[0] = val;
    }
    __syncthreads();
    return s[0];
}

__device__ __forceinline__ void warp_atomic_add_row(float val, int r, float* row_sum, int lane) {
    if (r < 0) return;
    unsigned active = __activemask();
    unsigned mask = __match_any_sync(active, r);
    int leader = __ffs(mask) - 1;

    if (mask == 0xffffffff) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(mask, val, offset);
        }
        if (lane == 0) atomicAdd(&row_sum[r], val);
    } else {
        float sum = 0.0f;
        for (unsigned temp_mask = mask; temp_mask != 0; temp_mask &= temp_mask - 1) {
            int src_lane = __ffs(temp_mask) - 1;
            sum += __shfl_sync(mask, val, src_lane);
        }
        if (lane == leader) atomicAdd(&row_sum[r], sum);
    }
}

__device__ __forceinline__ float dequant_v(unsigned char q, float scale, float min_log) {
    return (q == 0) ? 0.0f : exp2f((float)(q - 1) * INV_254 * scale + min_log);
}

__device__ __forceinline__ unsigned char quant_v(float vlog, float min_log, float inv_scale) {
    if (vlog <= -1e20f) return 0;
    int qi = __float2int_rn((vlog - min_log) * inv_scale) + 1;
    return (unsigned char)max(1, min(255, qi));
}

static inline int calc_threads(int block_size, int m_bits) {
    int elems = (m_bits == 8) ? 1 : 2;
    int slots = block_size / elems;
    return min(256, max(32, ((slots + 31) / 32) * 32));
}

static constexpr int SMEM_QMAP = 256 + 16;

// ==========================================
// 2. Fused Log-Quantize Lerp (standalone V EMA)
// ==========================================
template <typename T, int V_BITS>
__global__ void fused_log_quantize_lerp_kernel(
    void* __restrict__ q_void, float* __restrict__ scale, float* __restrict__ min_log_out,
    const T* __restrict__ new_val, float* __restrict__ fp32_out,
    float beta, int block_size, bool square_input, float eps1, int N, float log_floor) {
    int block_id = blockIdx.x, tid = threadIdx.x, stride = blockDim.x;
    int start = block_id * block_size, num_warps = stride / 32;
    extern __shared__ float shared_mem[];
    float* local_logs = shared_mem;
    float* s_reduce = &shared_mem[block_size];
    float old_scale = scale[block_id], old_min_log = min_log_out[block_id];
    float one_minus_b = 1.0f - beta;
    float thread_max = LOG_ZERO, thread_min = LOG_MAX;

    if constexpr (V_BITS == 8) {
        uchar4* q_vec = reinterpret_cast<uchar4*>(reinterpret_cast<unsigned char*>(q_void) + start);
        int vec_iters = (block_size / 4) / stride;
        for (int i = 0; i < vec_iters; i++) {
            int idx = tid + i * stride, base_idx = start + idx * 4;
            float vx, vy, vz, vw;
            if (base_idx + 3 < N) {
                if constexpr (sizeof(T) == 4) { float4 nv = reinterpret_cast<const float4*>(new_val + base_idx)[0]; vx=nv.x; vy=nv.y; vz=nv.z; vw=nv.w; }
                else { vx=static_cast<float>(new_val[base_idx]); vy=static_cast<float>(new_val[base_idx+1]); vz=static_cast<float>(new_val[base_idx+2]); vw=static_cast<float>(new_val[base_idx+3]); }
                if (square_input) { vx=vx*vx+eps1; vy=vy*vy+eps1; vz=vz*vz+eps1; vw=vw*vw+eps1; }
                vx=(isnan(vx)||isinf(vx))?0.0f:vx; vy=(isnan(vy)||isinf(vy))?0.0f:vy;
                vz=(isnan(vz)||isinf(vz))?0.0f:vz; vw=(isnan(vw)||isinf(vw))?0.0f:vw;
            } else {
                float v0=(base_idx<N)?static_cast<float>(new_val[base_idx]):0.0f;
                float v1=(base_idx+1<N)?static_cast<float>(new_val[base_idx+1]):0.0f;
                float v2=(base_idx+2<N)?static_cast<float>(new_val[base_idx+2]):0.0f;
                float v3=(base_idx+3<N)?static_cast<float>(new_val[base_idx+3]):0.0f;
                if (square_input) { v0=v0*v0+eps1; v1=v1*v1+eps1; v2=v2*v2+eps1; v3=v3*v3+eps1; }
                vx=(isnan(v0)||isinf(v0))?0.0f:v0; vy=(isnan(v1)||isinf(v1))?0.0f:v1;
                vz=(isnan(v2)||isinf(v2))?0.0f:v2; vw=(isnan(v3)||isinf(v3))?0.0f:v3;
            }
            uchar4 qv = q_vec[idx];
            float o0=dequant_v(qv.x,old_scale,old_min_log), o1=dequant_v(qv.y,old_scale,old_min_log);
            float o2=dequant_v(qv.z,old_scale,old_min_log), o3=dequant_v(qv.w,old_scale,old_min_log);
            float u0=fmaxf(o0*one_minus_b+vx*beta,0.0f);
            float u1=fmaxf(o1*one_minus_b+vy*beta,0.0f);
            float u2=fmaxf(o2*one_minus_b+vz*beta,0.0f);
            float u3=fmaxf(o3*one_minus_b+vw*beta,0.0f);
            if (fp32_out) {
                if (base_idx < N)     fp32_out[base_idx]     = u0;
                if (base_idx + 1 < N) fp32_out[base_idx + 1] = u1;
                if (base_idx + 2 < N) fp32_out[base_idx + 2] = u2;
                if (base_idx + 3 < N) fp32_out[base_idx + 3] = u3;
            }
            float l0=(u0>0.0f)?log2f(u0):LOG_ZERO, l1=(u1>0.0f)?log2f(u1):LOG_ZERO;
            float l2=(u2>0.0f)?log2f(u2):LOG_ZERO, l3=(u3>0.0f)?log2f(u3):LOG_ZERO;
            local_logs[idx*4]=l0; local_logs[idx*4+1]=l1; local_logs[idx*4+2]=l2; local_logs[idx*4+3]=l3;
            thread_max = fmaxf(thread_max, fmaxf(fmaxf(l0,l1),fmaxf(l2,l3)));
            float local_min = LOG_MAX;
            if(u0>0.0f) local_min=fminf(local_min,l0); if(u1>0.0f) local_min=fminf(local_min,l1);
            if(u2>0.0f) local_min=fminf(local_min,l2); if(u3>0.0f) local_min=fminf(local_min,l3);
            thread_min = fminf(thread_min, local_min);
        }
        float raw_max, raw_min;
        block_reduce_minmax(thread_max, thread_min, s_reduce, s_reduce + num_warps,
                            tid, num_warps, LOG_ZERO, LOG_MAX, &raw_max, &raw_min);
        float max_log = fminf(raw_max, LOG_MAX);
        float min_log = fmaxf(raw_min, log_floor);
        if (min_log >= max_log) min_log = max_log - 1.0f;
        float new_scale = fmaxf(max_log - min_log, MIN_SCALE), inv_s = 254.0f / new_scale;
        for (int i = 0; i < vec_iters; i++) {
            int idx = tid + i * stride;
            uchar4 out;
            out.x=quant_v(local_logs[idx*4],min_log,inv_s);
            out.y=quant_v(local_logs[idx*4+1],min_log,inv_s);
            out.z=quant_v(local_logs[idx*4+2],min_log,inv_s);
            out.w=quant_v(local_logs[idx*4+3],min_log,inv_s);
            q_vec[idx] = out;
        }
        if (tid == 0) { scale[block_id] = new_scale; min_log_out[block_id] = min_log; }
    } else {
        unsigned short* q_ptr = reinterpret_cast<unsigned short*>(q_void);
        int vec_iters = (block_size / 2) / stride;
        for (int i = 0; i < vec_iters; i++) {
            int idx = tid + i * stride;
            int base_idx = start + idx * 2;
            float v0, v1;
            if (base_idx + 1 < N) {
                if constexpr (sizeof(T) == 4) {
                    float2 nv = reinterpret_cast<const float2*>(new_val + base_idx)[0];
                    v0 = nv.x; v1 = nv.y;
                } else if constexpr (std::is_same_v<T, at::Half>) {
                    half2 nv = reinterpret_cast<const half2*>(new_val + base_idx)[0];
                    v0 = __half2float(nv.x); v1 = __half2float(nv.y);
                } else {
                    __nv_bfloat162 nv = reinterpret_cast<const __nv_bfloat162*>(new_val + base_idx)[0];
                    v0 = __bfloat162float(nv.x); v1 = __bfloat162float(nv.y);
                }
            } else {
                v0 = (base_idx < N) ? static_cast<float>(new_val[base_idx]) : 0.0f;
                v1 = 0.0f;
            }
            if (square_input) { v0 = v0*v0+eps1; v1 = v1*v1+eps1; }
            v0 = (isnan(v0)||isinf(v0)) ? 0.0f : v0;
            v1 = (isnan(v1)||isinf(v1)) ? 0.0f : v1;

            unsigned short q0 = (base_idx < N) ? q_ptr[base_idx] : 0;
            unsigned short q1 = (base_idx + 1 < N) ? q_ptr[base_idx + 1] : 0;
            
            float o0 = (q0 == 0) ? 0.0f : exp2f((float)(q0 - 1) * INV_65534 * old_scale + old_min_log);
            float o1 = (q1 == 0) ? 0.0f : exp2f((float)(q1 - 1) * INV_65534 * old_scale + old_min_log);
            
            float u0 = fmaxf(o0 * one_minus_b + v0 * beta, 0.0f);
            float u1 = fmaxf(o1 * one_minus_b + v1 * beta, 0.0f);
            
            if (fp32_out) {
                if (base_idx < N)     fp32_out[base_idx]     = u0;
                if (base_idx + 1 < N) fp32_out[base_idx + 1] = u1;
            }
            
            float l0 = (u0 > 0.0f) ? log2f(u0) : LOG_ZERO;
            float l1 = (u1 > 0.0f) ? log2f(u1) : LOG_ZERO;
            
            local_logs[idx*2] = l0; local_logs[idx*2+1] = l1;
            thread_max = fmaxf(thread_max, fmaxf(l0, l1));
            float local_min = LOG_MAX;
            if(u0 > 0.0f) local_min = fminf(local_min, l0);
            if(u1 > 0.0f) local_min = fminf(local_min, l1);
            thread_min = fminf(thread_min, local_min);
        }
        float raw_max, raw_min;
        block_reduce_minmax(thread_max, thread_min, s_reduce, s_reduce + num_warps,
                            tid, num_warps, LOG_ZERO, LOG_MAX, &raw_max, &raw_min);
        float max_log = fminf(raw_max, LOG_MAX);
        float min_log = fmaxf(raw_min, log_floor);
        if (min_log >= max_log) min_log = max_log - 1.0f;
        float new_scale = fmaxf(max_log - min_log, MIN_SCALE), inv_s = 65534.0f / new_scale;
        for (int i = 0; i < vec_iters; i++) {
            int idx = tid + i * stride;
            int base_idx = start + idx * 2;
            unsigned short out0 = (local_logs[idx*2] <= -1e20f) ? 0 : max(1, min(65535, __float2int_rn((local_logs[idx*2] - min_log) * inv_s) + 1));
            unsigned short out1 = (local_logs[idx*2+1] <= -1e20f) ? 0 : max(1, min(65535, __float2int_rn((local_logs[idx*2+1] - min_log) * inv_s) + 1));
            if (base_idx < N) q_ptr[base_idx] = out0;
            if (base_idx + 1 < N) q_ptr[base_idx + 1] = out1;
        }
        if (tid == 0) { scale[block_id] = new_scale; min_log_out[block_id] = min_log; }
    }
}

void fused_log_quantize_lerp_cuda(torch::Tensor q, torch::Tensor scale, torch::Tensor min_log_out,
    torch::Tensor new_val, c10::optional<torch::Tensor> fp32_out,
    float beta, int block_size, bool square_input, float eps1, int N, float log_floor, int v_bits) {
    TORCH_CHECK(q.is_cuda() && scale.is_cuda() && min_log_out.is_cuda() && new_val.is_cuda());
    if (v_bits == 8) {
        TORCH_CHECK(q.scalar_type() == at::kByte, "v_bits=8 requires q to be torch.uint8");
    } else {
        TORCH_CHECK(q.scalar_type() == at::kShort, "v_bits=16 requires q to be torch.int16");
    }
    TORCH_CHECK(scale.scalar_type() == at::kFloat && min_log_out.scalar_type() == at::kFloat, 
                "scale and min_log_out must be torch.float32");
    int threads = min(256, block_size/4); threads = max(32, (threads/32)*32);
    if (v_bits == 8) {
        TORCH_CHECK(block_size>=128 && block_size%4==0 && block_size%(4*threads)==0);
    } else {
        TORCH_CHECK(block_size>=128 && block_size%2==0 && block_size%(2*threads)==0);
    }
    int num_blocks=(N+block_size-1)/block_size, num_warps=threads/32;
    size_t smem=(block_size+2*num_warps)*sizeof(float);
    float* fp32_ptr = fp32_out.has_value() ? fp32_out.value().data_ptr<float>() : nullptr;
    
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        new_val.scalar_type(), "fused_log_quantize_lerp", ([&] {
            if (v_bits == 8) {
                fused_log_quantize_lerp_kernel<scalar_t, 8><<<num_blocks, threads, smem>>>(
                    q.data_ptr(), scale.data_ptr<float>(), min_log_out.data_ptr<float>(),
                    new_val.data_ptr<scalar_t>(), fp32_ptr,
                    beta, block_size, square_input, eps1, N, log_floor);
            } else {
                fused_log_quantize_lerp_kernel<scalar_t, 16><<<num_blocks, threads, smem>>>(
                    q.data_ptr(), scale.data_ptr<float>(), min_log_out.data_ptr<float>(),
                    new_val.data_ptr<scalar_t>(), fp32_ptr,
                    beta, block_size, square_input, eps1, N, log_floor);
            }
        }));
}

// ==========================================
// 3. Standalone M EMA (all modes, for fallback/migration)
// ==========================================
template <typename T>
__global__ void fused_m_quantize_lerp_kernel(
    unsigned char* __restrict__ q, float* __restrict__ scale,
    const T* __restrict__ new_val, float* __restrict__ fp32_out, float beta,
    int block_size, int N, int m_bits, int m_mode) {
    int block_id=blockIdx.x, tid=threadIdx.x, stride=blockDim.x;
    int start=block_id*block_size, num_warps=stride/32;
    extern __shared__ float shared_mem[];
    float* local_m=shared_mem; float* s_max=&shared_mem[block_size];
    float* s_qmap8=&shared_mem[block_size+num_warps];
    float* s_qmap4=s_qmap8+256;
    load_codebook_smem(s_qmap8, s_qmap4, tid, stride);
    __syncthreads();

    float thread_max=0.0f, one_minus_b=1.0f-beta;
    int elems=(m_bits==8)?1:2;
    int total_slots=block_size/elems, slot_iters=(total_slots+stride-1)/stride;
    for (int i=0;i<slot_iters;i++) {
        int slot=tid+i*stride; if(slot>=total_slots) break;
        int idx0=start+slot*elems;
        float m_new_0=0.0f, m_new_1=0.0f;
        if (idx0<N) {
            float g0=load_grad(new_val,idx0);
            float m_old0=dequant_m(q,idx0,scale,block_size,m_bits,m_mode,s_qmap8,s_qmap4);
            m_new_0=beta*m_old0+one_minus_b*g0;
            thread_max=fmaxf(thread_max,fabsf(m_new_0));
        }
        if (elems==2 && idx0+1<N) {
            float g1=load_grad(new_val,idx0+1);
            float m_old1=dequant_m(q,idx0+1,scale,block_size,m_bits,m_mode,s_qmap8,s_qmap4);
            m_new_1=beta*m_old1+one_minus_b*g1;
            thread_max=fmaxf(thread_max,fabsf(m_new_1));
        }
        local_m[slot*elems]=m_new_0;
        if(elems==2) local_m[slot*elems+1]=m_new_1;
        if (fp32_out) {
            if (idx0 < N) fp32_out[idx0] = m_new_0;
            if (elems == 2 && idx0 + 1 < N) fp32_out[idx0 + 1] = m_new_1;
        }
    }
    float abs_max=fmaxf(block_reduce_max(thread_max,s_max,tid,num_warps),MIN_SCALE);
    float inv_abs=1.0f/abs_max;
    for (int i=0;i<slot_iters;i++) {
        int slot=tid+i*stride; if(slot>=total_slots) break;
        int idx0=start+slot*elems;
        float m0=local_m[slot*elems];
        float m1=(elems==2)?local_m[slot*elems+1]:0.0f;
        if(idx0<N) quant_m_store(q,idx0,m0,m1,inv_abs,m_bits,m_mode,N,s_qmap8,s_qmap4);
    }
    if(tid==0) scale[block_id]=abs_max;
}

void fused_m_quantize_lerp_cuda(torch::Tensor q, torch::Tensor scale, torch::Tensor new_val,
    c10::optional<torch::Tensor> fp32_out,
    float beta, int block_size, int N, int m_bits, int m_mode) {
    TORCH_CHECK(q.scalar_type()==at::kByte && scale.scalar_type()==at::kFloat);
    int threads=calc_threads(block_size, m_bits);
    int num_blocks=(N+block_size-1)/block_size, num_warps=threads/32;
    size_t smem=(block_size+num_warps+SMEM_QMAP)*sizeof(float);
    float* fp32_ptr = fp32_out.has_value() ? fp32_out.value().data_ptr<float>() : nullptr;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        new_val.scalar_type(), "fused_m_quantize_lerp", ([&] {
            fused_m_quantize_lerp_kernel<scalar_t><<<num_blocks, threads, smem>>>(
                q.data_ptr<unsigned char>(), scale.data_ptr<float>(), new_val.data_ptr<scalar_t>(),
                fp32_ptr, beta, block_size, N, m_bits, m_mode);
        }));
}

// ==========================================
// 4. Fused 1D Update (V-EMA fused, exact v for update)
// ==========================================
template <typename T, int MODE, int V_BITS>
__global__ void fused_update_1d_kernel(
    T* __restrict__ param, const T* __restrict__ grad,
    void* __restrict__ m_q, float* __restrict__ m_scale,
    void* __restrict__ v_q_void, float* __restrict__ v_scale, float* __restrict__ v_min_log,
    float* __restrict__ v_buf,
    float* __restrict__ norm_buf, const float* __restrict__ alpha,
    float beta1, float beta_val, float eps_for_denom, bool use_adam_denom,
    float log_eps_sq, float eps_for_grad_sq,
    int numel, float sqrt_numel, int block_size, int m_block_size, int m_bits, int m_mode,
    const float* __restrict__ prev_norm_ptr, float d, bool momentum_only) {
    int block_id=blockIdx.x, tid=threadIdx.x, num_warps=blockDim.x/32;
    int start=block_id*block_size;
    extern __shared__ float shared_mem[];

    // Shared memory layout (codebook appended at end):
    // NOCLIP/LAG: [smem_m | smem_vlog | s_reduce(2*nw) | s_qmap8 | s_qmap4]
    // NORM:       [smem_vlog | s_reduce(2*nw) | s_qmap8 | s_qmap4]
    // APPLY:      [smem_m | s_reduce(nw) | s_qmap8 | s_qmap4]
    float* smem_m;
    float* smem_vlog;
    float* s_reduce;
    float* s_qmap8;
    float* s_qmap4;
    if constexpr (MODE == MODE_NORM) {
        smem_m = nullptr;
        smem_vlog = shared_mem;
        s_reduce = &shared_mem[block_size];
        s_qmap8 = &shared_mem[block_size + 2*num_warps];
        s_qmap4 = s_qmap8 + 256;
    } else if constexpr (MODE == MODE_APPLY) {
        smem_m = shared_mem;
        smem_vlog = nullptr;
        s_reduce = &shared_mem[block_size];
        s_qmap8 = &shared_mem[block_size + num_warps];
        s_qmap4 = s_qmap8 + 256;
    } else {
        smem_m = shared_mem;
        smem_vlog = &shared_mem[block_size];
        s_reduce = &shared_mem[2 * block_size];
        s_qmap8 = &shared_mem[2*block_size + 2*num_warps];
        s_qmap4 = s_qmap8 + 256;
    }
    load_codebook_smem(s_qmap8, s_qmap4, tid, blockDim.x);
    __syncthreads();

    float one_minus_b1=1.0f-beta1, one_minus_bv=1.0f-beta_val;
    int elems=(m_bits==8)?1:2;
    float old_v_scale=v_scale[block_id], old_v_min_log=v_min_log[block_id];

    float step_scale=0.0f;
    float ut_clip=1.0f;
    if constexpr (MODE==MODE_NOCLIP) { step_scale=*alpha; }
    else if constexpr (MODE==MODE_APPLY) {
        float denom=(d>0.0f)?fmaxf(1.0f,sqrtf(*norm_buf)/(sqrt_numel*d)):1.0f;
        if (momentum_only) { step_scale=*alpha; ut_clip=denom; }
        else { step_scale=*alpha/denom; }
    } else if constexpr (MODE==MODE_LAG) {
        float pn=*prev_norm_ptr;
        float denom=(d>0.0f)?fmaxf(1.0f,pn/(sqrt_numel*d)):1.0f;
        if (momentum_only) { step_scale=*alpha; ut_clip=denom; }
        else { step_scale=*alpha/denom; }
    }

    int total_slots=block_size/elems;
    int slot_iters=(total_slots+blockDim.x-1)/blockDim.x;
    float thread_max_m=0.0f, thread_max_vlog=LOG_ZERO, thread_min_vlog=LOG_MAX, local_sq=0.0f;

    for (int iter=0; iter<slot_iters; iter++) {
        int slot=tid+iter*blockDim.x; if(slot>=total_slots) break;
        int idx0=start+slot*elems;
        float m_new_0=0.0f, m_new_1=0.0f, vlog_0=LOG_ZERO, vlog_1=LOG_ZERO;

        if (idx0<numel) {
            float g0=load_grad(grad,idx0);
            float inv_std0;

            if constexpr (MODE == MODE_APPLY) {
                float v_exact=v_buf[idx0];
                inv_std0=use_adam_denom ? 1.0f/(sqrtf(v_exact)+eps_for_denom)
                                        : rsqrtf(v_exact);
            } else {
                float v_old0;
                if constexpr (V_BITS == 8) {
                    unsigned char* v_q = reinterpret_cast<unsigned char*>(v_q_void);
                    v_old0 = dequant_v(v_q[idx0], old_v_scale, old_v_min_log);
                } else {
                    unsigned short* v_q = reinterpret_cast<unsigned short*>(v_q_void);
                    unsigned short q_val = v_q[idx0];
                    v_old0 = (q_val == 0) ? 0.0f : exp2f((float)(q_val - 1) * INV_65534 * old_v_scale + old_v_min_log);
                }
                float v_new0=v_old0*one_minus_bv+(g0*g0+eps_for_grad_sq)*beta_val;
                vlog_0=log2f(v_new0);
                thread_min_vlog=fminf(thread_min_vlog,vlog_0);
                inv_std0=use_adam_denom ? 1.0f/(sqrtf(v_new0)+eps_for_denom)
                                        : rsqrtf(v_new0);
                if constexpr (MODE==MODE_NORM) { v_buf[idx0]=v_new0; }
            }

            float m_old0=dequant_m(m_q,idx0,m_scale,m_block_size,m_bits,m_mode,s_qmap8,s_qmap4);
            float u0;
            if (momentum_only) {
                float ut0=(g0*inv_std0)/ut_clip;
                m_new_0=beta1*m_old0+one_minus_b1*ut0;
                u0=m_new_0;
            } else {
                m_new_0=beta1*m_old0+one_minus_b1*g0;
                u0=m_new_0*inv_std0;
            }

            if constexpr (MODE==MODE_NORM) { local_sq+=momentum_only?(g0*inv_std0)*(g0*inv_std0):u0*u0; }
            else { float p=static_cast<float>(param[idx0]); param[idx0]=static_cast<T>(p-step_scale*u0); }
            if constexpr (MODE==MODE_LAG) { local_sq+=momentum_only?(g0*inv_std0)*(g0*inv_std0):u0*u0; }
            thread_max_m=fmaxf(thread_max_m,fabsf(m_new_0));
            thread_max_vlog=fmaxf(thread_max_vlog,vlog_0);
        }

        if (elems==2 && idx0+1<numel) {
            float g1=load_grad(grad,idx0+1);
            float inv_std1;

            if constexpr (MODE == MODE_APPLY) {
                float v_exact=v_buf[idx0+1];
                inv_std1=use_adam_denom ? 1.0f/(sqrtf(v_exact)+eps_for_denom)
                                        : rsqrtf(v_exact);
            } else {
                float v_old1;
                if constexpr (V_BITS == 8) {
                    unsigned char* v_q = reinterpret_cast<unsigned char*>(v_q_void);
                    v_old1 = dequant_v(v_q[idx0+1], old_v_scale, old_v_min_log);
                } else {
                    unsigned short* v_q = reinterpret_cast<unsigned short*>(v_q_void);
                    unsigned short q_val = v_q[idx0+1];
                    v_old1 = (q_val == 0) ? 0.0f : exp2f((float)(q_val - 1) * INV_65534 * old_v_scale + old_v_min_log);
                }
                float v_new1=v_old1*one_minus_bv+(g1*g1+eps_for_grad_sq)*beta_val;
                vlog_1=log2f(v_new1);
                thread_min_vlog=fminf(thread_min_vlog,vlog_1);
                inv_std1=use_adam_denom ? 1.0f/(sqrtf(v_new1)+eps_for_denom)
                                        : rsqrtf(v_new1);
                if constexpr (MODE==MODE_NORM) { v_buf[idx0+1]=v_new1; }
            }

            float m_old1=dequant_m(m_q,idx0+1,m_scale,m_block_size,m_bits,m_mode,s_qmap8,s_qmap4);
            float u1;
            if (momentum_only) {
                float ut1=(g1*inv_std1)/ut_clip;
                m_new_1=beta1*m_old1+one_minus_b1*ut1;
                u1=m_new_1;
            } else {
                m_new_1=beta1*m_old1+one_minus_b1*g1;
                u1=m_new_1*inv_std1;
            }

            if constexpr (MODE==MODE_NORM) { local_sq+=momentum_only?(g1*inv_std1)*(g1*inv_std1):u1*u1; }
            else { float p=static_cast<float>(param[idx0+1]); param[idx0+1]=static_cast<T>(p-step_scale*u1); }
            if constexpr (MODE==MODE_LAG) { local_sq+=momentum_only?(g1*inv_std1)*(g1*inv_std1):u1*u1; }
            thread_max_m=fmaxf(thread_max_m,fabsf(m_new_1));
            thread_max_vlog=fmaxf(thread_max_vlog,vlog_1);
        }

        if constexpr (MODE==MODE_NOCLIP || MODE==MODE_LAG) {
            smem_m[slot*elems]=m_new_0; smem_vlog[slot*elems]=vlog_0;
            if(elems==2){smem_m[slot*elems+1]=m_new_1; smem_vlog[slot*elems+1]=vlog_1;}
        } else if constexpr (MODE==MODE_NORM) {
            smem_vlog[slot*elems]=vlog_0;
            if(elems==2) smem_vlog[slot*elems+1]=vlog_1;
        } else {
            smem_m[slot*elems]=m_new_0;
            if(elems==2) smem_m[slot*elems+1]=m_new_1;
        }
    }

    if constexpr (MODE != MODE_APPLY) {
        float raw_max_v, raw_min_v;
        block_reduce_minmax(thread_max_vlog, thread_min_vlog,
                            s_reduce, s_reduce + num_warps,
                            tid, num_warps, LOG_ZERO, LOG_MAX, &raw_max_v, &raw_min_v);
        float max_log_v=fminf(raw_max_v,LOG_MAX);
        float min_log_v=fmaxf(raw_min_v,log_eps_sq);
        if(min_log_v>=max_log_v) min_log_v=max_log_v-1.0f;
        float new_v_scale=fmaxf(max_log_v-min_log_v,MIN_SCALE), inv_v_s=254.0f/new_v_scale;
        float inv_v_s_16 = 65534.0f / new_v_scale;
        
        for (int i = tid; i < block_size; i += blockDim.x) {
            int gi = start + i;
            if (gi >= numel) break;
            float vl = smem_vlog[i];
            
            if constexpr (V_BITS == 8) {
                unsigned char* v_q = reinterpret_cast<unsigned char*>(v_q_void);
                v_q[gi] = quant_v(vl, min_log_v, inv_v_s);
            } else {
                unsigned short* v_q = reinterpret_cast<unsigned short*>(v_q_void);
                v_q[gi] = (vl <= -1e20f) ? 0 : max(1, min(65535, __float2int_rn((vl - min_log_v) * inv_v_s_16) + 1));
            }
        }
        if(tid==0) { v_scale[block_id]=new_v_scale; v_min_log[block_id]=min_log_v; }
    }

    if constexpr (MODE != MODE_NORM) {
        int num_m_blocks = block_size / m_block_size;
        int m_slots_per_block = m_block_size / elems;
        for (int mb = 0; mb < num_m_blocks; mb++) {
            int mb_slot_start = mb * m_slots_per_block;
            float local_max = 0.0f;
            for (int j = tid; j < m_slots_per_block; j += blockDim.x) {
                int si = mb_slot_start + j;
                local_max = fmaxf(local_max, fabsf(smem_m[si * elems]));
                if (elems == 2) local_max = fmaxf(local_max, fabsf(smem_m[si * elems + 1]));
            }
            float abs_max_m = fmaxf(block_reduce_max(local_max, s_reduce, tid, num_warps), MIN_SCALE);
            float inv_abs_m = 1.0f / abs_max_m;
            for (int j = tid; j < m_slots_per_block; j += blockDim.x) {
                int si = mb_slot_start + j;
                int idx0 = start + si * elems;
                float m0 = smem_m[si * elems];
                float m1 = (elems == 2) ? smem_m[si * elems + 1] : 0.0f;
                if (idx0 < numel) quant_m_store(m_q, idx0, m0, m1, inv_abs_m, m_bits, m_mode, numel, s_qmap8, s_qmap4);
            }
            if (tid == 0) m_scale[block_id * num_m_blocks + mb] = abs_max_m;
        }
    }

    if constexpr (MODE==MODE_NORM || MODE==MODE_LAG) {
        float total_sq=block_reduce_sum(local_sq,s_reduce,tid,num_warps);
        if(tid==0) atomicAdd(norm_buf,total_sq);
    }
}

#define LAUNCH_1D(T,M,V) fused_update_1d_kernel<T,M,V><<<blocks,threads,smem>>>

void fused_update_1d_noclip_cuda(
    torch::Tensor param, torch::Tensor grad, torch::Tensor m_q, torch::Tensor m_scale,
    torch::Tensor v_q, torch::Tensor v_scale, torch::Tensor v_min_log, torch::Tensor alpha,
    float beta1, float beta_val, float eps_for_denom, bool use_adam_denom,
    float log_eps_sq, float eps_for_grad_sq, int numel, int block_size, int m_block_size, int m_bits, int m_mode, int v_bits, bool momentum_only) {
    if (v_bits == 8) {
        TORCH_CHECK(v_q.scalar_type() == at::kByte, "v_bits=8 requires v_q to be torch.uint8");
    } else {
        TORCH_CHECK(v_q.scalar_type() == at::kShort, "v_bits=16 requires v_q to be torch.int16");
    }
    int threads=calc_threads(block_size,m_bits), num_warps=threads/32;
    size_t smem=(2*block_size+2*num_warps+SMEM_QMAP)*sizeof(float);
    int blocks=(numel+block_size-1)/block_size;
    float sqrt_numel=sqrtf((float)numel);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16, param.scalar_type(),"fused_update_1d_noclip",([&]{
        if (v_bits == 8) LAUNCH_1D(scalar_t,MODE_NOCLIP,8)(param.data_ptr<scalar_t>(),grad.data_ptr<scalar_t>(),m_q.data_ptr(),m_scale.data_ptr<float>(),v_q.data_ptr(),v_scale.data_ptr<float>(),v_min_log.data_ptr<float>(),nullptr,nullptr,alpha.data_ptr<float>(),beta1,beta_val,eps_for_denom,use_adam_denom,log_eps_sq,eps_for_grad_sq,numel,sqrt_numel,block_size,m_block_size,m_bits,m_mode,nullptr,0.0f,momentum_only);
        else LAUNCH_1D(scalar_t,MODE_NOCLIP,16)(param.data_ptr<scalar_t>(),grad.data_ptr<scalar_t>(),m_q.data_ptr(),m_scale.data_ptr<float>(),v_q.data_ptr(),v_scale.data_ptr<float>(),v_min_log.data_ptr<float>(),nullptr,nullptr,alpha.data_ptr<float>(),beta1,beta_val,eps_for_denom,use_adam_denom,log_eps_sq,eps_for_grad_sq,numel,sqrt_numel,block_size,m_block_size,m_bits,m_mode,nullptr,0.0f,momentum_only);
    }));
}

void fused_update_1d_norm_cuda(
    torch::Tensor grad, torch::Tensor m_q, torch::Tensor m_scale,
    torch::Tensor v_q, torch::Tensor v_scale, torch::Tensor v_min_log,
    torch::Tensor v_buf, torch::Tensor norm_buf,
    float beta1, float beta_val, float eps_for_denom, bool use_adam_denom,
    float log_eps_sq, float eps_for_grad_sq, int numel, int block_size, int m_block_size, int m_bits, int m_mode, int v_bits, bool momentum_only) {
    if (v_bits == 8) {
        TORCH_CHECK(v_q.scalar_type() == at::kByte, "v_bits=8 requires v_q to be torch.uint8");
    } else {
        TORCH_CHECK(v_q.scalar_type() == at::kShort, "v_bits=16 requires v_q to be torch.int16");
    }
    int threads=calc_threads(block_size,m_bits), num_warps=threads/32;
    size_t smem=(block_size+2*num_warps+SMEM_QMAP)*sizeof(float);
    int blocks=(numel+block_size-1)/block_size;
    float sqrt_numel=sqrtf((float)numel);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16, grad.scalar_type(),"fused_update_1d_norm",([&]{
        if (v_bits == 8) LAUNCH_1D(scalar_t,MODE_NORM,8)(nullptr,grad.data_ptr<scalar_t>(),m_q.data_ptr(),m_scale.data_ptr<float>(),v_q.data_ptr(),v_scale.data_ptr<float>(),v_min_log.data_ptr<float>(),v_buf.data_ptr<float>(),norm_buf.data_ptr<float>(),nullptr,beta1,beta_val,eps_for_denom,use_adam_denom,log_eps_sq,eps_for_grad_sq,numel,sqrt_numel,block_size,m_block_size,m_bits,m_mode,nullptr,0.0f,momentum_only);
        else LAUNCH_1D(scalar_t,MODE_NORM,16)(nullptr,grad.data_ptr<scalar_t>(),m_q.data_ptr(),m_scale.data_ptr<float>(),v_q.data_ptr(),v_scale.data_ptr<float>(),v_min_log.data_ptr<float>(),v_buf.data_ptr<float>(),norm_buf.data_ptr<float>(),nullptr,beta1,beta_val,eps_for_denom,use_adam_denom,log_eps_sq,eps_for_grad_sq,numel,sqrt_numel,block_size,m_block_size,m_bits,m_mode,nullptr,0.0f,momentum_only);
    }));
}

void fused_update_1d_apply_cuda(
    torch::Tensor param, torch::Tensor grad, torch::Tensor m_q, torch::Tensor m_scale,
    torch::Tensor v_q, torch::Tensor v_scale, torch::Tensor v_min_log,
    torch::Tensor v_buf, torch::Tensor norm_buf, torch::Tensor alpha,
    float beta1, float beta_val, float eps_for_denom, bool use_adam_denom,
    float log_eps_sq, float eps_for_grad_sq, int numel, int block_size, int m_block_size, int m_bits, int m_mode, float d, int v_bits, bool momentum_only) {
    if (v_bits == 8) {
        TORCH_CHECK(v_q.scalar_type() == at::kByte, "v_bits=8 requires v_q to be torch.uint8");
    } else {
        TORCH_CHECK(v_q.scalar_type() == at::kShort, "v_bits=16 requires v_q to be torch.int16");
    }
    int threads=calc_threads(block_size,m_bits), num_warps=threads/32;
    size_t smem=(block_size+num_warps+SMEM_QMAP)*sizeof(float);
    int blocks=(numel+block_size-1)/block_size;
    float sqrt_numel=sqrtf((float)numel);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16, param.scalar_type(),"fused_update_1d_apply",([&]{
        if (v_bits == 8) LAUNCH_1D(scalar_t,MODE_APPLY,8)(param.data_ptr<scalar_t>(),grad.data_ptr<scalar_t>(),m_q.data_ptr(),m_scale.data_ptr<float>(),v_q.data_ptr(),v_scale.data_ptr<float>(),v_min_log.data_ptr<float>(),v_buf.data_ptr<float>(),norm_buf.data_ptr<float>(),alpha.data_ptr<float>(),beta1,beta_val,eps_for_denom,use_adam_denom,log_eps_sq,eps_for_grad_sq,numel,sqrt_numel,block_size,m_block_size,m_bits,m_mode,nullptr,d,momentum_only);
        else LAUNCH_1D(scalar_t,MODE_APPLY,16)(param.data_ptr<scalar_t>(),grad.data_ptr<scalar_t>(),m_q.data_ptr(),m_scale.data_ptr<float>(),v_q.data_ptr(),v_scale.data_ptr<float>(),v_min_log.data_ptr<float>(),v_buf.data_ptr<float>(),norm_buf.data_ptr<float>(),alpha.data_ptr<float>(),beta1,beta_val,eps_for_denom,use_adam_denom,log_eps_sq,eps_for_grad_sq,numel,sqrt_numel,block_size,m_block_size,m_bits,m_mode,nullptr,d,momentum_only);
    }));
}

void fused_update_1d_lag_cuda(
    torch::Tensor param, torch::Tensor grad, torch::Tensor m_q, torch::Tensor m_scale,
    torch::Tensor v_q, torch::Tensor v_scale, torch::Tensor v_min_log,
    torch::Tensor norm_buf, torch::Tensor alpha, torch::Tensor prev_norm,
    float beta1, float beta_val, float eps_for_denom, bool use_adam_denom,
    float log_eps_sq, float eps_for_grad_sq, int numel, int block_size, int m_block_size, int m_bits, int m_mode, float d, int v_bits, bool momentum_only) {
    if (v_bits == 8) {
        TORCH_CHECK(v_q.scalar_type() == at::kByte, "v_bits=8 requires v_q to be torch.uint8");
    } else {
        TORCH_CHECK(v_q.scalar_type() == at::kShort, "v_bits=16 requires v_q to be torch.int16");
    }
    int threads=calc_threads(block_size,m_bits), num_warps=threads/32;
    size_t smem=(2*block_size+2*num_warps+SMEM_QMAP)*sizeof(float);
    int blocks=(numel+block_size-1)/block_size;
    float sqrt_numel=sqrtf((float)numel);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16, param.scalar_type(),"fused_update_1d_lag",([&]{
        if (v_bits == 8) LAUNCH_1D(scalar_t,MODE_LAG,8)(param.data_ptr<scalar_t>(),grad.data_ptr<scalar_t>(),m_q.data_ptr(),m_scale.data_ptr<float>(),v_q.data_ptr(),v_scale.data_ptr<float>(),v_min_log.data_ptr<float>(),nullptr,norm_buf.data_ptr<float>(),alpha.data_ptr<float>(),beta1,beta_val,eps_for_denom,use_adam_denom,log_eps_sq,eps_for_grad_sq,numel,sqrt_numel,block_size,m_block_size,m_bits,m_mode,prev_norm.data_ptr<float>(),d,momentum_only);
        else LAUNCH_1D(scalar_t,MODE_LAG,16)(param.data_ptr<scalar_t>(),grad.data_ptr<scalar_t>(),m_q.data_ptr(),m_scale.data_ptr<float>(),v_q.data_ptr(),v_scale.data_ptr<float>(),v_min_log.data_ptr<float>(),nullptr,norm_buf.data_ptr<float>(),alpha.data_ptr<float>(),beta1,beta_val,eps_for_denom,use_adam_denom,log_eps_sq,eps_for_grad_sq,numel,sqrt_numel,block_size,m_block_size,m_bits,m_mode,prev_norm.data_ptr<float>(),d,momentum_only);
    }));
}
#undef LAUNCH_1D

// ==========================================
// 5. Fused 1D V-only Update (beta1=None, V-EMA fused)
// ==========================================
template <typename T, int MODE, int V_BITS>
__global__ void fused_update_1d_vonly_kernel(
    T* __restrict__ param, const T* __restrict__ grad,
    void* __restrict__ v_q_void, float* __restrict__ v_scale, float* __restrict__ v_min_log,
    float* __restrict__ v_buf,
    float* __restrict__ norm_buf, const float* __restrict__ alpha,
    float beta_val, float eps_for_denom, bool use_adam_denom,
    float log_eps_sq, float eps_for_grad_sq,
    int numel, float sqrt_numel, int block_size,
    const float* __restrict__ prev_norm_ptr, float d) {
    int block_id=blockIdx.x, tid=threadIdx.x, num_warps=blockDim.x/32;
    int start=block_id*block_size;
    extern __shared__ float shared_mem[];
    float* smem_vlog;
    float* s_reduce;
    if constexpr (MODE == MODE_APPLY) {
        smem_vlog = nullptr;
        s_reduce = shared_mem;
    } else {
        smem_vlog = shared_mem;
        s_reduce = &shared_mem[block_size];
    }

    float one_minus_bv=1.0f-beta_val;
    float old_v_scale=v_scale[block_id], old_v_min_log=v_min_log[block_id];

    float step_scale=0.0f;
    if constexpr (MODE==MODE_NOCLIP) { step_scale=*alpha; }
    else if constexpr (MODE==MODE_APPLY) {
        float denom=(d>0.0f)?fmaxf(1.0f,sqrtf(*norm_buf)/(sqrt_numel*d)):1.0f;
        step_scale=*alpha/denom;
    } else if constexpr (MODE==MODE_LAG) {
        float pn=*prev_norm_ptr;
        float denom=(d>0.0f)?fmaxf(1.0f,pn/(sqrt_numel*d)):1.0f;
        step_scale=*alpha/denom;
    }

    float thread_max_vlog=LOG_ZERO, thread_min_vlog=LOG_MAX, local_sq=0.0f;

    for (int i=tid; i<block_size; i+=blockDim.x) {
        int gi=start+i;
        if (gi>=numel) break;
        float g=load_grad(grad,gi);
        float inv_std;
        if constexpr (MODE==MODE_APPLY) {
            float v_exact=v_buf[gi];
            inv_std=use_adam_denom?1.0f/(sqrtf(v_exact)+eps_for_denom):rsqrtf(v_exact);
        } else {
            float v_old;
            if constexpr (V_BITS == 8) {
                unsigned char* v_q = reinterpret_cast<unsigned char*>(v_q_void);
                v_old = dequant_v(v_q[gi], old_v_scale, old_v_min_log);
            } else {
                unsigned short* v_q = reinterpret_cast<unsigned short*>(v_q_void);
                unsigned short q_val = v_q[gi];
                v_old = (q_val == 0) ? 0.0f : exp2f((float)(q_val - 1) * INV_65534 * old_v_scale + old_v_min_log);
            }
            float v_new=v_old*one_minus_bv+(g*g+eps_for_grad_sq)*beta_val;
            float vl=log2f(v_new);
            smem_vlog[i]=vl;
            thread_max_vlog=fmaxf(thread_max_vlog,vl);
            thread_min_vlog=fminf(thread_min_vlog,vl);
            inv_std=use_adam_denom?1.0f/(sqrtf(v_new)+eps_for_denom):rsqrtf(v_new);
            if constexpr (MODE==MODE_NORM) { v_buf[gi]=v_new; }
        }
        float u=g*inv_std;
        if constexpr (MODE==MODE_NORM) { local_sq+=u*u; }
        else { float p=static_cast<float>(param[gi]); param[gi]=static_cast<T>(p-step_scale*u); }
        if constexpr (MODE==MODE_LAG) { local_sq+=u*u; }
    }

    if constexpr (MODE!=MODE_APPLY) {
        float raw_max_v, raw_min_v;
        block_reduce_minmax(thread_max_vlog, thread_min_vlog,
                            s_reduce, s_reduce + num_warps,
                            tid, num_warps, LOG_ZERO, LOG_MAX, &raw_max_v, &raw_min_v);
        float max_log_v=fminf(raw_max_v,LOG_MAX);
        float min_log_v=fmaxf(raw_min_v,log_eps_sq);
        if(min_log_v>=max_log_v) min_log_v=max_log_v-1.0f;
        float new_v_scale=fmaxf(max_log_v-min_log_v,MIN_SCALE), inv_v_s=254.0f/new_v_scale;
        float inv_v_s_16 = 65534.0f / new_v_scale;
        for (int i=tid; i<block_size; i+=blockDim.x) {
            int gi=start+i;
            if(gi>=numel) break;
            if constexpr (V_BITS == 8) {
                unsigned char* v_q = reinterpret_cast<unsigned char*>(v_q_void);
                v_q[gi]=quant_v(smem_vlog[i],min_log_v,inv_v_s);
            } else {
                unsigned short* v_q = reinterpret_cast<unsigned short*>(v_q_void);
                float vl = smem_vlog[i];
                v_q[gi] = (vl <= -1e20f) ? 0 : max(1, min(65535, __float2int_rn((vl - min_log_v) * inv_v_s_16) + 1));
            }
        }
        if(tid==0) { v_scale[block_id]=new_v_scale; v_min_log[block_id]=min_log_v; }
    }

    if constexpr (MODE==MODE_NORM || MODE==MODE_LAG) {
        float total_sq=block_reduce_sum(local_sq,s_reduce,tid,num_warps);
        if(tid==0) atomicAdd(norm_buf,total_sq);
    }
}

void fused_update_1d_vonly_noclip_cuda(
    torch::Tensor param, torch::Tensor grad, torch::Tensor v_q, torch::Tensor v_scale,
    torch::Tensor v_min_log, torch::Tensor alpha,
    float beta_val, float eps_for_denom, bool use_adam_denom,
    float log_eps_sq, float eps_for_grad_sq, int numel, int block_size, int v_bits) {
    if (v_bits == 8) {
        TORCH_CHECK(v_q.scalar_type() == at::kByte, "v_bits=8 requires v_q to be torch.uint8");
    } else {
        TORCH_CHECK(v_q.scalar_type() == at::kShort, "v_bits=16 requires v_q to be torch.int16");
    }
    int threads=min(256,max(32,block_size/4)); threads=(threads/32)*32;
    int num_warps=threads/32;
    size_t smem=(block_size+2*num_warps)*sizeof(float);
    int blocks=(numel+block_size-1)/block_size;
    float sqrt_numel=sqrtf((float)numel);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16, param.scalar_type(),"fused_update_1d_vonly_noclip",([&]{
        if (v_bits == 8) fused_update_1d_vonly_kernel<scalar_t,MODE_NOCLIP,8><<<blocks,threads,smem>>>(param.data_ptr<scalar_t>(),grad.data_ptr<scalar_t>(),v_q.data_ptr(),v_scale.data_ptr<float>(),v_min_log.data_ptr<float>(),nullptr,nullptr,alpha.data_ptr<float>(),beta_val,eps_for_denom,use_adam_denom,log_eps_sq,eps_for_grad_sq,numel,sqrt_numel,block_size,nullptr,0.0f);
        else fused_update_1d_vonly_kernel<scalar_t,MODE_NOCLIP,16><<<blocks,threads,smem>>>(param.data_ptr<scalar_t>(),grad.data_ptr<scalar_t>(),v_q.data_ptr(),v_scale.data_ptr<float>(),v_min_log.data_ptr<float>(),nullptr,nullptr,alpha.data_ptr<float>(),beta_val,eps_for_denom,use_adam_denom,log_eps_sq,eps_for_grad_sq,numel,sqrt_numel,block_size,nullptr,0.0f);
    }));
}

void fused_update_1d_vonly_norm_cuda(
    torch::Tensor grad, torch::Tensor v_q, torch::Tensor v_scale, torch::Tensor v_min_log,
    torch::Tensor v_buf, torch::Tensor norm_buf,
    float beta_val, float eps_for_denom, bool use_adam_denom,
    float log_eps_sq, float eps_for_grad_sq, int numel, int block_size, int v_bits) {
    if (v_bits == 8) {
        TORCH_CHECK(v_q.scalar_type() == at::kByte, "v_bits=8 requires v_q to be torch.uint8");
    } else {
        TORCH_CHECK(v_q.scalar_type() == at::kShort, "v_bits=16 requires v_q to be torch.int16");
    }
    int threads=min(256,max(32,block_size/4)); threads=(threads/32)*32;
    int num_warps=threads/32;
    size_t smem=(block_size+2*num_warps)*sizeof(float);
    int blocks=(numel+block_size-1)/block_size;
    float sqrt_numel=sqrtf((float)numel);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16, grad.scalar_type(),"fused_update_1d_vonly_norm",([&]{
        if (v_bits == 8) fused_update_1d_vonly_kernel<scalar_t,MODE_NORM,8><<<blocks,threads,smem>>>(nullptr,grad.data_ptr<scalar_t>(),v_q.data_ptr(),v_scale.data_ptr<float>(),v_min_log.data_ptr<float>(),v_buf.data_ptr<float>(),norm_buf.data_ptr<float>(),nullptr,beta_val,eps_for_denom,use_adam_denom,log_eps_sq,eps_for_grad_sq,numel,sqrt_numel,block_size,nullptr,0.0f);
        else fused_update_1d_vonly_kernel<scalar_t,MODE_NORM,16><<<blocks,threads,smem>>>(nullptr,grad.data_ptr<scalar_t>(),v_q.data_ptr(),v_scale.data_ptr<float>(),v_min_log.data_ptr<float>(),v_buf.data_ptr<float>(),norm_buf.data_ptr<float>(),nullptr,beta_val,eps_for_denom,use_adam_denom,log_eps_sq,eps_for_grad_sq,numel,sqrt_numel,block_size,nullptr,0.0f);
    }));
}

void fused_update_1d_vonly_apply_cuda(
    torch::Tensor param, torch::Tensor grad, torch::Tensor v_q, torch::Tensor v_scale,
    torch::Tensor v_min_log, torch::Tensor v_buf, torch::Tensor norm_buf, torch::Tensor alpha,
    float beta_val, float eps_for_denom, bool use_adam_denom,
    float log_eps_sq, float eps_for_grad_sq, int numel, int block_size, float d, int v_bits) {
    if (v_bits == 8) {
        TORCH_CHECK(v_q.scalar_type() == at::kByte, "v_bits=8 requires v_q to be torch.uint8");
    } else {
        TORCH_CHECK(v_q.scalar_type() == at::kShort, "v_bits=16 requires v_q to be torch.int16");
    }
    int threads=min(256,max(32,block_size/4)); threads=(threads/32)*32;
    int num_warps=threads/32;
    size_t smem=num_warps*sizeof(float);
    int blocks=(numel+block_size-1)/block_size;
    float sqrt_numel=sqrtf((float)numel);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16, param.scalar_type(),"fused_update_1d_vonly_apply",([&]{
        if (v_bits == 8) fused_update_1d_vonly_kernel<scalar_t,MODE_APPLY,8><<<blocks,threads,smem>>>(param.data_ptr<scalar_t>(),grad.data_ptr<scalar_t>(),v_q.data_ptr(),v_scale.data_ptr<float>(),v_min_log.data_ptr<float>(),v_buf.data_ptr<float>(),norm_buf.data_ptr<float>(),alpha.data_ptr<float>(),beta_val,eps_for_denom,use_adam_denom,log_eps_sq,eps_for_grad_sq,numel,sqrt_numel,block_size,nullptr,d);
        else fused_update_1d_vonly_kernel<scalar_t,MODE_APPLY,16><<<blocks,threads,smem>>>(param.data_ptr<scalar_t>(),grad.data_ptr<scalar_t>(),v_q.data_ptr(),v_scale.data_ptr<float>(),v_min_log.data_ptr<float>(),v_buf.data_ptr<float>(),norm_buf.data_ptr<float>(),alpha.data_ptr<float>(),beta_val,eps_for_denom,use_adam_denom,log_eps_sq,eps_for_grad_sq,numel,sqrt_numel,block_size,nullptr,d);
    }));
}

void fused_update_1d_vonly_lag_cuda(
    torch::Tensor param, torch::Tensor grad, torch::Tensor v_q, torch::Tensor v_scale,
    torch::Tensor v_min_log,
    torch::Tensor norm_buf, torch::Tensor alpha, torch::Tensor prev_norm,
    float beta_val, float eps_for_denom, bool use_adam_denom,
    float log_eps_sq, float eps_for_grad_sq, int numel, int block_size, float d, int v_bits) {
    if (v_bits == 8) {
        TORCH_CHECK(v_q.scalar_type() == at::kByte, "v_bits=8 requires v_q to be torch.uint8");
    } else {
        TORCH_CHECK(v_q.scalar_type() == at::kShort, "v_bits=16 requires v_q to be torch.int16");
    }
    int threads=min(256,max(32,block_size/4)); threads=(threads/32)*32;
    int num_warps=threads/32;
    size_t smem=(block_size+2*num_warps)*sizeof(float);
    int blocks=(numel+block_size-1)/block_size;
    float sqrt_numel=sqrtf((float)numel);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16, param.scalar_type(),"fused_update_1d_vonly_lag",([&]{
        if (v_bits == 8) fused_update_1d_vonly_kernel<scalar_t,MODE_LAG,8><<<blocks,threads,smem>>>(param.data_ptr<scalar_t>(),grad.data_ptr<scalar_t>(),v_q.data_ptr(),v_scale.data_ptr<float>(),v_min_log.data_ptr<float>(),nullptr,norm_buf.data_ptr<float>(),alpha.data_ptr<float>(),beta_val,eps_for_denom,use_adam_denom,log_eps_sq,eps_for_grad_sq,numel,sqrt_numel,block_size,prev_norm.data_ptr<float>(),d);
        else fused_update_1d_vonly_kernel<scalar_t,MODE_LAG,16><<<blocks,threads,smem>>>(param.data_ptr<scalar_t>(),grad.data_ptr<scalar_t>(),v_q.data_ptr(),v_scale.data_ptr<float>(),v_min_log.data_ptr<float>(),nullptr,norm_buf.data_ptr<float>(),alpha.data_ptr<float>(),beta_val,eps_for_denom,use_adam_denom,log_eps_sq,eps_for_grad_sq,numel,sqrt_numel,block_size,prev_norm.data_ptr<float>(),d);
    }));
}

// ==========================================
// 6. Fused 1D Full Precision Update (fp32 V, optional fp32 M)
// ==========================================
template <typename T, int MODE, bool HAS_M>
__global__ void fused_update_1d_full_kernel(
    T* __restrict__ param, const T* __restrict__ grad,
    float* __restrict__ variance, float* __restrict__ m,
    float* __restrict__ norm_buf, const float* __restrict__ alpha,
    float beta1, float beta_val, float eps_sq, float eps_for_denom, bool use_adam_denom, float eps_for_grad_sq,
    int numel, float sqrt_numel,
    const float* __restrict__ prev_norm_ptr, float d, bool momentum_only) {

    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int num_warps = blockDim.x / 32;
    float local_sq = 0.0f;

    int idx = blockIdx.x * blockDim.x + tid;
    if (idx < numel) {
        float g = load_grad(grad, idx);

        float one_minus_b1 = 1.0f - beta1;
        float one_minus_bv = 1.0f - beta_val;

        float v_new;
        float inv_std;

        if constexpr (MODE == MODE_APPLY) {
            // V was already updated by NORM pass, just read it
            v_new = variance[idx];
            inv_std = use_adam_denom ? 1.0f / (sqrtf(v_new) + eps_for_denom) : rsqrtf(fmaxf(v_new, eps_sq));
        } else {
            // NOCLIP, NORM, LAG: compute and write V
            float v_old = variance[idx];
            v_new = one_minus_bv * v_old + beta_val * (g * g + eps_for_grad_sq);
            variance[idx] = v_new;
            inv_std = use_adam_denom ? 1.0f / (sqrtf(v_new) + eps_for_denom) : rsqrtf(fmaxf(v_new, eps_sq));
        }

        float step_scale = 0.0f;
        float ut_clip = 1.0f;
        if constexpr (MODE == MODE_NOCLIP) {
            step_scale = *alpha;
        } else if constexpr (MODE == MODE_APPLY) {
            float denom = (d > 0.0f) ? fmaxf(1.0f, sqrtf(*norm_buf) / (sqrt_numel * d)) : 1.0f;
            if (momentum_only) { step_scale = *alpha; ut_clip = denom; }
            else { step_scale = *alpha / denom; }
        } else if constexpr (MODE == MODE_LAG) {
            float pn = *prev_norm_ptr;
            float denom = (d > 0.0f) ? fmaxf(1.0f, pn / (sqrt_numel * d)) : 1.0f;
            if (momentum_only) { step_scale = *alpha; ut_clip = denom; }
            else { step_scale = *alpha / denom; }
        }

        float u;
        if constexpr (HAS_M) {
            if constexpr (MODE == MODE_APPLY) {
                if (momentum_only) {
                    // momentum_only: NORM pass did NOT update M.
                    // Re-compute U_t with correct clip, then update M here.
                    float ut = (g * inv_std) / ut_clip;
                    float m_old = m[idx];
                    float m_new = beta1 * m_old + one_minus_b1 * ut;
                    m[idx] = m_new;
                    u = m_new;
                } else {
                    // Adam style: M was already updated by NORM pass, just read it
                    float m_val = m[idx];
                    u = m_val * inv_std;
                }
            } else if constexpr (MODE == MODE_NORM) {
                if (momentum_only) {
                    // momentum_only: do NOT update M in NORM pass.
                    // Just compute unclipped U_t for norm accumulation.
                    u = g * inv_std;
                } else {
                    // Adam style: compute and write M in NORM pass
                    float m_old = m[idx];
                    float m_new = beta1 * m_old + one_minus_b1 * g;
                    m[idx] = m_new;
                    u = m_new * inv_std;
                }
            } else {
                // NOCLIP, LAG: compute and write M (single pass)
                float m_old = m[idx];
                float m_new;
                if (momentum_only) {
                    float ut = (g * inv_std) / ut_clip;
                    m_new = beta1 * m_old + one_minus_b1 * ut;
                    u = m_new;
                } else {
                    m_new = beta1 * m_old + one_minus_b1 * g;
                    u = m_new * inv_std;
                }
                m[idx] = m_new;
            }
        } else {
            u = (g * inv_std) / ut_clip;
        }

        if constexpr (MODE == MODE_NORM || MODE == MODE_LAG) {
            float norm_val = momentum_only ? (g * inv_std) * (g * inv_std) : u * u;
            local_sq = norm_val;
        }

        if constexpr (MODE != MODE_NORM) {
            float p = static_cast<float>(param[idx]);
            param[idx] = static_cast<T>(p - step_scale * u);
        }
    }

    if constexpr (MODE == MODE_NORM || MODE == MODE_LAG) {
        float total_sq = block_reduce_sum(local_sq, shared_mem, tid, num_warps);
        if (tid == 0) atomicAdd(norm_buf, total_sq);
    }
}

void fused_update_1d_full_noclip_cuda(
    torch::Tensor param, torch::Tensor grad, torch::Tensor variance, c10::optional<torch::Tensor> m,
    torch::Tensor alpha,
    float beta1, float beta_val, float eps_sq, float eps_for_denom, bool use_adam_denom, float eps_for_grad_sq,
    int numel, bool momentum_only) {
    int threads = 256, blocks = (numel + threads - 1) / threads;
    float sqrt_numel = sqrtf((float)numel);
    float* m_ptr = m.has_value() ? m.value().data_ptr<float>() : nullptr;
    if (m_ptr) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, param.scalar_type(), "fused_update_1d_full_noclip_m", ([&] {
            fused_update_1d_full_kernel<scalar_t, MODE_NOCLIP, true><<<blocks, threads>>>(param.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), variance.data_ptr<float>(), m_ptr, nullptr, alpha.data_ptr<float>(), beta1, beta_val, eps_sq, eps_for_denom, use_adam_denom, eps_for_grad_sq, numel, sqrt_numel, nullptr, 0.0f, momentum_only);
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, param.scalar_type(), "fused_update_1d_full_noclip", ([&] {
            fused_update_1d_full_kernel<scalar_t, MODE_NOCLIP, false><<<blocks, threads>>>(param.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), variance.data_ptr<float>(), nullptr, nullptr, alpha.data_ptr<float>(), beta1, beta_val, eps_sq, eps_for_denom, use_adam_denom, eps_for_grad_sq, numel, sqrt_numel, nullptr, 0.0f, momentum_only);
        }));
    }
}

void fused_update_1d_full_norm_cuda(
    torch::Tensor grad, torch::Tensor variance, c10::optional<torch::Tensor> m,
    torch::Tensor norm_buf,
    float beta1, float beta_val, float eps_sq, float eps_for_denom, bool use_adam_denom, float eps_for_grad_sq,
    int numel, bool momentum_only) {
    int threads = 256, num_warps = threads / 32, blocks = (numel + threads - 1) / threads;
    size_t smem = num_warps * sizeof(float);
    float sqrt_numel = sqrtf((float)numel);
    float* m_ptr = m.has_value() ? m.value().data_ptr<float>() : nullptr;
    if (m_ptr) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grad.scalar_type(), "fused_update_1d_full_norm_m", ([&] {
            fused_update_1d_full_kernel<scalar_t, MODE_NORM, true><<<blocks, threads, smem>>>(nullptr, grad.data_ptr<scalar_t>(), variance.data_ptr<float>(), m_ptr, norm_buf.data_ptr<float>(), nullptr, beta1, beta_val, eps_sq, eps_for_denom, use_adam_denom, eps_for_grad_sq, numel, sqrt_numel, nullptr, 0.0f, momentum_only);
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grad.scalar_type(), "fused_update_1d_full_norm", ([&] {
            fused_update_1d_full_kernel<scalar_t, MODE_NORM, false><<<blocks, threads, smem>>>(nullptr, grad.data_ptr<scalar_t>(), variance.data_ptr<float>(), nullptr, norm_buf.data_ptr<float>(), nullptr, beta1, beta_val, eps_sq, eps_for_denom, use_adam_denom, eps_for_grad_sq, numel, sqrt_numel, nullptr, 0.0f, momentum_only);
        }));
    }
}

void fused_update_1d_full_apply_cuda(
    torch::Tensor param, torch::Tensor grad, torch::Tensor variance, c10::optional<torch::Tensor> m,
    torch::Tensor norm_buf, torch::Tensor alpha,
    float beta1, float beta_val, float eps_sq, float eps_for_denom, bool use_adam_denom, float eps_for_grad_sq,
    int numel, float d, bool momentum_only) {
    int threads = 256, num_warps = threads / 32, blocks = (numel + threads - 1) / threads;
    size_t smem = num_warps * sizeof(float);
    float sqrt_numel = sqrtf((float)numel);
    float* m_ptr = m.has_value() ? m.value().data_ptr<float>() : nullptr;
    if (m_ptr) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, param.scalar_type(), "fused_update_1d_full_apply_m", ([&] {
            fused_update_1d_full_kernel<scalar_t, MODE_APPLY, true><<<blocks, threads, smem>>>(param.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), variance.data_ptr<float>(), m_ptr, norm_buf.data_ptr<float>(), alpha.data_ptr<float>(), beta1, beta_val, eps_sq, eps_for_denom, use_adam_denom, eps_for_grad_sq, numel, sqrt_numel, nullptr, d, momentum_only);
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, param.scalar_type(), "fused_update_1d_full_apply", ([&] {
            fused_update_1d_full_kernel<scalar_t, MODE_APPLY, false><<<blocks, threads, smem>>>(param.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), variance.data_ptr<float>(), nullptr, norm_buf.data_ptr<float>(), alpha.data_ptr<float>(), beta1, beta_val, eps_sq, eps_for_denom, use_adam_denom, eps_for_grad_sq, numel, sqrt_numel, nullptr, d, momentum_only);
        }));
    }
}

void fused_update_1d_full_lag_cuda(
    torch::Tensor param, torch::Tensor grad, torch::Tensor variance, c10::optional<torch::Tensor> m,
    torch::Tensor norm_buf, torch::Tensor alpha, torch::Tensor prev_norm,
    float beta1, float beta_val, float eps_sq, float eps_for_denom, bool use_adam_denom, float eps_for_grad_sq,
    int numel, float d, bool momentum_only) {
    int threads = 256, num_warps = threads / 32, blocks = (numel + threads - 1) / threads;
    size_t smem = num_warps * sizeof(float);
    float sqrt_numel = sqrtf((float)numel);
    float* m_ptr = m.has_value() ? m.value().data_ptr<float>() : nullptr;
    if (m_ptr) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, param.scalar_type(), "fused_update_1d_full_lag_m", ([&] {
            fused_update_1d_full_kernel<scalar_t, MODE_LAG, true><<<blocks, threads, smem>>>(param.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), variance.data_ptr<float>(), m_ptr, norm_buf.data_ptr<float>(), alpha.data_ptr<float>(), beta1, beta_val, eps_sq, eps_for_denom, use_adam_denom, eps_for_grad_sq, numel, sqrt_numel, prev_norm.data_ptr<float>(), d, momentum_only);
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, param.scalar_type(), "fused_update_1d_full_lag", ([&] {
            fused_update_1d_full_kernel<scalar_t, MODE_LAG, false><<<blocks, threads, smem>>>(param.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), variance.data_ptr<float>(), nullptr, norm_buf.data_ptr<float>(), alpha.data_ptr<float>(), beta1, beta_val, eps_sq, eps_for_denom, use_adam_denom, eps_for_grad_sq, numel, sqrt_numel, prev_norm.data_ptr<float>(), d, momentum_only);
        }));
    }
}

// ==========================================
// 7. Fused 2D Update (Unified: supports both quantized M and fp32 M)
//    QUANTIZED_M=true: uses shared memory, codebook, sub-block reduce.
//    QUANTIZED_M=false: uses global memory for M, zero shared memory overhead.
// ==========================================
template <typename T, int MODE, bool QUANTIZED_M>
__global__ void fused_update_2d_kernel(
    T* __restrict__ param, const T* __restrict__ grad,
    void* __restrict__ m_ptr, float* __restrict__ m_scale,
    const float* __restrict__ row_fp32, const float* __restrict__ col_fp32,
    const float* __restrict__ row_mean_val_ptr,
    float* __restrict__ norm_buf, const float* __restrict__ alpha,
    float beta1, int R, int C, int numel, float sqrt_numel,
    int m_block_size, int m_bits, int m_mode,
    const float* __restrict__ prev_norm_ptr, float d) {
    
    int block_id=blockIdx.x, tid=threadIdx.x, num_warps=blockDim.x/32;
    int start=block_id*m_block_size;
    extern __shared__ float shared_mem[];
    
    float* smem_m = nullptr;
    float* s_reduce = shared_mem;
    float* s_qmap8 = nullptr;
    float* s_qmap4 = nullptr;
    
    if constexpr (QUANTIZED_M) {
        smem_m = (MODE==MODE_NORM) ? nullptr : shared_mem;
        s_reduce = (MODE==MODE_NORM) ? shared_mem : &shared_mem[m_block_size];
        int reduce_base = (MODE==MODE_NORM) ? 0 : m_block_size;
        s_qmap8 = &shared_mem[reduce_base + num_warps];
        s_qmap4 = s_qmap8 + 256;
        load_codebook_smem(s_qmap8, s_qmap4, tid, blockDim.x);
    }
    __syncthreads();

    float one_minus_b1=1.0f-beta1;
    int elems = QUANTIZED_M ? ((m_bits==8)?1:2) : 1;
    
    float step_scale=0.0f;
    if constexpr (MODE==MODE_NOCLIP) { step_scale=*alpha; }
    else if constexpr (MODE==MODE_APPLY) {
        float denom=(d>0.0f)?fmaxf(1.0f,sqrtf(*norm_buf)/(sqrt_numel*d)):1.0f;
        step_scale=*alpha/denom;
    } else if constexpr (MODE==MODE_LAG) {
        float pn=*prev_norm_ptr;
        float denom=(d>0.0f)?fmaxf(1.0f,pn/(sqrt_numel*d)):1.0f;
        step_scale=*alpha/denom;
    }

    int total_slots=m_block_size/elems;
    int slot_iters=(total_slots+blockDim.x-1)/blockDim.x;
    float thread_max=0.0f, local_sq=0.0f;

    for (int iter=0;iter<slot_iters;iter++) {
        int slot=tid+iter*blockDim.x; if(slot>=total_slots) break;
        int idx0=start+slot*elems;
        float m_new_0=0.0f, m_new_1=0.0f;
        
        if (idx0<numel) {
            int b0=idx0/(R*C), r0=(idx0/C)%R, c0=idx0%C;
            float g0=load_grad(grad,idx0);
            float m_old0;
            if constexpr (QUANTIZED_M) {
                m_old0 = dequant_m(m_ptr,idx0,m_scale,m_block_size,m_bits,m_mode,s_qmap8,s_qmap4);
            } else {
                m_old0 = (m_ptr != nullptr) ? ((float*)m_ptr)[idx0] : 0.0f;
            }
            m_new_0=beta1*m_old0+one_minus_b1*g0;
            
            int r_idx0=b0*R+r0, c_idx0=b0*C+c0;
            float inv_std0=sqrtf(fmaxf(row_mean_val_ptr[b0],MIN_VAL))*rsqrtf(fmaxf(row_fp32[r_idx0],MIN_VAL))*rsqrtf(fmaxf(col_fp32[c_idx0],MIN_VAL));
            float u0=m_new_0*inv_std0;
            
            if constexpr (MODE==MODE_NORM) { local_sq+=u0*u0; }
            else { 
                float p=static_cast<float>(param[idx0]); 
                param[idx0]=static_cast<T>(p-step_scale*u0); 
                if constexpr (!QUANTIZED_M) {
                    if (m_ptr != nullptr) ((float*)m_ptr)[idx0] = m_new_0;
                }
            }
            if constexpr (MODE==MODE_LAG) { local_sq+=u0*u0; }
            
            if constexpr (QUANTIZED_M) {
                thread_max=fmaxf(thread_max,fabsf(m_new_0));
            }
        }
        
        if constexpr (QUANTIZED_M) {
            if (elems==2 && idx0+1<numel) {
                int idx1=idx0+1, b1=idx1/(R*C), r1=(idx1/C)%R, c1=idx1%C;
                float g1=load_grad(grad,idx1);
                float m_old1=dequant_m(m_ptr,idx1,m_scale,m_block_size,m_bits,m_mode,s_qmap8,s_qmap4);
                m_new_1=beta1*m_old1+one_minus_b1*g1;
                int r_idx1=b1*R+r1, c_idx1=b1*C+c1;
                float inv_std1=sqrtf(fmaxf(row_mean_val_ptr[b1],MIN_VAL))*rsqrtf(fmaxf(row_fp32[r_idx1],MIN_VAL))*rsqrtf(fmaxf(col_fp32[c_idx1],MIN_VAL));
                float u1=m_new_1*inv_std1;
                if constexpr (MODE==MODE_NORM) { local_sq+=u1*u1; }
                else { float p=static_cast<float>(param[idx1]); param[idx1]=static_cast<T>(p-step_scale*u1); }
                if constexpr (MODE==MODE_LAG) { local_sq+=u1*u1; }
                thread_max=fmaxf(thread_max,fabsf(m_new_1));
            }
            if constexpr (MODE != MODE_NORM) {
                smem_m[slot*elems]=m_new_0;
                if(elems==2) smem_m[slot*elems+1]=m_new_1;
            }
        }
    }
    
    if constexpr (QUANTIZED_M && MODE!=MODE_NORM) {
        float abs_max=fmaxf(block_reduce_max(thread_max,s_reduce,tid,num_warps),MIN_SCALE);
        float inv_abs=1.0f/abs_max;
        for (int iter=0;iter<slot_iters;iter++) {
            int slot=tid+iter*blockDim.x; if(slot>=total_slots) break;
            int idx0=start+slot*elems;
            float m0=smem_m[slot*elems];
            float m1=(elems==2)?smem_m[slot*elems+1]:0.0f;
            if(idx0<numel) quant_m_store(m_ptr,idx0,m0,m1,inv_abs,m_bits,m_mode,numel,s_qmap8,s_qmap4);
        }
        if(tid==0) m_scale[block_id]=abs_max;
    }
    if constexpr (MODE==MODE_NORM || MODE==MODE_LAG) {
        float total_sq=block_reduce_sum(local_sq,s_reduce,tid,num_warps);
        if(tid==0) atomicAdd(norm_buf,total_sq);
    }
}

#define LAUNCH_2D(T,M,Q) fused_update_2d_kernel<T,M,Q><<<blocks,threads,smem>>>

void fused_update_2d_noclip_cuda(
    torch::Tensor param, torch::Tensor grad, torch::Tensor m_q, torch::Tensor m_scale,
    torch::Tensor row_fp32, torch::Tensor col_fp32,
    torch::Tensor row_mean_val, torch::Tensor alpha,
    float beta1, int R, int C, int numel, int m_block_size, int m_bits, int m_mode) {
    int threads=calc_threads(m_block_size,m_bits), num_warps=threads/32;
    size_t smem=(m_block_size+num_warps+SMEM_QMAP)*sizeof(float);
    int blocks=(numel+m_block_size-1)/m_block_size;
    float sqrt_numel=sqrtf((float)numel);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16, param.scalar_type(),"fused_update_2d_noclip",([&]{
        LAUNCH_2D(scalar_t,MODE_NOCLIP,true)(param.data_ptr<scalar_t>(),grad.data_ptr<scalar_t>(),m_q.data_ptr(),m_scale.data_ptr<float>(),row_fp32.data_ptr<float>(),col_fp32.data_ptr<float>(),row_mean_val.data_ptr<float>(),nullptr,alpha.data_ptr<float>(),beta1,R,C,numel,sqrt_numel,m_block_size,m_bits,m_mode,nullptr,0.0f);
    }));
}

void fused_update_2d_norm_cuda(
    torch::Tensor grad, torch::Tensor m_q, torch::Tensor m_scale,
    torch::Tensor row_fp32, torch::Tensor col_fp32,
    torch::Tensor row_mean_val, torch::Tensor norm_buf,
    float beta1, int R, int C, int numel, int m_block_size, int m_bits, int m_mode) {
    int threads=calc_threads(m_block_size,m_bits), num_warps=threads/32;
    size_t smem=(num_warps+SMEM_QMAP)*sizeof(float);
    int blocks=(numel+m_block_size-1)/m_block_size;
    float sqrt_numel=sqrtf((float)numel);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16, grad.scalar_type(),"fused_update_2d_norm",([&]{
        LAUNCH_2D(scalar_t,MODE_NORM,true)(nullptr,grad.data_ptr<scalar_t>(),m_q.data_ptr(),m_scale.data_ptr<float>(),row_fp32.data_ptr<float>(),col_fp32.data_ptr<float>(),row_mean_val.data_ptr<float>(),norm_buf.data_ptr<float>(),nullptr,beta1,R,C,numel,sqrt_numel,m_block_size,m_bits,m_mode,nullptr,0.0f);
    }));
}

void fused_update_2d_apply_cuda(
    torch::Tensor param, torch::Tensor grad, torch::Tensor m_q, torch::Tensor m_scale,
    torch::Tensor row_fp32, torch::Tensor col_fp32,
    torch::Tensor row_mean_val, torch::Tensor norm_buf, torch::Tensor alpha,
    float beta1, int R, int C, int numel, int m_block_size, int m_bits, int m_mode, float d) {
    int threads=calc_threads(m_block_size,m_bits), num_warps=threads/32;
    size_t smem=(m_block_size+num_warps+SMEM_QMAP)*sizeof(float);
    int blocks=(numel+m_block_size-1)/m_block_size;
    float sqrt_numel=sqrtf((float)numel);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16, param.scalar_type(),"fused_update_2d_apply",([&]{
        LAUNCH_2D(scalar_t,MODE_APPLY,true)(param.data_ptr<scalar_t>(),grad.data_ptr<scalar_t>(),m_q.data_ptr(),m_scale.data_ptr<float>(),row_fp32.data_ptr<float>(),col_fp32.data_ptr<float>(),row_mean_val.data_ptr<float>(),norm_buf.data_ptr<float>(),alpha.data_ptr<float>(),beta1,R,C,numel,sqrt_numel,m_block_size,m_bits,m_mode,nullptr,d);
    }));
}

void fused_update_2d_lag_cuda(
    torch::Tensor param, torch::Tensor grad, torch::Tensor m_q, torch::Tensor m_scale,
    torch::Tensor row_fp32, torch::Tensor col_fp32,
    torch::Tensor row_mean_val, torch::Tensor norm_buf, torch::Tensor alpha, torch::Tensor prev_norm,
    float beta1, int R, int C, int numel, int m_block_size, int m_bits, int m_mode, float d) {
    int threads=calc_threads(m_block_size,m_bits), num_warps=threads/32;
    size_t smem=(m_block_size+num_warps+SMEM_QMAP)*sizeof(float);
    int blocks=(numel+m_block_size-1)/m_block_size;
    float sqrt_numel=sqrtf((float)numel);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16, param.scalar_type(),"fused_update_2d_lag",([&]{
        LAUNCH_2D(scalar_t,MODE_LAG,true)(param.data_ptr<scalar_t>(),grad.data_ptr<scalar_t>(),m_q.data_ptr(),m_scale.data_ptr<float>(),row_fp32.data_ptr<float>(),col_fp32.data_ptr<float>(),row_mean_val.data_ptr<float>(),norm_buf.data_ptr<float>(),alpha.data_ptr<float>(),beta1,R,C,numel,sqrt_numel,m_block_size,m_bits,m_mode,prev_norm.data_ptr<float>(),d);
    }));
}

void fused_update_2d_full_noclip_cuda(
    torch::Tensor param, torch::Tensor grad, c10::optional<torch::Tensor> m,
    torch::Tensor row_fp32, torch::Tensor col_fp32,
    torch::Tensor row_mean_val, torch::Tensor alpha,
    float beta1, int R, int C, int numel, int m_block_size) {
    int threads = 256, num_warps = threads / 32;
    size_t smem = num_warps * sizeof(float);
    int blocks=(numel+m_block_size-1)/m_block_size;
    float sqrt_numel=sqrtf((float)numel);
    float* m_ptr = m.has_value() ? m.value().data_ptr<float>() : nullptr;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16, param.scalar_type(),"fused_update_2d_full_noclip",([&]{
        LAUNCH_2D(scalar_t,MODE_NOCLIP,false)(param.data_ptr<scalar_t>(),grad.data_ptr<scalar_t>(),m_ptr,nullptr,row_fp32.data_ptr<float>(),col_fp32.data_ptr<float>(),row_mean_val.data_ptr<float>(),nullptr,alpha.data_ptr<float>(),beta1,R,C,numel,sqrt_numel,m_block_size,0,0,nullptr,0.0f);
    }));
}

void fused_update_2d_full_norm_cuda(
    torch::Tensor grad, c10::optional<torch::Tensor> m,
    torch::Tensor row_fp32, torch::Tensor col_fp32,
    torch::Tensor row_mean_val, torch::Tensor norm_buf,
    float beta1, int R, int C, int numel, int m_block_size) {
    int threads = 256, num_warps = threads / 32;
    size_t smem = num_warps * sizeof(float);
    int blocks=(numel+m_block_size-1)/m_block_size;
    float sqrt_numel=sqrtf((float)numel);
    float* m_ptr = m.has_value() ? m.value().data_ptr<float>() : nullptr;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16, grad.scalar_type(),"fused_update_2d_full_norm",([&]{
        LAUNCH_2D(scalar_t,MODE_NORM,false)(nullptr,grad.data_ptr<scalar_t>(),m_ptr,nullptr,row_fp32.data_ptr<float>(),col_fp32.data_ptr<float>(),row_mean_val.data_ptr<float>(),norm_buf.data_ptr<float>(),nullptr,beta1,R,C,numel,sqrt_numel,m_block_size,0,0,nullptr,0.0f);
    }));
}

void fused_update_2d_full_apply_cuda(
    torch::Tensor param, torch::Tensor grad, c10::optional<torch::Tensor> m,
    torch::Tensor row_fp32, torch::Tensor col_fp32,
    torch::Tensor row_mean_val, torch::Tensor norm_buf, torch::Tensor alpha,
    float beta1, int R, int C, int numel, int m_block_size, float d) {
    int threads = 256, num_warps = threads / 32;
    size_t smem = num_warps * sizeof(float);
    int blocks=(numel+m_block_size-1)/m_block_size;
    float sqrt_numel=sqrtf((float)numel);
    float* m_ptr = m.has_value() ? m.value().data_ptr<float>() : nullptr;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16, param.scalar_type(),"fused_update_2d_full_apply",([&]{
        LAUNCH_2D(scalar_t,MODE_APPLY,false)(param.data_ptr<scalar_t>(),grad.data_ptr<scalar_t>(),m_ptr,nullptr,row_fp32.data_ptr<float>(),col_fp32.data_ptr<float>(),row_mean_val.data_ptr<float>(),norm_buf.data_ptr<float>(),alpha.data_ptr<float>(),beta1,R,C,numel,sqrt_numel,m_block_size,0,0,nullptr,d);
    }));
}

void fused_update_2d_full_lag_cuda(
    torch::Tensor param, torch::Tensor grad, c10::optional<torch::Tensor> m,
    torch::Tensor row_fp32, torch::Tensor col_fp32,
    torch::Tensor row_mean_val, torch::Tensor norm_buf, torch::Tensor alpha, torch::Tensor prev_norm,
    float beta1, int R, int C, int numel, int m_block_size, float d) {
    int threads = 256, num_warps = threads / 32;
    size_t smem = num_warps * sizeof(float);
    int blocks=(numel+m_block_size-1)/m_block_size;
    float sqrt_numel=sqrtf((float)numel);
    float* m_ptr = m.has_value() ? m.value().data_ptr<float>() : nullptr;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16, param.scalar_type(),"fused_update_2d_full_lag",([&]{
        LAUNCH_2D(scalar_t,MODE_LAG,false)(param.data_ptr<scalar_t>(),grad.data_ptr<scalar_t>(),m_ptr,nullptr,row_fp32.data_ptr<float>(),col_fp32.data_ptr<float>(),row_mean_val.data_ptr<float>(),norm_buf.data_ptr<float>(),alpha.data_ptr<float>(),beta1,R,C,numel,sqrt_numel,m_block_size,0,0,prev_norm.data_ptr<float>(),d);
    }));
}
#undef LAUNCH_2D

// ==========================================
// 8. Factored Means (eliminates g_sq temporary)
// ==========================================
template <typename T>
__global__ void compute_factored_sums_kernel(
    const T* __restrict__ grad,
    float* __restrict__ row_sum, float* __restrict__ col_sum,
    int R, int C, int numel) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int lane = threadIdx.x % 32;

    for (int i = tid; i < numel; i += stride) {
        float g = load_grad(grad, i);
        float g_sq = g * g;
        int r = i / C;
        int c = i % C;
        int b = i / (R * C);
        
        warp_atomic_add_row(g_sq, r, row_sum, lane);
        atomicAdd(&col_sum[b * C + c], g_sq);
    }
}

void compute_factored_sums_cuda(torch::Tensor grad, torch::Tensor row_sum, torch::Tensor col_sum,
    int R, int C, int numel) {
    int threads = 256, blocks = min(1024, (numel + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "compute_factored_sums", ([&] {
            compute_factored_sums_kernel<scalar_t><<<blocks, threads>>>(
                grad.data_ptr<scalar_t>(),
                row_sum.data_ptr<float>(), col_sum.data_ptr<float>(),
                R, C, numel);
        }));
}

// ==========================================
// 9. CAME (Unified: supports both quantized M and fp32 M)
// ==========================================
template <typename T>
__global__ void compute_ut_rms_kernel(
    const T* __restrict__ grad,
    const float* __restrict__ row_fp32, const float* __restrict__ col_fp32,
    const float* __restrict__ row_mean_val_ptr,
    float* __restrict__ ut_sq_sum,
    int R, int C, int numel) {
    float sq=0.0f;
    int stride=gridDim.x*blockDim.x;
    for (int idx=blockIdx.x*blockDim.x+threadIdx.x; idx<numel; idx+=stride) {
        int b=idx/(R*C), r=(idx/C)%R, c=idx%C;
        float g=load_grad(grad,idx);
        int r_idx=b*R+r, c_idx=b*C+c;
        float r_val=fmaxf(row_fp32[r_idx],MIN_VAL);
        float c_val=fmaxf(col_fp32[c_idx],MIN_VAL);
        float rm_val=fmaxf(row_mean_val_ptr[b],MIN_VAL);
        float inv_std=sqrtf(rm_val)*rsqrtf(r_val)*rsqrtf(c_val);
        float u=g*inv_std;
        sq+=u*u;
    }
    __shared__ float s_sum[32];
    float total=block_reduce_sum(sq,s_sum,threadIdx.x,blockDim.x/32);
    if(threadIdx.x==0) atomicAdd(ut_sq_sum,total);
}

void compute_ut_rms_cuda(
    torch::Tensor grad,
    torch::Tensor row_fp32, torch::Tensor col_fp32,
    torch::Tensor row_mean_val, torch::Tensor ut_sq_sum,
    int R, int C, int numel) {
    int threads=256, blocks=min(1024,(numel+threads-1)/threads);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,
        grad.scalar_type(),"compute_ut_rms",([&]{
            compute_ut_rms_kernel<scalar_t><<<blocks,threads>>>(
                grad.data_ptr<scalar_t>(),
                row_fp32.data_ptr<float>(),col_fp32.data_ptr<float>(),
                row_mean_val.data_ptr<float>(),ut_sq_sum.data_ptr<float>(),
                R,C,numel);
        }));
}

template <typename T, bool QUANTIZED_M>
__global__ void fused_came_pass1_kernel(
    const T* __restrict__ grad,
    void* __restrict__ m_ptr, float* __restrict__ m_scale,
    const float* __restrict__ row_fp32, const float* __restrict__ col_fp32,
    const float* __restrict__ row_mean_val_ptr,
    float* __restrict__ res_row_sum, float* __restrict__ res_col_sum,
    const float* __restrict__ clip_factor_ptr,
    float beta1, float eps_came,
    int R, int C, int numel, int m_block_size, int m_bits, int m_mode) {
    
    int block_id=blockIdx.x, tid=threadIdx.x;
    int start=block_id*m_block_size;
    
    extern __shared__ float shared_mem[];
    float* s_qmap8 = nullptr;
    float* s_qmap4 = nullptr;
    if constexpr (QUANTIZED_M) {
        s_qmap8 = shared_mem;
        s_qmap4 = s_qmap8 + 256;
        load_codebook_smem(s_qmap8, s_qmap4, tid, blockDim.x);
    }
    __syncthreads();

    float one_minus_b1=1.0f-beta1;
    int elems = QUANTIZED_M ? ((m_bits==8)?1:2) : 1;
    float clip_factor=*clip_factor_ptr;
    int total_slots=m_block_size/elems;
    int slot_iters=(total_slots+blockDim.x-1)/blockDim.x;

    for (int iter=0;iter<slot_iters;iter++) {
        int slot=tid+iter*blockDim.x;
        bool valid=(slot<total_slots);
        int idx0=valid?(start+slot*elems):0;
        float res_0=0.0f, res_1=0.0f;

        if (valid && idx0<numel) {
            int b0=idx0/(R*C), r0=(idx0/C)%R, c0=idx0%C;
            float g=load_grad(grad,idx0);
            int r_idx=b0*R+r0, c_idx=b0*C+c0;
            float r_val=fmaxf(row_fp32[r_idx],MIN_VAL);
            float c_val=fmaxf(col_fp32[c_idx],MIN_VAL);
            float rm_val=fmaxf(row_mean_val_ptr[b0],MIN_VAL);
            float inv_std=sqrtf(rm_val)*rsqrtf(r_val)*rsqrtf(c_val);
            float u_t=g*inv_std/clip_factor;
            
            float m_old, m_new;
            if constexpr (QUANTIZED_M) {
                m_old = dequant_m(m_ptr,idx0,m_scale,m_block_size,m_bits,m_mode,s_qmap8,s_qmap4);
            } else {
                m_old = (m_ptr != nullptr) ? ((float*)m_ptr)[idx0] : 0.0f;
            }
            m_new=beta1*m_old+one_minus_b1*u_t;
            
            if constexpr (!QUANTIZED_M) {
                if (m_ptr != nullptr) ((float*)m_ptr)[idx0] = m_new;
            }
            
            float diff=u_t-m_new;
            res_0=diff*diff+eps_came;
            atomicAdd(&res_col_sum[b0*C+c0],res_0);
        }
        
        if constexpr (QUANTIZED_M) {
            if (valid && elems==2 && idx0+1<numel) {
                int idx1=idx0+1, b1=idx1/(R*C), r1=(idx1/C)%R, c1=idx1%C;
                float g=load_grad(grad,idx1);
                int r_idx=b1*R+r1, c_idx=b1*C+c1;
                float r_val=fmaxf(row_fp32[r_idx],MIN_VAL);
                float c_val=fmaxf(col_fp32[c_idx],MIN_VAL);
                float rm_val=fmaxf(row_mean_val_ptr[b1],MIN_VAL);
                float inv_std=sqrtf(rm_val)*rsqrtf(r_val)*rsqrtf(c_val);
                float u_t=g*inv_std/clip_factor;
                float m_old=dequant_m(m_ptr,idx1,m_scale,m_block_size,m_bits,m_mode,s_qmap8,s_qmap4);
                float m_new=beta1*m_old+one_minus_b1*u_t;
                float diff=u_t-m_new;
                res_1=diff*diff+eps_came;
                atomicAdd(&res_col_sum[b1*C+c1],res_1);
            }
        }

        int lane=tid%32;
        int r_idx0=(valid&&idx0<numel)?(idx0/(R*C))*R+(idx0/C)%R:-1;
        warp_atomic_add_row((valid&&idx0<numel)?res_0:0.0f, r_idx0, res_row_sum, lane);

        if constexpr (QUANTIZED_M) {
            if(elems==2){
                int r_idx1=(valid&&idx0+1<numel)?((idx0+1)/(R*C))*R+((idx0+1)/C)%R:-1;
                warp_atomic_add_row((valid&&idx0+1<numel)?res_1:0.0f, r_idx1, res_row_sum, lane);
            }
        }
    }
}

template <typename T, int MODE, bool QUANTIZED_M>
__global__ void fused_came_pass2_kernel(
    T* __restrict__ param, const T* __restrict__ grad,
    void* __restrict__ m_ptr, float* __restrict__ m_scale,
    const float* __restrict__ row_fp32, const float* __restrict__ col_fp32,
    const float* __restrict__ row_mean_val_ptr,
    const float* __restrict__ conf_row_fp32, const float* __restrict__ conf_col_fp32,
    const float* __restrict__ conf_row_mean_ptr,
    const float* __restrict__ clip_factor_ptr,
    const float* __restrict__ alpha, float* __restrict__ norm_buf,
    float beta1,
    int R, int C, int numel, int m_block_size, int m_bits, int m_mode) {
    
    int block_id=blockIdx.x, tid=threadIdx.x, num_warps=blockDim.x/32;
    int start=block_id*m_block_size;
    extern __shared__ float shared_mem[];
    
    float* smem_m = nullptr;
    // s_reduce is always needed for MODE_NORM (and quantized M's max reduce)
    float* s_reduce = shared_mem; 
    float* s_qmap8 = nullptr;
    float* s_qmap4 = nullptr;
    
    if constexpr (QUANTIZED_M) {
        smem_m = (MODE==MODE_NORM) ? nullptr : shared_mem;
        s_reduce = (MODE==MODE_NORM) ? shared_mem : &shared_mem[m_block_size];
        int reduce_base = (MODE==MODE_NORM) ? 0 : m_block_size;
        s_qmap8 = &shared_mem[reduce_base + num_warps];
        s_qmap4 = s_qmap8 + 256;
        load_codebook_smem(s_qmap8, s_qmap4, tid, blockDim.x);
    }
    __syncthreads();

    float one_minus_b1=1.0f-beta1;
    int elems = QUANTIZED_M ? ((m_bits==8)?1:2) : 1;
    float clip_factor = (clip_factor_ptr != nullptr) ? (*clip_factor_ptr) : 1.0f;
    float alpha_val = (MODE == MODE_APPLY || MODE == MODE_NOCLIP) ? (*alpha) : 0.0f;
    
    int total_slots=m_block_size/elems;
    int slot_iters=(total_slots+blockDim.x-1)/blockDim.x;
    float thread_max=0.0f, local_sq=0.0f;

    for (int iter=0;iter<slot_iters;iter++) {
        int slot=tid+iter*blockDim.x; if(slot>=total_slots) break;
        int idx0=start+slot*elems;
        float m_val_0=0.0f, m_val_1=0.0f;

        if (idx0<numel) {
            int b=idx0/(R*C), r=(idx0/C)%R, c=idx0%C;
            float g=load_grad(grad,idx0);
            int r_idx=b*R+r, c_idx=b*C+c;
            
            float inv_row=sqrtf(fmaxf(conf_row_mean_ptr[b],MIN_VAL))*rsqrtf(fmaxf(conf_row_fp32[r_idx],MIN_VAL));
            float inv_col_c=rsqrtf(fmaxf(conf_col_fp32[c_idx],MIN_VAL));
            
            if constexpr (QUANTIZED_M) {
                float r_val=fmaxf(row_fp32[r_idx],MIN_VAL);
                float c_val=fmaxf(col_fp32[c_idx],MIN_VAL);
                float rm_val=fmaxf(row_mean_val_ptr[b],MIN_VAL);
                float inv_std=sqrtf(rm_val)*rsqrtf(r_val)*rsqrtf(c_val);
                float u_t=g*inv_std/clip_factor;
                float m_old=dequant_m(m_ptr,idx0,m_scale,m_block_size,m_bits,m_mode,s_qmap8,s_qmap4);
                m_val_0=beta1*m_old+one_minus_b1*u_t;
            } else {
                m_val_0 = (m_ptr != nullptr) ? ((float*)m_ptr)[idx0] : 0.0f;
            }
            
            float u_final=m_val_0*inv_row*inv_col_c;
            if constexpr (MODE==MODE_NORM) { local_sq+=u_final*u_final; }
            else { float p=static_cast<float>(param[idx0]); param[idx0]=static_cast<T>(p-alpha_val*u_final); }
            
            if constexpr (QUANTIZED_M) {
                thread_max=fmaxf(thread_max,fabsf(m_val_0));
            }
        }
        
        if constexpr (QUANTIZED_M) {
            if (elems==2 && idx0+1<numel) {
                int idx1=idx0+1, b=idx1/(R*C), r=(idx1/C)%R, c=idx1%C;
                float g=load_grad(grad,idx1);
                int r_idx=b*R+r, c_idx=b*C+c;
                float r_val=fmaxf(row_fp32[r_idx],MIN_VAL);
                float c_val=fmaxf(col_fp32[c_idx],MIN_VAL);
                float rm_val=fmaxf(row_mean_val_ptr[b],MIN_VAL);
                float inv_std=sqrtf(rm_val)*rsqrtf(r_val)*rsqrtf(c_val);
                float u_t=g*inv_std/clip_factor;
                float m_old=dequant_m(m_ptr,idx1,m_scale,m_block_size,m_bits,m_mode,s_qmap8,s_qmap4);
                m_val_1=beta1*m_old+one_minus_b1*u_t;
                float inv_row=sqrtf(fmaxf(conf_row_mean_ptr[b],MIN_VAL))*rsqrtf(fmaxf(conf_row_fp32[r_idx],MIN_VAL));
                float inv_col_c=rsqrtf(fmaxf(conf_col_fp32[c_idx],MIN_VAL));
                float u_final=m_val_1*inv_row*inv_col_c;
                if constexpr (MODE==MODE_NORM) { local_sq+=u_final*u_final; }
                else { float p=static_cast<float>(param[idx1]); param[idx1]=static_cast<T>(p-alpha_val*u_final); }
                thread_max=fmaxf(thread_max,fabsf(m_val_1));
            }
            if constexpr (MODE==MODE_APPLY) {
                smem_m[slot*elems]=m_val_0;
                if(elems==2) smem_m[slot*elems+1]=m_val_1;
            }
        }
    }
    if constexpr (QUANTIZED_M && MODE==MODE_APPLY) {
        float abs_max=fmaxf(block_reduce_max(thread_max,s_reduce,tid,num_warps),MIN_SCALE);
        float inv_abs=1.0f/abs_max;
        for (int iter=0;iter<slot_iters;iter++) {
            int slot=tid+iter*blockDim.x; if(slot>=total_slots) break;
            int idx0=start+slot*elems;
            float m0=smem_m[slot*elems];
            float m1=(elems==2)?smem_m[slot*elems+1]:0.0f;
            if(idx0<numel) quant_m_store(m_ptr,idx0,m0,m1,inv_abs,m_bits,m_mode,numel,s_qmap8,s_qmap4);
        }
        if(tid==0) m_scale[block_id]=abs_max;
    }
    if constexpr (MODE==MODE_NORM) {
        float total_sq=block_reduce_sum(local_sq,s_reduce,tid,num_warps);
        if(tid==0) atomicAdd(norm_buf,total_sq);
    }
}

void fused_came_pass1_cuda(
    torch::Tensor grad, torch::Tensor m_q, torch::Tensor m_scale,
    torch::Tensor row_fp32, torch::Tensor col_fp32,
    torch::Tensor row_mean_val,
    torch::Tensor res_row_sum, torch::Tensor res_col_sum, torch::Tensor clip_factor,
    float beta1, float eps_came,
    int R, int C, int numel, int m_block_size, int m_bits, int m_mode) {
    int threads=calc_threads(m_block_size,m_bits);
    int blocks=(numel+m_block_size-1)/m_block_size;
    size_t smem = SMEM_QMAP * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16, grad.scalar_type(),"fused_came_pass1",([&]{
        fused_came_pass1_kernel<scalar_t, true><<<blocks,threads,smem>>>(
            grad.data_ptr<scalar_t>(),m_q.data_ptr(),m_scale.data_ptr<float>(),
            row_fp32.data_ptr<float>(),col_fp32.data_ptr<float>(),
            row_mean_val.data_ptr<float>(),
            res_row_sum.data_ptr<float>(),res_col_sum.data_ptr<float>(),
            clip_factor.data_ptr<float>(),
            beta1,eps_came,R,C,numel,m_block_size,m_bits,m_mode);
    }));
}

void fused_came_pass2_norm_cuda(
    torch::Tensor grad, torch::Tensor m_q, torch::Tensor m_scale,
    torch::Tensor row_fp32, torch::Tensor col_fp32,
    torch::Tensor row_mean_val,
    torch::Tensor conf_row_fp32, torch::Tensor conf_col_fp32,
    torch::Tensor conf_row_mean,
    torch::Tensor clip_factor, torch::Tensor norm_buf,
    float beta1, int R, int C, int numel, int m_block_size, int m_bits, int m_mode) {
    int threads=calc_threads(m_block_size,m_bits), num_warps=threads/32;
    size_t smem=(num_warps+SMEM_QMAP)*sizeof(float);
    int blocks=(numel+m_block_size-1)/m_block_size;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16, grad.scalar_type(),"fused_came_pass2_norm",([&]{
        fused_came_pass2_kernel<scalar_t,MODE_NORM,true><<<blocks,threads,smem>>>(
            nullptr,grad.data_ptr<scalar_t>(),
            m_q.data_ptr(),m_scale.data_ptr<float>(),
            row_fp32.data_ptr<float>(),col_fp32.data_ptr<float>(),
            row_mean_val.data_ptr<float>(),
            conf_row_fp32.data_ptr<float>(),conf_col_fp32.data_ptr<float>(),
            conf_row_mean.data_ptr<float>(),
            clip_factor.data_ptr<float>(),nullptr,norm_buf.data_ptr<float>(),
            beta1,R,C,numel,m_block_size,m_bits,m_mode);
    }));
}

void fused_came_pass2_cuda(
    torch::Tensor param, torch::Tensor grad, torch::Tensor m_q, torch::Tensor m_scale,
    torch::Tensor row_fp32, torch::Tensor col_fp32,
    torch::Tensor row_mean_val,
    torch::Tensor conf_row_fp32, torch::Tensor conf_col_fp32,
    torch::Tensor conf_row_mean,
    torch::Tensor clip_factor, torch::Tensor alpha,
    float beta1, int R, int C, int numel, int m_block_size, int m_bits, int m_mode) {
    int threads=calc_threads(m_block_size,m_bits), num_warps=threads/32;
    size_t smem=(m_block_size+num_warps+SMEM_QMAP)*sizeof(float);
    int blocks=(numel+m_block_size-1)/m_block_size;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16, param.scalar_type(),"fused_came_pass2",([&]{
        fused_came_pass2_kernel<scalar_t,MODE_APPLY,true><<<blocks,threads,smem>>>(
            param.data_ptr<scalar_t>(),grad.data_ptr<scalar_t>(),
            m_q.data_ptr(),m_scale.data_ptr<float>(),
            row_fp32.data_ptr<float>(),col_fp32.data_ptr<float>(),
            row_mean_val.data_ptr<float>(),
            conf_row_fp32.data_ptr<float>(),conf_col_fp32.data_ptr<float>(),
            conf_row_mean.data_ptr<float>(),
            clip_factor.data_ptr<float>(),alpha.data_ptr<float>(),nullptr,
            beta1,R,C,numel,m_block_size,m_bits,m_mode);
    }));
}

void fused_came_full_pass1_cuda(
    torch::Tensor grad, torch::Tensor m,
    torch::Tensor row_fp32, torch::Tensor col_fp32,
    torch::Tensor row_mean_val,
    torch::Tensor res_row_sum, torch::Tensor res_col_sum,
    torch::Tensor clip_factor,
    float beta1, float eps_came,
    int R, int C, int numel, int m_block_size) {
    int threads = 256;
    int blocks=(numel+m_block_size-1)/m_block_size;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16, grad.scalar_type(),"fused_came_full_pass1",([&]{
        fused_came_pass1_kernel<scalar_t, false><<<blocks,threads,0>>>(
            grad.data_ptr<scalar_t>(),m.data_ptr<float>(),nullptr,
            row_fp32.data_ptr<float>(),col_fp32.data_ptr<float>(),
            row_mean_val.data_ptr<float>(),
            res_row_sum.data_ptr<float>(),res_col_sum.data_ptr<float>(),
            clip_factor.data_ptr<float>(),
            beta1,eps_came,R,C,numel,m_block_size,0,0);
    }));
}

void fused_came_full_pass2_cuda(
    torch::Tensor param, torch::Tensor m,
    torch::Tensor conf_row_fp32, torch::Tensor conf_col_fp32,
    torch::Tensor conf_row_mean,
    c10::optional<torch::Tensor> norm_buf, torch::Tensor alpha,
    int R, int C, int numel, int m_block_size, bool norm_only) {
    int threads = 256, num_warps = threads / 32;
    size_t smem = num_warps * sizeof(float);
    int blocks=(numel+m_block_size-1)/m_block_size;
    float* norm_ptr = norm_buf.has_value() ? norm_buf.value().data_ptr<float>() : nullptr;
    
    if (norm_only) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16, param.scalar_type(),"fused_came_full_pass2_norm",([&]{
            fused_came_pass2_kernel<scalar_t,MODE_NORM,false><<<blocks,threads,smem>>>(
                param.data_ptr<scalar_t>(),nullptr,
                m.data_ptr<float>(),nullptr,
                nullptr,nullptr,nullptr,
                conf_row_fp32.data_ptr<float>(),conf_col_fp32.data_ptr<float>(),
                conf_row_mean.data_ptr<float>(),
                nullptr,alpha.data_ptr<float>(),norm_ptr,
                0.0f,R,C,numel,m_block_size,0,0);
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16, param.scalar_type(),"fused_came_full_pass2",([&]{
            fused_came_pass2_kernel<scalar_t,MODE_NOCLIP,false><<<blocks,threads,smem>>>(
                param.data_ptr<scalar_t>(),nullptr,
                m.data_ptr<float>(),nullptr,
                nullptr,nullptr,nullptr,
                conf_row_fp32.data_ptr<float>(),conf_col_fp32.data_ptr<float>(),
                conf_row_mean.data_ptr<float>(),
                nullptr,alpha.data_ptr<float>(),norm_ptr,
                0.0f,R,C,numel,m_block_size,0,0);
        }));
    }
}

// ==========================================
// 10. Apollo Kernels
// ==========================================
__global__ void compute_apollo_norms_kernel(
    const float* __restrict__ m_fp32,
    const float* __restrict__ v_fp32,
    const float* __restrict__ grad_low,
    float* __restrict__ norm_update, float* __restrict__ norm_grad,
    int N, int D, int stride_N, int stride_D,
    float apollo_eps) {
    int row=blockIdx.x; if(row>=N) return;
    int tid=threadIdx.x, stride=blockDim.x;

    float sum_u2=0.0f, sum_g2=0.0f;
    for (int i=tid;i<D;i+=stride) {
        int gi=row*stride_N+i*stride_D;
        float m_val=m_fp32[gi];
        float v_val=v_fp32[gi];
        float u_val=m_val/(sqrtf(v_val)+apollo_eps);
        float g_val=grad_low[gi];
        sum_u2+=u_val*u_val; sum_g2+=g_val*g_val;
    }
    for(int o=16;o>0;o/=2){sum_u2+=__shfl_down_sync(0xffffffff,sum_u2,o);sum_g2+=__shfl_down_sync(0xffffffff,sum_g2,o);}
    __shared__ float s_u2[32]; __shared__ float s_g2[32];
    int lane=tid%32, wid=tid/32, nw=blockDim.x/32;
    if(lane==0){s_u2[wid]=sum_u2;s_g2[wid]=sum_g2;}
    __syncthreads();
    if(wid==0){
        sum_u2=(lane<nw)?s_u2[lane]:0.0f; sum_g2=(lane<nw)?s_g2[lane]:0.0f;
        for(int o=16;o>0;o/=2){sum_u2+=__shfl_down_sync(0xffffffff,sum_u2,o);sum_g2+=__shfl_down_sync(0xffffffff,sum_g2,o);}
        if(lane==0){norm_update[row]=sqrtf(sum_u2);norm_grad[row]=sqrtf(sum_g2);}
    }
}

void compute_apollo_norms_cuda(torch::Tensor m_fp32,
    torch::Tensor v_fp32, torch::Tensor grad_low,
    torch::Tensor norm_update, torch::Tensor norm_grad,
    int N, int D, int stride_N, int stride_D,
    float apollo_eps) {
    int threads=256; if(D<256)threads=128; if(D<128)threads=64; if(D<64)threads=32;
    compute_apollo_norms_kernel<<<N,threads>>>(
        m_fp32.data_ptr<float>(),
        v_fp32.data_ptr<float>(),
        grad_low.data_ptr<float>(),
        norm_update.data_ptr<float>(),norm_grad.data_ptr<float>(),
        N,D,stride_N,stride_D,apollo_eps);
}

__global__ void compute_apollo_came_rms_kernel(
    const float* __restrict__ grad_low,
    const float* __restrict__ v_fp32,
    float* __restrict__ sum_u2, float eps_sq, int numel) {
    float sq=0.0f; int stride=gridDim.x*blockDim.x;
    for (int idx=blockIdx.x*blockDim.x+threadIdx.x;idx<numel;idx+=stride) {
        float v_val=v_fp32[idx];
        float g=(isnan(grad_low[idx])||isinf(grad_low[idx]))?0.0f:grad_low[idx];
        float u=g*rsqrtf(fmaxf(v_val,eps_sq)); sq+=u*u;
    }
    __shared__ float s_sum[32];
    float total=block_reduce_sum(sq,s_sum,threadIdx.x,blockDim.x/32);
    if(threadIdx.x==0) atomicAdd(sum_u2,total);
}

void compute_apollo_came_rms_cuda(torch::Tensor grad_low, torch::Tensor v_fp32,
    torch::Tensor sum_u2, float eps_sq, int numel) {
    int threads=256, blocks=min(1024,(numel+threads-1)/threads);
    compute_apollo_came_rms_kernel<<<blocks,threads>>>(
        grad_low.data_ptr<float>(),v_fp32.data_ptr<float>(),
        sum_u2.data_ptr<float>(),eps_sq,numel);
}

__global__ void apollo_came_compute_m_res_kernel(
    const float* __restrict__ grad_low,
    const float* __restrict__ v_fp32,
    const void* __restrict__ m_q_old, const float* __restrict__ m_scale_old,
    const float* __restrict__ res_fp32_old,
    float* __restrict__ m_temp, float* __restrict__ res_temp,
    const float* __restrict__ clip_factor_ptr,
    float beta1, float beta3, float eps_came, float eps_sq,
    int m_block_size, int numel, int m_bits, int m_mode) {
    float clip_factor=*clip_factor_ptr;
    int stride=gridDim.x*blockDim.x;
    __shared__ float s_qmap8[256];
    __shared__ float s_qmap4[16];
    load_codebook_smem(s_qmap8, s_qmap4, threadIdx.x, blockDim.x);
    __syncthreads();

    float one_minus_b1=1.0f-beta1, one_minus_b3=1.0f-beta3;
    for (int idx=blockIdx.x*blockDim.x+threadIdx.x;idx<numel;idx+=stride) {
        float v_val=v_fp32[idx];
        float g=(isnan(grad_low[idx])||isinf(grad_low[idx]))?0.0f:grad_low[idx];
        float u_t=g*rsqrtf(fmaxf(v_val,eps_sq))/clip_factor;
        float m_old=dequant_m(m_q_old,idx,m_scale_old,m_block_size,m_bits,m_mode,s_qmap8,s_qmap4);
        float m_new=beta1*m_old+one_minus_b1*u_t;
        m_temp[idx]=m_new;
        float diff=u_t-m_new;
        float c_old=res_fp32_old[idx];
        res_temp[idx]=fmaxf(beta3*c_old+one_minus_b3*(diff*diff+eps_came),MIN_VAL);
    }
}

void apollo_came_compute_m_res_cuda(torch::Tensor grad_low, torch::Tensor v_fp32,
    torch::Tensor m_q_old, torch::Tensor m_scale_old,
    torch::Tensor res_fp32_old,
    torch::Tensor m_temp, torch::Tensor res_temp, torch::Tensor clip_factor,
    float beta1, float beta3, float eps_came, float eps_sq,
    int m_block_size, int numel, int m_bits, int m_mode) {
    int threads=256, blocks=min(1024,(numel+threads-1)/threads);
    apollo_came_compute_m_res_kernel<<<blocks,threads>>>(
        grad_low.data_ptr<float>(),v_fp32.data_ptr<float>(),
        m_q_old.data_ptr(),m_scale_old.data_ptr<float>(),
        res_fp32_old.data_ptr<float>(),
        m_temp.data_ptr<float>(),res_temp.data_ptr<float>(),clip_factor.data_ptr<float>(),
        beta1,beta3,eps_came,eps_sq,m_block_size,numel,m_bits,m_mode);
}

__global__ void apollo_came_compute_update_norms_kernel(
    const float* __restrict__ m_fp32_new,
    const float* __restrict__ res_fp32_new,
    const float* __restrict__ grad_low,
    float* __restrict__ norm_update, float* __restrict__ norm_grad,
    float eps_sq, int N, int D, int stride_N, int stride_D, int numel) {
    int channel=blockIdx.x; if(channel>=N) return;
    int tid=threadIdx.x, stride=blockDim.x;

    float sum_u2=0.0f, sum_g2=0.0f;
    for (int i=tid;i<D;i+=stride) {
        int idx=channel*stride_N+i*stride_D; if(idx>=numel) break;
        float m_val=m_fp32_new[idx];
        float res_val=res_fp32_new[idx];
        float u_final=m_val*rsqrtf(fmaxf(res_val,eps_sq));
        sum_u2+=u_final*u_final;
        float g=(isnan(grad_low[idx])||isinf(grad_low[idx]))?0.0f:grad_low[idx];
        sum_g2+=g*g;
    }
    for(int o=16;o>0;o/=2){sum_u2+=__shfl_down_sync(0xffffffff,sum_u2,o);sum_g2+=__shfl_down_sync(0xffffffff,sum_g2,o);}
    __shared__ float s_u2[32]; __shared__ float s_g2[32];
    int lane=tid%32, wid=tid/32, nw=blockDim.x/32;
    if(lane==0){s_u2[wid]=sum_u2;s_g2[wid]=sum_g2;}
    __syncthreads();
    if(wid==0){
        sum_u2=(lane<nw)?s_u2[lane]:0.0f; sum_g2=(lane<nw)?s_g2[lane]:0.0f;
        for(int o=16;o>0;o/=2){sum_u2+=__shfl_down_sync(0xffffffff,sum_u2,o);sum_g2+=__shfl_down_sync(0xffffffff,sum_g2,o);}
        if(lane==0){norm_update[channel]=sqrtf(sum_u2);norm_grad[channel]=sqrtf(sum_g2);}
    }
}

void apollo_came_compute_update_norms_cuda(torch::Tensor m_fp32_new,
    torch::Tensor res_fp32_new,
    torch::Tensor grad_low,
    torch::Tensor norm_update, torch::Tensor norm_grad,
    float eps_sq, int N, int D, int stride_N, int stride_D, int numel) {
    int threads=256; if(D<256)threads=128; if(D<128)threads=64; if(D<64)threads=32;
    apollo_came_compute_update_norms_kernel<<<N,threads>>>(
        m_fp32_new.data_ptr<float>(),
        res_fp32_new.data_ptr<float>(),
        grad_low.data_ptr<float>(),
        norm_update.data_ptr<float>(),norm_grad.data_ptr<float>(),
        eps_sq,N,D,stride_N,stride_D,numel);
}

// ==========================================
// 11. Dequantize Utilities
// ==========================================
__global__ void dequantize_dynamic_kernel(
    float* __restrict__ output, const unsigned char* __restrict__ q,
    const float* __restrict__ scale, int numel, int block_size, int m_bits) {
    __shared__ float s_qmap8[256];
    __shared__ float s_qmap4[16];
    load_codebook_smem(s_qmap8, s_qmap4, threadIdx.x, blockDim.x);
    __syncthreads();

    int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=numel) return;
    float s=scale[idx/block_size];
    if (m_bits==8) output[idx]=s_qmap8[q[idx]]*s;
    else { unsigned char packed=q[idx/2]; int qi=(idx&1)?(packed&0x0F):(packed>>4); output[idx]=s_qmap4[qi]*s; }
}

void dequantize_dynamic_cuda(torch::Tensor output, torch::Tensor q, torch::Tensor scale,
    int numel, int block_size, int m_bits) {
    int threads=256, blocks=(numel+threads-1)/threads;
    dequantize_dynamic_kernel<<<blocks,threads>>>(
        output.data_ptr<float>(),q.data_ptr<unsigned char>(),scale.data_ptr<float>(),numel,block_size,m_bits);
}

template <int V_BITS>
__global__ void dequantize_log_nonneg_kernel(
    float* __restrict__ output, const void* __restrict__ q_void,
    const float* __restrict__ scale, const float* __restrict__ min_log,
    int numel, int block_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    int bid = idx / block_size;
    float s = scale[bid];
    float ml = min_log[bid];
    if constexpr (V_BITS == 8) {
        const unsigned char* q = reinterpret_cast<const unsigned char*>(q_void);
        unsigned char qi = q[idx];
        output[idx] = (qi == 0) ? 0.0f : exp2f((float)(qi - 1) * INV_254 * s + ml);
    } else {
        const unsigned short* q = reinterpret_cast<const unsigned short*>(q_void);
        unsigned short qi = q[idx];
        output[idx] = (qi == 0) ? 0.0f : exp2f((float)(qi - 1) * INV_65534 * s + ml);
    }
}

void dequantize_log_nonneg_cuda(torch::Tensor output, torch::Tensor q,
    torch::Tensor scale, torch::Tensor min_log, int numel, int block_size, int v_bits) {
    if (v_bits == 8) {
        TORCH_CHECK(q.scalar_type() == at::kByte, "v_bits=8 requires q to be torch.uint8");
    } else {
        TORCH_CHECK(q.scalar_type() == at::kShort, "v_bits=16 requires q to be torch.int16");
    }
    TORCH_CHECK(scale.scalar_type() == at::kFloat && min_log.scalar_type() == at::kFloat, 
                "scale and min_log must be torch.float32");
    int threads = 256, blocks = (numel + threads - 1) / threads;
    if (v_bits == 8) {
        dequantize_log_nonneg_kernel<8><<<blocks, threads>>>(
            output.data_ptr<float>(), q.data_ptr(),
            scale.data_ptr<float>(), min_log.data_ptr<float>(), numel, block_size);
    } else {
        dequantize_log_nonneg_kernel<16><<<blocks, threads>>>(
            output.data_ptr<float>(), q.data_ptr(),
            scale.data_ptr<float>(), min_log.data_ptr<float>(), numel, block_size);
    }
}

// ==========================================
// PYBIND11
// ==========================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("set_qmap", &set_qmap_cuda);
    m.def("fused_log_quantize_lerp", &fused_log_quantize_lerp_cuda);
    m.def("fused_m_quantize_lerp", &fused_m_quantize_lerp_cuda);
    m.def("fused_update_1d_noclip", &fused_update_1d_noclip_cuda);
    m.def("fused_update_1d_norm", &fused_update_1d_norm_cuda);
    m.def("fused_update_1d_apply", &fused_update_1d_apply_cuda);
    m.def("fused_update_1d_lag", &fused_update_1d_lag_cuda);
    m.def("fused_update_1d_vonly_noclip", &fused_update_1d_vonly_noclip_cuda);
    m.def("fused_update_1d_vonly_norm", &fused_update_1d_vonly_norm_cuda);
    m.def("fused_update_1d_vonly_apply", &fused_update_1d_vonly_apply_cuda);
    m.def("fused_update_1d_vonly_lag", &fused_update_1d_vonly_lag_cuda);
    m.def("fused_update_1d_full_noclip", &fused_update_1d_full_noclip_cuda);
    m.def("fused_update_1d_full_norm", &fused_update_1d_full_norm_cuda);
    m.def("fused_update_1d_full_apply", &fused_update_1d_full_apply_cuda);
    m.def("fused_update_1d_full_lag", &fused_update_1d_full_lag_cuda);
    m.def("fused_update_2d_noclip", &fused_update_2d_noclip_cuda);
    m.def("fused_update_2d_norm", &fused_update_2d_norm_cuda);
    m.def("fused_update_2d_apply", &fused_update_2d_apply_cuda);
    m.def("fused_update_2d_lag", &fused_update_2d_lag_cuda);
    m.def("fused_update_2d_full_noclip", &fused_update_2d_full_noclip_cuda);
    m.def("fused_update_2d_full_norm", &fused_update_2d_full_norm_cuda);
    m.def("fused_update_2d_full_apply", &fused_update_2d_full_apply_cuda);
    m.def("fused_update_2d_full_lag", &fused_update_2d_full_lag_cuda);
    m.def("compute_factored_sums", &compute_factored_sums_cuda);
    m.def("compute_ut_rms", &compute_ut_rms_cuda);
    m.def("fused_came_pass1", &fused_came_pass1_cuda);
    m.def("fused_came_pass2_norm", &fused_came_pass2_norm_cuda);
    m.def("fused_came_pass2", &fused_came_pass2_cuda);
    m.def("fused_came_full_pass1", &fused_came_full_pass1_cuda);
    m.def("fused_came_full_pass2", &fused_came_full_pass2_cuda);
    m.def("compute_apollo_norms", &compute_apollo_norms_cuda);
    m.def("compute_apollo_came_rms", &compute_apollo_came_rms_cuda);
    m.def("apollo_came_compute_m_res", &apollo_came_compute_m_res_cuda);
    m.def("apollo_came_compute_update_norms", &apollo_came_compute_update_norms_cuda);
    m.def("dequantize_dynamic", &dequantize_dynamic_cuda);
    m.def("dequantize_log_nonneg", &dequantize_log_nonneg_cuda);
}