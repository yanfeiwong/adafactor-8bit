// Copyright (c) 2026 WANG YAN
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include <torch/extension.h>

__device__ constexpr float INV_255 = 1.0f / 255.0f;
__device__ constexpr float MIN_LOG = -126.0f; 
__device__ constexpr float MIN_VAL = 1.17549435e-38f; 

__device__ __forceinline__ float dequant_uniform_4bit(int q_int, float scale) {
    return ((float)q_int - 8.0f) * (scale * 0.125f);
}

__device__ __forceinline__ int quant_uniform_4bit(float x, float inv_scale) {
    int q = __float2int_rn(x * inv_scale);
    q = max(-8, min(7, q));
    return q + 8;
}

// ==========================================
// 1. Fused Log-Quantize Lerp (EMA Update for V_t)
// ==========================================
__global__ void fused_log_quantize_lerp_kernel(
    unsigned char* __restrict__ q,
    float* __restrict__ scale,
    const float* __restrict__ new_val,
    const float beta,
    int block_size,
    bool square_input,
    float eps1,
    int N)
{
    int block_id = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    int start = block_id * block_size;
    int num_warps = stride / 32;

    extern __shared__ float shared_mem[];
    float* local_logs = shared_mem;           
    float* s_max = &shared_mem[block_size];

    float old_scale = scale[block_id];
    float thread_max = MIN_LOG;
    float one_minus_b = 1.0f - beta;
    
    const float4* new_val_vec = reinterpret_cast<const float4*>(new_val + start);
    uchar4* q_vec = reinterpret_cast<uchar4*>(q + start);

    int vec_iters = (block_size / 4) / stride; 

    for (int i = 0; i < vec_iters; i++) {
        int idx = tid + i * stride; 
        int base_idx = start + idx * 4;
        
        float val_x, val_y, val_z, val_w;
        
        if (base_idx + 3 < N) {
            // Vectorized read for aligned elements
            float4 nv = new_val_vec[idx];
            if (square_input) {
                nv.x = nv.x * nv.x + eps1;
                nv.y = nv.y * nv.y + eps1;
                nv.z = nv.z * nv.z + eps1;
                nv.w = nv.w * nv.w + eps1;
            }
            val_x = (isnan(nv.x) || isinf(nv.x)) ? 0.0f : nv.x;
            val_y = (isnan(nv.y) || isinf(nv.y)) ? 0.0f : nv.y;
            val_z = (isnan(nv.z) || isinf(nv.z)) ? 0.0f : nv.z;
            val_w = (isnan(nv.w) || isinf(nv.w)) ? 0.0f : nv.w;
        } else {
            // Scalar fallback for boundary elements
            float v0 = (base_idx + 0 < N) ? new_val[base_idx + 0] : 0.0f;
            float v1 = (base_idx + 1 < N) ? new_val[base_idx + 1] : 0.0f;
            float v2 = (base_idx + 2 < N) ? new_val[base_idx + 2] : 0.0f;
            float v3 = (base_idx + 3 < N) ? new_val[base_idx + 3] : 0.0f;
            if (square_input) {
                v0 = v0 * v0 + eps1; v1 = v1 * v1 + eps1; v2 = v2 * v2 + eps1; v3 = v3 * v3 + eps1;
            }
            val_x = (isnan(v0) || isinf(v0)) ? 0.0f : v0;
            val_y = (isnan(v1) || isinf(v1)) ? 0.0f : v1;
            val_z = (isnan(v2) || isinf(v2)) ? 0.0f : v2;
            val_w = (isnan(v3) || isinf(v3)) ? 0.0f : v3;
        }

        uchar4 q_val = q_vec[idx];

        float v_old0 = exp2f((float)q_val.x * INV_255 * old_scale + MIN_LOG);
        float v_old1 = exp2f((float)q_val.y * INV_255 * old_scale + MIN_LOG);
        float v_old2 = exp2f((float)q_val.z * INV_255 * old_scale + MIN_LOG);
        float v_old3 = exp2f((float)q_val.w * INV_255 * old_scale + MIN_LOG);

        float v_upd0 = fmaxf(v_old0 * one_minus_b + fmaxf(val_x, MIN_VAL) * beta, MIN_VAL);
        float v_upd1 = fmaxf(v_old1 * one_minus_b + fmaxf(val_y, MIN_VAL) * beta, MIN_VAL);
        float v_upd2 = fmaxf(v_old2 * one_minus_b + fmaxf(val_z, MIN_VAL) * beta, MIN_VAL);
        float v_upd3 = fmaxf(v_old3 * one_minus_b + fmaxf(val_w, MIN_VAL) * beta, MIN_VAL);

        float log0 = log2f(v_upd0);
        float log1 = log2f(v_upd1);
        float log2 = log2f(v_upd2);
        float log3 = log2f(v_upd3);

        local_logs[idx * 4 + 0] = log0;
        local_logs[idx * 4 + 1] = log1;
        local_logs[idx * 4 + 2] = log2;
        local_logs[idx * 4 + 3] = log3;

        thread_max = fmaxf(thread_max, fmaxf(fmaxf(log0, log1), fmaxf(log2, log3)));
    }

    // Warp-level reduce
    float val = thread_max;
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    if (tid % 32 == 0) s_max[tid / 32] = val;
    __syncthreads();

    // Block-level reduce
    if (tid < 32) { 
        if (tid < num_warps) val = s_max[tid];
        else val = MIN_LOG;
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_down_sync(0xffffffff, val, offset);
            if (tid + offset < num_warps) val = fmaxf(val, other);
        }
        if (tid == 0) s_max[0] = val;
    }
    __syncthreads(); 

    float max_log = fminf(fmaxf(s_max[0], MIN_LOG + 1e-12f), 126.0f);
    float new_scale = max_log - MIN_LOG; 
    float inv_scale = 255.0f / (max_log - MIN_LOG);

    for (int i = 0; i < vec_iters; i++) {
        int idx = tid + i * stride;
        
        float l0 = local_logs[idx * 4 + 0];
        float l1 = local_logs[idx * 4 + 1];
        float l2 = local_logs[idx * 4 + 2];
        float l3 = local_logs[idx * 4 + 3];

        uchar4 out_q;
        out_q.x = (unsigned char)fminf(fmaxf((l0 - MIN_LOG) * inv_scale + 0.5f, 0.0f), 255.0f);
        out_q.y = (unsigned char)fminf(fmaxf((l1 - MIN_LOG) * inv_scale + 0.5f, 0.0f), 255.0f);
        out_q.z = (unsigned char)fminf(fmaxf((l2 - MIN_LOG) * inv_scale + 0.5f, 0.0f), 255.0f);
        out_q.w = (unsigned char)fminf(fmaxf((l3 - MIN_LOG) * inv_scale + 0.5f, 0.0f), 255.0f);

        q_vec[idx] = out_q; 
    }

    if (tid == 0) scale[block_id] = new_scale;
}

torch::Tensor fused_log_quantize_lerp_cuda(
    torch::Tensor q, torch::Tensor scale, torch::Tensor new_val, 
    float beta, int block_size, bool square_input, float eps1, int N)
{
    TORCH_CHECK(q.scalar_type() == at::kByte && scale.scalar_type() == at::kFloat && new_val.scalar_type() == at::kFloat);
    TORCH_CHECK(q.is_cuda() && scale.is_cuda() && new_val.is_cuda());
    TORCH_CHECK(q.is_contiguous() && scale.is_contiguous() && new_val.is_contiguous());
    
    int threads = 256;
    TORCH_CHECK(block_size >= threads && block_size % 4 == 0 && block_size % (4 * threads) == 0,
                "block_size must be a multiple of 4 * threads for vectorization.");

    int num_blocks = (N + block_size - 1) / block_size;
    int num_warps = threads / 32;
    size_t shared_mem = (block_size + num_warps) * sizeof(float); 

    TORCH_CHECK(shared_mem <= 49152, "block_size is too large, exceeding 48KB shared memory limit.");
    
    fused_log_quantize_lerp_kernel<<<num_blocks, threads, shared_mem>>>(
        q.data_ptr<unsigned char>(), scale.data_ptr<float>(), new_val.data_ptr<float>(), 
        beta, block_size, square_input, eps1, N
    );
    return q;
}

// ==========================================
// 2. M_t (Momentum) EMA Update (Physical 4-bit Packed Quantization)
// ==========================================
__global__ void fused_4bit_quantize_lerp_kernel(
    unsigned char* __restrict__ q, 
    float* __restrict__ scale,
    const float* __restrict__ new_val,
    const float beta,
    int block_size,
    int N) 
{
    int block_id = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    int start = block_id * block_size;
    int num_warps = stride / 32;

    extern __shared__ float shared_mem[];
    float* local_m = shared_mem;           
    float* s_max = &shared_mem[block_size];

    float old_scale = scale[block_id];
    float thread_max = 0.0f;
    float one_minus_b = 1.0f - beta;
    
    const float4* new_val_vec = reinterpret_cast<const float4*>(new_val + start);
    uchar2* q_vec = reinterpret_cast<uchar2*>(q + (start / 2)); 

    int total_vecs = block_size / 4;
    int vec_iters = (total_vecs + stride - 1) / stride;

    for (int i = 0; i < vec_iters; i++) {
        int idx = tid + i * stride; 
        if (idx >= total_vecs) break;
        
        int base_idx = start + idx * 4;
        float val_x, val_y, val_z, val_w;

        if (base_idx + 3 < N) {
            float4 nv = new_val_vec[idx];
            val_x = (isnan(nv.x) || isinf(nv.x)) ? 0.0f : nv.x;
            val_y = (isnan(nv.y) || isinf(nv.y)) ? 0.0f : nv.y;
            val_z = (isnan(nv.z) || isinf(nv.z)) ? 0.0f : nv.z;
            val_w = (isnan(nv.w) || isinf(nv.w)) ? 0.0f : nv.w;
        } else {
            float v0 = (base_idx + 0 < N) ? new_val[base_idx + 0] : 0.0f;
            float v1 = (base_idx + 1 < N) ? new_val[base_idx + 1] : 0.0f;
            float v2 = (base_idx + 2 < N) ? new_val[base_idx + 2] : 0.0f;
            float v3 = (base_idx + 3 < N) ? new_val[base_idx + 3] : 0.0f;
            val_x = (isnan(v0) || isinf(v0)) ? 0.0f : v0;
            val_y = (isnan(v1) || isinf(v1)) ? 0.0f : v1;
            val_z = (isnan(v2) || isinf(v2)) ? 0.0f : v2;
            val_w = (isnan(v3) || isinf(v3)) ? 0.0f : v3;
        }

        uchar2 old_q = q_vec[idx];

        float m_old0 = dequant_uniform_4bit(old_q.x >> 4, old_scale);
        float m_old1 = dequant_uniform_4bit(old_q.x & 0x0F, old_scale);
        float m_old2 = dequant_uniform_4bit(old_q.y >> 4, old_scale);
        float m_old3 = dequant_uniform_4bit(old_q.y & 0x0F, old_scale);

        float m_new0 = beta * m_old0 + one_minus_b * val_x;
        float m_new1 = beta * m_old1 + one_minus_b * val_y;
        float m_new2 = beta * m_old2 + one_minus_b * val_z;
        float m_new3 = beta * m_old3 + one_minus_b * val_w;

        local_m[idx * 4 + 0] = m_new0;
        local_m[idx * 4 + 1] = m_new1;
        local_m[idx * 4 + 2] = m_new2;
        local_m[idx * 4 + 3] = m_new3;

        thread_max = fmaxf(thread_max, fabsf(m_new0));
        thread_max = fmaxf(thread_max, fabsf(m_new1));
        thread_max = fmaxf(thread_max, fabsf(m_new2));
        thread_max = fmaxf(thread_max, fabsf(m_new3));
    }

    float val = thread_max;
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));

    if (tid % 32 == 0) s_max[tid / 32] = val;
    __syncthreads();

    if (tid < 32) { 
        if (tid < num_warps) val = s_max[tid];
        else val = 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_down_sync(0xffffffff, val, offset);
            if (tid + offset < num_warps) val = fmaxf(val, other);
        }
        if (tid == 0) s_max[0] = val;
    }
    __syncthreads(); 

    float abs_max = fmaxf(s_max[0], 1e-12f);
    float new_scale = abs_max;
    float inv_scale = 8.0f / new_scale;

    for (int i = 0; i < vec_iters; i++) {
        int idx = tid + i * stride;
        if (idx >= total_vecs) break;
        
        float m0 = local_m[idx * 4 + 0];
        float m1 = local_m[idx * 4 + 1];
        float m2 = local_m[idx * 4 + 2];
        float m3 = local_m[idx * 4 + 3];

        int q0 = quant_uniform_4bit(m0, inv_scale);
        int q1 = quant_uniform_4bit(m1, inv_scale);
        int q2 = quant_uniform_4bit(m2, inv_scale);
        int q3 = quant_uniform_4bit(m3, inv_scale);

        uchar2 out_q;
        out_q.x = (unsigned char)((q0 << 4) | q1);
        out_q.y = (unsigned char)((q2 << 4) | q3);

        q_vec[idx] = out_q;
    }

    if (tid == 0) scale[block_id] = new_scale;
}

void fused_4bit_quantize_lerp_cuda(
    torch::Tensor q, torch::Tensor scale, torch::Tensor new_val, 
    float beta, int block_size, int N)
{
    TORCH_CHECK(q.scalar_type() == at::kByte && scale.scalar_type() == at::kFloat && new_val.scalar_type() == at::kFloat);
    TORCH_CHECK(q.is_cuda() && scale.is_cuda() && new_val.is_cuda());
    TORCH_CHECK(q.is_contiguous() && scale.is_contiguous() && new_val.is_contiguous());
    
    int total_vecs = block_size / 4;
    int threads = min(256, ((total_vecs + 31) / 32) * 32);
    if (threads == 0) threads = 32;

    TORCH_CHECK(block_size >= 32 && block_size % 4 == 0, 
                "block_size must be a multiple of 4 and >= 32 for 4-bit quantization.");

    int num_blocks = (N + block_size - 1) / block_size;
    int num_warps = threads / 32;
    size_t shared_mem = (block_size + num_warps) * sizeof(float); 

    TORCH_CHECK(shared_mem <= 49152, "block_size is too large, exceeding 48KB shared memory limit.");
    
    fused_4bit_quantize_lerp_kernel<<<num_blocks, threads, shared_mem>>>(
        q.data_ptr<unsigned char>(), scale.data_ptr<float>(), new_val.data_ptr<float>(), 
        beta, block_size, N
    );
}

// ==========================================
// 3. Phase 1: Compute Update Norm (2D, V_t only)
// ==========================================
__global__ void compute_update_norm_2d_kernel(
    const unsigned char* __restrict__ row_var_q, const float* __restrict__ row_var_scale,
    const unsigned char* __restrict__ col_var_q, const float* __restrict__ col_var_scale,
    const float* __restrict__ grad, float* __restrict__ total_sum_sq,
    const float* __restrict__ row_mean_val_ptr, float log_eps_sq, int R, int C, int numel, int block_size)
{
    float sq = 0.0f;
    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        int b = idx / (R * C);
        int r = (idx / C) % R;
        int c = idx % C;
        
        float log_r = (float)row_var_q[b * R + r] * INV_255 * row_var_scale[(b * R + r) / block_size] + MIN_LOG;
        float log_c = (float)col_var_q[b * C + c] * INV_255 * col_var_scale[(b * C + c) / block_size] + MIN_LOG;
        float log_row_mean = log2f(fmaxf(row_mean_val_ptr[b], MIN_VAL));

        float log_v_ij = log_r + log_c - log_row_mean; 
        
        float max_log = fmaxf(log_v_ij, log_eps_sq);
        max_log = fmaxf(max_log, -53.0f); 
        float inv_std = exp2f(-0.5f * max_log); 
        
        float g_val = (isnan(grad[idx]) || isinf(grad[idx])) ? 0.0f : grad[idx];
        float u_ij = g_val * inv_std;
        
        sq += u_ij * u_ij;
    }
    
    for (int offset = 16; offset > 0; offset /= 2) sq += __shfl_down_sync(0xffffffff, sq, offset);
    int lane = threadIdx.x % 32; int wid = threadIdx.x / 32; int num_warps = blockDim.x / 32;
    __shared__ float s_sum[32];
    if (lane == 0) s_sum[wid] = sq;
    __syncthreads();
    if (wid == 0) {
        if (lane < num_warps) sq = s_sum[lane];
        else sq = 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_down_sync(0xffffffff, sq, offset);
            if (lane + offset < num_warps) sq += other;
        }
        if (lane == 0) atomicAdd(total_sum_sq, sq);
    }
}

void compute_update_norm_2d_cuda(
    torch::Tensor row_var_q, torch::Tensor row_var_scale,
    torch::Tensor col_var_q, torch::Tensor col_var_scale,
    torch::Tensor grad, torch::Tensor total_sum_sq,
    torch::Tensor row_mean_val, float log_eps_sq, int R, int C, int numel, int block_size)
{
    int threads = 256;
    int blocks = min(1024, (numel + threads - 1) / threads);
    compute_update_norm_2d_kernel<<<blocks, threads>>>(
        row_var_q.data_ptr<unsigned char>(), row_var_scale.data_ptr<float>(),
        col_var_q.data_ptr<unsigned char>(), col_var_scale.data_ptr<float>(),
        grad.data_ptr<float>(), total_sum_sq.data_ptr<float>(),
        row_mean_val.data_ptr<float>(), log_eps_sq, R, C, numel, block_size
    );
}

// ==========================================
// 4. Phase 2: Apply Update (2D, V_t only)
// ==========================================
template <typename scalar_t>
__global__ void apply_update_2d_kernel(
    scalar_t* __restrict__ param, const float* __restrict__ grad,
    const unsigned char* __restrict__ row_var_q, const float* __restrict__ row_var_scale,
    const unsigned char* __restrict__ col_var_q, const float* __restrict__ col_var_scale,
    const float* __restrict__ sum_sq_ptr, const float* __restrict__ alpha,
    const float* __restrict__ row_mean_val_ptr, float d, float log_eps_sq, int R, int C, int numel, int block_size)
{
    float sum_sq_val = *sum_sq_ptr;
    float alpha_val = *alpha;
    float denom = fmaxf(1.0f, sqrtf(sum_sq_val) / (sqrtf((float)numel) * d));
    float step_size = alpha_val / denom;

    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        int b = idx / (R * C);
        int r = (idx / C) % R;
        int c = idx % C;
        
        float log_r = (float)row_var_q[b * R + r] * INV_255 * row_var_scale[(b * R + r) / block_size] + MIN_LOG;
        float log_c = (float)col_var_q[b * C + c] * INV_255 * col_var_scale[(b * C + c) / block_size] + MIN_LOG;
        float log_row_mean = log2f(fmaxf(row_mean_val_ptr[b], MIN_VAL));

        float log_v_ij = log_r + log_c - log_row_mean; 
        
        float max_log = fmaxf(log_v_ij, log_eps_sq);
        max_log = fmaxf(max_log, -53.0f); 
        float inv_std = exp2f(-0.5f * max_log); 
        
        float g_val = (isnan(grad[idx]) || isinf(grad[idx])) ? 0.0f : grad[idx];
        float u_ij = g_val * inv_std;
        
        float p_val = static_cast<float>(param[idx]);
        p_val -= step_size * u_ij;
        param[idx] = static_cast<scalar_t>(p_val);
    }
}

void apply_update_2d_cuda(
    torch::Tensor param, torch::Tensor grad,
    torch::Tensor row_var_q, torch::Tensor row_var_scale,
    torch::Tensor col_var_q, torch::Tensor col_var_scale,
    torch::Tensor sum_sq, torch::Tensor alpha,
    torch::Tensor row_mean_val, float d, float log_eps_sq, int R, int C, int numel, int block_size)
{
    int threads = 256;
    int blocks = min(1024, (numel + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, 
        param.scalar_type(), "apply_update_2d_cuda", ([&] {
            apply_update_2d_kernel<scalar_t><<<blocks, threads>>>(
                param.data_ptr<scalar_t>(), grad.data_ptr<float>(),
                row_var_q.data_ptr<unsigned char>(), row_var_scale.data_ptr<float>(),
                col_var_q.data_ptr<unsigned char>(), col_var_scale.data_ptr<float>(),
                sum_sq.data_ptr<float>(), alpha.data_ptr<float>(),
                row_mean_val.data_ptr<float>(), d, log_eps_sq, R, C, numel, block_size
            );
        }));
}

// ==========================================
// 5. 1D Vector Variance Kernels (V_t only)
// ==========================================
__global__ void compute_update_norm_1d_kernel(
    const unsigned char* __restrict__ variance_q, const float* __restrict__ variance_scale,
    const float* __restrict__ grad, float* __restrict__ total_sum_sq,
    float log_eps_sq, int numel, int block_size)
{
    float sq = 0.0f;
    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        float log_v = (float)variance_q[idx] * INV_255 * variance_scale[idx / block_size] + MIN_LOG;
        
        float max_log = fmaxf(log_v, log_eps_sq);
        max_log = fmaxf(max_log, -53.0f); 
        float inv_std = exp2f(-0.5f * max_log); 
        
        float g_val = (isnan(grad[idx]) || isinf(grad[idx])) ? 0.0f : grad[idx];
        float u_val = g_val * inv_std;
        
        sq += u_val * u_val;
    }
    for (int offset = 16; offset > 0; offset /= 2) sq += __shfl_down_sync(0xffffffff, sq, offset);
    int lane = threadIdx.x % 32; int wid = threadIdx.x / 32; int num_warps = blockDim.x / 32;
    __shared__ float s_sum[32];
    if (lane == 0) s_sum[wid] = sq;
    __syncthreads();
    if (wid == 0) {
        if (lane < num_warps) sq = s_sum[lane];
        else sq = 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_down_sync(0xffffffff, sq, offset);
            if (lane + offset < num_warps) sq += other;
        }
        if (lane == 0) atomicAdd(total_sum_sq, sq);
    }
}

void compute_update_norm_1d_cuda(
    torch::Tensor variance_q, torch::Tensor variance_scale,
    torch::Tensor grad, torch::Tensor total_sum_sq, float log_eps_sq, int numel, int block_size)
{
    int threads = 256;
    int blocks = min(1024, (numel + threads - 1) / threads);
    compute_update_norm_1d_kernel<<<blocks, threads>>>(
        variance_q.data_ptr<unsigned char>(), variance_scale.data_ptr<float>(),
        grad.data_ptr<float>(), total_sum_sq.data_ptr<float>(), log_eps_sq, numel, block_size
    );
}

template <typename scalar_t>
__global__ void apply_update_1d_kernel(
    scalar_t* __restrict__ param, const float* __restrict__ grad,
    const unsigned char* __restrict__ variance_q, const float* __restrict__ variance_scale,
    const float* __restrict__ sum_sq_ptr, const float* __restrict__ alpha,
    float d, float log_eps_sq, int numel, int block_size)
{
    float sum_sq_val = *sum_sq_ptr;
    float alpha_val = *alpha;
    float denom = fmaxf(1.0f, sqrtf(sum_sq_val) / (sqrtf((float)numel) * d));
    float step_size = alpha_val / denom;

    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        float log_v = (float)variance_q[idx] * INV_255 * variance_scale[idx / block_size] + MIN_LOG;
        
        float max_log = fmaxf(log_v, log_eps_sq);
        max_log = fmaxf(max_log, -53.0f); 
        float inv_std = exp2f(-0.5f * max_log); 
        
        float g_val = (isnan(grad[idx]) || isinf(grad[idx])) ? 0.0f : grad[idx];
        float u_val = g_val * inv_std;

        float p_val = static_cast<float>(param[idx]);
        p_val -= step_size * u_val;
        param[idx] = static_cast<scalar_t>(p_val);
    }
}

void apply_update_1d_cuda(
    torch::Tensor param, torch::Tensor grad,
    torch::Tensor variance_q, torch::Tensor variance_scale,
    torch::Tensor sum_sq, torch::Tensor alpha, float d, float log_eps_sq, int numel, int block_size)
{
    int threads = 256;
    int blocks = min(1024, (numel + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, 
        param.scalar_type(), "apply_update_1d_cuda", ([&] {
            apply_update_1d_kernel<scalar_t><<<blocks, threads>>>(
                param.data_ptr<scalar_t>(), grad.data_ptr<float>(),
                variance_q.data_ptr<unsigned char>(), variance_scale.data_ptr<float>(),
                sum_sq.data_ptr<float>(), alpha.data_ptr<float>(), d, log_eps_sq, numel, block_size
            );
        }));
}

// ==========================================
// 6. Phase 1: Compute Update Norm with M_t (2D, M_t + V_t)
// ==========================================
__global__ void compute_update_norm_m_2d_kernel(
    const unsigned char* __restrict__ m_q, const float* __restrict__ m_scale,
    const unsigned char* __restrict__ row_var_q, const float* __restrict__ row_var_scale,
    const unsigned char* __restrict__ col_var_q, const float* __restrict__ col_var_scale,
    float* __restrict__ total_sum_sq,
    const float* __restrict__ row_mean_val_ptr, float log_eps_sq, int R, int C, int numel, int m_block_size, int v_block_size)
{
    float sq = 0.0f;
    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        int b = idx / (R * C);
        int r = (idx / C) % R;
        int c = idx % C;
        
        // Unpack 4-bit m_t
        unsigned char packed = m_q[idx / 2];
        int q_int = (idx & 1) ? (packed & 0x0F) : (packed >> 4);
        float m_val = dequant_uniform_4bit(q_int, m_scale[idx / m_block_size]);
        
        float log_r = (float)row_var_q[b * R + r] * INV_255 * row_var_scale[(b * R + r) / v_block_size] + MIN_LOG;
        float log_c = (float)col_var_q[b * C + c] * INV_255 * col_var_scale[(b * C + c) / v_block_size] + MIN_LOG;
        float log_row_mean = log2f(fmaxf(row_mean_val_ptr[b], MIN_VAL));
        float log_v_ij = log_r + log_c - log_row_mean; 
        
        float max_log = fmaxf(log_v_ij, log_eps_sq);
        max_log = fmaxf(max_log, -53.0f); 
        float inv_std = exp2f(-0.5f * max_log); 
        
        float u_ij = m_val * inv_std; 
        sq += u_ij * u_ij;
    }
    
    for (int offset = 16; offset > 0; offset /= 2) sq += __shfl_down_sync(0xffffffff, sq, offset);
    int lane = threadIdx.x % 32; int wid = threadIdx.x / 32; int num_warps = blockDim.x / 32;
    __shared__ float s_sum[32];
    if (lane == 0) s_sum[wid] = sq;
    __syncthreads();
    if (wid == 0) {
        if (lane < num_warps) sq = s_sum[lane];
        else sq = 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_down_sync(0xffffffff, sq, offset);
            if (lane + offset < num_warps) sq += other;
        }
        if (lane == 0) atomicAdd(total_sum_sq, sq);
    }
}

void compute_update_norm_m_2d_cuda(
    torch::Tensor m_q, torch::Tensor m_scale,
    torch::Tensor row_var_q, torch::Tensor row_var_scale,
    torch::Tensor col_var_q, torch::Tensor col_var_scale,
    torch::Tensor total_sum_sq,
    torch::Tensor row_mean_val, float log_eps_sq, int R, int C, int numel, int m_block_size, int v_block_size)
{
    int threads = 256;
    int blocks = min(1024, (numel + threads - 1) / threads);
    compute_update_norm_m_2d_kernel<<<blocks, threads>>>(
        m_q.data_ptr<unsigned char>(), m_scale.data_ptr<float>(),
        row_var_q.data_ptr<unsigned char>(), row_var_scale.data_ptr<float>(),
        col_var_q.data_ptr<unsigned char>(), col_var_scale.data_ptr<float>(),
        total_sum_sq.data_ptr<float>(),
        row_mean_val.data_ptr<float>(), log_eps_sq, R, C, numel, m_block_size, v_block_size
    );
}

// ==========================================
// 7. Phase 2: Apply Final Update with M_t (2D, M_t + V_t)
// ==========================================
template <typename scalar_t>
__global__ void apply_update_m_2d_kernel(
    scalar_t* __restrict__ param, 
    const unsigned char* __restrict__ m_q, const float* __restrict__ m_scale,
    const unsigned char* __restrict__ row_var_q, const float* __restrict__ row_var_scale,
    const unsigned char* __restrict__ col_var_q, const float* __restrict__ col_var_scale,
    const float* __restrict__ sum_sq_ptr, const float* __restrict__ alpha,
    const float* __restrict__ row_mean_val_ptr, float d, float log_eps_sq, int R, int C, int numel, int m_block_size, int v_block_size)
{
    float sum_sq_val = *sum_sq_ptr;
    float alpha_val = *alpha;
    float denom = fmaxf(1.0f, sqrtf(sum_sq_val) / (sqrtf((float)numel) * d));
    float step_size = alpha_val / denom;

    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        int b = idx / (R * C);
        int r = (idx / C) % R;
        int c = idx % C;
        
        // Unpack 4-bit m_t
        unsigned char packed = m_q[idx / 2];
        int q_int = (idx & 1) ? (packed & 0x0F) : (packed >> 4);
        float m_val = dequant_uniform_4bit(q_int, m_scale[idx / m_block_size]);
        
        float log_r = (float)row_var_q[b * R + r] * INV_255 * row_var_scale[(b * R + r) / v_block_size] + MIN_LOG;
        float log_c = (float)col_var_q[b * C + c] * INV_255 * col_var_scale[(b * C + c) / v_block_size] + MIN_LOG;
        float log_row_mean = log2f(fmaxf(row_mean_val_ptr[b], MIN_VAL));
        float log_v_ij = log_r + log_c - log_row_mean; 
        
        float max_log = fmaxf(log_v_ij, log_eps_sq);
        max_log = fmaxf(max_log, -53.0f); 
        float inv_std = exp2f(-0.5f * max_log); 
        
        float u_ij = m_val * inv_std; 
        
        float p_val = static_cast<float>(param[idx]);
        p_val -= step_size * u_ij;
        param[idx] = static_cast<scalar_t>(p_val);
    }
}

void apply_update_m_2d_cuda(
    torch::Tensor param, 
    torch::Tensor m_q, torch::Tensor m_scale,
    torch::Tensor row_var_q, torch::Tensor row_var_scale,
    torch::Tensor col_var_q, torch::Tensor col_var_scale,
    torch::Tensor sum_sq, torch::Tensor alpha,
    torch::Tensor row_mean_val, float d, float log_eps_sq, int R, int C, int numel, int m_block_size, int v_block_size)
{
    int threads = 256;
    int blocks = min(1024, (numel + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, 
        param.scalar_type(), "apply_update_m_2d_cuda", ([&] {
            apply_update_m_2d_kernel<scalar_t><<<blocks, threads>>>(
                param.data_ptr<scalar_t>(), 
                m_q.data_ptr<unsigned char>(), m_scale.data_ptr<float>(),
                row_var_q.data_ptr<unsigned char>(), row_var_scale.data_ptr<float>(),
                col_var_q.data_ptr<unsigned char>(), col_var_scale.data_ptr<float>(),
                sum_sq.data_ptr<float>(), alpha.data_ptr<float>(),
                row_mean_val.data_ptr<float>(), d, log_eps_sq, R, C, numel, m_block_size, v_block_size
            );
        }));
}

// ==========================================
// 8. Phase 1 & 2: 1D Compute Norm & Apply Update with M_t (M_t + V_t)
// ==========================================
__global__ void compute_update_norm_m_1d_kernel(
    const unsigned char* __restrict__ m_q, const float* __restrict__ m_scale,
    const unsigned char* __restrict__ variance_q, const float* __restrict__ variance_scale,
    float* __restrict__ total_sum_sq, float log_eps_sq, int numel, int m_block_size, int v_block_size)
{
    float sq = 0.0f;
    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        // Unpack 4-bit m_t
        unsigned char packed = m_q[idx / 2];
        int q_int = (idx & 1) ? (packed & 0x0F) : (packed >> 4);
        float m_val = dequant_uniform_4bit(q_int, m_scale[idx / m_block_size]);
        
        float log_v = (float)variance_q[idx] * INV_255 * variance_scale[idx / v_block_size] + MIN_LOG;
        
        float max_log = fmaxf(log_v, log_eps_sq);
        max_log = fmaxf(max_log, -53.0f); 
        float inv_std = exp2f(-0.5f * max_log); 
        
        float u_val = m_val * inv_std;
        sq += u_val * u_val;
    }
    for (int offset = 16; offset > 0; offset /= 2) sq += __shfl_down_sync(0xffffffff, sq, offset);
    int lane = threadIdx.x % 32; int wid = threadIdx.x / 32; int num_warps = blockDim.x / 32;
    __shared__ float s_sum[32];
    if (lane == 0) s_sum[wid] = sq;
    __syncthreads();
    if (wid == 0) {
        if (lane < num_warps) sq = s_sum[lane];
        else sq = 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_down_sync(0xffffffff, sq, offset);
            if (lane + offset < num_warps) sq += other;
        }
        if (lane == 0) atomicAdd(total_sum_sq, sq);
    }
}

void compute_update_norm_m_1d_cuda(
    torch::Tensor m_q, torch::Tensor m_scale,
    torch::Tensor variance_q, torch::Tensor variance_scale,
    torch::Tensor total_sum_sq, float log_eps_sq, int numel, int m_block_size, int v_block_size)
{
    int threads = 256;
    int blocks = min(1024, (numel + threads - 1) / threads);
    compute_update_norm_m_1d_kernel<<<blocks, threads>>>(
        m_q.data_ptr<unsigned char>(), m_scale.data_ptr<float>(),
        variance_q.data_ptr<unsigned char>(), variance_scale.data_ptr<float>(),
        total_sum_sq.data_ptr<float>(), log_eps_sq, numel, m_block_size, v_block_size
    );
}

template <typename scalar_t>
__global__ void apply_update_m_1d_kernel(
    scalar_t* __restrict__ param, 
    const unsigned char* __restrict__ m_q, const float* __restrict__ m_scale,
    const unsigned char* __restrict__ variance_q, const float* __restrict__ variance_scale,
    const float* __restrict__ sum_sq_ptr, const float* __restrict__ alpha,
    float d, float log_eps_sq, int numel, int m_block_size, int v_block_size)
{
    float sum_sq_val = *sum_sq_ptr;
    float alpha_val = *alpha;
    float denom = fmaxf(1.0f, sqrtf(sum_sq_val) / (sqrtf((float)numel) * d));
    float step_size = alpha_val / denom;

    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        // Unpack 4-bit m_t
        unsigned char packed = m_q[idx / 2];
        int q_int = (idx & 1) ? (packed & 0x0F) : (packed >> 4);
        float m_val = dequant_uniform_4bit(q_int, m_scale[idx / m_block_size]);
        
        float log_v = (float)variance_q[idx] * INV_255 * variance_scale[idx / v_block_size] + MIN_LOG;
        
        float max_log = fmaxf(log_v, log_eps_sq);
        max_log = fmaxf(max_log, -53.0f); 
        float inv_std = exp2f(-0.5f * max_log); 
        
        float u_val = m_val * inv_std;

        float p_val = static_cast<float>(param[idx]);
        p_val -= step_size * u_val;
        param[idx] = static_cast<scalar_t>(p_val);
    }
}

void apply_update_m_1d_cuda(
    torch::Tensor param, 
    torch::Tensor m_q, torch::Tensor m_scale,
    torch::Tensor variance_q, torch::Tensor variance_scale,
    torch::Tensor sum_sq, torch::Tensor alpha, float d, float log_eps_sq, int numel, int m_block_size, int v_block_size)
{
    int threads = 256;
    int blocks = min(1024, (numel + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, 
        param.scalar_type(), "apply_update_m_1d_cuda", ([&] {
            apply_update_m_1d_kernel<scalar_t><<<blocks, threads>>>(
                param.data_ptr<scalar_t>(), 
                m_q.data_ptr<unsigned char>(), m_scale.data_ptr<float>(),
                variance_q.data_ptr<unsigned char>(), variance_scale.data_ptr<float>(),
                sum_sq.data_ptr<float>(), alpha.data_ptr<float>(), d, log_eps_sq, numel, m_block_size, v_block_size
            );
        }));
}

// ==========================================
// 9. Apollo Channel Norms (4-bit m + 8-bit v)
// ==========================================
__global__ void compute_apollo_norms_kernel(
    const unsigned char* __restrict__ m_q, const float* __restrict__ m_scale,
    const unsigned char* __restrict__ v_q, const float* __restrict__ v_scale,
    const float* __restrict__ grad_low,
    float* __restrict__ norm_update, float* __restrict__ norm_grad,
    int N, int D, int stride_N, int stride_D,
    int m_block_size, int v_block_size, float apollo_eps)
{
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    float sum_u2 = 0.0f;
    float sum_g2 = 0.0f;
    for (int i = tid; i < D; i += stride) {
        int global_idx = row * stride_N + i * stride_D;
        // 4-bit m
        unsigned char m_byte = m_q[global_idx / 2];
        int m_int = (global_idx & 1) ? (m_byte & 0x0F) : (m_byte >> 4);
        float m_val = dequant_uniform_4bit(m_int, m_scale[global_idx / m_block_size]);
        // 8-bit log v
        unsigned char v_byte = v_q[global_idx];
        float log_v = (float)v_byte * INV_255 * v_scale[global_idx / v_block_size] + MIN_LOG;

        // Clamp log variance to prevent numerical overflow
        float max_log = fmaxf(log_v, -53.0f); 
        float v_val = exp2f(max_log);
        
        float u_val = m_val / (sqrtf(v_val) + apollo_eps);
        float g_val = grad_low[global_idx];
        sum_u2 += u_val * u_val;
        sum_g2 += g_val * g_val;
    }
    // Warp-level reduce
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_u2 += __shfl_down_sync(0xffffffff, sum_u2, offset);
        sum_g2 += __shfl_down_sync(0xffffffff, sum_g2, offset);
    }
    // Block-level reduce
    __shared__ float s_u2[32];
    __shared__ float s_g2[32];
    int lane = tid % 32;
    int wid = tid / 32;
    int num_warps = blockDim.x / 32;
    if (lane == 0) {
        s_u2[wid] = sum_u2;
        s_g2[wid] = sum_g2;
    }
    __syncthreads();
    if (wid == 0) {
        if (lane < num_warps) {
            sum_u2 = s_u2[lane];
            sum_g2 = s_g2[lane];
        } else {
            sum_u2 = 0.0f;
            sum_g2 = 0.0f;
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_u2 += __shfl_down_sync(0xffffffff, sum_u2, offset);
            sum_g2 += __shfl_down_sync(0xffffffff, sum_g2, offset);
        }
        if (lane == 0) {
            norm_update[row] = sqrtf(sum_u2);
            norm_grad[row] = sqrtf(sum_g2);
        }
    }
}

void compute_apollo_norms_cuda(
    torch::Tensor m_q, torch::Tensor m_scale,
    torch::Tensor v_q, torch::Tensor v_scale,
    torch::Tensor grad_low,
    torch::Tensor norm_update, torch::Tensor norm_grad,
    int N, int D, int stride_N, int stride_D,
    int m_block_size, int v_block_size, float apollo_eps)
{
    int threads = 256;
    if (D < 256) threads = 128;
    if (D < 128) threads = 64;
    if (D < 64) threads = 32;
    compute_apollo_norms_kernel<<<N, threads>>>(
        m_q.data_ptr<unsigned char>(), m_scale.data_ptr<float>(),
        v_q.data_ptr<unsigned char>(), v_scale.data_ptr<float>(),
        grad_low.data_ptr<float>(),
        norm_update.data_ptr<float>(), norm_grad.data_ptr<float>(),
        N, D, stride_N, stride_D,
        m_block_size, v_block_size, apollo_eps
    );
}

// ==========================================
// 10. Dequantize 4-bit (Tensor-wise fallback)
// ==========================================
__global__ void dequantize_4bit_kernel(
    float* __restrict__ output,
    const unsigned char* __restrict__ q,
    const float* __restrict__ scale,
    int numel, int block_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    unsigned char packed = q[idx / 2];
    int q_int = (idx & 1) ? (packed & 0x0F) : (packed >> 4);
    output[idx] = dequant_uniform_4bit(q_int, scale[idx / block_size]);
}

void dequantize_4bit_cuda(
    torch::Tensor output, torch::Tensor q, torch::Tensor scale,
    int numel, int block_size)
{
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    dequantize_4bit_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(), q.data_ptr<unsigned char>(),
        scale.data_ptr<float>(), numel, block_size
    );
}


// ==========================================
// 11. Full Precision 1D Variance Update & Norm Compute
// ==========================================
__global__ void compute_update_norm_1d_full_kernel(
    float* __restrict__ variance,
    const float* __restrict__ grad,
    float* __restrict__ total_sum_sq,
    float beta, float eps_sq, int numel)
{
    float sq = 0.0f;
    int stride = gridDim.x * blockDim.x;
    float one_minus_b = 1.0f - beta;
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        float g = (isnan(grad[idx]) || isinf(grad[idx])) ? 0.0f : grad[idx];
        float g2 = g * g;
        float v = one_minus_b * variance[idx] + beta * g2;
        variance[idx] = v;
        float inv_std = rsqrtf(fmaxf(v, eps_sq));
        float u = g * inv_std;
        sq += u * u;
    }
    
    for (int offset = 16; offset > 0; offset /= 2) sq += __shfl_down_sync(0xffffffff, sq, offset);
    int lane = threadIdx.x % 32; int wid = threadIdx.x / 32; int num_warps = blockDim.x / 32;
    __shared__ float s_sum[32];
    if (lane == 0) s_sum[wid] = sq;
    __syncthreads();
    if (wid == 0) {
        if (lane < num_warps) sq = s_sum[lane];
        else sq = 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_down_sync(0xffffffff, sq, offset);
            if (lane + offset < num_warps) sq += other;
        }
        if (lane == 0) atomicAdd(total_sum_sq, sq);
    }
}

void compute_update_norm_1d_full_cuda(
    torch::Tensor variance, torch::Tensor grad, torch::Tensor total_sum_sq, 
    float beta, float eps_sq, int numel)
{
    int threads = 256;
    int blocks = min(1024, (numel + threads - 1) / threads);
    compute_update_norm_1d_full_kernel<<<blocks, threads>>>(
        variance.data_ptr<float>(), grad.data_ptr<float>(), total_sum_sq.data_ptr<float>(),
        beta, eps_sq, numel
    );
}

// ==========================================
// 12. Full Precision 1D Apply Update
// ==========================================
template <typename scalar_t>
__global__ void apply_update_1d_full_kernel(
    scalar_t* __restrict__ param,
    const float* __restrict__ variance,
    const float* __restrict__ grad,
    const float* __restrict__ sum_sq_ptr,
    const float* __restrict__ alpha,
    float d, float eps_sq, int numel)
{
    float sum_sq_val = *sum_sq_ptr;
    float alpha_val = *alpha;
    float denom = fmaxf(1.0f, sqrtf(sum_sq_val) / (sqrtf((float)numel) * d));
    float step_size = alpha_val / denom;

    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        float g = (isnan(grad[idx]) || isinf(grad[idx])) ? 0.0f : grad[idx];
        float v = variance[idx];
        float inv_std = rsqrtf(fmaxf(v, eps_sq));
        float u = g * inv_std;

        float p_val = static_cast<float>(param[idx]);
        p_val -= step_size * u;
        param[idx] = static_cast<scalar_t>(p_val);
    }
}

void apply_update_1d_full_cuda(
    torch::Tensor param, torch::Tensor variance, torch::Tensor grad,
    torch::Tensor sum_sq, torch::Tensor alpha, float d, float eps_sq, int numel)
{
    int threads = 256;
    int blocks = min(1024, (numel + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, 
        param.scalar_type(), "apply_update_1d_full_cuda", ([&] {
            apply_update_1d_full_kernel<scalar_t><<<blocks, threads>>>(
                param.data_ptr<scalar_t>(), variance.data_ptr<float>(), grad.data_ptr<float>(),
                sum_sq.data_ptr<float>(), alpha.data_ptr<float>(), d, eps_sq, numel
            );
        }));
}



// ==========================================
// 13. Full Precision 1D Variance & Momentum Update + Norm Compute
// ==========================================
__global__ void compute_update_norm_1d_full_m_kernel(
    float* __restrict__ variance,
    float* __restrict__ m,
    const float* __restrict__ grad,
    float* __restrict__ total_sum_sq,
    float beta1, float beta_val, float eps_sq, int numel)
{
    float sq = 0.0f;
    int stride = gridDim.x * blockDim.x;
    float one_minus_b1 = 1.0f - beta1;
    float one_minus_bv = 1.0f - beta_val;
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        float g = (isnan(grad[idx]) || isinf(grad[idx])) ? 0.0f : grad[idx];
        float g2 = g * g;

        float v = one_minus_bv * variance[idx] + beta_val * g2;
        variance[idx] = v;
        
        float m_new = beta1 * m[idx] + one_minus_b1 * g;
        m[idx] = m_new;
        
        float inv_std = rsqrtf(fmaxf(v, eps_sq));
        float u = m_new * inv_std;
        sq += u * u;
    }
    
    for (int offset = 16; offset > 0; offset /= 2) sq += __shfl_down_sync(0xffffffff, sq, offset);
    int lane = threadIdx.x % 32; int wid = threadIdx.x / 32; int num_warps = blockDim.x / 32;
    __shared__ float s_sum[32];
    if (lane == 0) s_sum[wid] = sq;
    __syncthreads();
    if (wid == 0) {
        if (lane < num_warps) sq = s_sum[lane];
        else sq = 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_down_sync(0xffffffff, sq, offset);
            if (lane + offset < num_warps) sq += other;
        }
        if (lane == 0) atomicAdd(total_sum_sq, sq);
    }
}

void compute_update_norm_1d_full_m_cuda(
    torch::Tensor variance, torch::Tensor m, torch::Tensor grad, torch::Tensor total_sum_sq, 
    float beta1, float beta_val, float eps_sq, int numel)
{
    int threads = 256;
    int blocks = min(1024, (numel + threads - 1) / threads);
    compute_update_norm_1d_full_m_kernel<<<blocks, threads>>>(
        variance.data_ptr<float>(), m.data_ptr<float>(), grad.data_ptr<float>(), total_sum_sq.data_ptr<float>(),
        beta1, beta_val, eps_sq, numel
    );
}

// ==========================================
// 14. Full Precision 1D Apply Update with Momentum
// ==========================================
template <typename scalar_t>
__global__ void apply_update_1d_full_m_kernel(
    scalar_t* __restrict__ param,
    const float* __restrict__ variance,
    const float* __restrict__ m,
    const float* __restrict__ sum_sq_ptr,
    const float* __restrict__ alpha,
    float d, float eps_sq, int numel)
{
    float sum_sq_val = *sum_sq_ptr;
    float alpha_val = *alpha;
    float denom = fmaxf(1.0f, sqrtf(sum_sq_val) / (sqrtf((float)numel) * d));
    float step_size = alpha_val / denom;

    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        float v = variance[idx];
        float m_val = m[idx];
        float inv_std = rsqrtf(fmaxf(v, eps_sq));
        float u = m_val * inv_std;

        float p_val = static_cast<float>(param[idx]);
        p_val -= step_size * u;
        param[idx] = static_cast<scalar_t>(p_val);
    }
}

void apply_update_1d_full_m_cuda(
    torch::Tensor param, torch::Tensor variance, torch::Tensor m,
    torch::Tensor sum_sq, torch::Tensor alpha, float d, float eps_sq, int numel)
{
    int threads = 256;
    int blocks = min(1024, (numel + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, 
        param.scalar_type(), "apply_update_1d_full_m_cuda", ([&] {
            apply_update_1d_full_m_kernel<scalar_t><<<blocks, threads>>>(
                param.data_ptr<scalar_t>(), variance.data_ptr<float>(), m.data_ptr<float>(),
                sum_sq.data_ptr<float>(), alpha.data_ptr<float>(), d, eps_sq, numel
            );
        }));
}


// ==========================================
// 15. CAME: Compute Residual Variance (Row & Col)
// ==========================================
__global__ void came_compute_residual_2d_kernel(
    const unsigned char* __restrict__ m_q, const float* __restrict__ m_scale,
    const float* __restrict__ u_t,                             
    float* __restrict__ res_row_sum, float* __restrict__ res_col_sum,
    float eps_came, int R, int C, int numel, int m_block_size)   
{
    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        int b = idx / (R * C);
        int r = (idx / C) % R;
        int c = idx % C;
        
        unsigned char packed = m_q[idx / 2];
        int q_int = (idx & 1) ? (packed & 0x0F) : (packed >> 4);
        float m_val = dequant_uniform_4bit(q_int, m_scale[idx / m_block_size]);
        
        float u_val = u_t[idx];
        float diff = u_val - m_val;
        float res = diff * diff + eps_came;
        
        atomicAdd(&res_col_sum[b * C + c], res);
        
        int row_idx = b * R + r;
        int lane = threadIdx.x % 32;
        
        for (int offset = 16; offset > 0; offset /= 2) {
            int other_row_idx = __shfl_down_sync(0xffffffff, row_idx, offset);
            float other_res = __shfl_down_sync(0xffffffff, res, offset);
            if (lane + offset < 32 && row_idx == other_row_idx) {
                res += other_res;
            }
        }
        
        int prev_row_idx = __shfl_up_sync(0xffffffff, row_idx, 1);
        bool is_first_in_row = (lane == 0) || (row_idx != prev_row_idx);
        
        if (is_first_in_row) {
            atomicAdd(&res_row_sum[row_idx], res);
        }
    }
}

void came_compute_residual_2d_cuda(
    torch::Tensor m_q, torch::Tensor m_scale,
    torch::Tensor u_t, 
    torch::Tensor res_row_sum, torch::Tensor res_col_sum,
    float eps_came, int R, int C, int numel, int m_block_size)
{
    int threads = 256;
    int blocks = min(1024, (numel + threads - 1) / threads);
    came_compute_residual_2d_kernel<<<blocks, threads>>>(
        m_q.data_ptr<unsigned char>(), m_scale.data_ptr<float>(),
        u_t.data_ptr<float>(),
        res_row_sum.data_ptr<float>(), res_col_sum.data_ptr<float>(),
        eps_came, R, C, numel, m_block_size
    );
}



// ==========================================
// 16. Apollo+CAME Phase 1: Compute RMS for U_t
// ==========================================
__global__ void compute_apollo_came_rms_kernel(
    const float* __restrict__ grad_low,
    const unsigned char* __restrict__ v_q, const float* __restrict__ v_scale,
    float* __restrict__ sum_u2,
    float eps_sq, int numel, int v_block_size)
{
    float sq = 0.0f;
    int stride = gridDim.x * blockDim.x;
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        float log_v = (float)v_q[idx] * INV_255 * v_scale[idx / v_block_size] + MIN_LOG;
        float v_val = exp2f(fmaxf(log_v, -53.0f));
        
        float g_val = (isnan(grad_low[idx]) || isinf(grad_low[idx])) ? 0.0f : grad_low[idx];
        float u_t = g_val * rsqrtf(fmaxf(v_val, eps_sq));
        sq += u_t * u_t;
    }
    
    for (int offset = 16; offset > 0; offset /= 2) sq += __shfl_down_sync(0xffffffff, sq, offset);
    int lane = threadIdx.x % 32; int wid = threadIdx.x / 32; int num_warps = blockDim.x / 32;
    __shared__ float s_sum[32];
    if (lane == 0) s_sum[wid] = sq;
    __syncthreads();
    if (wid == 0) {
        if (lane < num_warps) sq = s_sum[lane];
        else sq = 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_down_sync(0xffffffff, sq, offset);
            if (lane + offset < num_warps) sq += other;
        }
        if (lane == 0) atomicAdd(sum_u2, sq);
    }
}

void compute_apollo_came_rms_cuda(
    torch::Tensor grad_low, torch::Tensor v_q, torch::Tensor v_scale,
    torch::Tensor sum_u2, float eps_sq, int numel, int v_block_size)
{
    int threads = 256;
    int blocks = min(1024, (numel + threads - 1) / threads);
    compute_apollo_came_rms_kernel<<<blocks, threads>>>(
        grad_low.data_ptr<float>(), v_q.data_ptr<unsigned char>(), v_scale.data_ptr<float>(),
        sum_u2.data_ptr<float>(), eps_sq, numel, v_block_size
    );
}

// ==========================================
// 17. Apollo+CAME Phase 2: Compute M_new & Res
// ==========================================
__global__ void apollo_came_compute_m_res_kernel(
    const float* __restrict__ grad_low,
    const unsigned char* __restrict__ v_q, const float* __restrict__ v_scale,
    const unsigned char* __restrict__ m_q_old, const float* __restrict__ m_scale_old,
    const unsigned char* __restrict__ res_q_old, const float* __restrict__ res_scale_old,
    float* __restrict__ m_temp, float* __restrict__ res_temp,
    const float* __restrict__ clip_factor_ptr, 
    float beta1, float beta3, float eps_came, float eps_sq,
    int v_block_size, int m_block_size, int res_block_size, int numel)
{
    float clip_factor = *clip_factor_ptr;
    int stride = gridDim.x * blockDim.x;
    const float one_minus_b1 = 1.0f - beta1;
    const float one_minus_b3 = 1.0f - beta3;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        float log_v = (float)v_q[idx] * INV_255 * v_scale[idx / v_block_size] + MIN_LOG;
        float v_val = exp2f(fmaxf(log_v, -53.0f));
        float g_val = (isnan(grad_low[idx]) || isinf(grad_low[idx])) ? 0.0f : grad_low[idx];
        float u_t = g_val * rsqrtf(fmaxf(v_val, eps_sq));
        u_t /= clip_factor;

        unsigned char m_byte = m_q_old[idx / 2];
        int m_int = (idx & 1) ? (m_byte & 0x0F) : (m_byte >> 4);
        float m_old = dequant_uniform_4bit(m_int, m_scale_old[idx / m_block_size]);
        float m_new = beta1 * m_old + one_minus_b1 * u_t;
        m_temp[idx] = m_new;

        float diff = u_t - m_new;
        float res_raw = diff * diff + eps_came;
        float c_old_log = (float)res_q_old[idx] * INV_255 * res_scale_old[idx / res_block_size] + MIN_LOG;
        float c_old = exp2f(c_old_log);
        float c_new = beta3 * c_old + one_minus_b3 * res_raw;
        res_temp[idx] = fmaxf(c_new, MIN_VAL);
    }
}

void apollo_came_compute_m_res_cuda(
    torch::Tensor grad_low,
    torch::Tensor v_q, torch::Tensor v_scale,
    torch::Tensor m_q_old, torch::Tensor m_scale_old,
    torch::Tensor res_q_old, torch::Tensor res_scale_old,
    torch::Tensor m_temp, torch::Tensor res_temp,
    torch::Tensor clip_factor,
    float beta1, float beta3, float eps_came, float eps_sq,
    int v_block_size, int m_block_size, int res_block_size, int numel)
{
    int threads = 256;
    int blocks = min(1024, (numel + threads - 1) / threads);
    apollo_came_compute_m_res_kernel<<<blocks, threads>>>(
        grad_low.data_ptr<float>(),
        v_q.data_ptr<unsigned char>(), v_scale.data_ptr<float>(),
        m_q_old.data_ptr<unsigned char>(), m_scale_old.data_ptr<float>(),
        res_q_old.data_ptr<unsigned char>(), res_scale_old.data_ptr<float>(),
        m_temp.data_ptr<float>(), res_temp.data_ptr<float>(),
        clip_factor.data_ptr<float>(),
        beta1, beta3, eps_came, eps_sq,
        v_block_size, m_block_size, res_block_size, numel
    );
}

// ==========================================
// 18. Apollo+CAME Phase 3: Compute Final Update & Norms
// ==========================================
__global__ void apollo_came_compute_update_norms_kernel(
    const unsigned char* __restrict__ m_q_new, const float* __restrict__ m_scale_new,
    const unsigned char* __restrict__ res_q_new, const float* __restrict__ res_scale_new,
    const float* __restrict__ grad_low,
    float* __restrict__ norm_update, float* __restrict__ norm_grad,
    float eps_sq, int N, int D, int stride_N, int stride_D, int numel,
    int m_block_size, int res_block_size)
{
    int channel = blockIdx.x;
    if (channel >= N) return;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    float sum_u2 = 0.0f;
    float sum_g2 = 0.0f;
    
    for (int i = tid; i < D; i += stride) {
        int idx = channel * stride_N + i * stride_D;
        if (idx >= numel) break;
        
        unsigned char m_byte = m_q_new[idx / 2];
        int m_int = (idx & 1) ? (m_byte & 0x0F) : (m_byte >> 4);
        float m_val = dequant_uniform_4bit(m_int, m_scale_new[idx / m_block_size]);

        float log_res = (float)res_q_new[idx] * INV_255 * res_scale_new[idx / res_block_size] + MIN_LOG;
        float res_val = exp2f(fmaxf(log_res, -53.0f));

        float u_final = m_val * rsqrtf(fmaxf(res_val, eps_sq));
        sum_u2 += u_final * u_final;
        
        float g_val = (isnan(grad_low[idx]) || isinf(grad_low[idx])) ? 0.0f : grad_low[idx];
        sum_g2 += g_val * g_val;
    }
    
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_u2 += __shfl_down_sync(0xffffffff, sum_u2, offset);
        sum_g2 += __shfl_down_sync(0xffffffff, sum_g2, offset);
    }
    __shared__ float s_u2[32];
    __shared__ float s_g2[32];
    int lane = tid % 32;
    int wid = tid / 32;
    int num_warps = blockDim.x / 32;
    if (lane == 0) {
        s_u2[wid] = sum_u2;
        s_g2[wid] = sum_g2;
    }
    __syncthreads();
    
    if (wid == 0) {
        if (lane < num_warps) {
            sum_u2 = s_u2[lane];
            sum_g2 = s_g2[lane];
        } else {
            sum_u2 = 0.0f;
            sum_g2 = 0.0f;
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_u2 += __shfl_down_sync(0xffffffff, sum_u2, offset);
            sum_g2 += __shfl_down_sync(0xffffffff, sum_g2, offset);
        }
        if (lane == 0) {
            norm_update[channel] = sqrtf(sum_u2);
            norm_grad[channel] = sqrtf(sum_g2);
        }
    }
}

void apollo_came_compute_update_norms_cuda(
    torch::Tensor m_q_new, torch::Tensor m_scale_new,
    torch::Tensor res_q_new, torch::Tensor res_scale_new,
    torch::Tensor grad_low,
    torch::Tensor norm_update, torch::Tensor norm_grad,
    float eps_sq, int N, int D, int stride_N, int stride_D, int numel,
    int m_block_size, int res_block_size)
{
    int threads = 256;
    if (D < 256) threads = 128;
    if (D < 128) threads = 64;
    if (D < 64) threads = 32;
    apollo_came_compute_update_norms_kernel<<<N, threads>>>(
        m_q_new.data_ptr<unsigned char>(), m_scale_new.data_ptr<float>(),
        res_q_new.data_ptr<unsigned char>(), res_scale_new.data_ptr<float>(),
        grad_low.data_ptr<float>(),
        norm_update.data_ptr<float>(), norm_grad.data_ptr<float>(),
        eps_sq, N, D, stride_N, stride_D, numel,
        m_block_size, res_block_size
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_log_quantize_lerp", &fused_log_quantize_lerp_cuda, "Fused log quantize lerp (CUDA)");
    m.def("fused_4bit_quantize_lerp", &fused_4bit_quantize_lerp_cuda, "Fused 4-bit packed quantize lerp for m_t (CUDA)");
    
    m.def("compute_update_norm_2d", &compute_update_norm_2d_cuda, "Compute update norm 2D (CUDA)");
    m.def("apply_update_2d", &apply_update_2d_cuda, "Apply update 2D (CUDA)");
    m.def("compute_update_norm_1d", &compute_update_norm_1d_cuda, "Compute update norm 1D (CUDA)");
    m.def("apply_update_1d", &apply_update_1d_cuda, "Apply update 1D (CUDA)");
    
    m.def("compute_update_norm_m_2d", &compute_update_norm_m_2d_cuda, "Compute update norm with m_t 2D (CUDA)");
    m.def("apply_update_m_2d", &apply_update_m_2d_cuda, "Apply update with m_t 2D (CUDA)");
    m.def("compute_update_norm_m_1d", &compute_update_norm_m_1d_cuda, "Compute update norm with m_t 1D (CUDA)");
    m.def("apply_update_m_1d", &apply_update_m_1d_cuda, "Apply update with m_t 1D (CUDA)");

    m.def("compute_apollo_norms", &compute_apollo_norms_cuda, "Compute Apollo norms from packed 4-bit/8-bit (CUDA)");
    m.def("dequantize_4bit", &dequantize_4bit_cuda, "Dequantize 4-bit packed tensor (CUDA)");

    m.def("compute_update_norm_1d_full", &compute_update_norm_1d_full_cuda, "Compute update norm 1D full precision (CUDA)");
    m.def("apply_update_1d_full", &apply_update_1d_full_cuda, "Apply update 1D full precision (CUDA)");

    m.def("compute_update_norm_1d_full_m", &compute_update_norm_1d_full_m_cuda, "Compute update norm 1D full precision with momentum (CUDA)");
    m.def("apply_update_1d_full_m", &apply_update_1d_full_m_cuda, "Apply update 1D full precision with momentum (CUDA)");

    m.def("came_compute_residual_2d", &came_compute_residual_2d_cuda, "Compute CAME residual row/col sums (CUDA)");

    m.def("compute_apollo_came_rms", &compute_apollo_came_rms_cuda, "Compute Apollo CAME RMS (CUDA)");
    m.def("apollo_came_compute_m_res", &apollo_came_compute_m_res_cuda, "Apollo CAME compute M and Res (CUDA)");
    m.def("apollo_came_compute_update_norms", &apollo_came_compute_update_norms_cuda, "Apollo CAME compute update and norms (CUDA)");
}