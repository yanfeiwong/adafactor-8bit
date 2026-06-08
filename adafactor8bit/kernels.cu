// Copyright (c) 2026 WANG YAN
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include <torch/extension.h>

__device__ constexpr float INV_255 = 1.0f / 255.0f;
__device__ constexpr float MIN_LOG = -126.0f; // 对应 FP32 最小正规格化数 2^-126
__device__ constexpr float MIN_VAL = 1.17549435e-38f; // 2^-126

// ==========================================
// 1. Fused Log-Quantize Lerp (EMA Update)
// ==========================================
__global__ void fused_log_quantize_lerp_kernel(
    unsigned char* __restrict__ q,
    float* __restrict__ scale,
    const float* __restrict__ new_val,
    const float beta,
    int block_size)
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
        
        float4 nv = new_val_vec[idx];
        uchar4 q_val = q_vec[idx];

        // 1. 反量化旧值到线性空间 (exp2)
        float v_old0 = exp2f((float)q_val.x * INV_255 * old_scale + MIN_LOG);
        float v_old1 = exp2f((float)q_val.y * INV_255 * old_scale + MIN_LOG);
        float v_old2 = exp2f((float)q_val.z * INV_255 * old_scale + MIN_LOG);
        float v_old3 = exp2f((float)q_val.w * INV_255 * old_scale + MIN_LOG);

        // 2. 在线性空间做 EMA (Lerp)，保证数学正确性
        float v_upd0 = v_old0 * one_minus_b + fmaxf(nv.x, MIN_VAL) * beta;
        float v_upd1 = v_old1 * one_minus_b + fmaxf(nv.y, MIN_VAL) * beta;
        float v_upd2 = v_old2 * one_minus_b + fmaxf(nv.z, MIN_VAL) * beta;
        float v_upd3 = v_old3 * one_minus_b + fmaxf(nv.w, MIN_VAL) * beta;

        // 3. 转回对数空间 (log2) 并寻找 Max
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

    // Warp & Block Reduce 找 max_log
    float val = thread_max;
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    if (tid % 32 == 0) s_max[tid / 32] = val;
    __syncthreads();

    if (tid < 32) { 
        val = (tid < num_warps) ? s_max[tid] : MIN_LOG; 
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_down_sync(0xffffffff, val, offset);
            if (tid + offset < num_warps) val = fmaxf(val, other);
        }
        if (tid == 0) s_max[0] = val;
    }
    __syncthreads(); 

    float max_log = fmaxf(s_max[0], MIN_LOG + 1e-12f);
    float new_scale = (max_log - MIN_LOG) / 255.0f;
    float inv_scale = 255.0f / (max_log - MIN_LOG);

    // 4. 量化并写回
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
    torch::Tensor q, torch::Tensor scale, torch::Tensor new_val, float beta, int block_size)
{
    TORCH_CHECK(q.scalar_type() == at::kByte && scale.scalar_type() == at::kFloat && new_val.scalar_type() == at::kFloat);
    TORCH_CHECK(q.is_cuda() && scale.is_cuda() && new_val.is_cuda());
    TORCH_CHECK(q.is_contiguous() && scale.is_contiguous() && new_val.is_contiguous());
    
    int threads = 256;
    TORCH_CHECK(block_size >= threads && block_size % 4 == 0 && block_size % (4 * threads) == 0);

    int num_blocks = scale.size(0);   
    int num_warps = threads / 32;
    size_t shared_mem = (block_size + num_warps) * sizeof(float); 
    
    fused_log_quantize_lerp_kernel<<<num_blocks, threads, shared_mem>>>(
        q.data_ptr<unsigned char>(), scale.data_ptr<float>(), new_val.data_ptr<float>(), beta, block_size
    );
    return q;
}

// ==========================================
// 2. Phase 1: Compute Update Norm (2D)
// ==========================================
__global__ void compute_update_norm_2d_kernel(
    const unsigned char* __restrict__ row_var_q, const float* __restrict__ row_var_scale,
    const unsigned char* __restrict__ col_var_q, const float* __restrict__ col_var_scale,
    const float* __restrict__ grad, float* __restrict__ total_sum_sq,
    const float* __restrict__ row_mean_val_ptr, float eps, int R, int C, int numel, int block_size)
{
    float sq = 0.0f;
    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        int b = idx / (R * C);
        int r = (idx / C) % R;
        int c = idx % C;
        
        // 对数空间反量化 (exp2)
        float r_val = exp2f((float)row_var_q[b * R + r] * INV_255 * row_var_scale[(b * R + r) / block_size] + MIN_LOG);
        float c_val = exp2f((float)col_var_q[b * C + c] * INV_255 * col_var_scale[(b * C + c) / block_size] + MIN_LOG);
        
        float row_mean_val = fmaxf(row_mean_val_ptr[b], MIN_VAL);
        float v_ij = (r_val * c_val) / row_mean_val;
        
        // 限制放大倍数 & 限制最终更新量
        float inv_std = rsqrtf(fmaxf(v_ij, MIN_VAL));
        inv_std = fminf(inv_std, 1e8f); 
        float u_ij = grad[idx] * inv_std;
        u_ij = fmaxf(fminf(u_ij, 1e4f), -1e4f); 
        
        sq += u_ij * u_ij;
    }
    
    // Warp & Block Reduce
    for (int offset = 16; offset > 0; offset /= 2) sq += __shfl_down_sync(0xffffffff, sq, offset);
    int lane = threadIdx.x % 32; int wid = threadIdx.x / 32; int num_warps = blockDim.x / 32;
    __shared__ float s_sum[32];
    if (lane == 0) s_sum[wid] = sq;
    __syncthreads();
    if (wid == 0) {
        sq = (lane < num_warps) ? s_sum[lane] : 0.0f;
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
    torch::Tensor row_mean_val, float eps, int R, int C, int numel, int block_size)
{
    int threads = 256;
    int blocks = min(1024, (numel + threads - 1) / threads);
    compute_update_norm_2d_kernel<<<blocks, threads>>>(
        row_var_q.data_ptr<unsigned char>(), row_var_scale.data_ptr<float>(),
        col_var_q.data_ptr<unsigned char>(), col_var_scale.data_ptr<float>(),
        grad.data_ptr<float>(), total_sum_sq.data_ptr<float>(),
        row_mean_val.data_ptr<float>(), eps, R, C, numel, block_size
    );
}

// ==========================================
// 3. Phase 2: Apply Update (2D)
// ==========================================
template <typename scalar_t>
__global__ void apply_update_2d_kernel(
    scalar_t* __restrict__ param, const float* __restrict__ grad,
    const unsigned char* __restrict__ row_var_q, const float* __restrict__ row_var_scale,
    const unsigned char* __restrict__ col_var_q, const float* __restrict__ col_var_scale,
    const float* __restrict__ sum_sq_ptr, const float* __restrict__ alpha,
    const float* __restrict__ row_mean_val_ptr, float d, float eps, int R, int C, int numel, int block_size)
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
        
        float r_val = exp2f((float)row_var_q[b * R + r] * INV_255 * row_var_scale[(b * R + r) / block_size] + MIN_LOG);
        float c_val = exp2f((float)col_var_q[b * C + c] * INV_255 * col_var_scale[(b * C + c) / block_size] + MIN_LOG);
        
        float row_mean_val = fmaxf(row_mean_val_ptr[b], MIN_VAL);
        float v_ij = (r_val * c_val) / row_mean_val;
        
        float inv_std = rsqrtf(fmaxf(v_ij, MIN_VAL));
        inv_std = fminf(inv_std, 1e8f);
        float u_ij = grad[idx] * inv_std;
        u_ij = fmaxf(fminf(u_ij, 1e4f), -1e4f);
        
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
    torch::Tensor row_mean_val, float d, float eps, int R, int C, int numel, int block_size)
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
                row_mean_val.data_ptr<float>(), d, eps, R, C, numel, block_size
            );
        }));
}

// ==========================================
// 4. 1D Vector Variance Kernels
// ==========================================
__global__ void compute_update_norm_1d_kernel(
    const unsigned char* __restrict__ variance_q, const float* __restrict__ variance_scale,
    const float* __restrict__ grad, float* __restrict__ total_sum_sq,
    float eps, int numel, int block_size)
{
    float sq = 0.0f;
    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        float v_val = exp2f((float)variance_q[idx] * INV_255 * variance_scale[idx / block_size] + MIN_LOG);
        float inv_std = rsqrtf(fmaxf(v_val, MIN_VAL));
        inv_std = fminf(inv_std, 1e8f);
        float u_val = grad[idx] * inv_std;
        u_val = fmaxf(fminf(u_val, 1e4f), -1e4f);
        sq += u_val * u_val;
    }
    for (int offset = 16; offset > 0; offset /= 2) sq += __shfl_down_sync(0xffffffff, sq, offset);
    int lane = threadIdx.x % 32; int wid = threadIdx.x / 32; int num_warps = blockDim.x / 32;
    __shared__ float s_sum[32];
    if (lane == 0) s_sum[wid] = sq;
    __syncthreads();
    if (wid == 0) {
        sq = (lane < num_warps) ? s_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_down_sync(0xffffffff, sq, offset);
            if (lane + offset < num_warps) sq += other;
        }
        if (lane == 0) atomicAdd(total_sum_sq, sq);
    }
}

void compute_update_norm_1d_cuda(
    torch::Tensor variance_q, torch::Tensor variance_scale,
    torch::Tensor grad, torch::Tensor total_sum_sq, float eps, int numel, int block_size)
{
    int threads = 256;
    int blocks = min(1024, (numel + threads - 1) / threads);
    compute_update_norm_1d_kernel<<<blocks, threads>>>(
        variance_q.data_ptr<unsigned char>(), variance_scale.data_ptr<float>(),
        grad.data_ptr<float>(), total_sum_sq.data_ptr<float>(), eps, numel, block_size
    );
}

template <typename scalar_t>
__global__ void apply_update_1d_kernel(
    scalar_t* __restrict__ param, const float* __restrict__ grad,
    const unsigned char* __restrict__ variance_q, const float* __restrict__ variance_scale,
    const float* __restrict__ sum_sq_ptr, const float* __restrict__ alpha,
    float d, float eps, int numel, int block_size)
{
    float sum_sq_val = *sum_sq_ptr;
    float alpha_val = *alpha;
    float denom = fmaxf(1.0f, sqrtf(sum_sq_val) / (sqrtf((float)numel) * d));
    float step_size = alpha_val / denom;

    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        float v_val = exp2f((float)variance_q[idx] * INV_255 * variance_scale[idx / block_size] + MIN_LOG);
        float inv_std = rsqrtf(fmaxf(v_val, MIN_VAL));
        inv_std = fminf(inv_std, 1e8f);
        float u_val = grad[idx] * inv_std;
        u_val = fmaxf(fminf(u_val, 1e4f), -1e4f);
        
        float p_val = static_cast<float>(param[idx]);
        p_val -= step_size * u_val;
        param[idx] = static_cast<scalar_t>(p_val);
    }
}

void apply_update_1d_cuda(
    torch::Tensor param, torch::Tensor grad,
    torch::Tensor variance_q, torch::Tensor variance_scale,
    torch::Tensor sum_sq, torch::Tensor alpha, float d, float eps, int numel, int block_size)
{
    int threads = 256;
    int blocks = min(1024, (numel + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, 
        param.scalar_type(), "apply_update_1d_cuda", ([&] {
            apply_update_1d_kernel<scalar_t><<<blocks, threads>>>(
                param.data_ptr<scalar_t>(), grad.data_ptr<float>(),
                variance_q.data_ptr<unsigned char>(), variance_scale.data_ptr<float>(),
                sum_sq.data_ptr<float>(), alpha.data_ptr<float>(), d, eps, numel, block_size
            );
        }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_log_quantize_lerp", &fused_log_quantize_lerp_cuda, "Fused log quantize lerp (CUDA)");
    m.def("compute_update_norm_2d", &compute_update_norm_2d_cuda, "Compute update norm 2D (CUDA)");
    m.def("apply_update_2d", &apply_update_2d_cuda, "Apply update 2D (CUDA)");
    m.def("compute_update_norm_1d", &compute_update_norm_1d_cuda, "Compute update norm 1D (CUDA)");
    m.def("apply_update_1d", &apply_update_1d_cuda, "Apply update 1D (CUDA)");
}