// Copyright (c) 2026 WANG YAN
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include <torch/extension.h>

__device__ constexpr float INV_255 = 1.0f / 255.0f;

// ==========================================
// 1. Fused Quantize Lerp (EMA Update)
// ==========================================
__global__ void fused_quantize_lerp_kernel(
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
    float* local_vals = shared_mem;           
    float* s_max = &shared_mem[block_size];

    float old_scale = scale[block_id];
    float thread_max = 0.0f;
    float one_minus_b = 1.0f - beta;
    
    const float4* new_val_vec = reinterpret_cast<const float4*>(new_val + start);
    uchar4* q_vec = reinterpret_cast<uchar4*>(q + start);

    int vec_iters = (block_size / 4) / stride; 

    for (int i = 0; i < vec_iters; i++) {
        int idx = tid + i * stride; 
        
        float4 nv = new_val_vec[idx];
        uchar4 q_val = q_vec[idx];

        float upd0 = ((float)q_val.x * INV_255 * old_scale) * one_minus_b + nv.x * beta;
        local_vals[idx * 4 + 0] = upd0;
        thread_max = fmaxf(thread_max, upd0);

        float upd1 = ((float)q_val.y * INV_255 * old_scale) * one_minus_b + nv.y * beta;
        local_vals[idx * 4 + 1] = upd1;
        thread_max = fmaxf(thread_max, upd1);

        float upd2 = ((float)q_val.z * INV_255 * old_scale) * one_minus_b + nv.z * beta;
        local_vals[idx * 4 + 2] = upd2;
        thread_max = fmaxf(thread_max, upd2);

        float upd3 = ((float)q_val.w * INV_255 * old_scale) * one_minus_b + nv.w * beta;
        local_vals[idx * 4 + 3] = upd3;
        thread_max = fmaxf(thread_max, upd3);
    }

    float val = thread_max;
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    if (tid % 32 == 0) {
        s_max[tid / 32] = val;
    }
    __syncthreads();

    if (tid < 32) { 
        val = (tid < num_warps) ? s_max[tid] : 0.0f; 
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_down_sync(0xffffffff, val, offset);
            if (tid + offset < num_warps) {
                val = fmaxf(val, other);
            }
        }
        if (tid == 0) {
            s_max[0] = val;
        }
    }
    __syncthreads(); 

    float new_scale = fmaxf(s_max[0], 1e-12f);
    float inv_scale = 255.0f / new_scale;

    for (int i = 0; i < vec_iters; i++) {
        int idx = tid + i * stride;
        
        float v0 = local_vals[idx * 4 + 0];
        float v1 = local_vals[idx * 4 + 1];
        float v2 = local_vals[idx * 4 + 2];
        float v3 = local_vals[idx * 4 + 3];

        uchar4 out_q;
        out_q.x = (unsigned char)fminf(fmaxf(v0 * inv_scale + 0.5f, 0.0f), 255.0f);
        out_q.y = (unsigned char)fminf(fmaxf(v1 * inv_scale + 0.5f, 0.0f), 255.0f);
        out_q.z = (unsigned char)fminf(fmaxf(v2 * inv_scale + 0.5f, 0.0f), 255.0f);
        out_q.w = (unsigned char)fminf(fmaxf(v3 * inv_scale + 0.5f, 0.0f), 255.0f);

        q_vec[idx] = out_q;
    }

    if (tid == 0) {
        scale[block_id] = new_scale;
    }
}

torch::Tensor fused_quantize_lerp_cuda(
    torch::Tensor q, torch::Tensor scale, torch::Tensor new_val, float beta, int block_size)
{
    TORCH_CHECK(q.scalar_type() == at::kByte && scale.scalar_type() == at::kFloat && new_val.scalar_type() == at::kFloat);
    TORCH_CHECK(q.is_cuda() && scale.is_cuda() && new_val.is_cuda());
    TORCH_CHECK(q.is_contiguous() && scale.is_contiguous() && new_val.is_contiguous());
    
    int threads = 256;
    TORCH_CHECK(block_size >= threads && block_size % 4 == 0 && block_size % (4 * threads) == 0,
                "block_size must be a multiple of 4 * threads for vectorization.");

    int num_blocks = scale.size(0);   
    int num_warps = threads / 32;
    size_t shared_mem = (block_size + num_warps) * sizeof(float); 
    
    TORCH_CHECK(shared_mem <= 49152, "block_size is too large, exceeding 48KB shared memory limit.");

    fused_quantize_lerp_kernel<<<num_blocks, threads, shared_mem>>>(
        q.data_ptr<unsigned char>(), scale.data_ptr<float>(), new_val.data_ptr<float>(), beta, block_size
    );
    return q;
}


// ==========================================
// 2. Phase 1: Compute Update Norm (N-D Matrix mapped to 2D)
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
        
        int row_var_idx = b * R + r;
        int col_var_idx = b * C + c;
        
        float r_val = (float)row_var_q[row_var_idx] * INV_255 * row_var_scale[row_var_idx / block_size];
        float c_val = (float)col_var_q[col_var_idx] * INV_255 * col_var_scale[col_var_idx / block_size];
        
        float row_mean_val = row_mean_val_ptr[b];
        
        float v_ij = (r_val * c_val) / row_mean_val;
        float u_ij = grad[idx] * rsqrtf(fmaxf(v_ij, eps));
        sq += u_ij * u_ij;
    }
    
    for (int offset = 16; offset > 0; offset /= 2) {
        sq += __shfl_down_sync(0xffffffff, sq, offset);
    }
    
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
    
    __shared__ float s_sum[32];
    if (lane == 0) s_sum[wid] = sq;
    __syncthreads();
    
    if (wid == 0) {
        sq = (lane < num_warps) ? s_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_down_sync(0xffffffff, sq, offset);
            if (lane + offset < num_warps) {
                sq += other;
            }
        }
        if (lane == 0) {
            atomicAdd(total_sum_sq, sq);
        }
    }
}

void compute_update_norm_2d_cuda(
    torch::Tensor row_var_q, torch::Tensor row_var_scale,
    torch::Tensor col_var_q, torch::Tensor col_var_scale,
    torch::Tensor grad, torch::Tensor total_sum_sq,
    torch::Tensor row_mean_val, float eps, int R, int C, int numel, int block_size)
{
    int threads = 256;
    int max_blocks = 1024; 
    int blocks = min(max_blocks, (numel + threads - 1) / threads);
    compute_update_norm_2d_kernel<<<blocks, threads>>>(
        row_var_q.data_ptr<unsigned char>(), row_var_scale.data_ptr<float>(),
        col_var_q.data_ptr<unsigned char>(), col_var_scale.data_ptr<float>(),
        grad.data_ptr<float>(), total_sum_sq.data_ptr<float>(),
        row_mean_val.data_ptr<float>(), eps, R, C, numel, block_size
    );
}


// ==========================================
// 3. Phase 2: Apply Update (N-D Matrix mapped to 2D)
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
        
        int row_var_idx = b * R + r;
        int col_var_idx = b * C + c;
        
        float r_val = (float)row_var_q[row_var_idx] * INV_255 * row_var_scale[row_var_idx / block_size];
        float c_val = (float)col_var_q[col_var_idx] * INV_255 * col_var_scale[col_var_idx / block_size];
        
        float row_mean_val = row_mean_val_ptr[b];
        
        float v_ij = (r_val * c_val) / row_mean_val;
        float u_ij = grad[idx] * rsqrtf(fmaxf(v_ij, eps));
        
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
    int max_blocks = 1024;
    int blocks = min(max_blocks, (numel + threads - 1) / threads);
    
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
        float v_val = (float)variance_q[idx] * INV_255 * variance_scale[idx / block_size];
        float u_val = grad[idx] * rsqrtf(fmaxf(v_val, eps));
        sq += u_val * u_val;
    }
    
    for (int offset = 16; offset > 0; offset /= 2) sq += __shfl_down_sync(0xffffffff, sq, offset);
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
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
    int max_blocks = 1024;
    int blocks = min(max_blocks, (numel + threads - 1) / threads);
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
        float v_val = (float)variance_q[idx] * INV_255 * variance_scale[idx / block_size];
        float u_val = grad[idx] * rsqrtf(fmaxf(v_val, eps));
        
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
    int max_blocks = 1024;
    int blocks = min(max_blocks, (numel + threads - 1) / threads);
    
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
    m.def("fused_quantize_lerp", &fused_quantize_lerp_cuda, "Fused quantize lerp (CUDA)");
    m.def("compute_update_norm_2d", &compute_update_norm_2d_cuda, "Compute update norm 2D (CUDA)");
    m.def("apply_update_2d", &apply_update_2d_cuda, "Apply update 2D (CUDA)");
    m.def("compute_update_norm_1d", &compute_update_norm_1d_cuda, "Compute update norm 1D (CUDA)");
    m.def("apply_update_1d", &apply_update_1d_cuda, "Apply update 1D (CUDA)");
}