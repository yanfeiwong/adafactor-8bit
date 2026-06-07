// Copyright (c) 2026 WANG YAN
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include <torch/extension.h>

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

        float upd0 = ((float)q_val.x / 255.0f * old_scale) * one_minus_b + nv.x * beta;
        local_vals[idx * 4 + 0] = upd0;
        thread_max = fmaxf(thread_max, upd0);

        float upd1 = ((float)q_val.y / 255.0f * old_scale) * one_minus_b + nv.y * beta;
        local_vals[idx * 4 + 1] = upd1;
        thread_max = fmaxf(thread_max, upd1);

        float upd2 = ((float)q_val.z / 255.0f * old_scale) * one_minus_b + nv.z * beta;
        local_vals[idx * 4 + 2] = upd2;
        thread_max = fmaxf(thread_max, upd2);

        float upd3 = ((float)q_val.w / 255.0f * old_scale) * one_minus_b + nv.w * beta;
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

        val = (tid < 8) ? s_max[tid] : 0.0f; 
        
        for (int offset = 4; offset > 0; offset /= 2) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
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
    torch::Tensor q,
    torch::Tensor scale,
    torch::Tensor new_val, 
    float beta,
    int block_size)
{
    TORCH_CHECK(q.scalar_type() == at::kByte, "q must be uint8");
    TORCH_CHECK(scale.scalar_type() == at::kFloat, "scale must be float32");
    TORCH_CHECK(new_val.scalar_type() == at::kFloat, "new_val must be float32");
    TORCH_CHECK(q.is_cuda() && scale.is_cuda() && new_val.is_cuda(), "tensors must be on CUDA");
    TORCH_CHECK(q.is_contiguous() && scale.is_contiguous() && new_val.is_contiguous(), "tensors must be contiguous");
    
    int threads = 256;
    TORCH_CHECK(block_size >= threads, "block_size must be >= threads");
    TORCH_CHECK(block_size % 4 == 0, "block_size must be multiple of 4 for vectorization");

    int num_blocks = scale.size(0);   
    size_t shared_mem = (block_size + 8) * sizeof(float); 

    fused_quantize_lerp_kernel<<<num_blocks, threads, shared_mem>>>(
        q.data_ptr<unsigned char>(),
        scale.data_ptr<float>(),
        new_val.data_ptr<float>(),
        beta, 
        block_size
    );

    return q;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_quantize_lerp", &fused_quantize_lerp_cuda, "Fused quantize lerp (CUDA)");
}