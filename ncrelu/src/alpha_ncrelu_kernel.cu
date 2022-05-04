#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#include "alpha_ncrelu_kernel.cuh"

#define BLOCKSIZE 1024

__global__ void alpha_ncrelu_fwd_cuda(const float *input, float *ret, 
                                      const size_t channel,
                                      const size_t height,
                                      const size_t width,
                                      const size_t size)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
    {
        auto value = input[idx];    // 寻找原始值
        const size_t chw = channel * height * width;
        ret[idx + idx / chw * chw] = value >= 0 ? value : 0;             // 前一半通道为正值
        ret[idx + idx / chw * chw + chw] = value >= 0 ? 0 : value;       // 后一半通道为负值
    }
}

__global__ void alpha_ncrelu_bwd_cuda(const float *grad_out, 
                                      const float* input, 
                                      float *ret, 
                                      const size_t channel,
                                      const size_t height,
                                      const size_t width,
                                      size_t size)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
    {
        // 这里反传的逻辑瞎写的, 数学不好不会推~~~~~! 如果你知道怎么推请告诉我
        // ret[idx] = input[idx] * grad_out[idx] - 0.5 * input[idx] * grad_out[idx + size];
        const size_t chw = channel * height * width;
        // ret[idx + idx / chw * chw] = input[idx + idx / chw * chw] * 
        //                              (grad_out[idx + idx / chw * chw] - 
        //                              0.5 * grad_out[idx + idx / chw * chw + chw]);
        ret[idx + idx / chw * chw] = grad_out[idx + idx / chw * chw] - 0.5 * grad_out[idx + idx / chw * chw + chw];
    }
}

void AlphaNCReLUForwardCudaInterface(const float *input, 
                                     const size_t batch, 
                                     const size_t channel, 
                                     const size_t height, 
                                     const size_t width,
                                     float *ret)
{
    const size_t size = batch * channel * height * width;
    size_t nblock = (size + BLOCKSIZE - 1) / BLOCKSIZE;
    alpha_ncrelu_fwd_cuda<<<nblock, BLOCKSIZE>>>(input, ret, channel, height, width, size);
}

void AlphaNCReLUBackwardCudaInterface(const float *grad_output,
                                      const float *input,
                                      const size_t batch, 
                                      const size_t channel, 
                                      const size_t height, 
                                      const size_t width,
                                      float *ret)
{
    size_t size = batch * channel * height * width;
    size_t nblock = (size + BLOCKSIZE - 1) / BLOCKSIZE;
    alpha_ncrelu_bwd_cuda<<<nblock, BLOCKSIZE>>>(grad_output, input, ret, channel, height, width, size);
}