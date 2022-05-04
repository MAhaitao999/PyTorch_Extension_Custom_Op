#include <iostream>
#include <cstdlib>
#include "alpha_ncrelu.h"
#include "alpha_ncrelu_kernel.cuh"

void AlphaNCReLUForwardCudaInterface(const float *input, 
                                     const size_t batch, 
                                     const size_t channel, 
                                     const size_t height, 
                                     const size_t width,
                                     const float *ret);

void AlphaNCReLUBackwardCudaInterface(const float *grad_output,
                                      const float *input,
                                      const size_t batch, 
                                      const size_t channel, 
                                      const size_t height, 
                                      const size_t width,
                                      float *ret);

torch::Tensor AlphaNCReLUForward(const torch::Tensor input)
{
    if (input.device().type() == torch::kCPU)
    {
        auto pos = input.clamp_min(0);
        auto neg = input.clamp_max(0);
        return torch::cat({pos, 0.5 * neg}, 1);
    }
    else if (input.device().type() == torch::kCUDA)
    {
        TORCH_CHECK(input.dtype() == torch::kFloat32,
                   "DataType not implemented");
        size_t batch = input.size(0);
        size_t channel = input.size(1);
        size_t height = input.size(2);
        size_t width = input.size(3);

        auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device().type());

	    auto ret = torch::zeros({batch, channel * 2, height, width}, options);
	    AlphaNCReLUForwardCudaInterface(input.data_ptr<float>(),
			                            batch,
                                        channel,
                                        height,
                                        width,
                                        ret.data_ptr<float>());

        return ret;
            
    }
    else
    {
        AT_ERROR("No such device: ", input.device());
    }
}


torch::Tensor AlphaNCReLUBackward(torch::Tensor grad_out, torch::Tensor input)
{
    if (input.device().type() == torch::kCPU)
    {
        size_t size = input.numel();
        auto ret = torch::zeros_like(input);

        float *grad_out_data_ptr = grad_out.data_ptr<float>();
        float *input_data_ptr = input.data_ptr<float>();
        float *ret_data_ptr = ret.data_ptr<float>();
        for (size_t i = 0; i < size; i++)
        {
            *(ret_data_ptr + i) = *(input_data_ptr + i) * *(grad_out_data_ptr + i) - 
            0.5 * *(input_data_ptr + i) * *(grad_out_data_ptr + i + size);
        }

        return ret;
    }

    else if (input.device().type() == torch::kCUDA)
    {
        TORCH_CHECK(input.dtype() == torch::kFloat32,
                    "DataType not implemented");
	    size_t batch = input.size(0);
        size_t channel = input.size(1);
        size_t height = input.size(2);
        size_t width = input.size(3);
        auto ret = torch::zeros_like(input);
        AlphaNCReLUBackwardCudaInterface(grad_out.data_ptr<float>(),
                                         input.data_ptr<float>(),
                                         batch,
                                         channel,
                                         height,
                                         width,
                                         ret.data_ptr<float>());

        return ret;
    }
    else
    {
        AT_ERROR("No such device: ", input.device());
    }

}

static auto registry = torch::RegisterOperators("haitao::AlphaNCReLU", &AlphaNCReLUForward);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {	// 绑定部分
    m.def("forward", &AlphaNCReLUForward, "AlphaNCReLU forward");
    m.def("backward", &AlphaNCReLUBackward, "AlphaNCReLU backward");
}
