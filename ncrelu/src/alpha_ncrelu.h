#ifndef __ALPHA_NCRELU_H__
#define __ALPHA_NCRELU_H__

#include <torch/extension.h>

torch::Tensor AlphaNCReLUForward(const torch::Tensor input);

torch::Tensor AlphaNCReLUBackward(torch::Tensor grad_out, torch::Tensor input);

#endif // __ALPHA_NCRELU_H__