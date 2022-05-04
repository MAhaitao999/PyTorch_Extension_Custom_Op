#ifndef __ALPHA_NCRELU_KERNEL_H__
#define __ALPHA_NCRELU_KERNEL_H__

void AlphaNCReLUForwardCudaInterface(const float *input, 
                                     const size_t batch, 
                                     const size_t channel, 
                                     const size_t height, 
                                     const size_t width,
                                     float *ret);

void AlphaNCReLUBackwardCudaInterface(const float *grad_output,
                                      const float *input,
                                      const size_t batch, 
                                      const size_t channel, 
                                      const size_t height, 
                                      const size_t width,
                                      float *ret);

#endif // __ALPHA_NCRELU_KERNEL_H__