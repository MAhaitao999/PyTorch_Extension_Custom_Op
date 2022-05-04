import torch
import numpy as np
import alpha_ncrelu


class AlphaNCReLUModuleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return alpha_ncrelu.forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        return alpha_ncrelu.backward(grad_output, ctx.input)

    @staticmethod
    def symbolic(g, *inputs):
        return g.op("AlphaNCReLU", inputs[0])
