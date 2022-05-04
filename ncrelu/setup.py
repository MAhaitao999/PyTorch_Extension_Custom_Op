from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='alpha_ncrelu',
      ext_modules=[cpp_extension.CUDAExtension('alpha_ncrelu',
                      ['src/alpha_ncrelu.cc', 'src/alpha_ncrelu_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
