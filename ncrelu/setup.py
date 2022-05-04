from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='alpha_ncrelu',
      version='0.1.0',
      description='PTorch Custom defined Op: alphaNCReLU',
      author='Henry Ma',
      author_email='769413715@qq.com',
      ext_modules=[cpp_extension.CUDAExtension('alpha_ncrelu',
                      ['src/alpha_ncrelu.cc', 'src/alpha_ncrelu_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
