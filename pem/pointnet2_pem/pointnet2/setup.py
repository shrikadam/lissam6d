import os
import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_ext_src_root = "_ext_src"
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)

setup(
    name='pointnet2',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',
            sources=_ext_sources,
            # Let PyTorch automatically find the system CUDA headers/libs
            include_dirs=[os.path.join(_ext_src_root, "include")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3", 
                    "-DCUDA_HAS_FP16=1",
                    "-D__CUDA_NO_HALF_OPERATORS__",
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "-D__CUDA_NO_HALF2_OPERATORS__",
                ]
            },
        )
    ],
    # use_ninja=True makes the compilation much faster by parallelizing it
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=True)}
)