# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

# Get conda environment path
conda_env_path = os.environ.get('CONDA_PREFIX')
if not conda_env_path:
    raise RuntimeError("Cannot find conda environment. Make sure the environment is activated.")

# Define cuda paths for conda installation on Windows
cuda_include_path = os.path.join(conda_env_path, 'Library', 'include')
cuda_lib_path = os.path.join(conda_env_path, 'Library', 'lib')

_ext_src_root = "_ext_src"
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

setup(
    name='pointnet2',
    packages = find_packages(),
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',
            sources=_ext_sources,
            include_dirs = [os.path.join(_ext_src_root, "include"), cuda_include_path],
            library_dirs=[cuda_lib_path],
            libraries=['cudart'],
            extra_compile_args={
                # "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
                # "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
                "cxx": [],
                "nvcc": ["-O3", 
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]},)
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=True)}
)