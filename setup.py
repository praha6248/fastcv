from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fastcv",
    ext_modules=[
        CUDAExtension(
            name="fastcv",
            sources=[
                "kernels/grayscale.cu",
                "kernels/box_blur.cu",
                "kernels/sobel.cu",
                "kernels/dilation.cu",
                "kernels/erosion.cu",
                "kernels/adaptive_threshold.cu",
                "kernels/module.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3", 
                    "--use_fast_math", 
                    "--extended-lambda" 
                ]
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
