# Copyright (c) 2026 WANG YAN
# Licensed under the MIT License.

from setuptools import setup, find_packages

setup(
    name="adafactor8bit",
    version="0.1.0",
    description="8-bit Adafactor Optimizer with Fused CUDA Kernels",
    author="WANG YAN",
    author_email="yanfeiwong1997@outlook.com",
    url="https://github.com/yanfeiwong/adafactor-8bit",
    packages=find_packages(),
    package_data={"adafactor8bit": ["*.cu"]}, 
    install_requires=[
        "torch",
        "ninja",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)