# Copyright (c) 2026 WANG YAN
# Licensed under the MIT License.

from pathlib import Path
from setuptools import setup, find_packages

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="adafactor8bit",
    version="0.2.9",
    description="8-bit Adafactor Optimizer with Fused CUDA Kernels",
    author="WANG YAN",
    author_email="yanfeiwong1997@outlook.com",
    url="https://github.com/yanfeiwong/adafactor-8bit",
    packages=find_packages(),
    include_package_data=True,
    package_data={"adafactor8bit": ["*.cu"]}, 
    install_requires=[
        "torch>=2.1",
        "ninja",
    ],
    python_requires=">=3.10",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)