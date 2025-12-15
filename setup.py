"""
Setup script for opinfer package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="opinfer",
    version="0.1.0",
    author="Opinfer Contributors",
    author_email="",
    description="Optimized Inference with Adaptive Motion Gating and Queuing for Vision Transformers and VLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ruturajdixit99/Opinfer",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Multimedia :: Video",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "timm>=0.6.0",
        "transformers>=4.20.0",
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "Pillow>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    keywords=[
        "computer-vision",
        "vision-transformer",
        "motion-gating",
        "video-inference",
        "optimization",
        "vlm",
        "vision-language-models",
        "real-time-inference",
        "batch-processing",
        "frame-queuing",
    ],
    project_urls={
        "Bug Reports": "https://github.com/ruturajdixit99/Opinfer/issues",
        "Source": "https://github.com/ruturajdixit99/Opinfer",
        "Documentation": "https://github.com/ruturajdixit99/Opinfer#readme",
    },
    entry_points={
        "console_scripts": [
            "opinfer=opinfer.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

