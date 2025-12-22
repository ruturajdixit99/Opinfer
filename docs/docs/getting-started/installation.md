# Installation

This page will help you install Opinfer on your system.

## Requirements

- Python 3.7+
- PyTorch 1.12.0+
- CUDA (optional, for GPU acceleration)

## Install from GitHub (Recommended)

The easiest way to install Opinfer is directly from GitHub:

```bash
pip install git+https://github.com/ruturajdixit99/Opinfer.git
```

## Install from PyPI

Once the package is published on PyPI:

```bash
pip install opinfer
```

## Install from Source (Development)

If you want to contribute or need the latest development version:

```bash
# Clone the repository
git clone https://github.com/ruturajdixit99/Opinfer.git
cd Opinfer

# Install in development mode
pip install -e .
```

## Verify Installation

After installation, verify that Opinfer is correctly installed:

```python
import opinfer
print(opinfer.__version__)  # Should print the version number
```

## Dependencies

Opinfer automatically installs the following dependencies:

- `torch>=1.12.0` - PyTorch framework
- `torchvision>=0.13.0` - Vision utilities
- `timm>=0.6.0` - PyTorch Image Models
- `transformers>=4.20.0` - Hugging Face Transformers (for OWL-ViT)
- `opencv-python>=4.5.0` - Computer vision library
- `numpy>=1.21.0` - Numerical computing
- `Pillow>=8.0.0` - Image processing

## Troubleshooting

### CUDA Issues

If you encounter CUDA-related errors:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Import Errors

If you get import errors, make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### GPU Not Detected

Verify CUDA is available:

```python
import torch
print(torch.cuda.is_available())  # Should print True if GPU is available
```





