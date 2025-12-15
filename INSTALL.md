# Installation Guide - Opinfer

## 📦 Installation Methods

### Method 1: From PyPI (Recommended)

Once released to PyPI:

```bash
pip install opinfer
```

### Method 2: From Source (Current)

```bash
# Clone repository
git clone https://github.com/ruturajdixit99/Opinfer.git
cd opinfer

# Install in development mode
pip install -e .

# Or install directly
pip install .
```

### Method 3: Direct Installation

```bash
# From local directory
cd SARD/MotionGated
pip install -e .
```

---

## ✅ Verify Installation

```bash
# Test import
python -c "from opinfer import OptimizedInference; print('✅ Installed!')"

# Test CLI
opinfer list-models

# Run tests
python simple_test.py
```

---

## 📋 Requirements

The package requires:
- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA (optional, but recommended for GPU acceleration)

All dependencies are automatically installed with:
```bash
pip install opinfer
```

---

## 🔧 Development Installation

For development:

```bash
git clone https://github.com/ruturajdixit99/Opinfer.git
cd opinfer
pip install -e ".[dev]"
```

This installs development dependencies (pytest, black, etc.)

---

## 🐛 Troubleshooting

### CUDA Not Available
The package will automatically use CPU if CUDA is not available. This is slower but works.

### Import Errors
```bash
# Reinstall
pip uninstall opinfer
pip install opinfer
```

### Permission Errors
```bash
# Use user installation
pip install --user opinfer
```

---

## 📚 Next Steps

After installation:
1. Read `README.md` for usage examples
2. Check `VLM_USAGE_GUIDE.md` for VLM integration
3. Run `python test_with_videos.py` to test with your videos

---

**Ready to use!** 🚀

