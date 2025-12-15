# Release Guide for Opinfer Package

This guide explains how to prepare and release the opinfer package for pip installation.

---

## 📦 Pre-Release Checklist

- [x] Package structure is complete
- [x] All modules are implemented
- [x] Tests are working
- [x] Documentation is complete
- [x] LICENSE file added
- [x] setup.py configured
- [x] .gitignore created
- [x] README.md is comprehensive

---

## 🚀 Step 1: Prepare Repository

### Initialize Git Repository

```bash
cd SARD/MotionGated
git init
git add .
git commit -m "Initial release: opinfer v0.1.0"
```

### Create GitHub Repository

1. Go to GitHub and create a new repository named `opinfer`
2. Add remote:
```bash
git remote add origin https://github.com/ruturajdixit99/Opinfer.git
git branch -M main
git push -u origin main
```

---

## 📝 Step 2: Update Package Metadata

Edit `setup.py` and update:
- `author` and `author_email`
- `url` with your GitHub repository URL
- `project_urls` with correct links

---

## 🏗️ Step 3: Build Package

### Install Build Tools

```bash
pip install build twine
```

### Build Distribution

```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Build source distribution and wheel
python -m build

# This creates:
# - dist/opinfer-0.1.0.tar.gz (source distribution)
# - dist/opinfer-0.1.0-py3-none-any.whl (wheel)
```

### Verify Build

```bash
# Check contents
tar -tzf dist/opinfer-0.1.0.tar.gz

# Test installation
pip install dist/opinfer-0.1.0-py3-none-any.whl
```

---

## 🧪 Step 4: Test Installation

### Test Local Installation

```bash
# Install from local wheel
pip install dist/opinfer-0.1.0-py3-none-any.whl

# Test import
python -c "from opinfer import OptimizedInference; print('✅ Package installed correctly!')"

# Test CLI
opinfer list-models
```

### Test from Source

```bash
# Install in development mode
pip install -e .

# Run tests
python simple_test.py
python test_with_videos.py
```

---

## 📤 Step 5: Release to PyPI

### Test on TestPyPI First

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ opinfer
```

### Release to PyPI

```bash
# Upload to PyPI
python -m twine upload dist/*

# Users can now install with:
# pip install opinfer
```

**Note**: You'll need PyPI credentials. Create account at https://pypi.org

---

## 🏷️ Step 6: Create GitHub Release

1. Go to GitHub repository
2. Click "Releases" → "Create a new release"
3. Tag: `v0.1.0`
4. Title: `Opinfer v0.1.0 - Initial Release`
5. Description:
```markdown
## Features
- Adaptive motion gating with automatic optimization
- Frame queuing for batch processing
- Support for ViT/DeiT classifiers
- Support for OWL-ViT detectors
- Works with custom Vision-Language Models
- Real-time video inference

## Installation
```bash
pip install opinfer
```

## Quick Start
```python
from opinfer import OptimizedInference

inf = OptimizedInference(
    model_name="vit_base_patch16_224",
    technique="motion_gating",
    auto_optimize=True,
)

results = inf.process_video_file("video.mp4")
```
```

---

## 📋 Step 7: Update Documentation

### Update README Installation Section

```markdown
## 📦 Installation

### From PyPI (Recommended)

```bash
pip install opinfer
```

### From Source

```bash
git clone https://github.com/ruturajdixit99/Opinfer.git
cd opinfer
pip install -e .
```
```

---

## ✅ Post-Release

### Verify Installation

```bash
# In a clean environment
pip install opinfer

# Test
python -c "from opinfer import OptimizedInference; print('✅ Success!')"
```

### Update Version

For future releases, update version in:
- `setup.py` (version="0.1.0")
- `opinfer/__init__.py` (__version__ = "0.1.0")

---

## 🎯 Quick Release Commands

```bash
# 1. Clean and build
rm -rf build/ dist/ *.egg-info
python -m build

# 2. Test locally
pip install dist/opinfer-0.1.0-py3-none-any.whl

# 3. Upload to PyPI
python -m twine upload dist/*

# 4. Create git tag
git tag v0.1.0
git push origin v0.1.0
```

---

## 📝 Package Structure for Release

```
opinfer/
├── opinfer/
│   ├── __init__.py
│   ├── core.py
│   ├── adaptive.py
│   ├── queue.py
│   ├── detectors.py
│   ├── optimizer.py
│   ├── models.py
│   ├── api.py
│   └── cli.py
├── setup.py
├── requirements.txt
├── README.md
├── LICENSE
├── MANIFEST.in
└── .gitignore
```

---

## 🐛 Troubleshooting

### "Package not found"
- Check PyPI upload was successful
- Wait a few minutes for PyPI to index
- Verify package name is correct

### "Import errors"
- Check all dependencies are in `install_requires`
- Verify `__init__.py` exports are correct

### "CLI not working"
- Check `entry_points` in setup.py
- Verify `opinfer/cli.py` exists and has `main()` function

---

## 📚 Additional Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [PyPI Upload Guide](https://packaging.python.org/tutorials/packaging-projects/)
- [Semantic Versioning](https://semver.org/)

---

**Your package is ready for release!** 🚀

