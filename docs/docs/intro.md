# Welcome to Opinfer

**Opinfer** (Optimized Inference) is a high-performance Python package designed to dramatically improve the efficiency of video inference for Vision Transformers (ViTs) and Vision-Language Models (VLMs). By intelligently analyzing video characteristics and automatically optimizing inference parameters, Opinfer reduces computational overhead while maintaining accuracy across diverse real-world scenarios—from static traffic cameras to fast-moving drone footage.

## What Makes Opinfer Special?

Opinfer addresses the critical challenge of processing video streams efficiently by:

- **Skipping redundant frames** when scene changes are minimal (motion gating)
- **Batching frames** for optimal GPU utilization (queuing)
- **Automatically adapting** to different video characteristics (lighting, motion patterns, scene stability)
- **Supporting multiple model types** including ViT/DeiT classifiers and OWL-ViT detectors
- **Providing up to 50%+ performance improvements** in real-world scenarios

## Key Features

- **Two Optimization Techniques**: Choose between Motion Gating or Queuing
- **Adaptive Motion Gating**: Automatically adjusts parameters based on video characteristics
- **Frame Queuing**: Batch processing for consistent frame rates
- **Multi-Scenario Support**: Optimized for traffic cams, night drives, and fast-motion scenarios
- **Multiple Model Support**: Works with ViT/DeiT classifiers and OWL-ViT detectors
- **Automatic Optimization**: Finds optimal parameters through iterative search (motion gating)
- **Video Analysis**: Detects motion patterns, lighting conditions, and scene stability
- **Easy-to-Use API**: Simple interface for quick integration

## Quick Example

```python
from opinfer import OptimizedInference

# Initialize with motion gating (default)
inf = OptimizedInference(
    model_name="vit_base_patch16_224",
    model_type="classifier",
    technique="motion_gating",
    auto_optimize=True,
)

# Process a video file
results = inf.process_video_file("path/to/video.mp4", max_frames=500)

print(f"Skip rate: {results['stats']['skip_rate_pct']:.1f}%")
print(f"Effective FPS: {results['stats']['effective_fps']:.2f}")
```

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/ruturajdixit99/Opinfer.git

# Or from PyPI (once released)
pip install opinfer
```

## Next Steps

- [Installation Guide](./getting-started/installation.md)
- [Quick Start Tutorial](./getting-started/quickstart.md)
- [Core Concepts](./concepts/motion-gating.md)
- [API Reference](./api/intro.md)





