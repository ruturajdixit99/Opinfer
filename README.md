# Opinfer: Optimized Inference with Adaptive Motion Gating

**Opinfer** is a Python package for efficient real-time video inference using adaptive motion-gated Vision Transformers. It automatically optimizes motion gating parameters based on video characteristics, making it effective across diverse scenarios from static traffic cameras to fast-moving drone footage.

## 🚀 Features

- **Two Optimization Techniques**: Choose between Motion Gating or Queuing
- **Adaptive Motion Gating**: Automatically adjusts parameters based on video characteristics
- **Frame Queuing**: Batch processing for consistent frame rates
- **Multi-Scenario Support**: Optimized for traffic cams, night drives, and fast-motion scenarios
- **Multiple Model Support**: Works with ViT/DeiT classifiers and OWL-ViT detectors
- **Automatic Optimization**: Finds optimal parameters through iterative search (motion gating)
- **Video Analysis**: Detects motion patterns, lighting conditions, and scene stability
- **Easy-to-Use API**: Simple interface for quick integration

## 📦 Installation

### From PyPI (Once Released)

```bash
pip install opinfer
```

### From Source

```bash
# Clone repository
git clone https://github.com/ruturajdixit99/Opinfer.git
cd opinfer

# Install in development mode
pip install -e .

# Or install directly
pip install .
```

### Local Development

```bash
cd SARD/MotionGated
pip install -e .
```

## 🎯 Quick Start

### Using with Vision-Language Models (VLMs)

```python
from opinfer import OptimizedInference

# OWL-ViT (built-in VLM support)
inf = OptimizedInference(
    model_name="owlvit-base",
    model_type="detector",
    technique="motion_gating",
    auto_optimize=True,
)

results = inf.process_video_file("video.mp4")
```

**See `VLM_USAGE_GUIDE.md` for detailed VLM integration examples.**

### Basic Usage - Motion Gating (Default)

```python
from opinfer import OptimizedInference

# Initialize with motion gating (default)
inf = OptimizedInference(
    model_name="vit_base_patch16_224",
    model_type="classifier",
    technique="motion_gating",  # or omit (default)
    auto_optimize=True,
)

# Process a video file
results = inf.process_video_file("path/to/video.mp4", max_frames=500)

print(f"Skip rate: {results['stats']['skip_rate_pct']:.1f}%")
print(f"Effective FPS: {results['stats']['effective_fps']:.2f}")
```

### Basic Usage - Queuing

```python
from opinfer import OptimizedInference

# Initialize with queuing technique
inf = OptimizedInference(
    model_name="vit_base_patch16_224",
    model_type="classifier",
    technique="queuing",
    queue_size=4,
    batch_size=4,
    max_queue_wait_ms=33.0,
)

# Process a video file
results = inf.process_video_file("path/to/video.mp4", max_frames=500)

print(f"Effective FPS: {results['stats']['effective_fps']:.2f}")
print(f"Batches processed: {results['stats']['batches_processed']}")
```

### Advanced Usage

```python
from opinfer import AdaptiveMotionGater, ModelLoader

# Load a model
model = ModelLoader.load_classifier("vit_base_patch16_224", device="cuda")

# Create adaptive gater
gater = AdaptiveMotionGater(
    model=model,
    device="cuda",
    auto_optimize=True,
    target_skip_rate=40.0,
)

# Load frames
import cv2
cap = cv2.VideoCapture("video.mp4")
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

# Process with automatic optimization
results = gater.process_video(frames, analyze_first=True)
```

## 🔬 How It Works

### 1. Video Analysis
The system analyzes video characteristics:
- **Motion Pattern**: Static, slow, moderate, fast, or very fast
- **Lighting Conditions**: Bright, normal, low, or very low
- **Contrast Level**: Detects low-contrast scenes (e.g., night drives)
- **Scene Stability**: Measures frame-to-frame consistency

### 2. Adaptive Parameter Optimization
Based on detected characteristics, the system:
- Adjusts motion threshold for different motion levels
- Reduces threshold for low-contrast scenes (night drives)
- Increases threshold for fast-motion scenarios (drones)
- Optimizes minimum frames between calls

### 3. Motion Gating
For each frame:
1. Computes motion score (grayscale frame differencing)
2. Decides whether to call the model or reuse previous prediction
3. Tracks statistics for performance monitoring

## 📊 Supported Models

### Classifiers (timm)
- `vit_tiny_patch16_224`
- `vit_small_patch16_224`
- `vit_base_patch16_224`
- `vit_large_patch16_224`
- `deit_tiny_patch16_224`
- `deit_small_patch16_224`
- `deit_base_patch16_224`

### Detectors (OWL-ViT)
- `owlvit-base`
- `owlvit-large`

## 🎬 Scenario-Specific Optimizations

### Traffic Camera (Static Scenes)
- **Challenge**: Steady feed with minimal motion
- **Solution**: High skip rate (50-70%), optimized thresholds
- **Result**: Excellent performance, minimal model calls

### Night Drive (Low Contrast)
- **Challenge**: Reduced optimization due to lighting/contrast issues
- **Solution**: Automatic threshold reduction (50% for low contrast), lighting-aware adjustments
- **Result**: 50% improvement over fixed thresholds

### Drone Footage (Fast Motion)
- **Challenge**: Fast motion makes traditional gating ineffective
- **Solution**: Dynamic threshold adjustment, reduced min_frames, motion-aware optimization
- **Result**: Effective gating even with rapid scene changes

## 📈 Benchmarking

Run comprehensive benchmarks on all models and scenarios:

```bash
python benchmark_all.py
```

This will:
- Test all available models
- Process traffic cam, night drive, and drone videos
- Generate detailed performance reports
- Save results to `benchmark_results/`

## 🛠️ API Reference

### OptimizedInference

Main high-level API class.

```python
inf = OptimizedInference(
    model_name: str,
    model_type: str = "classifier",
    device: str = "cuda",
    auto_optimize: bool = True,
    target_skip_rate: float = 40.0,
)
```

**Methods:**
- `process_video_file(video_path, max_frames=None)`: Process video file
- `process_video_frames(frames, analyze_first=True)`: Process frame list
- `process_frame(frame)`: Process single frame (streaming)
- `benchmark_all_models(video_path, max_frames=500)`: Benchmark all models

### AdaptiveMotionGater

Lower-level adaptive gating system.

```python
gater = AdaptiveMotionGater(
    model: nn.Module,
    device: str = "cuda",
    auto_optimize: bool = True,
    optimization_sample_frames: int = 200,
    target_skip_rate: float = 40.0,
)
```

**Methods:**
- `analyze_and_optimize(frames)`: Analyze video and optimize parameters
- `process_video(frames, analyze_first=True)`: Process video with gating
- `process_frame(frame)`: Process single frame
- `get_current_stats()`: Get performance statistics

## 📝 Example Results

### Traffic Camera
- Skip Rate: 60-70%
- Speedup: 2.5-3.5x
- Stability: 95%+

### Night Drive
- Skip Rate: 30-40% (improved from ~20% with fixed params)
- Speedup: 1.8-2.2x
- Stability: 90%+

### Drone Footage
- Skip Rate: 15-25% (improved from ~5% with fixed params)
- Speedup: 1.3-1.6x
- Stability: 85%+

## 🔧 Configuration

### Motion Gating Parameters

- `motion_threshold`: Motion score threshold (auto-optimized)
- `min_frames_between_calls`: Minimum frames between model calls (auto-optimized)
- `target_skip_rate`: Target percentage of frames to skip (default: 40%)

### Optimization Settings

- `max_iterations`: Maximum optimization iterations (default: 50)
- `patience`: Early stopping patience (default: 10)
- `optimization_sample_frames`: Frames to use for optimization (default: 200)

## 🐛 Troubleshooting

### Low Skip Rates
- Check video characteristics: fast motion may limit skipping
- Adjust `target_skip_rate` to be more realistic
- Increase `optimization_sample_frames` for better analysis

### Poor Performance
- Ensure CUDA is available: `torch.cuda.is_available()`
- Reduce `max_frames` for faster testing
- Check video quality and frame rate

### Memory Issues
- Reduce `optimization_sample_frames`
- Process videos in chunks
- Use smaller models (vit_tiny, vit_small)

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Opinfer** - Making Vision Transformer inference efficient and adaptive. 🚀

