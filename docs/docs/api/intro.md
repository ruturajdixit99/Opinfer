# API Reference

Complete API reference for the Opinfer package.

## Main Classes

### OptimizedInference

High-level API for optimized inference with adaptive motion gating or queuing.

```python
from opinfer import OptimizedInference

inf = OptimizedInference(
    model_name: str,
    model_type: str = "classifier",
    device: str = "cuda",
    technique: str = "motion_gating",
    auto_optimize: bool = False,
    target_skip_rate: float = 40.0,
    queue_size: int = 4,
    batch_size: int = 4,
    max_queue_wait_ms: float = 33.0,
)
```

**Methods**:
- `process_video_file(video_path, max_frames=None)`
- `process_video_frames(frames, analyze_first=True)`
- `process_frame(frame)`
- `initialize_streaming(sample_frames=None, num_sample_frames=30)`

See [OptimizedInference API](./optimized-inference.md) for details.

### AdaptiveMotionGater

Lower-level adaptive motion gating system.

```python
from opinfer import AdaptiveMotionGater

gater = AdaptiveMotionGater(
    model: nn.Module,
    device: str = "cuda",
    auto_optimize: bool = False,
    optimization_sample_frames: int = 200,
    target_skip_rate: float = 40.0,
    processor=None,
    text_queries=None,
)
```

**Methods**:
- `analyze_and_optimize(frames)`
- `process_video(frames, analyze_first=True)`
- `process_frame(frame)`
- `get_current_stats()`

See [Core Classes API](./core-classes.md) for details.

### MotionGatedInference

Core motion-gated inference engine.

```python
from opinfer import MotionGatedInference

engine = MotionGatedInference(
    model: nn.Module,
    device: str = "cuda",
    img_size: int = 224,
    motion_threshold: float = 4.0,
    min_frames_between_calls: int = 2,
    motion_downsample: Tuple[int, int] = (64, 64),
)
```

**Methods**:
- `infer(frame_bgr, processor=None, text_queries=None)`
- `reset()`
- `get_stats()`
- `update_parameters(motion_threshold=None, min_frames_between_calls=None)`

See [Core Classes API](./core-classes.md) for details.

## Helper Functions

### load_video_frames

Load frames from a video file.

```python
from opinfer import load_video_frames

frames = load_video_frames(
    video_path: str,
    max_frames: Optional[int] = None,
) -> List[np.ndarray]
```

### ModelLoader

Load and manage various model types.

```python
from opinfer.models import ModelLoader

# Load classifier
model = ModelLoader.load_classifier(
    model_name: str,
    device: str = "cuda",
    pretrained: bool = True,
)

# Load detector
model, processor = ModelLoader.load_detector(
    model_name: str,
    device: str = "cuda",
)

# List available models
models = ModelLoader.list_models()
```

## Next Steps

- [OptimizedInference API](./optimized-inference.md)
- [Core Classes API](./core-classes.md)





