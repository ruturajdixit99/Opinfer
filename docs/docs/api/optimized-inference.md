# OptimizedInference API

Complete API reference for the `OptimizedInference` class.

## Class Definition

```python
class OptimizedInference:
    """High-level API for optimized inference with adaptive motion gating."""
    
    def __init__(
        self,
        model_name: str,
        model_type: str = "classifier",
        device: str = "cuda",
        technique: str = "motion_gating",
        auto_optimize: bool = False,
        target_skip_rate: float = 40.0,
        queue_size: int = 4,
        batch_size: int = 4,
        max_queue_wait_ms: float = 33.0,
    ):
        ...
```

## Parameters

### model_name

Model identifier string.

**Classifiers** (timm):
- `"vit_tiny_patch16_224"`
- `"vit_small_patch16_224"`
- `"vit_base_patch16_224"` (default)
- `"vit_large_patch16_224"`
- `"deit_tiny_patch16_224"`
- `"deit_small_patch16_224"`
- `"deit_base_patch16_224"`

**Detectors** (OWL-ViT):
- `"owlvit-base"`
- `"owlvit-large"`

### model_type

Type of model: `"classifier"` or `"detector"`.

### device

Device for inference: `"cuda"` or `"cpu"`.

### technique

Optimization technique: `"motion_gating"` or `"queuing"`.

### auto_optimize

Whether to automatically optimize parameters (motion gating only).

- `True`: Runs optimization (30-60 seconds), better performance
- `False`: Uses recommended params instantly, faster startup

### target_skip_rate

Target percentage of frames to skip (motion gating only). Default: `40.0`.

### queue_size

Maximum queue size (queuing only). Default: `4`.

### batch_size

Batch size for processing (queuing only). Default: `4`.

### max_queue_wait_ms

Maximum wait time before processing queue in milliseconds (queuing only). Default: `33.0` (30 FPS).

## Methods

### process_video_file

Process a video file.

```python
def process_video_file(
    self,
    video_path: str,
    max_frames: Optional[int] = None,
    analyze_sample: int = 200,
) -> Dict[str, Any]:
    """
    Process a video file with optimized inference.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum frames to process (None = all)
        analyze_sample: Frames to use for analysis (motion gating)
    
    Returns:
        Dictionary containing:
        - 'stats': Performance statistics
        - 'outputs': Model outputs for each frame
        - 'video_characteristics': Detected video properties
        - 'optimization_result': Optimization results (if enabled)
    """
```

**Example**:

```python
results = inf.process_video_file("video.mp4", max_frames=500)
stats = results['stats']
print(f"Skip rate: {stats['skip_rate_pct']:.1f}%")
```

### process_video_frames

Process a list of frames.

```python
def process_video_frames(
    self,
    frames: List[np.ndarray],
    analyze_first: bool = True,
) -> Dict[str, Any]:
    """
    Process a list of video frames.
    
    Args:
        frames: List of BGR frames (numpy arrays)
        analyze_first: Whether to analyze/optimize first (motion gating)
    
    Returns:
        Dictionary with stats, outputs, etc.
    """
```

### process_frame

Process a single frame (for streaming).

```python
def process_frame(
    self,
    frame: np.ndarray,
) -> Tuple[Any, Dict[str, float]]:
    """
    Process a single frame.
    
    Args:
        frame: BGR frame (numpy array)
    
    Returns:
        Tuple of (output, stats_dict):
        - output: Model prediction or cached output
        - stats_dict: Statistics including:
            - 'motion_score': Motion score for frame
            - 'did_call': Whether model was called
            - 'inference_ms': Inference time (if called)
    """
```

**Example**:

```python
output, stats = inf.process_frame(frame)
if stats['did_call']:
    print(f"Model called, inference: {stats['inference_ms']:.1f}ms")
```

### initialize_streaming

Initialize for streaming/live feed processing.

```python
def initialize_streaming(
    self,
    sample_frames: Optional[List[np.ndarray]] = None,
    num_sample_frames: int = 30,
) -> None:
    """
    Initialize for streaming mode with fast defaults.
    
    Args:
        sample_frames: Optional sample frames for quick analysis
        num_sample_frames: Number of sample frames to use
    """
```

**Example**:

```python
inf.initialize_streaming()  # Uses default params instantly
# or
inf.initialize_streaming(sample_frames=frames[:50])  # Quick analysis
```

## Attributes

### model

The loaded PyTorch model.

### processor

The processor (for detector models like OWL-ViT).

### initialized

Whether the system has been initialized.

### streaming_mode

Whether the instance is in streaming mode.

## Example Usage

```python
from opinfer import OptimizedInference

# Initialize
inf = OptimizedInference(
    model_name="vit_base_patch16_224",
    model_type="classifier",
    technique="motion_gating",
    auto_optimize=True,
)

# Process video file
results = inf.process_video_file("video.mp4", max_frames=500)

# Access results
stats = results['stats']
print(f"Skip rate: {stats['skip_rate_pct']:.1f}%")
print(f"Effective FPS: {stats['effective_fps']:.2f}")
```

## Next Steps

- [Core Classes API](./core-classes.md)
- [Getting Started Guide](../getting-started/quickstart.md)





