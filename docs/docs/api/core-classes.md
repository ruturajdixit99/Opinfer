# Core Classes API

API reference for core Opinfer classes.

## AdaptiveMotionGater

Adaptive motion gating system that automatically adjusts parameters.

```python
class AdaptiveMotionGater:
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        auto_optimize: bool = False,
        optimization_sample_frames: int = 200,
        target_skip_rate: float = 40.0,
        processor=None,
        text_queries=None,
    ):
        ...
```

### Methods

#### analyze_and_optimize

Analyze video and optimize parameters.

```python
def analyze_and_optimize(
    self,
    frames: List[np.ndarray],
) -> Dict[str, Any]:
    """
    Analyze video characteristics and optimize parameters.
    
    Returns:
        Dictionary with:
        - 'video_characteristics': Detected properties
        - 'optimization_result': Optimization results (if enabled)
        - 'final_threshold': Final motion threshold
        - 'final_min_frames': Final min frames between calls
    """
```

#### process_video

Process a video with motion gating.

```python
def process_video(
    self,
    frames: List[np.ndarray],
    analyze_first: bool = True,
) -> Dict[str, Any]:
    """
    Process video frames with adaptive motion gating.
    
    Args:
        frames: List of BGR frames
        analyze_first: Whether to analyze/optimize first
    
    Returns:
        Dictionary with stats and outputs
    """
```

#### process_frame

Process a single frame.

```python
def process_frame(
    self,
    frame: np.ndarray,
) -> Tuple[Any, Dict[str, float]]:
    """
    Process single frame with motion gating.
    
    Returns:
        (output, stats_dict)
    """
```

#### get_current_stats

Get current performance statistics.

```python
def get_current_stats(self) -> Dict[str, float]:
    """Get accumulated statistics."""
```

### Attributes

- `motion_threshold`: Current motion threshold
- `min_frames_between_calls`: Current min frames setting
- `engine`: The MotionGatedInference engine
- `video_chars`: Video characteristics (after analysis)

## MotionGatedInference

Core motion-gated inference engine.

```python
class MotionGatedInference:
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        img_size: int = 224,
        motion_threshold: float = 4.0,
        min_frames_between_calls: int = 2,
        motion_downsample: Tuple[int, int] = (64, 64),
    ):
        ...
```

### Methods

#### infer

Run motion-gated inference on a frame.

```python
def infer(
    self,
    frame_bgr: np.ndarray,
    processor=None,
    text_queries=None,
) -> Tuple[Any, Dict[str, float]]:
    """
    Run motion-gated inference.
    
    Args:
        frame_bgr: BGR frame
        processor: Optional processor (for detector models)
        text_queries: Optional text queries (for detector models)
    
    Returns:
        (output, stats_dict)
    """
```

#### reset

Reset state for new video sequence.

```python
def reset(self) -> None:
    """Reset all state variables."""
```

#### get_stats

Get accumulated statistics.

```python
def get_stats(self) -> Dict[str, float]:
    """
    Returns:
        Dictionary with:
        - 'total_frames': Total frames processed
        - 'model_calls': Number of model calls
        - 'skipped_frames': Number of skipped frames
        - 'skip_rate_pct': Skip rate percentage
        - 'avg_inference_ms': Average inference time
        - 'effective_fps': Effective FPS
    """
```

#### update_parameters

Update motion gating parameters dynamically.

```python
def update_parameters(
    self,
    motion_threshold: Optional[float] = None,
    min_frames_between_calls: Optional[int] = None,
) -> None:
    """Update parameters on the fly."""
```

## QueuedInference

Frame queuing-based inference engine.

```python
class QueuedInference:
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        queue_size: int = 4,
        batch_size: int = 4,
        max_queue_wait_ms: float = 33.0,
    ):
        ...
```

### Methods

#### infer

Add frame to queue and process if needed.

```python
def infer(
    self,
    frame_bgr: np.ndarray,
    processor=None,
    text_queries=None,
) -> Tuple[Optional[Any], Dict[str, float]]:
    """
    Add frame to queue, process batch if queue full.
    
    Returns:
        (output, stats_dict) - output may be None if frame in queue
    """
```

#### flush

Process remaining frames in queue.

```python
def flush(self) -> List[Tuple[int, Any]]:
    """Process all remaining frames in queue."""
```

#### reset

Reset queue state.

```python
def reset(self) -> None:
    """Reset queue and statistics."""
```

#### get_stats

Get queue statistics.

```python
def get_stats(self) -> Dict[str, float]:
    """
    Returns:
        Dictionary with queue and batch statistics
    """
```

## Next Steps

- [OptimizedInference API](./optimized-inference.md)
- [Getting Started Guide](../getting-started/quickstart.md)





