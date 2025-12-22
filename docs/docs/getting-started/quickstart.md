# Quick Start

Get up and running with Opinfer in minutes!

## Basic Usage - Motion Gating

Motion gating is the default technique that skips redundant frames based on motion detection:

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

## Basic Usage - Queuing

Frame queuing batches frames for consistent processing:

```python
from opinfer import OptimizedInference

# Initialize with queuing technique
inf = OptimizedInference(
    model_name="vit_base_patch16_224",
    model_type="classifier",
    technique="queuing",
    queue_size=4,
    batch_size=4,
)

# Process video
results = inf.process_video_file("path/to/video.mp4")
```

## Fast Start (Recommended for Real-Time)

For real-time applications, use fast defaults (no optimization delay):

```python
from opinfer import OptimizedInference

# Fast start - works immediately!
inf = OptimizedInference(
    model_name="vit_base_patch16_224",
    model_type="classifier",
    auto_optimize=False,  # Fast defaults, instant start
)

# Process video immediately
results = inf.process_video_file("path/to/video.mp4")
```

## Using with Vision-Language Models (VLMs)

Opinfer works great with OWL-ViT and other VLMs:

```python
from opinfer import OptimizedInference

# OWL-ViT detector
inf = OptimizedInference(
    model_name="owlvit-base",
    model_type="detector",
    technique="motion_gating",
    auto_optimize=True,
)

# Process video
results = inf.process_video_file("video.mp4", max_frames=200)
```

## Processing Frame by Frame

For live feeds or streaming:

```python
from opinfer import OptimizedInference
import cv2

# Initialize
inf = OptimizedInference(
    model_name="vit_base_patch16_224",
    model_type="classifier",
    auto_optimize=False,  # Fast start
)

# Initialize for streaming
inf.initialize_streaming()

# Process frames one by one
cap = cv2.VideoCapture(0)  # Webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    output, stats = inf.process_frame(frame)
    
    if stats['did_call']:
        print(f"Model called at frame {stats['motion_score']:.2f}")
```

## What to Expect

### Performance Improvements

- **Static Scenes** (traffic cameras): 60-70% skip rate
- **Moderate Motion** (normal driving): 40-50% skip rate  
- **Fast Motion** (drones, racing): 20-30% skip rate

### Typical Results

```python
results = {
    'stats': {
        'total_frames': 500,
        'model_calls': 250,
        'skipped_frames': 250,
        'skip_rate_pct': 50.0,
        'avg_inference_ms': 45.2,
        'effective_fps': 22.1
    }
}
```

## Next Steps

- Learn about [Motion Gating](./../concepts/motion-gating.md)
- Explore [Advanced Usage](./../guides/vlm-integration.md)
- Check the [API Reference](./../api/intro.md)





