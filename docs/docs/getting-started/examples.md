# Examples

Real-world examples demonstrating Opinfer usage.

## Example 1: Traffic Camera Analysis

Process a static traffic camera feed:

```python
from opinfer import OptimizedInference

inf = OptimizedInference(
    model_name="vit_base_patch16_224",
    model_type="classifier",
    auto_optimize=True,
)

results = inf.process_video_file("traffic_cam.mp4", max_frames=500)

print(f"Skip rate: {results['stats']['skip_rate_pct']:.1f}%")
print(f"Model calls: {results['stats']['model_calls']}/{results['stats']['total_frames']}")
```

**Expected**: High skip rate (60-70%) due to static scenes.

## Example 2: Object Detection with OWL-ViT

Detect objects in video using OWL-ViT:

```python
from opinfer import OptimizedInference

inf = OptimizedInference(
    model_name="owlvit-base",
    model_type="detector",
    technique="motion_gating",
    auto_optimize=True,
)

results = inf.process_video_file("street_scene.mp4")

# Access detection results
for frame_output in results.get('outputs', []):
    if frame_output:
        print(f"Detections: {len(frame_output.get('boxes', []))}")
```

## Example 3: Real-Time Webcam Processing

Process webcam feed in real-time:

```python
from opinfer import OptimizedInference
import cv2

inf = OptimizedInference(
    model_name="vit_base_patch16_224",
    model_type="classifier",
    auto_optimize=False,  # Fast start for real-time
)

inf.initialize_streaming()

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    output, stats = inf.process_frame(frame)
    
    # Display results
    cv2.imshow('Frame', frame)
    if stats['did_call']:
        print(f"Motion: {stats['motion_score']:.2f}, Inference: {stats['inference_ms']:.1f}ms")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Example 4: Batch Processing Multiple Videos

Process multiple videos efficiently:

```python
from opinfer import OptimizedInference
from pathlib import Path

inf = OptimizedInference(
    model_name="vit_base_patch16_224",
    model_type="classifier",
    auto_optimize=True,
)

video_dir = Path("videos")
for video_path in video_dir.glob("*.mp4"):
    print(f"Processing {video_path.name}...")
    results = inf.process_video_file(str(video_path))
    print(f"  Skip rate: {results['stats']['skip_rate_pct']:.1f}%")
```

## Example 5: Custom Text Queries for OWL-ViT

Use custom object queries:

```python
from opinfer import OptimizedInference

inf = OptimizedInference(
    model_name="owlvit-base",
    model_type="detector",
)

# Custom queries will be used automatically
# The package supports custom queries through the API
```





