# VLM Integration Guide

This guide shows how to use Opinfer with Vision-Language Models (VLMs) for efficient video inference.

## Quick Start with OWL-ViT

OWL-ViT is a Vision-Language Model for open-vocabulary object detection. It's fully supported out of the box:

```python
from opinfer import OptimizedInference

# Initialize with OWL-ViT detector
inf = OptimizedInference(
    model_name="owlvit-base",  # or "owlvit-large"
    model_type="detector",
    technique="motion_gating",
    auto_optimize=True,
)

# Process video
results = inf.process_video_file("video.mp4", max_frames=200)

print(f"Skip rate: {results['stats']['skip_rate_pct']:.1f}%")
print(f"Effective FPS: {results['stats']['effective_fps']:.2f}")
```

## Supported VLM Models

### OWL-ViT

- **owlvit-base**: Faster, smaller model
- **owlvit-large**: More accurate, larger model

Both models support text queries for object detection.

## Text Queries

OWL-ViT uses text queries to detect objects. Default queries include common objects:

```python
default_queries = ["person", "car", "road", "building", "tree", "traffic light", "sky"]
```

These are automatically used, but you can customize them through the API.

## Processing Detection Results

When using detector models, results contain bounding boxes, scores, and labels:

```python
from opinfer import OptimizedInference

inf = OptimizedInference(
    model_name="owlvit-base",
    model_type="detector",
)

results = inf.process_video_file("video.mp4")

# Access detection outputs
for frame_idx, output in enumerate(results.get('outputs', [])):
    if output and isinstance(output, dict):
        boxes = output.get('boxes', [])  # Bounding boxes
        scores = output.get('scores', [])  # Confidence scores
        labels = output.get('labels', [])  # Label indices
        
        print(f"Frame {frame_idx}: {len(boxes)} detections")
        for box, score, label in zip(boxes, scores, labels):
            print(f"  Detection: score={score:.2f}, label={label}")
```

## Frame-by-Frame Processing

For real-time detection:

```python
from opinfer import OptimizedInference
import cv2

inf = OptimizedInference(
    model_name="owlvit-base",
    model_type="detector",
    auto_optimize=False,  # Fast start
)

inf.initialize_streaming()

cap = cv2.VideoCapture(0)  # Webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    output, stats = inf.process_frame(frame)
    
    if output and isinstance(output, dict):
        boxes = output.get('boxes', [])
        scores = output.get('scores', [])
        labels = output.get('labels', [])
        
        # Draw bounding boxes
        for box, score, label in zip(boxes, scores, labels):
            if score > 0.3:  # Confidence threshold
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {score:.2f}", 
                           (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Custom VLM Models

To use a custom VLM model, you'll need to wrap it appropriately. The model should:

1. Accept image input (BGR format)
2. Return predictions in a consistent format
3. Support batch processing (optional)

```python
from opinfer import AdaptiveMotionGater, ModelLoader
import torch.nn as nn

# Your custom VLM model
class YourVLM(nn.Module):
    def __init__(self):
        super().__init__()
        # Your model architecture
    
    def forward(self, images):
        # Process images
        return predictions

# Load your model
model = YourVLM()
model.load_state_dict(torch.load("your_model.pth"))
model.eval()

# Use with opinfer
# Note: You may need to implement custom processor
```

## Performance Tips

### For Real-Time Detection

1. Use `auto_optimize=False` for instant start
2. Use `owlvit-base` instead of `owlvit-large` for speed
3. Process at lower resolution if needed
4. Use motion gating to skip redundant frames

### For Batch Processing

1. Enable `auto_optimize=True` for optimal parameters
2. Process videos offline for best results
3. Use larger batch sizes if GPU memory allows

## Example: Traffic Monitoring

```python
from opinfer import OptimizedInference

# Initialize for traffic monitoring
inf = OptimizedInference(
    model_name="owlvit-base",
    model_type="detector",
    technique="motion_gating",
    auto_optimize=True,
)

# Process traffic camera feed
results = inf.process_video_file("traffic_cam.mp4")

# Analyze results
total_detections = 0
for output in results.get('outputs', []):
    if output:
        total_detections += len(output.get('boxes', []))

print(f"Total detections: {total_detections}")
print(f"Average per frame: {total_detections / results['stats']['total_frames']:.2f}")
```

## Troubleshooting

### Low Detection Scores

- Check video quality and lighting
- Adjust text queries to match scene
- Lower confidence threshold for more detections

### Performance Issues

- Use `owlvit-base` instead of `owlvit-large`
- Enable motion gating to skip frames
- Process at lower resolution

### Memory Issues

- Reduce batch size
- Process videos in chunks
- Use smaller models

## Next Steps

- Learn about [Performance Tuning](./performance-tuning.md)
- Check [API Reference](../api/intro.md)
- Explore [Examples](../getting-started/examples.md)





