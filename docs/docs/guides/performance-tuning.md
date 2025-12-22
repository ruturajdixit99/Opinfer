# Performance Tuning

Optimize Opinfer for your specific use case and hardware.

## Understanding Performance Metrics

### Key Metrics

- **Skip Rate**: Percentage of frames skipped (higher = more efficient)
- **Effective FPS**: Processing speed including skipped frames
- **Model Calls**: Number of actual inference calls
- **Inference Time**: Average time per model call

### Target Performance

- **Static Scenes**: 60-70% skip rate, 2-3x speedup
- **Moderate Motion**: 40-50% skip rate, 1.5-2x speedup
- **Fast Motion**: 20-30% skip rate, 1.3-1.6x speedup

## Optimization Strategies

### 1. Enable Auto-Optimization

For best results, enable automatic parameter optimization:

```python
inf = OptimizedInference(
    model_name="vit_base_patch16_224",
    auto_optimize=True,  # Automatically finds optimal parameters
)
```

**When to use**: 
- Processing videos offline
- Need maximum performance
- Have time for optimization (30-60 seconds)

### 2. Use Fast Defaults

For real-time applications, use fast defaults:

```python
inf = OptimizedInference(
    model_name="vit_base_patch16_224",
    auto_optimize=False,  # Uses recommended params instantly
)
```

**When to use**:
- Real-time processing
- Live feeds
- Need instant startup

### 3. Adjust Target Skip Rate

Control how aggressive the optimization should be:

```python
inf = OptimizedInference(
    model_name="vit_base_patch16_224",
    target_skip_rate=50.0,  # Target 50% skip rate
)
```

**Guidelines**:
- **30-40%**: Conservative (more accuracy)
- **40-50%**: Balanced (default)
- **50-60%**: Aggressive (more speed)

### 4. Choose Appropriate Model

Larger models = better accuracy but slower:

```python
# Fast but less accurate
model_name="vit_tiny_patch16_224"

# Balanced (default)
model_name="vit_base_patch16_224"

# Slower but more accurate
model_name="vit_large_patch16_224"
```

## Scenario-Specific Tuning

### Static Scenes (Traffic Cameras)

```python
inf = OptimizedInference(
    model_name="vit_base_patch16_224",
    auto_optimize=True,
    target_skip_rate=60.0,  # High skip rate for static scenes
)
```

**Expected**: 60-70% skip rate, 2.5-3.5x speedup

### Fast Motion (Drones, Racing)

```python
inf = OptimizedInference(
    model_name="vit_base_patch16_224",
    auto_optimize=True,
    target_skip_rate=30.0,  # Lower skip rate for fast motion
)
```

**Expected**: 20-30% skip rate, 1.3-1.6x speedup

### Low Light (Night Drives)

The system automatically adjusts for low contrast scenes:

```python
inf = OptimizedInference(
    model_name="vit_base_patch16_224",
    auto_optimize=True,  # Automatically reduces threshold for low contrast
)
```

**Expected**: System automatically reduces threshold by 50% for low contrast

## Hardware Optimization

### GPU Memory

If you encounter out-of-memory errors:

```python
# Use smaller models
model_name="vit_tiny_patch16_224"  # or vit_small_patch16_224

# Reduce batch sizes (for queuing)
queue_size=2
batch_size=2
```

### CPU Processing

For CPU-only environments:

```python
inf = OptimizedInference(
    model_name="vit_tiny_patch16_224",  # Smaller model
    device="cpu",
    auto_optimize=False,  # Skip optimization for speed
)
```

## Advanced Tuning

### Manual Parameter Adjustment

For fine-grained control:

```python
from opinfer import AdaptiveMotionGater, ModelLoader

model = ModelLoader.load_classifier("vit_base_patch16_224", device="cuda")

gater = AdaptiveMotionGater(
    model=model,
    auto_optimize=False,  # Disable auto-optimization
)

# Manually set parameters
gater.motion_threshold = 5.0
gater.min_frames_between_calls = 3

# Use custom parameters
results = gater.process_video(frames)
```

### Optimization Sample Size

Control how many frames to use for optimization:

```python
gater = AdaptiveMotionGater(
    model=model,
    auto_optimize=True,
    optimization_sample_frames=100,  # Use fewer frames (faster optimization)
)
```

**Trade-offs**:
- **Fewer frames**: Faster optimization, less accurate
- **More frames**: Slower optimization, more accurate

## Monitoring Performance

Track performance metrics:

```python
results = inf.process_video_file("video.mp4")

stats = results['stats']
print(f"Skip Rate: {stats['skip_rate_pct']:.1f}%")
print(f"Effective FPS: {stats['effective_fps']:.2f}")
print(f"Model Calls: {stats['model_calls']}/{stats['total_frames']}")
print(f"Avg Inference: {stats['avg_inference_ms']:.2f} ms")
```

## Benchmarking

Compare different configurations:

```python
# Test configuration 1
inf1 = OptimizedInference(model_name="vit_base_patch16_224", auto_optimize=True)
results1 = inf1.process_video_file("video.mp4")

# Test configuration 2
inf2 = OptimizedInference(model_name="vit_tiny_patch16_224", auto_optimize=False)
results2 = inf2.process_video_file("video.mp4")

# Compare
print(f"Base + Optimize: {results1['stats']['skip_rate_pct']:.1f}% skip rate")
print(f"Tiny + Fast: {results2['stats']['skip_rate_pct']:.1f}% skip rate")
```

## Troubleshooting Performance

### Low Skip Rates

- Check video characteristics (may have high motion)
- Lower `target_skip_rate` for more realistic expectations
- Verify optimization completed successfully

### Poor Speedup

- Ensure CUDA is available: `torch.cuda.is_available()`
- Check GPU utilization
- Try smaller models
- Reduce video resolution if possible

### High Memory Usage

- Use smaller models
- Reduce `optimization_sample_frames`
- Process videos in chunks
- Use CPU if GPU memory is limited

## Next Steps

- Learn about [Optimization Process](./optimization.md)
- Check [API Reference](../api/intro.md)
- Explore [Examples](../getting-started/examples.md)





