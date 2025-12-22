# Optimization Guide

Understand how Opinfer's automatic parameter optimization works.

## What is Optimization?

Opinfer automatically finds the **best motion threshold and min_frames parameters** for your specific video. This ensures optimal performance - maximizing speed while maintaining accuracy.

## Optimization Process

### Step 1: Video Analysis

The system analyzes your video to understand:

- **Motion Pattern**: How much movement is typical (static, slow, moderate, fast, very_fast)
- **Lighting Conditions**: Bright, normal, low, or very low
- **Contrast Level**: How much detail/variation in the scene (0-1)
- **Scene Stability**: How consistent the scene is (0-1)

Based on this analysis, it calculates a **recommended threshold range** to search.

### Step 2: Parameter Search

Opinfer tests different combinations of:

- **Motion Threshold**: How much motion is needed to trigger model inference
  - Lower threshold = More model calls (more accurate, slower)
  - Higher threshold = Fewer model calls (faster, may skip important frames)
  
- **Min Frames Between Calls**: Minimum frames before forcing a refresh
  - Prevents stale predictions
  - Ensures model is called periodically even with low motion

### Step 3: Evaluation

For each parameter combination, the system:

1. Runs inference on sample frames with those parameters
2. Measures performance:
   - **Skip Rate**: % of frames skipped (target: ~40%)
   - **Effective FPS**: Processing speed achieved
   - **Model Call Ratio**: Balance between calls and skips

3. Calculates a **Score** (higher is better) that balances:
   - Skip rate (40% weight) - should be close to target
   - FPS (30% weight) - should be high
   - Call ratio (30% weight) - should be balanced (40-60% of frames)

### Step 4: Iterative Refinement

The optimizer:

- Tries many combinations in the first iteration (grid search)
- Identifies the best configuration so far
- Refines search around the best parameters
- Stops early if no improvement (early stopping with patience)

## Understanding Optimization Output

When optimization runs, you'll see output like:

```
Step 2: Optimizing parameters...
🔍 Starting optimization (max 50 iterations)...
   Threshold range: [1.00, 50.00]
   Min frames candidates: [1, 2, 3, 4, 5]

✓ Iter 1: threshold=4.23, min_frames=2, score=84.92, skip=38.0%, fps=45.12
✓ Iter 3: threshold=3.45, min_frames=1, score=97.54, skip=39.0%, fps=47.24
✓ Iter 5: threshold=3.67, min_frames=2, score=98.12, skip=40.2%, fps=48.56
...
```

### Reading the Output

Each line shows:
- **threshold**: Motion threshold being tested
- **min_frames**: Minimum frames between calls
- **score**: Quality score (higher = better configuration)
- **skip**: Percentage of frames skipped
- **fps**: Effective frames per second achieved

The optimizer explores the parameter space to find optimal settings. As it finds better configs, scores increase.

## Optimization Settings

### Control Optimization Speed

```python
gater = AdaptiveMotionGater(
    model=model,
    auto_optimize=True,
    optimization_sample_frames=200,  # Frames to use for optimization
)
```

**Trade-offs**:
- **More frames**: Better optimization, slower (200-500 frames)
- **Fewer frames**: Faster optimization, less accurate (50-100 frames)

### Fast Mode (No Optimization)

For real-time applications, skip optimization:

```python
inf = OptimizedInference(
    model_name="vit_base_patch16_224",
    auto_optimize=False,  # Uses recommended params based on analysis
)
```

The system still analyzes the video and uses recommended parameters, but skips the iterative search (instant start).

## When to Use Optimization

### Use Auto-Optimization When:

- ✅ Processing videos offline
- ✅ Need maximum performance
- ✅ Have time for optimization (30-60 seconds)
- ✅ Processing multiple similar videos

### Skip Optimization When:

- ✅ Real-time processing needed
- ✅ Live feeds or streaming
- ✅ Need instant startup
- ✅ Videos are very different (optimization won't help)

## Performance Impact

### With Optimization

- **Adapts to your video**: Fast video gets higher threshold, slow video gets lower
- **Balances speed and accuracy**: Finds the sweet spot
- **Maximizes performance**: Gets the best FPS for your use case
- **50% better skip rates** in low-contrast scenarios (night drives)
- **3-5x improvement** in fast-motion scenarios (drones)

### Without Optimization

- Uses recommended parameters based on analysis
- Still adapts to video characteristics
- Instant startup (no 30-60 second delay)
- Good performance for most use cases

## Example Scenarios

### Scenario 1: Slow Traffic Video

- **Detected**: Slow motion, stable scene
- **Optimized**: Lower threshold (2.0), higher min_frames (3)
- **Result**: 65% skip rate, 2.8x speedup

### Scenario 2: Night Drive Video

- **Detected**: Fast motion, low contrast, low lighting
- **Optimized**: Reduced threshold (50% reduction), lower min_frames (1)
- **Result**: 35% skip rate (vs 20% without optimization), 1.9x speedup

### Scenario 3: Drone Footage

- **Detected**: Very fast motion, high contrast
- **Optimized**: Higher threshold (25.0), lower min_frames (1)
- **Result**: 22% skip rate (vs 5% without optimization), 1.4x speedup

## Monitoring Optimization

Check if optimization improved performance:

```python
results = inf.process_video_file("video.mp4")

# Check optimization results
if 'optimization_result' in results:
    opt = results['optimization_result']
    print(f"Best threshold: {opt.best_threshold:.2f}")
    print(f"Best min_frames: {opt.best_min_frames}")
    print(f"Final score: {opt.best_score:.2f}")
```

## Next Steps

- Learn about [Performance Tuning](./performance-tuning.md)
- Check [Adaptive System](../concepts/adaptive-system.md)
- Explore [API Reference](../api/intro.md)





