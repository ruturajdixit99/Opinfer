# Motion Gating

Motion gating is the core optimization technique in Opinfer that skips redundant model inference calls by only processing frames when there's significant motion in the scene.

## Overview

**Motion Gating** is an optimization technique that skips redundant model inference calls by only processing frames when there's significant motion in the scene. The system has two main components:

1. **Core Motion Gating Engine** (`MotionGatedInference`) - Main inference logic that decides when to run the model
2. **Adaptive System** (`AdaptiveMotionGater`) - Dynamically adjusts parameters based on video characteristics

## Why Motion Gating?

- **Problem**: Running vision models on every frame is computationally expensive
- **Insight**: Most consecutive frames in a video are very similar
- **Solution**: Only process frames when the scene has changed significantly (motion detected)
- **Result**: 30-50% reduction in inference calls while maintaining accuracy

## How It Works

### Motion Score Computation

For each frame, the system computes a motion score:

1. Convert frame to grayscale
2. Downsample to 64x64 (for speed: less pixels to process)
3. Compare with previous frame using absolute difference
4. Calculate mean difference = motion score

```python
def compute_motion_score(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, (64, 64))
    
    if prev_frame is not None:
        diff = cv2.absdiff(gray_small, prev_frame)
        motion_score = np.mean(diff)
    else:
        motion_score = 0.0
    
    prev_frame = gray_small
    return motion_score
```

### Decision Logic

The system decides whether to call the model based on:

1. **Motion Threshold**: If motion_score > threshold → call model
2. **Minimum Frames**: Must have at least `min_frames_between_calls` since last call
3. **Previous Output**: If conditions not met → reuse previous prediction

```python
def should_call_model(motion_score):
    # Must exceed threshold
    if motion_score < motion_threshold:
        return False
    
    # Must wait minimum frames between calls
    if frames_since_last_call < min_frames_between_calls:
        return False
    
    return True
```

## Key Parameters

### Motion Threshold

Controls sensitivity to motion:
- **Low threshold** (1-3): More sensitive, calls model more often
- **Medium threshold** (4-8): Balanced (default)
- **High threshold** (10+): Less sensitive, fewer calls

### Minimum Frames Between Calls

Ensures minimum spacing between model calls:
- **Low value** (1-2): Allows more frequent calls
- **Medium value** (2-4): Balanced (default)
- **High value** (5+): Forces longer gaps between calls

## Adaptive Optimization

The system automatically optimizes these parameters based on video characteristics:

- **Static scenes**: Higher threshold, higher min_frames
- **Fast motion**: Lower threshold, lower min_frames
- **Low contrast**: Reduced threshold (50% reduction)
- **Scene stability**: Adjusted based on frame-to-frame consistency

See [Adaptive System](./adaptive-system.md) for details.

## Performance Impact

### Typical Skip Rates

- **Static scenes** (traffic cameras): 60-70% skip rate
- **Moderate motion** (normal driving): 40-50% skip rate
- **Fast motion** (drones, racing): 20-30% skip rate

### Speed Improvements

- **2-3x speedup** in static scenarios
- **1.5-2x speedup** in moderate motion
- **1.3-1.6x speedup** in fast motion

## Limitations

Motion gating works best when:
- ✅ Scenes have periods of relative stability
- ✅ Motion is predictable (e.g., camera movement, object movement)
- ✅ Frame rate is consistent

Less effective when:
- ❌ Constant rapid motion
- ❌ Random frame drops or inconsistent frame rates
- ❌ Very low frame rates (< 10 FPS)

## Next Steps

- Learn about [Adaptive System](./adaptive-system.md)
- Explore [Frame Queuing](../concepts/techniques.md#queuing)
- Check [Performance Tuning Guide](../guides/performance-tuning.md)





