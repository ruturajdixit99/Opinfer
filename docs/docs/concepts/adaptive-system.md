# Adaptive System

The adaptive system automatically optimizes motion gating parameters based on detected video characteristics.

## Overview

The adaptive system (`AdaptiveMotionGater`) analyzes video properties and automatically tunes motion gating parameters for optimal performance. This eliminates the need for manual parameter tuning and adapts to different scenarios.

## Video Analysis

The system detects several video characteristics:

### Motion Pattern

Categorized based on average motion scores:
- **Static**: < 1.0 (traffic cameras, security cams)
- **Slow**: 1.0 - 5.0 (slow walking, minimal movement)
- **Moderate**: 5.0 - 15.0 (normal driving, moderate activity)
- **Fast**: 15.0 - 30.0 (fast driving, rapid movement)
- **Very Fast**: > 30.0 (drones, racing, fast pans)

### Lighting Conditions

Detected from average frame brightness:
- **Bright**: > 0.6 (daylight, well-lit scenes)
- **Normal**: 0.4 - 0.6 (normal lighting)
- **Low**: 0.2 - 0.4 (dusk, dim lighting)
- **Very Low**: < 0.2 (night, dark scenes)

### Contrast Level

Measures scene contrast (0-1):
- Higher contrast = clearer scenes
- Lower contrast = hazy, low-visibility scenes
- Affects threshold adjustments

### Scene Stability

Measures frame-to-frame consistency (0-1):
- High stability = static scenes
- Low stability = dynamic, changing scenes

## Parameter Optimization

Based on detected characteristics, the system automatically adjusts:

### Motion Threshold

```python
# Base threshold from motion pattern
if motion_pattern == "static":
    base_threshold = 2.0
elif motion_pattern == "slow":
    base_threshold = 4.0
elif motion_pattern == "moderate":
    base_threshold = 8.0
elif motion_pattern == "fast":
    base_threshold = 15.0
else:  # very_fast
    base_threshold = 25.0

# Adjust for low contrast (e.g., night drives)
if contrast_level < 0.2:
    threshold *= 0.5  # Reduce by 50%

# Adjust for lighting
if lighting_condition == "very_low":
    threshold *= 0.8  # Slight reduction
```

### Minimum Frames Between Calls

```python
if motion_pattern == "static":
    min_frames = 3
elif motion_pattern == "slow":
    min_frames = 2
elif motion_pattern in ["moderate", "fast"]:
    min_frames = 2
else:  # very_fast
    min_frames = 1
```

## Optimization Process

The system uses iterative search to find optimal parameters:

1. **Initial Range**: Based on video characteristics
2. **Grid Search**: Tests multiple threshold values
3. **Evaluation**: Measures skip rate and performance
4. **Refinement**: Narrows search space
5. **Early Stopping**: Stops when no improvement

### Optimization Settings

```python
gater = AdaptiveMotionGater(
    model=model,
    auto_optimize=True,  # Enable optimization
    optimization_sample_frames=200,  # Frames to use
    target_skip_rate=40.0,  # Target skip percentage
)
```

### Fast Mode (No Optimization)

For real-time applications, use fast defaults:

```python
gater = AdaptiveMotionGater(
    model=model,
    auto_optimize=False,  # Use recommended params only
)
```

## Recommended Parameters

When `auto_optimize=False`, the system uses recommended parameters based on analysis:

- **Static**: threshold=2.0, min_frames=3
- **Slow**: threshold=4.0, min_frames=2
- **Moderate**: threshold=8.0, min_frames=2
- **Fast**: threshold=15.0, min_frames=2
- **Very Fast**: threshold=25.0, min_frames=1

These are adjusted for lighting and contrast automatically.

## Performance Impact

Adaptive optimization provides:

- **50% better skip rates** in low-contrast scenarios (night drives)
- **3-5x improvement** in fast-motion scenarios (drones)
- **Automatic adaptation** to different video types
- **No manual tuning** required

## Example

```python
from opinfer import OptimizedInference

# Adaptive optimization enabled
inf = OptimizedInference(
    model_name="vit_base_patch16_224",
    auto_optimize=True,  # Automatically optimizes parameters
)

# System analyzes video and optimizes
results = inf.process_video_file("video.mp4")

# Parameters are automatically tuned based on:
# - Motion pattern detected
# - Lighting conditions
# - Contrast levels
# - Scene stability
```

## Next Steps

- Learn about [Motion Gating](./motion-gating.md)
- Explore [Performance Tuning](../guides/performance-tuning.md)
- Check [Optimization Guide](../guides/optimization.md)





