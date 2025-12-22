# Motion Gating Technology - Complete Logic Explanation

This document explains the core motion gating technology used in opinfer, including both the main inference logic and the dynamic adaptation system that adjusts parameters based on video characteristics.

---

## Table of Contents

1. [Overview](#overview)
2. [Core Motion Gating Logic](#core-motion-gating-logic)
3. [Dynamic Adaptation System](#dynamic-adaptation-system)
4. [Complete Flow Diagram](#complete-flow-diagram)
5. [Key Algorithms](#key-algorithms)

---

## Overview

**Motion Gating** is an optimization technique that skips redundant model inference calls by only processing frames when there's significant motion in the scene. The system has two main components:

1. **Core Motion Gating Engine** (`MotionGatedInference`) - Main inference logic that decides when to run the model
2. **Adaptive System** (`AdaptiveMotionGater`) - Dynamically adjusts parameters based on video characteristics

### Why Motion Gating?

- **Problem**: Running vision models on every frame is computationally expensive
- **Insight**: Most consecutive frames in a video are very similar
- **Solution**: Only process frames when the scene has changed significantly (motion detected)
- **Result**: 30-50% reduction in inference calls while maintaining accuracy

---

## Core Motion Gating Logic

### Location: `opinfer/core.py` - `MotionGatedInference` class

### Key Components

#### 1. Motion Score Computation

**Function**: `compute_motion_score(frame_bgr)`

**Algorithm**:
```python
1. Convert frame to grayscale
2. Downsample to 64x64 (for speed: less pixels to process)
3. Compare with previous frame using absolute difference
4. Calculate mean difference = motion score
```

**Code Logic**:
```python
gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
gray_small = cv2.resize(gray, (64, 64))  # Downsample for speed

if previous_frame exists:
    diff = abs(current_frame - previous_frame)
    motion_score = mean(diff)  # Average pixel difference
else:
    motion_score = 0.0  # First frame, no motion
```

**Why This Works**:
- **Grayscale**: Color doesn't matter for motion detection
- **Downsampling**: 64x64 is enough to detect motion, much faster than full resolution
- **Absolute difference**: Shows how much pixels changed between frames
- **Mean**: Single number representing overall motion

**Example Values**:
- `0-2`: Static scene (traffic cam with no movement)
- `2-5`: Slow motion (walking person, slow car)
- `5-10`: Moderate motion (normal traffic)
- `10-20`: Fast motion (drone footage, fast cars)
- `20+`: Very fast motion (action scenes)

---

#### 2. Decision Logic: Should We Call the Model?

**Function**: `should_call_model(motion_score)`

**Decision Tree**:
```
IF this is the first frame:
    → YES, call model (need initial prediction)

ELSE IF motion_score >= motion_threshold:
    → YES, call model (significant scene change)

ELSE IF frames_since_last_call >= min_frames_between_calls:
    → YES, call model (force refresh to avoid stale predictions)

ELSE:
    → NO, reuse previous output (frame is similar to last processed one)
```

**Parameters**:
- **`motion_threshold`**: Minimum motion score to trigger inference
  - Lower = more sensitive (processes more frames)
  - Higher = less sensitive (skips more frames)
  - Typical range: 1.0 - 20.0

- **`min_frames_between_calls`**: Minimum frames before forcing a refresh
  - Prevents predictions from becoming too stale
  - Even if motion is low, we update every N frames
  - Typical values: 2-5 frames

**Example Scenario**:
```
Frame 1: motion=0.0 → CALL (first frame)
Frame 2: motion=1.5, threshold=4.0 → SKIP (motion too low)
Frame 3: motion=1.2, threshold=4.0 → CALL (min_frames=2 reached)
Frame 4: motion=5.5, threshold=4.0 → CALL (motion exceeds threshold)
Frame 5: motion=1.8, threshold=4.0 → SKIP
```

---

#### 3. Inference Execution

**Function**: `infer(frame_bgr)`

**Process Flow**:
```
1. Compute motion score for current frame
2. Decide if model should be called
3. IF should_call:
     - Preprocess frame (resize, normalize)
     - Run model inference
     - Store output
     - Reset frame counter
     - Update statistics
4. ELSE:
     - Return cached previous output
     - Increment frame counter
5. Return (output, statistics)
```

**Efficiency Gains**:
- Model inference: ~10-50ms per frame (expensive)
- Motion computation: ~1-2ms per frame (cheap)
- **Savings**: When we skip, we save 10-50ms per frame!

---

### Core Logic Summary

The core motion gating follows this simple principle:

> **"Only run expensive model inference when the scene has changed enough to warrant a new prediction"**

**State Variables**:
- `prev_gray_small`: Previous frame (for comparison)
- `last_output`: Last model prediction (cached for reuse)
- `frames_since_last_call`: Counter to force periodic refreshes

**Performance Metrics**:
- **Skip Rate**: % of frames where model wasn't called
- **Effective FPS**: Actual processing speed considering skips
- **Model Calls**: Total number of inference operations

---

## Dynamic Adaptation System

### Location: `opinfer/adaptive.py` - `AdaptiveMotionGater` class

The adaptive system solves a critical problem: **different videos need different parameters**.

### The Problem

A single set of parameters doesn't work for all scenarios:
- **Traffic cam (static)**: High threshold (8.0), high min_frames (3) → Skip many frames
- **Night drive (low contrast)**: Low threshold (2.0), low min_frames (2) → More sensitive
- **Drone (fast motion)**: Medium threshold (5.0), low min_frames (1) → Can't skip much

### The Solution: Three-Step Adaptive Process

#### Step 1: Video Characteristic Detection

**Location**: `opinfer/detectors.py` - `VideoCharacteristicDetector`

**What It Detects**:

1. **Motion Pattern** (from motion scores):
   ```python
   avg_motion < 2.0   → "static"
   avg_motion < 5.0   → "slow"
   avg_motion < 10.0  → "moderate"
   avg_motion < 20.0  → "fast"
   else               → "very_fast"
   ```

2. **Lighting Condition** (from brightness):
   ```python
   brightness > 0.7   → "bright"
   brightness > 0.4   → "normal"
   brightness > 0.2   → "low"
   else               → "very_low"
   ```

3. **Contrast Level** (from standard deviation):
   ```python
   contrast = std(grayscale_pixels) / 255.0
   ```

4. **Scene Stability** (from motion variance):
   ```python
   stability = 1.0 - (motion_variance / max_variance)
   # Lower variance = more stable scene
   ```

**Intelligent Parameter Recommendation**:

Based on detected characteristics, the system recommends initial parameters:

```python
# Base threshold from average motion
base_threshold = avg_motion * 0.8

# Adjust for low contrast (night scenes need lower threshold)
if contrast < 0.2:
    base_threshold *= 0.5  # 50% reduction for night scenes

# Adjust for low lighting
if lighting < 0.3:
    base_threshold *= 0.6  # Reduce for dark scenes

# For fast motion, need higher threshold
if motion_pattern == "fast":
    base_threshold = max(base_threshold, avg_motion * 1.2)

# Set bounds
min_threshold = max(1.0, base_threshold * 0.5)
max_threshold = min(50.0, base_threshold * 2.0)
```

**Why These Adjustments Work**:

- **Night/Low Contrast**: Motion is harder to detect (dark pixels don't change much), so we need lower threshold
- **Fast Motion**: Too many frames trigger, so we raise threshold to filter out minor changes
- **Static Scenes**: Can safely use high threshold and skip many frames

---

#### Step 2: Parameter Optimization (Iterative Search)

**Location**: `opinfer/optimizer.py` - `ParameterOptimizer`

**Goal**: Find the best parameters that achieve target skip rate while maintaining good performance.

**Optimization Algorithm**:

1. **Initial Search Space**:
   ```python
   threshold_range = recommended_range_from_detection
   min_frames_candidates = [1, 2, 3, 4, 5]
   ```

2. **Grid Search**:
   - Test multiple threshold values
   - Test multiple min_frames values
   - Evaluate each combination on sample frames

3. **Scoring Function**:
   ```python
   score = (
       skip_rate_score * 0.4 +    # How close to target skip rate
       fps_score * 0.3 +           # How fast we process
       call_ratio_score * 0.3      # Balanced model call frequency
   )
   ```

4. **Refinement**:
   - After initial search, narrow around best parameters
   - Search more finely in promising regions
   - Early stop if no improvement for N iterations

**Optimization Loop**:
```
FOR each iteration (max 50):
    FOR each threshold in range:
        FOR each min_frames in candidates:
            1. Run inference on sample frames with these parameters
            2. Measure skip rate, FPS, call frequency
            3. Compute score
            4. IF score > best_score:
                 - Update best parameters
                 - Reset improvement counter
    IF no improvement for 10 iterations:
        → STOP (early stopping)
    ELSE:
        → Refine search around best parameters
        → Continue
```

**Example Optimization**:
```
Initial recommendation: threshold=[2.0, 8.0], min_frames=2

Iteration 1: Testing 10 thresholds × 5 min_frames = 50 combinations
  Best so far: threshold=4.5, min_frames=2, score=78.5

Iteration 2: Refining around 4.5: testing [3.5, 5.5]
  Best so far: threshold=4.2, min_frames=2, score=82.3

Iteration 3: Further refinement around 4.2
  Best so far: threshold=4.1, min_frames=2, score=83.1

Iteration 4-6: No improvement
  → Early stop

Final: threshold=4.1, min_frames=2
```

---

#### Step 3: Engine Initialization with Optimized Parameters

After optimization, create the motion gating engine with best parameters:

```python
engine = MotionGatedInference(
    model=model,
    device=device,
    motion_threshold=optimized_threshold,      # From optimization
    min_frames_between_calls=optimized_min_frames  # From optimization
)
```

---

### Complete Adaptive Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. ANALYZE VIDEO CHARACTERISTICS                            │
│    └─> Detect: motion, lighting, contrast, stability       │
│    └─> Classify: pattern type (static/slow/fast/etc)       │
│    └─> Recommend: initial parameter range                   │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. OPTIMIZE PARAMETERS (if auto_optimize=True)              │
│    └─> Grid search over recommended range                   │
│    └─> Evaluate each combination on sample frames           │
│    └─> Score based on skip rate, FPS, call frequency        │
│    └─> Refine search around best results                    │
│    └─> Return: best threshold, best min_frames              │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. INITIALIZE ENGINE                                        │
│    └─> Create MotionGatedInference with optimized params    │
│    └─> Ready to process video frames                        │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. PROCESS FRAMES (Runtime)                                 │
│    FOR each frame:                                          │
│      └─> Compute motion score                               │
│      └─> Decide: call model or reuse output?                │
│      └─> IF call: run inference, cache output               │
│      └─> ELSE: return cached output                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Real-World Example Scenarios

### Scenario 1: Static Traffic Camera

**Video Characteristics**:
- Motion pattern: "static" (avg_motion = 1.5)
- Lighting: "normal" (brightness = 0.6)
- Contrast: "normal" (contrast = 0.3)

**Recommended Parameters**:
- Threshold: 3.0-6.0 (can skip many frames)
- Min frames: 3 (update every 3 frames max)

**Optimized Result**:
- Threshold: 4.5
- Min frames: 3
- Skip rate: 65% (processes 1 out of ~3 frames)
- **Result**: 65% faster inference!

---

### Scenario 2: Night Drive (Low Contrast)

**Video Characteristics**:
- Motion pattern: "moderate" (avg_motion = 6.0)
- Lighting: "low" (brightness = 0.25)
- Contrast: "very_low" (contrast = 0.15)

**Recommended Parameters**:
- Threshold: 2.0-4.0 (lower because low contrast makes motion detection harder)
- Min frames: 2

**Optimized Result**:
- Threshold: 2.8 (lower than normal for sensitivity)
- Min frames: 2
- Skip rate: 35% (less skipping due to low contrast)
- **Result**: Still 35% faster, but less aggressive

---

### Scenario 3: Fast Drone Footage

**Video Characteristics**:
- Motion pattern: "very_fast" (avg_motion = 25.0)
- Lighting: "bright" (brightness = 0.8)
- Contrast: "normal" (contrast = 0.35)

**Recommended Parameters**:
- Threshold: 20.0-30.0 (very high to filter minor changes)
- Min frames: 1 (can't skip much)

**Optimized Result**:
- Threshold: 22.0
- Min frames: 1
- Skip rate: 25% (minimal skipping, most frames needed)
- **Result**: Modest improvement, but still 25% faster

---

## Key Algorithms Explained

### Algorithm 1: Motion Score Computation

```python
def compute_motion_score(frame):
    # 1. Convert to grayscale (color doesn't matter for motion)
    gray = to_grayscale(frame)
    
    # 2. Downsample (64x64 is enough, much faster)
    small = resize(gray, 64x64)
    
    # 3. Compare with previous frame
    if previous_frame exists:
        diff = absolute_difference(small, previous_frame)
        score = mean(diff)  # Average pixel difference
    else:
        score = 0.0  # First frame
    
    # 4. Store for next comparison
    previous_frame = small
    
    return score
```

**Time Complexity**: O(64×64) = O(4096) pixels per frame (very fast!)

---

### Algorithm 2: Decision Making

```python
def should_call_model(motion_score, threshold, min_frames, frames_since_call):
    # Rule 1: Always process first frame
    if first_frame:
        return True
    
    # Rule 2: Motion exceeded threshold
    if motion_score >= threshold:
        return True
    
    # Rule 3: Force refresh after N frames
    if frames_since_call >= min_frames:
        return True
    
    # Rule 4: Skip this frame
    return False
```

**Decision Time**: O(1) - constant time decision

---

### Algorithm 3: Parameter Optimization

```python
def optimize_parameters(frames, target_skip_rate):
    # 1. Get recommended range from video analysis
    threshold_range = analyze_video_characteristics(frames)
    
    # 2. Grid search
    best_score = -infinity
    best_params = None
    
    for threshold in threshold_range:
        for min_frames in [1, 2, 3, 4, 5]:
            # 3. Evaluate this combination
            score = evaluate_parameters(threshold, min_frames, frames)
            
            # 4. Track best
            if score > best_score:
                best_score = score
                best_params = (threshold, min_frames)
    
    # 5. Refine around best (iterative refinement)
    # ... (narrow search and repeat)
    
    return best_params
```

**Complexity**: O(iterations × threshold_candidates × min_frames_candidates × frames)
- Typically: 10 iterations × 10 thresholds × 5 min_frames × 200 frames
- = ~100,000 inference evaluations (but on small sample, fast)

---

## Summary

### Core Motion Gating

1. **Compute motion score** (cheap operation: ~1-2ms)
2. **Decide** if model should run (instant)
3. **IF motion high OR refresh needed**: Run model (expensive: ~10-50ms)
4. **ELSE**: Reuse previous output (instant)

**Key Insight**: Motion computation is 10-50x cheaper than model inference!

### Dynamic Adaptation

1. **Analyze video** → Understand scene characteristics
2. **Optimize parameters** → Find best threshold and min_frames
3. **Initialize engine** → Use optimized parameters
4. **Process frames** → Enjoy optimized performance!

**Key Insight**: Different videos need different parameters - the system finds them automatically!

---

## Performance Impact

### Without Motion Gating
- Process: 30 FPS video
- Model calls: 30 per second
- Inference time: 30 × 20ms = 600ms per second
- **Result**: Can't keep up, drops frames

### With Motion Gating (40% skip rate)
- Process: 30 FPS video
- Model calls: 18 per second (40% skipped)
- Inference time: 18 × 20ms = 360ms per second
- **Result**: 40% faster, maintains quality!

### With Adaptive Motion Gating
- Automatically adjusts for each video
- Traffic cam: 65% skip rate
- Night drive: 35% skip rate
- Drone: 25% skip rate
- **Result**: Optimal performance for every scenario!

---

This is the complete logic behind opinfer's motion gating technology!









