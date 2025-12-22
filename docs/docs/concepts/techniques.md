# Optimization Techniques

Opinfer provides two complementary optimization techniques: Motion Gating and Frame Queuing.

## Motion Gating

Motion gating skips redundant frames based on detected motion. This is the default and recommended technique for most use cases.

### How It Works

1. Computes motion score between consecutive frames
2. Calls model only when motion exceeds threshold
3. Reuses previous prediction when motion is low

### Best For

- ✅ Static or slow-moving scenes (traffic cameras, security feeds)
- ✅ Scenarios where accuracy is more important than consistent frame rate
- ✅ GPU-constrained environments
- ✅ Real-time processing with variable frame rates

### Advantages

- High skip rates (30-70%) in appropriate scenarios
- Maintains accuracy by processing key frames
- Automatic parameter optimization
- Lower computational overhead

### Example

```python
inf = OptimizedInference(
    model_name="vit_base_patch16_224",
    technique="motion_gating",  # Default
    auto_optimize=True,
)
```

## Frame Queuing

Frame queuing batches frames for processing to maintain consistent frame rates.

### How It Works

1. Accumulates frames in a queue
2. Processes batches when queue is full or timeout reached
3. Maintains consistent frame rate output

### Best For

- ✅ Applications requiring consistent frame rates
- ✅ Scenarios where latency can be tolerated
- ✅ Batch processing scenarios
- ✅ When you need predictable throughput

### Advantages

- Consistent frame rates
- Better GPU utilization through batching
- Predictable performance
- Lower per-frame overhead

### Example

```python
inf = OptimizedInference(
    model_name="vit_base_patch16_224",
    technique="queuing",
    queue_size=4,  # Max frames in queue
    batch_size=4,  # Frames per batch
    max_queue_wait_ms=33.0,  # Max wait time (30 FPS)
)
```

## Comparison

| Feature | Motion Gating | Queuing |
|---------|--------------|---------|
| Skip Rate | 30-70% | N/A (processes all) |
| Frame Rate | Variable | Consistent |
| Latency | Low (immediate) | Higher (batched) |
| GPU Usage | Lower | Higher (batched) |
| Best For | Real-time, accuracy | Consistency, throughput |

## Choosing a Technique

### Use Motion Gating When:

- You want maximum efficiency (fewer model calls)
- Frame rate can vary
- Real-time processing is needed
- You're processing static or slow-moving scenes

### Use Queuing When:

- You need consistent frame rates
- Latency is acceptable
- You want predictable throughput
- You're batch processing videos

## Combined Usage

You can use both techniques in different parts of your pipeline:

```python
# Use motion gating for real-time detection
detector = OptimizedInference(
    model_name="owlvit-base",
    model_type="detector",
    technique="motion_gating",
)

# Use queuing for post-processing
processor = OptimizedInference(
    model_name="vit_base_patch16_224",
    technique="queuing",
)
```

## Next Steps

- Learn about [Motion Gating](./motion-gating.md) in detail
- Explore [API Reference](../api/intro.md)
- Check [Performance Tuning](../guides/performance-tuning.md)





