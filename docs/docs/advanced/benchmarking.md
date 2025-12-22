# Benchmarking

How to benchmark Opinfer performance.

## Basic Benchmarking

Benchmark a single configuration:

```python
from opinfer import OptimizedInference
import time

inf = OptimizedInference(
    model_name="vit_base_patch16_224",
    auto_optimize=True,
)

start_time = time.time()
results = inf.process_video_file("video.mp4", max_frames=500)
elapsed = time.time() - start_time

stats = results['stats']
print(f"Total time: {elapsed:.2f}s")
print(f"Skip rate: {stats['skip_rate_pct']:.1f}%")
print(f"Effective FPS: {stats['effective_fps']:.2f}")
```

## Comparing Techniques

Compare motion gating vs queuing:

```python
from opinfer import OptimizedInference

# Motion gating
mg = OptimizedInference(
    model_name="vit_base_patch16_224",
    technique="motion_gating",
    auto_optimize=True,
)
results_mg = mg.process_video_file("video.mp4")

# Queuing
q = OptimizedInference(
    model_name="vit_base_patch16_224",
    technique="queuing",
)
results_q = q.process_video_file("video.mp4")

# Compare
print(f"Motion Gating - Skip: {results_mg['stats']['skip_rate_pct']:.1f}%")
print(f"Queuing - FPS: {results_q['stats']['effective_fps']:.2f}")
```

## Benchmarking Multiple Models

Test different model sizes:

```python
models = [
    "vit_tiny_patch16_224",
    "vit_small_patch16_224",
    "vit_base_patch16_224",
]

for model_name in models:
    inf = OptimizedInference(model_name=model_name, auto_optimize=True)
    results = inf.process_video_file("video.mp4", max_frames=200)
    stats = results['stats']
    print(f"{model_name}: {stats['skip_rate_pct']:.1f}% skip, {stats['effective_fps']:.2f} FPS")
```

## Using the Benchmark Script

Use the included benchmark script:

```bash
python benchmark_all.py
```

This will:
- Test all available models
- Process multiple video scenarios
- Generate detailed performance reports
- Save results to `benchmark_results/`

## Research Benchmarking

For detailed research benchmarking:

```bash
cd researchtest
python run_benchmark.py --fast-mode
```

This provides:
- Comprehensive VLM benchmarking
- Performance graphs
- Detailed metrics
- Comparison reports

## Next Steps

- [Performance Tuning](../guides/performance-tuning.md)
- [Examples](../getting-started/examples.md)





