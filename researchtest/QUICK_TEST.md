# Quick Testing Guide

The benchmark script can take a long time if running full optimization. Here are ways to speed it up:

## ⚡ Ultra-Fast Option (RECOMMENDED - 3-5 minutes)

```bash
python quick_benchmark.py
```

Or use the ultra-fast flag:
```bash
python vlm_benchmark.py --ultra-fast --frames 100
```

**Features:**
- ✅ Tests all models and scenarios
- ✅ Uses recommended parameters (from video analysis)
- ✅ Generates all graphs and reports
- ✅ Gets optimized values (recommended thresholds)
- ⏱️ **Completes in 3-5 minutes**

## Fast Options

### Option 1: Fast Mode
```bash
python vlm_benchmark.py --fast --frames 150
```
- Uses fewer optimization iterations (10 instead of 50)
- Processes fewer frames
- Still gets optimized results
- Time: 10-15 minutes

### Option 2: Skip Optimization
```bash
python vlm_benchmark.py --no-opt --frames 100
```
- Uses recommended parameters from video analysis
- Skips the iterative optimization step
- Time: 5-8 minutes

### Option 3: Ultra-Fast Mode (Best Balance)
```bash
python vlm_benchmark.py --ultra-fast --frames 100
```
- Analysis only, no optimization loop
- Still gets good parameters
- Time: 3-5 minutes ⚡

## Full Benchmark (Slow but Complete)
```bash
python vlm_benchmark.py
```
- Full optimization (50 iterations)
- Processes 300 frames per video
- Most accurate results
- Time: 30-60 minutes

## Expected Times

| Mode | Time | Accuracy | Optimization |
|------|------|----------|--------------|
| **Ultra-Fast** | **3-5 min** | **Good** | **Recommended params** |
| Fast Mode | 10-15 min | Very Good | 10 iterations |
| Skip Optimization | 5-8 min | Good | Recommended params |
| Full Benchmark | 30-60 min | Best | 50 iterations |

## What You Get in All Modes

✅ Performance metrics for all scenarios  
✅ Optimized parameter values  
✅ Comparative graphs  
✅ Detailed reports  
✅ JSON results for further analysis

