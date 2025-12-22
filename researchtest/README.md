# VLM Research Testing Suite

This folder contains comprehensive benchmarking and research tools for testing opinfer with various Vision-Language Models (VLMs).

## Overview

The research test suite evaluates opinfer's performance across:
- Multiple VLM models (OWL-ViT variants)
- Multiple classifier models (ViT/DeiT variants)
- Different video scenarios (static, moderate, fast motion)
- Performance metrics (skip rate, FPS, inference time)

## Files

- `vlm_benchmark.py` - Main benchmarking script
- `requirements.txt` - Additional dependencies for research
- `README.md` - This file

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: If you encounter NumPy compatibility errors (NumPy 2.x with TensorFlow), try:
```bash
pip install 'numpy<2' 'pandas<2' 'matplotlib<4'
```

2. Run the benchmark:
```bash
python vlm_benchmark.py
```

Or use the simplified runner:
```bash
python run_benchmark.py
```

3. Results will be generated in:
   - `vlm_benchmark_results.json` - Raw results data
   - `benchmark_report.md` - Markdown report
   - `graphs/` - Visualization graphs

## Outputs

### Graphs Generated:
1. `skip_rate_by_model.png` - Skip rate comparison across models
2. `fps_by_model.png` - Effective FPS comparison
3. `performance_by_scenario.png` - Performance by video type
4. `vlm_vs_classifier.png` - VLM vs Classifier comparison
5. `skip_rate_heatmap.png` - Heatmap of skip rates
6. `detailed_metrics.png` - Comprehensive performance metrics

### Reports:
- JSON results for programmatic analysis
- Markdown report with summary statistics

## Models Tested

### VLM Models (Detectors):
- OWL-ViT Base
- OWL-ViT Large

### Classifier Models:
- ViT Base
- ViT Large
- DeiT Base

## Video Scenarios

- Static Traffic Camera
- Moderate Motion
- Fast Motion (Drone)

