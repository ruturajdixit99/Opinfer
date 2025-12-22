"""
Simplified benchmark runner that works around NumPy compatibility issues.
Run this if you encounter NumPy/TensorFlow compatibility errors.
"""

import sys
import os
from pathlib import Path

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set environment variable to potentially avoid TensorFlow issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Try to run the benchmark
try:
    from vlm_benchmark import VLMResearchBenchmark
    
    print("Starting VLM Benchmark (Fast Mode - Recommended)...")
    print("Use --fast for even faster runs, --no-opt to skip optimization")
    benchmark = VLMResearchBenchmark(
        video_dir="../vdo",
        max_frames=150,  # Reduced for faster testing
        fast_mode=True,  # Enable fast mode by default
        skip_optimization=False,  # Still optimize but faster
    )
    
    # Run benchmarks
    benchmark.run_all_benchmarks(techniques=["motion_gating"])
    
    # Save results
    benchmark.save_results()
    
    # Try to generate graphs (will skip if matplotlib unavailable)
    try:
        benchmark.generate_graphs()
    except Exception as e:
        print(f"⚠️  Could not generate graphs: {e}")
        print("   Results saved to JSON, you can analyze them manually")
    
    # Generate report
    benchmark.generate_report()
    
    print("\n✅ Benchmark complete!")
    print("   Check vlm_benchmark_results.json and benchmark_report.md for results")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPossible solutions:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Check NumPy version compatibility")
    print("3. Try: pip install 'numpy<2' to use NumPy 1.x")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

