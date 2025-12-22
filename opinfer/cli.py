"""
Command-line interface for opinfer.
"""

import argparse
import sys
from pathlib import Path

from opinfer.api import OptimizedInference, quick_inference
from opinfer.models import ModelLoader


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Opinfer: Optimized Inference with Adaptive Motion Gating",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process video command
    process_parser = subparsers.add_parser("process", help="Process a video file")
    process_parser.add_argument("video_path", type=str, help="Path to video file")
    process_parser.add_argument(
        "--model",
        type=str,
        default="vit_base_patch16_224",
        help="Model name (default: vit_base_patch16_224)",
    )
    process_parser.add_argument(
        "--model-type",
        type=str,
        choices=["classifier", "detector"],
        default="classifier",
        help="Model type (default: classifier)",
    )
    process_parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process (default: all)",
    )
    process_parser.add_argument(
        "--target-skip-rate",
        type=float,
        default=40.0,
        help="Target skip rate percentage (default: 40.0)",
    )
    process_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (default: cuda)",
    )
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark all models")
    benchmark_parser.add_argument("video_path", type=str, help="Path to video file")
    benchmark_parser.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Maximum frames per model (default: 500)",
    )
    benchmark_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (default: cuda)",
    )
    
    # List models command
    list_parser = subparsers.add_parser("list-models", help="List available models")
    
    args = parser.parse_args()
    
    if args.command == "process":
        print(f"\n🎥 Processing video: {args.video_path}")
        print(f"   Model: {args.model} ({args.model_type})")
        
        try:
            results = quick_inference(
                video_path=args.video_path,
                model_name=args.model,
                model_type=args.model_type,
                max_frames=args.max_frames,
            )
            
            stats = results.get("stats", {})
            print("\n📊 Results:")
            print(f"   Total frames: {stats.get('total_frames', 0)}")
            print(f"   Model calls: {stats.get('model_calls', 0)}")
            print(f"   Skipped frames: {stats.get('skipped_frames', 0)}")
            print(f"   Skip rate: {stats.get('skip_rate_pct', 0):.1f}%")
            print(f"   Avg inference time: {stats.get('avg_inference_ms', 0):.2f} ms")
            print(f"   Effective FPS: {stats.get('effective_fps', 0):.2f}")
            
            params = results.get("parameters", {})
            print(f"\n⚙️  Optimized Parameters:")
            print(f"   Motion threshold: {params.get('motion_threshold', 0):.2f}")
            print(f"   Min frames between calls: {params.get('min_frames_between_calls', 0)}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)
    
    elif args.command == "benchmark":
        print(f"\n📊 Benchmarking all models on: {args.video_path}")
        
        try:
            inf = OptimizedInference(
                model_name="vit_base_patch16_224",  # Dummy, will be replaced
                model_type="classifier",
                device=args.device,
            )
            
            results = inf.benchmark_all_models(
                video_path=args.video_path,
                max_frames=args.max_frames,
            )
            
            print("\n✅ Benchmark complete!")
            print(f"   Models tested: {len(results)}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)
    
    elif args.command == "list-models":
        models = ModelLoader.list_models()
        
        print("\n📋 Available Models:\n")
        print("Classifiers:")
        for model in models["classifiers"]:
            print(f"   - {model}")
        
        print("\nDetectors:")
        for model in models["detectors"]:
            print(f"   - {model}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()









