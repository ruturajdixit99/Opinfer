"""
Example usage of opinfer package.
Demonstrates how to use the adaptive motion gating system.
"""

from opinfer import OptimizedInference
from pathlib import Path


def example_basic_usage():
    """Basic usage example."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 80)
    
    # Initialize
    inf = OptimizedInference(
        model_name="vit_base_patch16_224",
        model_type="classifier",
        auto_optimize=True,
        target_skip_rate=40.0,
    )
    
    # Process video
    video_path = "vdo/trafficcam30fps.mp4"
    if Path(video_path).exists():
        results = inf.process_video_file(video_path, max_frames=200)
        
        print(f"\nResults:")
        print(f"  Skip rate: {results['stats']['skip_rate_pct']:.1f}%")
        print(f"  Effective FPS: {results['stats']['effective_fps']:.2f}")
        print(f"  Motion threshold: {results['parameters']['motion_threshold']:.2f}")
    else:
        print(f"Video not found: {video_path}")


def example_all_scenarios():
    """Example testing all three scenarios."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Testing All Scenarios")
    print("=" * 80)
    
    scenarios = {
        "traffic_cam": "vdo/trafficcam30fps.mp4",
        "night_drive": "vdo/mediumpace30fps.mp4",
        "drone": "vdo/mediumspeeddrone30fps.mp4",
    }
    
    for scenario_name, video_path in scenarios.items():
        if not Path(video_path).exists():
            print(f"\n⚠️  Skipping {scenario_name}: video not found")
            continue
        
        print(f"\n🎬 Processing {scenario_name}...")
        
        inf = OptimizedInference(
            model_name="vit_base_patch16_224",
            model_type="classifier",
            auto_optimize=True,
        )
        
        try:
            results = inf.process_video_file(video_path, max_frames=300)
            
            print(f"  ✅ Skip rate: {results['stats']['skip_rate_pct']:.1f}%")
            print(f"  ✅ Effective FPS: {results['stats']['effective_fps']:.2f}")
            print(f"  ✅ Optimized threshold: {results['parameters']['motion_threshold']:.2f}")
            
            # Show video characteristics
            if results.get('video_characteristics'):
                vc = results['video_characteristics']
                print(f"  📊 Motion pattern: {vc.motion_pattern}")
                print(f"  📊 Lighting: {vc.lighting_condition}")
                print(f"  📊 Contrast: {vc.contrast_level:.2f}")
        
        except Exception as e:
            print(f"  ❌ Error: {e}")


def example_detector_model():
    """Example using detector model."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Detector Model (OWL-ViT)")
    print("=" * 80)
    
    inf = OptimizedInference(
        model_name="owlvit-base",
        model_type="detector",
        auto_optimize=True,
    )
    
    video_path = "vdo/trafficcam30fps.mp4"
    if Path(video_path).exists():
        results = inf.process_video_file(video_path, max_frames=200)
        
        print(f"\nResults:")
        print(f"  Skip rate: {results['stats']['skip_rate_pct']:.1f}%")
        print(f"  Effective FPS: {results['stats']['effective_fps']:.2f}")
    else:
        print(f"Video not found: {video_path}")


if __name__ == "__main__":
    print("\n🚀 Opinfer Examples\n")
    
    # Run examples
    example_basic_usage()
    example_all_scenarios()
    example_detector_model()
    
    print("\n✅ Examples complete!")

