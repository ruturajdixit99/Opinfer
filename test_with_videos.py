"""
Test opinfer with your actual video files.
This script will test all three scenarios: traffic cam, night drive, and drone.
"""

import sys
from pathlib import Path

from opinfer import OptimizedInference


def test_scenario(video_path: str, scenario_name: str, max_frames: int = 200):
    """Test a single video scenario."""
    print(f"\n{'='*80}")
    print(f"Testing: {scenario_name}")
    print(f"Video: {video_path}")
    print(f"{'='*80}")
    
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return None
    
    try:
        # Create inference instance
        inf = OptimizedInference(
            model_name="vit_base_patch16_224",
            model_type="classifier",
            device="cuda",
            auto_optimize=True,
            target_skip_rate=40.0,
        )
        
        # Process video
        print(f"\n🎬 Processing {max_frames} frames...")
        results = inf.process_video_file(video_path, max_frames=max_frames)
        
        # Display results
        stats = results.get("stats", {})
        params = results.get("parameters", {})
        vc = results.get("video_characteristics")
        
        print(f"\n📊 Results:")
        print(f"   Total frames: {stats.get('total_frames', 0)}")
        print(f"   Model calls: {stats.get('model_calls', 0)}")
        print(f"   Skipped frames: {stats.get('skipped_frames', 0)}")
        print(f"   Skip rate: {stats.get('skip_rate_pct', 0):.1f}%")
        print(f"   Avg inference time: {stats.get('avg_inference_ms', 0):.2f} ms")
        print(f"   Effective FPS: {stats.get('effective_fps', 0):.2f}")
        
        print(f"\n⚙️  Optimized Parameters:")
        print(f"   Motion threshold: {params.get('motion_threshold', 0):.2f}")
        print(f"   Min frames between calls: {params.get('min_frames_between_calls', 0)}")
        
        if vc:
            print(f"\n📹 Video Characteristics:")
            print(f"   Motion pattern: {vc.motion_pattern}")
            print(f"   Lighting condition: {vc.lighting_condition}")
            print(f"   Contrast level: {vc.contrast_level:.2f}")
            print(f"   Scene stability: {vc.scene_stability:.2f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Test all three scenarios."""
    print("\n" + "="*80)
    print("🚀 OPINFER VIDEO TESTING")
    print("="*80)
    
    # Video paths (adjust as needed)
    videos_dir = Path(__file__).parent / "vdo"
    
    scenarios = {
        "Traffic Camera": {
            "path": videos_dir / "trafficcam30fps.mp4",
            "expected": "High skip rate (60-70%), excellent performance"
        },
        "Night Drive": {
            "path": videos_dir / "mediumpace30fps.mp4",
            "expected": "Moderate skip rate (30-40%), improved from fixed params"
        },
        "Drone Footage": {
            "path": videos_dir / "mediumspeeddrone30fps.mp4",
            "expected": "Lower skip rate (15-25%), but much better than fixed params"
        },
    }
    
    results = {}
    
    for scenario_name, info in scenarios.items():
        video_path = str(info["path"])
        expected = info["expected"]
        
        print(f"\n📋 Expected for {scenario_name}: {expected}")
        
        result = test_scenario(video_path, scenario_name, max_frames=200)
        results[scenario_name] = result
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    for scenario_name, result in results.items():
        if result:
            stats = result.get("stats", {})
            skip_rate = stats.get("skip_rate_pct", 0)
            fps = stats.get("effective_fps", 0)
            print(f"\n{scenario_name}:")
            print(f"   Skip rate: {skip_rate:.1f}%")
            print(f"   Effective FPS: {fps:.2f}")
        else:
            print(f"\n{scenario_name}: ❌ Failed or video not found")
    
    print("\n✅ Testing complete!")
    print("\n💡 To test with more frames or different models, modify this script.")


if __name__ == "__main__":
    main()

