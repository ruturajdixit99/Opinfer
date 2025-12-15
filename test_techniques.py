"""
Test script to compare motion gating vs queuing techniques.
"""

from opinfer import OptimizedInference
from pathlib import Path


def compare_techniques(video_path: str, max_frames: int = 200):
    """Compare motion gating vs queuing techniques."""
    print("\n" + "=" * 100)
    print("🔬 TECHNIQUE COMPARISON: Motion Gating vs Queuing")
    print("=" * 100)
    
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    # Test Motion Gating
    print("\n" + "-" * 100)
    print("📊 TECHNIQUE 1: Motion Gating")
    print("-" * 100)
    
    inf_motion = OptimizedInference(
        model_name="vit_base_patch16_224",
        model_type="classifier",
        device="cuda",
        technique="motion_gating",
        auto_optimize=True,
        target_skip_rate=40.0,
    )
    
    results_motion = inf_motion.process_video_file(video_path, max_frames=max_frames)
    
    stats_motion = results_motion.get("stats", {})
    print(f"\n✅ Motion Gating Results:")
    print(f"   Skip rate: {stats_motion.get('skip_rate_pct', 0):.1f}%")
    print(f"   Effective FPS: {stats_motion.get('effective_fps', 0):.2f}")
    print(f"   Avg inference time: {stats_motion.get('avg_inference_ms', 0):.2f} ms")
    print(f"   Model calls: {stats_motion.get('model_calls', 0)}")
    
    # Test Queuing
    print("\n" + "-" * 100)
    print("📊 TECHNIQUE 2: Queuing")
    print("-" * 100)
    
    inf_queue = OptimizedInference(
        model_name="vit_base_patch16_224",
        model_type="classifier",
        device="cuda",
        technique="queuing",
        queue_size=4,
        batch_size=4,
        max_queue_wait_ms=33.0,
    )
    
    results_queue = inf_queue.process_video_file(video_path, max_frames=max_frames)
    
    stats_queue = results_queue.get("stats", {})
    print(f"\n✅ Queuing Results:")
    print(f"   Effective FPS: {stats_queue.get('effective_fps', 0):.2f}")
    print(f"   Avg batch latency: {stats_queue.get('avg_batch_latency_ms', 0):.2f} ms")
    print(f"   Batches processed: {stats_queue.get('batches_processed', 0)}")
    print(f"   Frames per batch: {stats_queue.get('frames_per_batch', 0):.2f}")
    
    # Comparison
    print("\n" + "=" * 100)
    print("📈 COMPARISON")
    print("=" * 100)
    
    fps_motion = stats_motion.get('effective_fps', 0)
    fps_queue = stats_queue.get('effective_fps', 0)
    
    print(f"\nFPS Comparison:")
    print(f"   Motion Gating: {fps_motion:.2f} FPS")
    print(f"   Queuing:       {fps_queue:.2f} FPS")
    
    if fps_motion > fps_queue:
        improvement = ((fps_motion - fps_queue) / fps_queue) * 100
        print(f"   → Motion Gating is {improvement:.1f}% faster")
    elif fps_queue > fps_motion:
        improvement = ((fps_queue - fps_motion) / fps_motion) * 100
        print(f"   → Queuing is {improvement:.1f}% faster")
    else:
        print(f"   → Both techniques perform similarly")
    
    print(f"\nLatency Comparison:")
    print(f"   Motion Gating: {stats_motion.get('avg_inference_ms', 0):.2f} ms per call")
    print(f"   Queuing:       {stats_queue.get('avg_batch_latency_ms', 0):.2f} ms per batch")
    
    print(f"\n💡 Recommendations:")
    if fps_motion > fps_queue * 1.1:
        print("   → Use Motion Gating for this scenario (better FPS)")
    elif fps_queue > fps_motion * 1.1:
        print("   → Use Queuing for this scenario (better FPS)")
    else:
        print("   → Both techniques work well. Choose based on:")
        print("     - Motion Gating: Better for static/low-motion scenes")
        print("     - Queuing: Better for consistent frame rates")


def main():
    """Test both techniques on all videos."""
    videos_dir = Path(__file__).parent / "vdo"
    
    scenarios = {
        "Traffic Camera": videos_dir / "trafficcam30fps.mp4",
        "Night Drive": videos_dir / "mediumpace30fps.mp4",
        "Drone Footage": videos_dir / "mediumspeeddrone30fps.mp4",
    }
    
    for scenario_name, video_path in scenarios.items():
        if not video_path.exists():
            print(f"\n⚠️  Skipping {scenario_name}: video not found")
            continue
        
        print("\n" + "=" * 100)
        print(f"🎬 SCENARIO: {scenario_name.upper()}")
        print("=" * 100)
        
        try:
            compare_techniques(str(video_path), max_frames=200)
        except Exception as e:
            print(f"\n❌ Error testing {scenario_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ Technique comparison complete!")


if __name__ == "__main__":
    main()

