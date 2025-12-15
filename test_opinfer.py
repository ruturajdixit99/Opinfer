"""
Test script for opinfer package.
Run this to verify installation and functionality.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test if all modules can be imported."""
    print("\n" + "=" * 80)
    print("TEST 1: Package Imports")
    print("=" * 80)
    
    try:
        from opinfer import MotionGatedInference, AdaptiveMotionGater, ModelLoader, ParameterOptimizer
        print("✅ All core modules imported successfully")
        
        from opinfer.api import OptimizedInference
        print("✅ API module imported successfully")
        
        from opinfer.detectors import VideoCharacteristicDetector
        print("✅ Detector module imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """Test model loading."""
    print("\n" + "=" * 80)
    print("TEST 2: Model Loading")
    print("=" * 80)
    
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Using device: {device}")
        
        from opinfer.models import ModelLoader
        
        # Test loading a small classifier
        print("\n   Testing classifier model loading...")
        model = ModelLoader.load_classifier("vit_tiny_patch16_224", device=device)
        print("   ✅ Classifier model loaded successfully")
        
        # Test listing models
        models = ModelLoader.list_models()
        print(f"   ✅ Available classifiers: {len(models['classifiers'])}")
        print(f"   ✅ Available detectors: {len(models['detectors'])}")
        
        return True
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_video_detection():
    """Test video characteristic detection."""
    print("\n" + "=" * 80)
    print("TEST 3: Video Characteristic Detection")
    print("=" * 80)
    
    try:
        import cv2
        import numpy as np
        from opinfer.detectors import VideoCharacteristicDetector
        
        # Create dummy frames for testing
        print("   Creating test frames...")
        frames = []
        for i in range(50):
            # Create a simple test frame
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frames.append(frame)
        
        detector = VideoCharacteristicDetector(sample_frames=50)
        chars = detector.analyze_video(frames)
        
        print(f"   ✅ Motion pattern: {chars.motion_pattern}")
        print(f"   ✅ Lighting condition: {chars.lighting_condition}")
        print(f"   ✅ Recommended threshold range: {chars.recommended_threshold_range}")
        print(f"   ✅ Recommended min_frames: {chars.recommended_min_frames}")
        
        return True
    except Exception as e:
        print(f"   ❌ Video detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_core_inference():
    """Test core motion-gated inference."""
    print("\n" + "=" * 80)
    print("TEST 4: Core Motion-Gated Inference")
    print("=" * 80)
    
    try:
        import torch
        import numpy as np
        from opinfer.core import MotionGatedInference
        from opinfer.models import ModelLoader
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Using device: {device}")
        
        # Load a small model
        print("   Loading model...")
        model = ModelLoader.load_classifier("vit_tiny_patch16_224", device=device)
        
        # Create inference engine
        engine = MotionGatedInference(
            model=model,
            device=device,
            motion_threshold=4.0,
            min_frames_between_calls=2,
        )
        
        # Create test frames
        print("   Creating test frames...")
        frames = []
        for i in range(20):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frames.append(frame)
        
        # Process frames
        print("   Processing frames...")
        for i, frame in enumerate(frames):
            output, stats = engine.infer(frame)
            if (i + 1) % 5 == 0:
                print(f"      Frame {i+1}: motion={stats['motion_score']:.2f}, called={stats['did_call']}")
        
        # Get stats
        final_stats = engine.get_stats()
        print(f"\n   ✅ Total frames: {final_stats['total_frames']}")
        print(f"   ✅ Model calls: {final_stats['model_calls']}")
        print(f"   ✅ Skip rate: {final_stats['skip_rate_pct']:.1f}%")
        
        return True
    except Exception as e:
        print(f"   ❌ Core inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adaptive_gating():
    """Test adaptive motion gating."""
    print("\n" + "=" * 80)
    print("TEST 5: Adaptive Motion Gating")
    print("=" * 80)
    
    try:
        import torch
        import numpy as np
        from opinfer.adaptive import AdaptiveMotionGater
        from opinfer.models import ModelLoader
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Using device: {device}")
        
        # Load model
        print("   Loading model...")
        model = ModelLoader.load_classifier("vit_tiny_patch16_224", device=device)
        
        # Create adaptive gater
        gater = AdaptiveMotionGater(
            model=model,
            device=device,
            auto_optimize=True,
            optimization_sample_frames=30,  # Small for testing
            target_skip_rate=40.0,
        )
        
        # Create test frames
        print("   Creating test frames...")
        frames = []
        for i in range(50):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frames.append(frame)
        
        # Process with adaptive gating
        print("   Processing with adaptive gating...")
        results = gater.process_video(frames, analyze_first=True)
        
        stats = results.get("stats", {})
        print(f"\n   ✅ Skip rate: {stats.get('skip_rate_pct', 0):.1f}%")
        print(f"   ✅ Effective FPS: {stats.get('effective_fps', 0):.2f}")
        print(f"   ✅ Optimized threshold: {results.get('parameters', {}).get('motion_threshold', 0):.2f}")
        
        return True
    except Exception as e:
        print(f"   ❌ Adaptive gating failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api():
    """Test high-level API."""
    print("\n" + "=" * 80)
    print("TEST 6: High-Level API")
    print("=" * 80)
    
    try:
        import numpy as np
        from opinfer.api import OptimizedInference
        
        print("   Creating OptimizedInference instance...")
        inf = OptimizedInference(
            model_name="vit_tiny_patch16_224",
            model_type="classifier",
            device="cuda",
            auto_optimize=True,
            target_skip_rate=40.0,
        )
        print("   ✅ OptimizedInference created")
        
        # Create test frames
        print("   Creating test frames...")
        frames = []
        for i in range(50):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frames.append(frame)
        
        # Process frames
        print("   Processing frames...")
        results = inf.process_video_frames(frames, analyze_first=True)
        
        stats = results.get("stats", {})
        print(f"\n   ✅ Skip rate: {stats.get('skip_rate_pct', 0):.1f}%")
        print(f"   ✅ Effective FPS: {stats.get('effective_fps', 0):.2f}")
        
        return True
    except Exception as e:
        print(f"   ❌ API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_real_video():
    """Test with real video file if available."""
    print("\n" + "=" * 80)
    print("TEST 7: Real Video Processing (if available)")
    print("=" * 80)
    
    # Check for video files
    videos_dir = Path(__file__).parent / "vdo"
    
    video_files = {
        "traffic_cam": videos_dir / "trafficcam30fps.mp4",
        "night_drive": videos_dir / "mediumpace30fps.mp4",
        "drone": videos_dir / "mediumspeeddrone30fps.mp4",
    }
    
    found_videos = []
    for name, path in video_files.items():
        if path.exists():
            found_videos.append((name, path))
            print(f"   ✅ Found: {name} at {path}")
        else:
            print(f"   ⚠️  Not found: {name} at {path}")
    
    if not found_videos:
        print("   ⚠️  No video files found. Skipping real video test.")
        print("   💡 To test with real videos, place them in ../vdo/")
        return True
    
    # Test with first available video
    try:
        from opinfer.api import OptimizedInference
        
        name, path = found_videos[0]
        print(f"\n   Testing with {name}...")
        
        inf = OptimizedInference(
            model_name="vit_tiny_patch16_224",
            model_type="classifier",
            device="cuda",
            auto_optimize=True,
        )
        
        results = inf.process_video_file(str(path), max_frames=100)
        
        stats = results.get("stats", {})
        print(f"\n   ✅ Processed {stats.get('total_frames', 0)} frames")
        print(f"   ✅ Skip rate: {stats.get('skip_rate_pct', 0):.1f}%")
        print(f"   ✅ Effective FPS: {stats.get('effective_fps', 0):.2f}")
        
        if results.get('video_characteristics'):
            vc = results['video_characteristics']
            print(f"   ✅ Motion pattern: {vc.motion_pattern}")
            print(f"   ✅ Lighting: {vc.lighting_condition}")
        
        return True
    except Exception as e:
        print(f"   ❌ Real video test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 100)
    print("🧪 OPINFER PACKAGE TEST SUITE")
    print("=" * 100)
    
    tests = [
        ("Package Imports", test_imports),
        ("Model Loading", test_model_loading),
        ("Video Detection", test_video_detection),
        ("Core Inference", test_core_inference),
        ("Adaptive Gating", test_adaptive_gating),
        ("High-Level API", test_api),
        ("Real Video", test_with_real_video),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 100)
    print("📊 TEST SUMMARY")
    print("=" * 100)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    print(f"\n   Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Package is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

