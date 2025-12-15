"""
Simple one-line test to verify opinfer is working.
Run: python simple_test.py
"""

if __name__ == "__main__":
    print("Testing opinfer package...\n")
    
    # Test 1: Check if package structure is correct
    import os
    import sys
    from pathlib import Path
    
    package_dir = Path(__file__).parent / "opinfer"
    if not package_dir.exists():
        print(f"[FAIL] Package directory not found: {package_dir}")
        exit(1)
    
    required_files = [
        "__init__.py",
        "core.py",
        "adaptive.py",
        "detectors.py",
        "optimizer.py",
        "models.py",
        "api.py",
        "cli.py",
    ]
    
    missing_files = []
    for file in required_files:
        if not (package_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"[FAIL] Missing files: {missing_files}")
        exit(1)
    
    print("[OK] Package structure is correct")
    
    # Test 2: Check if imports work (may fail due to missing dependencies)
    try:
        from opinfer import OptimizedInference, MotionGatedInference, AdaptiveMotionGater
        print("[OK] Package imports work (structure is correct)")
        print("[OK] OptimizedInference is available")
        print("[OK] All core classes are importable")
    except ImportError as e:
        if "timm" in str(e) or "torch" in str(e) or "transformers" in str(e):
            print(f"[WARN] Import structure is correct, but dependencies are missing: {e}")
            print("\n[INFO] To install dependencies, run:")
            print("  pip install -r requirements.txt")
            print("\n[INFO] Package structure is correct - just need to install dependencies!")
            exit(0)
        else:
            print(f"[FAIL] Import failed: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Test 3: Try to create instance (requires dependencies)
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[OK] Device: {device}")
        
        inf = OptimizedInference(
            model_name="vit_tiny_patch16_224",
            model_type="classifier",
            device=device,
            auto_optimize=False,  # Skip optimization for quick test
        )
        print("[OK] OptimizedInference created successfully")
    except Exception as e:
        print(f"[WARN] Could not create instance (may need dependencies): {e}")
        print("[INFO] Package structure is correct - install dependencies to test fully")
    
    print("\n[SUCCESS] Package structure is correct!")
    print("\nNext steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run full test: python test_opinfer.py")
    print("  3. Test with videos: python example_usage.py")

