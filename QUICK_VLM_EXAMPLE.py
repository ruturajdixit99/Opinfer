"""
Quick example: Using opinfer with Vision-Language Models
"""

from opinfer import OptimizedInference, load_video_frames

# ============================================
# Example 1: OWL-ViT (Built-in Support)
# ============================================

def example_owlvit():
    """Example using OWL-ViT detector (VLM for object detection)."""
    print("\n" + "="*80)
    print("Example 1: OWL-ViT (Vision-Language Model)")
    print("="*80)
    
    # Initialize with OWL-ViT
    inf = OptimizedInference(
        model_name="owlvit-base",
        model_type="detector",
        technique="motion_gating",
        auto_optimize=True,
    )
    
    # Process video
    video_path = "vdo/trafficcam30fps.mp4"
    results = inf.process_video_file(video_path, max_frames=100)
    
    print(f"\nResults:")
    print(f"  Skip rate: {results['stats']['skip_rate_pct']:.1f}%")
    print(f"  Effective FPS: {results['stats']['effective_fps']:.2f}")
    print(f"  Model calls: {results['stats']['model_calls']}")
    
    # OWL-ViT outputs contain detections with boxes, scores, labels
    # These are automatically handled by the package


# ============================================
# Example 2: Custom VLM Wrapper
# ============================================

def example_custom_vlm():
    """Example showing how to wrap a custom VLM."""
    print("\n" + "="*80)
    print("Example 2: Custom VLM Integration")
    print("="*80)
    
    import torch
    import torch.nn as nn
    from opinfer import AdaptiveMotionGater
    
    # Your custom VLM (example structure)
    class CustomVLM(nn.Module):
        def __init__(self):
            super().__init__()
            # Your VLM architecture here
            self.vision_encoder = nn.Sequential()  # Placeholder
            self.text_encoder = nn.Sequential()    # Placeholder
        
        def forward(self, image, text_features=None):
            # Your VLM forward pass
            vision_features = self.vision_encoder(image)
            # Combine with text if needed
            return vision_features
    
    # Load your VLM
    vlm_model = CustomVLM().to("cuda").eval()
    
    # Use with opinfer motion gating
    gater = AdaptiveMotionGater(
        model=vlm_model,
        device="cuda",
        auto_optimize=True,
    )
    
    # Process frames
    frames = load_video_frames("vdo/trafficcam30fps.mp4", max_frames=100)
    results = gater.process_video(frames, analyze_first=True)
    
    print(f"\nResults:")
    print(f"  Effective FPS: {results['stats']['effective_fps']:.2f}")
    print(f"  Skip rate: {results['stats']['skip_rate_pct']:.1f}%")


# ============================================
# Example 3: Real-Time VLM Processing
# ============================================

def example_realtime_vlm():
    """Example for real-time VLM processing."""
    print("\n" + "="*80)
    print("Example 3: Real-Time VLM Processing")
    print("="*80)
    
    import cv2
    
    # Initialize
    inf = OptimizedInference(
        model_name="owlvit-base",
        model_type="detector",
        technique="motion_gating",
        auto_optimize=True,
    )
    
    # For real-time, you'd initialize with sample frames first
    # Then process frame by frame
    print("\nFor real-time processing:")
    print("  1. Initialize with sample frames")
    print("  2. Use process_frame() for each new frame")
    print("  3. Get VLM outputs for each frame")
    
    # Example frame processing
    cap = cv2.VideoCapture(0)  # Webcam
    if cap.isOpened():
        # Get sample frames for initialization
        sample_frames = []
        for _ in range(30):
            ret, frame = cap.read()
            if ret:
                sample_frames.append(frame)
        
        if sample_frames:
            # Initialize
            inf.process_video_frames(sample_frames, analyze_first=True)
            print("  ✅ Initialized with sample frames")
            
            # Process a few frames
            for i in range(10):
                ret, frame = cap.read()
                if ret:
                    output, stats = inf.process_frame(frame)
                    print(f"  Frame {i+1}: motion={stats['motion_score']:.2f}, called={stats['did_call']}")
        
        cap.release()
    else:
        print("  ⚠️  Webcam not available, skipping real-time example")


# ============================================
# Main
# ============================================

if __name__ == "__main__":
    print("\n🚀 Opinfer VLM Usage Examples\n")
    
    # Run examples
    try:
        example_owlvit()
    except Exception as e:
        print(f"  ⚠️  OWL-ViT example failed: {e}")
    
    try:
        example_custom_vlm()
    except Exception as e:
        print(f"  ⚠️  Custom VLM example failed: {e}")
    
    try:
        example_realtime_vlm()
    except Exception as e:
        print(f"  ⚠️  Real-time example failed: {e}")
    
    print("\n✅ Examples complete!")
    print("\n💡 See VLM_USAGE_GUIDE.md for detailed documentation")

