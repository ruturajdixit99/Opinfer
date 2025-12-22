"""
Main API for opinfer package.
High-level interface for easy usage.
"""

import cv2
import numpy as np
import time
from typing import List, Optional, Union, Dict, Any
from pathlib import Path

from opinfer.adaptive import AdaptiveMotionGater
from opinfer.models import ModelLoader
from opinfer.detectors import VideoCharacteristicDetector
from opinfer.queue import QueuedInference


def load_video_frames(video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """
    Load frames from a video file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load (None = all)
        
    Returns:
        List of BGR frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    
    cap.release()
    return frames


class OptimizedInference:
    """
    High-level API for optimized inference with adaptive motion gating.
    """
    
    def __init__(
        self,
        model_name: str,
        model_type: str = "classifier",  # "classifier" or "detector"
        device: str = "cuda",
        technique: str = "motion_gating",  # "motion_gating" or "queuing"
        auto_optimize: bool = False,  # Default: Fast start, no optimization
        target_skip_rate: float = 40.0,
        # Queuing parameters
        queue_size: int = 4,
        batch_size: int = 4,
        max_queue_wait_ms: float = 33.0,
    ):
        """
        Initialize optimized inference system.
        
        Args:
            model_name: Name of the model to load
            model_type: "classifier" or "detector"
            device: Device for inference
            technique: "motion_gating" or "queuing" - which optimization technique to use
            auto_optimize: Whether to automatically optimize parameters (SLOW, 30-60s). 
                          Set False for instant start with good defaults (recommended for real-time).
            target_skip_rate: Target percentage of frames to skip (motion gating only)
            queue_size: Maximum queue size (queuing only)
            batch_size: Batch size for processing (queuing only)
            max_queue_wait_ms: Max wait time before processing queue (queuing only)
        """
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        self.technique = technique
        
        if technique not in ["motion_gating", "queuing"]:
            raise ValueError(f"Unknown technique: {technique}. Must be 'motion_gating' or 'queuing'")
        
        # Load model
        if model_type == "classifier":
            self.model = ModelLoader.load_classifier(model_name, device=device)
            self.processor = None
        elif model_type == "detector":
            self.model, self.processor = ModelLoader.load_detector(model_name, device=device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Create inference engine based on technique
        if technique == "motion_gating":
            self.gater = AdaptiveMotionGater(
                model=self.model,
                device=device,
                auto_optimize=auto_optimize,
                target_skip_rate=target_skip_rate,
                processor=self.processor,
                text_queries=["person", "car", "road", "building", "tree", "traffic light", "sky"] if model_type == "detector" else None,
            )
            self.queuer = None
        else:  # queuing
            if model_type == "detector":
                raise ValueError("Queuing technique is currently only supported for classifier models")
            self.queuer = QueuedInference(
                model=self.model,
                device=device,
                queue_size=queue_size,
                batch_size=batch_size,
                max_queue_wait_ms=max_queue_wait_ms,
            )
            self.gater = None
        
        self.initialized = False
        # For streaming mode: allow processing without pre-initialization
        self.streaming_mode = False
    
    def process_video_file(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        analyze_sample: int = 200,
    ) -> Dict[str, Any]:
        """
        Process a video file with selected technique.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process (None = all)
            analyze_sample: Number of frames to use for initial analysis (motion gating only)
            
        Returns:
            Dictionary with results and statistics
        """
        print(f"\n🎥 Processing video: {video_path}")
        print(f"   Technique: {self.technique}")
        
        # Load frames
        frames = load_video_frames(video_path, max_frames=max_frames)
        if not frames:
            raise ValueError(f"No frames loaded from {video_path}")
        
        print(f"   Loaded {len(frames)} frames")
        
        if self.technique == "motion_gating":
            # Initialize if needed
            if not self.initialized:
                if self.gater.auto_optimize:
                    # User wants optimization (slower but better)
                    sample_frames = frames[:min(analyze_sample, len(frames))]
                    self.gater.analyze_and_optimize(sample_frames)
                else:
                    # Fast start: quick analysis only (no optimization)
                    sample_frames = frames[:min(50, len(frames))]
                    self.gater.auto_optimize = False
                    self.gater.analyze_and_optimize(sample_frames)
                self.initialized = True
            
            # Process full video
            results = self.gater.process_video(frames, analyze_first=False)
        else:  # queuing
            # Process with queuing
            self.queuer.reset()
            print(f"\n🔄 Processing with queuing (queue_size={self.queuer.queue_size}, batch_size={self.queuer.batch_size})...")
            
            outputs = []
            last_output = None
            for i, frame in enumerate(frames):
                output, stats = self.queuer.infer(frame)
                
                # Handle None outputs (frames still in queue)
                if output is not None:
                    last_output = output
                    outputs.append(output)
                elif last_output is not None:
                    # Use last known output as fallback
                    outputs.append(last_output)
                else:
                    # No output yet, will be processed in batch
                    outputs.append(None)
                
                if (i + 1) % 100 == 0 or (i + 1) == len(frames):
                    print(f"   Frame {i+1}/{len(frames)}: queue_size={stats['queue_size']}, processed={stats.get('frames_processed', 0)}")
            
            # Flush remaining frames
            flush_results = self.queuer.flush()
            # Add flushed outputs
            for frame_id, output in flush_results:
                if frame_id < len(outputs):
                    outputs[frame_id] = output
            
            final_stats = self.queuer.get_stats()
            print("\n📈 Final Statistics:")
            print(f"   Total frames: {final_stats['total_frames']}")
            print(f"   Batches processed: {final_stats['batches_processed']}")
            print(f"   Avg batch latency: {final_stats['avg_batch_latency_ms']:.2f} ms")
            print(f"   Effective FPS: {final_stats['effective_fps']:.2f}")
            
            results = {
                "stats": final_stats,
                "technique": "queuing",
                "outputs": outputs,
                "parameters": {
                    "queue_size": self.queuer.queue_size,
                    "batch_size": self.queuer.batch_size,
                    "max_queue_wait_ms": self.queuer.max_queue_wait_ms,
                },
            }
        
        return results
    
    def process_video_frames(
        self,
        frames: List[np.ndarray],
        analyze_first: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a list of frames.
        
        Args:
            frames: List of BGR frames
            analyze_first: Whether to analyze and optimize first (motion gating only)
            
        Returns:
            Dictionary with results and statistics
        """
        if self.technique == "motion_gating":
            if analyze_first or not self.initialized:
                self.gater.analyze_and_optimize(frames)
                self.initialized = True
            
            results = self.gater.process_video(frames, analyze_first=False)
        else:  # queuing
            self.queuer.reset()
            outputs = []
            for i, frame in enumerate(frames):
                output, stats = self.queuer.infer(frame)
                outputs.append(output)
            
            self.queuer.flush()
            final_stats = self.queuer.get_stats()
            
            results = {
                "stats": final_stats,
                "technique": "queuing",
                "parameters": {
                    "queue_size": self.queuer.queue_size,
                    "batch_size": self.queuer.batch_size,
                    "max_queue_wait_ms": self.queuer.max_queue_wait_ms,
                },
            }
        
        return results
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame (for streaming/live feeds).
        
        This is the main method for real-time processing. Frames are processed
        one at a time as they arrive, perfect for webcam, camera, or streaming use cases.
        
        Args:
            frame: BGR frame (numpy array from OpenCV)
            
        Returns:
            (output, stats) tuple where:
                - output: Model prediction (logits, detections, etc.)
                - stats: Dictionary with motion_score, did_call, inference_ms, etc.
        
        Example:
            ```python
            infer = OptimizedInference(model_name="vit_base_patch16_224")
            
            # For streaming: initialize with default params or a few sample frames
            cap = cv2.VideoCapture(0)  # Webcam
            sample_frames = []
            for _ in range(30):  # Collect 30 frames for quick initialization
                ret, frame = cap.read()
                if ret:
                    sample_frames.append(frame)
                    if len(sample_frames) >= 30:
                        break
            
            if sample_frames:
                infer.initialize_streaming(sample_frames)
            
            # Now process live frames one at a time
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                output, stats = infer.process_frame(frame)
                # Use output and stats...
            ```
        """
        if self.technique == "motion_gating":
            # If not initialized and in streaming mode, use default parameters
            if not self.initialized:
                if self.streaming_mode:
                    # Initialize engine with default parameters for streaming
                    from opinfer.core import MotionGatedInference
                    self.gater.engine = MotionGatedInference(
                        model=self.model,
                        device=self.device,
                        motion_threshold=self.gater.motion_threshold,
                        min_frames_between_calls=self.gater.min_frames_between_calls,
                    )
                    self.gater.engine.processor = self.processor
                    self.gater.engine.text_queries = self.gater.text_queries
                    self.initialized = True
                else:
                    raise RuntimeError(
                        "Must initialize first. Use one of:\n"
                        "  1. initialize_streaming(sample_frames) - for live feeds\n"
                        "  2. process_video_file() - for video files\n"
                        "  3. analyze_and_optimize(frames) - manual initialization"
                    )
            return self.gater.process_frame(frame)
        else:  # queuing
            return self.queuer.infer(frame)
    
    def initialize_streaming(
        self,
        sample_frames: Optional[List[np.ndarray]] = None,
        num_sample_frames: int = 30,
    ) -> None:
        """
        Initialize for streaming mode (real-time/live feed processing).
        
        This allows you to start processing frames IMMEDIATELY without optimization.
        If sample_frames are provided and auto_optimize=True, it will do quick analysis.
        Otherwise, it uses default parameters (INSTANT start - recommended for real-time).
        
        Args:
            sample_frames: Optional list of sample frames for quick analysis.
                          If None, uses default parameters (instant start).
            num_sample_frames: Number of frames to use (not used if sample_frames provided)
        
        Example:
            ```python
            # Method 1: Instant start with defaults (RECOMMENDED for real-time)
            infer = OptimizedInference(model_name="owlvit-base", model_type="detector")
            infer.initialize_streaming()  # Instant - uses defaults!
            
            # Method 2: Quick analysis with sample frames (better defaults, still fast)
            cap = cv2.VideoCapture(0)
            sample_frames = [cap.read()[1] for _ in range(30)]
            infer.initialize_streaming(sample_frames)  # Fast analysis, no optimization
            ```
        """
        self.streaming_mode = True
        
        if self.technique == "motion_gating":
            if sample_frames and len(sample_frames) > 0 and self.gater.auto_optimize:
                # Quick analysis only (no optimization) - fast but better than defaults
                print(f"🔧 Initializing streaming mode (quick analysis, no optimization)...")
                self.gater.auto_optimize = False  # Force no optimization for speed
                self.gater.analyze_and_optimize(sample_frames)
                print(f"✅ Streaming mode ready! Motion threshold: {self.gater.motion_threshold:.2f}")
            else:
                # Use default parameters (INSTANT initialization, no analysis needed)
                print("🔧 Initializing streaming mode with defaults (instant start)...")
                from opinfer.core import MotionGatedInference
                default_threshold = 4.0  # Good default for most scenarios
                default_min_frames = 2
                
                self.gater.engine = MotionGatedInference(
                    model=self.model,
                    device=self.device,
                    motion_threshold=default_threshold,
                    min_frames_between_calls=default_min_frames,
                )
                self.gater.engine.processor = self.processor
                self.gater.engine.text_queries = self.gater.text_queries
                self.gater.motion_threshold = default_threshold
                self.gater.min_frames_between_calls = default_min_frames
                print(f"✅ Ready! Using defaults (threshold={default_threshold}, min_frames={default_min_frames})")
            
            self.initialized = True
        else:  # queuing doesn't need initialization
            self.initialized = True
    
    def process_live_feed(
        self,
        source: Union[int, str],
        max_frames: Optional[int] = None,
        show_video: bool = True,
        window_name: str = "Opinfer Live Feed",
    ) -> Dict[str, Any]:
        """
        Process a live video feed (webcam, camera, or video stream).
        
        This is the easiest way to process live feeds. It handles everything:
        - Opens the video source
        - Initializes with a few sample frames
        - Processes frames in real-time
        - Returns statistics
        
        Args:
            source: Video source:
                   - int: Webcam index (e.g., 0 for default camera)
                   - str: Video file path or RTSP/HTTP stream URL
            max_frames: Maximum frames to process (None = unlimited for live feeds)
            show_video: Whether to display video in OpenCV window
            window_name: Window name for display
        
        Returns:
            Dictionary with statistics
        
        Example:
            ```python
            infer = OptimizedInference(model_name="vit_base_patch16_224")
            
            # Process webcam feed
            stats = infer.process_live_feed(source=0, show_video=True)
            
            # Process video stream
            stats = infer.process_live_feed(source="rtsp://192.168.1.100:554/stream")
            ```
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        
        print(f"\n🎥 Processing live feed from: {source}")
        print(f"   Technique: {self.technique}")
        
        # Initialize for streaming (fast - uses defaults)
        if self.technique == "motion_gating" and not self.initialized:
            # Instant initialization with defaults (no waiting!)
            self.initialize_streaming()  # Uses defaults immediately
        
        # Process frames in real-time
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                output, stats = self.process_frame(frame)
                frame_count += 1
                
                # Display if requested
                if show_video:
                    display_frame = frame.copy()
                    
                    # Add overlay
                    status = "PROCESSED" if stats['did_call'] else "SKIPPED"
                    color = (0, 255, 0) if stats['did_call'] else (0, 165, 255)
                    
                    cv2.putText(display_frame, f"Frame: {frame_count}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Status: {status}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(display_frame, f"Motion: {stats['motion_score']:.2f}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    if self.initialized and self.technique == "motion_gating":
                        current_stats = self.gater.get_current_stats()
                        if current_stats:
                            cv2.putText(display_frame, 
                                       f"Calls: {current_stats.get('model_calls', 0)} / {current_stats.get('total_frames', 0)}", 
                                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            skip_rate = current_stats.get('skip_rate_pct', 0)
                            cv2.putText(display_frame, f"Skip Rate: {skip_rate:.1f}%", 
                                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    
                    cv2.imshow(window_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        print("\n   ⏹ Stopped by user")
                        break
                
                # Check max_frames
                if max_frames and frame_count >= max_frames:
                    break
        
        finally:
            cap.release()
            if show_video:
                cv2.destroyAllWindows()
        
        # Get final statistics
        total_time = time.time() - start_time
        fps = frame_count / total_time if total_time > 0 else 0
        
        if self.technique == "motion_gating" and self.initialized:
            final_stats = self.gater.get_current_stats()
            final_stats['total_time_seconds'] = total_time
            final_stats['fps'] = fps
        else:
            final_stats = {
                'total_frames': frame_count,
                'total_time_seconds': total_time,
                'fps': fps,
            }
        
        print(f"\n✅ Processing complete!")
        print(f"   Total frames: {frame_count}")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Average FPS: {fps:.2f}")
        
        return {"stats": final_stats}
    
    def benchmark_all_models(
        self,
        video_path: str,
        max_frames: Optional[int] = 500,
    ) -> Dict[str, Any]:
        """
        Benchmark all available models on a video.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process per model
            
        Returns:
            Dictionary with benchmark results for all models
        """
        frames = load_video_frames(video_path, max_frames=max_frames)
        if not frames:
            raise ValueError(f"No frames loaded from {video_path}")
        
        all_results = {}
        
        # Test all classifier models
        print("\n" + "=" * 80)
        print("📊 BENCHMARKING ALL CLASSIFIER MODELS")
        print("=" * 80)
        
        for model_name in ModelLoader.CLASSIFIER_MODELS:
            print(f"\n🔬 Testing {model_name}...")
            try:
                inf = OptimizedInference(
                    model_name=model_name,
                    model_type="classifier",
                    device=self.device,
                    auto_optimize=True,
                )
                results = inf.process_video_frames(frames, analyze_first=True)
                all_results[model_name] = {
                    "type": "classifier",
                    "results": results,
                }
            except Exception as e:
                print(f"   ❌ Error with {model_name}: {e}")
                all_results[model_name] = {"type": "classifier", "error": str(e)}
        
        # Test all detector models
        print("\n" + "=" * 80)
        print("📊 BENCHMARKING ALL DETECTOR MODELS")
        print("=" * 80)
        
        for model_name, _ in ModelLoader.DETECTOR_MODELS:
            print(f"\n🔬 Testing {model_name}...")
            try:
                inf = OptimizedInference(
                    model_name=model_name,
                    model_type="detector",
                    device=self.device,
                    auto_optimize=True,
                )
                results = inf.process_video_frames(frames, analyze_first=True)
                all_results[model_name] = {
                    "type": "detector",
                    "results": results,
                }
            except Exception as e:
                print(f"   ❌ Error with {model_name}: {e}")
                all_results[model_name] = {"type": "detector", "error": str(e)}
        
        return all_results


def quick_inference(
    video_path: str,
    model_name: str = "vit_base_patch16_224",
    model_type: str = "classifier",
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Quick inference function for simple use cases.
    
    Args:
        video_path: Path to video file
        model_name: Model to use
        model_type: "classifier" or "detector"
        max_frames: Maximum frames to process
        
    Returns:
        Results dictionary
    """
    inf = OptimizedInference(
        model_name=model_name,
        model_type=model_type,
        auto_optimize=True,
    )
    return inf.process_video_file(video_path, max_frames=max_frames)

