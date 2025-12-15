"""
Main API for opinfer package.
High-level interface for easy usage.
"""

import cv2
import numpy as np
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
        auto_optimize: bool = True,
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
            auto_optimize: Whether to automatically optimize parameters (motion gating only)
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
            # Analyze and optimize on sample
            if not self.initialized:
                sample_frames = frames[:min(analyze_sample, len(frames))]
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
        Process a single frame (for streaming).
        
        Args:
            frame: BGR frame
            
        Returns:
            (output, stats) tuple
        """
        if self.technique == "motion_gating":
            if not self.initialized:
                raise RuntimeError("Must initialize with analyze_and_optimize first")
            return self.gater.process_frame(frame)
        else:  # queuing
            return self.queuer.infer(frame)
    
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

