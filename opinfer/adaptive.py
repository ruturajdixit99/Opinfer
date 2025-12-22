"""
Adaptive motion gating system that automatically adjusts parameters
based on video characteristics and optimization results.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import cv2

from opinfer.core import MotionGatedInference
from opinfer.detectors import VideoCharacteristicDetector, VideoCharacteristics
from opinfer.optimizer import ParameterOptimizer, OptimizationResult


class AdaptiveMotionGater:
    """
    Adaptive motion gating system that:
    1. Analyzes video characteristics
    2. Optimizes parameters automatically
    3. Adapts to different scenarios (traffic cam, night drive, drone, etc.)
    """
    
    def __init__(
        self,
        model: any,
        device: str = "cuda",
        auto_optimize: bool = False,  # Default: Fast start, optimization is optional
        optimization_sample_frames: int = 200,
        target_skip_rate: float = 40.0,
        processor=None,
        text_queries=None,
    ):
        """
        Initialize adaptive motion gater.
        
        Args:
            model: PyTorch model
            device: Device for inference
            auto_optimize: Whether to automatically optimize parameters
            optimization_sample_frames: Number of frames to use for optimization
            target_skip_rate: Target percentage of frames to skip
            processor: Optional processor for detector models (OWL-ViT)
            text_queries: Optional text queries for detector models
        """
        self.model = model
        self.device = device
        self.auto_optimize = auto_optimize
        self.optimization_sample_frames = optimization_sample_frames
        self.target_skip_rate = target_skip_rate
        self.processor = processor
        self.text_queries = text_queries
        
        self.detector = VideoCharacteristicDetector()
        self.engine: Optional[MotionGatedInference] = None
        self.video_chars: Optional[VideoCharacteristics] = None
        self.optimization_result: Optional[OptimizationResult] = None
        
        # Current parameters
        self.motion_threshold = 4.0
        self.min_frames_between_calls = 2
    
    def analyze_and_optimize(self, frames: List[np.ndarray]) -> Dict[str, any]:
        """
        Analyze video and optimize parameters.
        
        Args:
            frames: List of BGR frames to analyze
            
        Returns:
            Dictionary with analysis and optimization results
        """
        print("\n" + "=" * 80)
        print("ADAPTIVE MOTION GATING: Analysis & Optimization")
        print("=" * 80)
        
        # Step 1: Analyze video characteristics
        print("\nStep 1: Analyzing video characteristics...")
        self.video_chars = self.detector.analyze_video(frames)
        
        print(f"   Motion pattern: {self.video_chars.motion_pattern}")
        print(f"   Avg motion score: {self.video_chars.avg_motion_score:.2f}")
        print(f"   Lighting condition: {self.video_chars.lighting_condition}")
        print(f"   Lighting level: {self.video_chars.lighting_level:.2f}")
        print(f"   Contrast level: {self.video_chars.contrast_level:.2f}")
        print(f"   Scene stability: {self.video_chars.scene_stability:.2f}")
        print(f"   Recommended threshold range: {self.video_chars.recommended_threshold_range}")
        print(f"   Recommended min_frames: {self.video_chars.recommended_min_frames}")
        
        # Step 2: Optimize parameters
        if self.auto_optimize:
            print("\nStep 2: Optimizing parameters...")
            
            # Use sample frames for optimization (faster)
            if len(frames) > self.optimization_sample_frames:
                step = len(frames) // self.optimization_sample_frames
                opt_frames = frames[::step][:self.optimization_sample_frames]
            else:
                opt_frames = frames
            
            # Adjust iterations based on sample size (faster for smaller samples)
            max_iter = 50
            patience = 10
            if len(opt_frames) <= 30:
                max_iter = 10  # Faster optimization for small samples
                patience = 5
            
            optimizer = ParameterOptimizer(
                model=self.model,
                frames=opt_frames,
                device=self.device,
                target_skip_rate=self.target_skip_rate,
                min_skip_rate=20.0,
                max_iterations=max_iter,
                patience=patience,
                processor=self.processor,
                text_queries=self.text_queries,
            )
            
            self.optimization_result = optimizer.optimize(
                initial_threshold_range=self.video_chars.recommended_threshold_range,
                initial_min_frames=self.video_chars.recommended_min_frames,
                video_chars=self.video_chars,
            )
            
            # Update parameters
            self.motion_threshold = self.optimization_result.best_threshold
            self.min_frames_between_calls = self.optimization_result.best_min_frames
            
            print(f"\nOptimized parameters:")
            print(f"   Motion threshold: {self.motion_threshold:.2f}")
            print(f"   Min frames between calls: {self.min_frames_between_calls}")
        else:
            # Use recommended parameters from analysis (fast, no optimization)
            self.motion_threshold = np.mean(self.video_chars.recommended_threshold_range)
            self.min_frames_between_calls = self.video_chars.recommended_min_frames
            print(f"\n✅ Using recommended parameters (fast mode, no optimization):")
            print(f"   Motion threshold: {self.motion_threshold:.2f}")
            print(f"   Min frames between calls: {self.min_frames_between_calls}")
        
        # Step 3: Create inference engine with parameters
        self.engine = MotionGatedInference(
            model=self.model,
            device=self.device,
            motion_threshold=self.motion_threshold,
            min_frames_between_calls=self.min_frames_between_calls,
        )
        
        # Store processor and queries for detector models
        self.engine.processor = self.processor
        self.engine.text_queries = self.text_queries
        
        return {
            "video_characteristics": self.video_chars,
            "optimization_result": self.optimization_result,
            "final_threshold": self.motion_threshold,
            "final_min_frames": self.min_frames_between_calls,
        }
    
    def process_video(
        self,
        frames: List[np.ndarray],
        analyze_first: bool = True,
    ) -> Dict[str, any]:
        """
        Process a video with adaptive motion gating.
        
        Args:
            frames: List of BGR frames
            analyze_first: Whether to analyze and optimize before processing
            
        Returns:
            Dictionary with results and statistics
        """
        if analyze_first or self.engine is None:
            self.analyze_and_optimize(frames)
        
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Call analyze_and_optimize first.")
        
        # Reset engine for new video
        self.engine.reset()
        
        print("\nProcessing video with optimized parameters...")
        
        # Process all frames
        for i, frame in enumerate(frames):
            output, stats = self.engine.infer(
                frame,
                processor=self.processor,
                text_queries=self.text_queries,
            )
            
            if (i + 1) % 100 == 0 or (i + 1) == len(frames):
                print(
                    f"   Frame {i+1}/{len(frames)}: "
                    f"motion={stats['motion_score']:.2f}, "
                    f"calls={self.engine.model_calls}, "
                    f"skipped={self.engine.total_frames - self.engine.model_calls}"
                )
        
        final_stats = self.engine.get_stats()
        
        print("\nFinal Statistics:")
        print(f"   Total frames: {final_stats['total_frames']}")
        print(f"   Model calls: {final_stats['model_calls']}")
        print(f"   Skipped frames: {final_stats['skipped_frames']} ({final_stats['skip_rate_pct']:.1f}%)")
        print(f"   Avg inference time: {final_stats['avg_inference_ms']:.2f} ms")
        print(f"   Effective FPS: {final_stats['effective_fps']:.2f}")
        
        return {
            "stats": final_stats,
            "video_characteristics": self.video_chars,
            "optimization_result": self.optimization_result,
            "parameters": {
                "motion_threshold": self.motion_threshold,
                "min_frames_between_calls": self.min_frames_between_calls,
            },
        }
    
    def process_frame(self, frame: np.ndarray) -> Tuple[any, Dict[str, float]]:
        """
        Process a single frame (for streaming/real-time use).
        
        Args:
            frame: BGR frame
            
        Returns:
            (output, stats) tuple
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Call analyze_and_optimize first with sample frames.")
        
        return self.engine.infer(
            frame,
            processor=self.processor,
            text_queries=self.text_queries,
        )
    
    def get_current_stats(self) -> Dict[str, float]:
        """Get current statistics from the engine."""
        if self.engine is None:
            return {}
        return self.engine.get_stats()

