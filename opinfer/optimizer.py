"""
Parameter optimizer for adaptive motion gating.
Finds optimal motion threshold and min_frames parameters through iterative optimization.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import time

from opinfer.detectors import VideoCharacteristics
from opinfer.core import MotionGatedInference


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    best_threshold: float
    best_min_frames: int
    best_score: float
    optimization_history: List[Dict[str, float]]
    final_stats: Dict[str, float]


class ParameterOptimizer:
    """Optimizes motion gating parameters for a given video and model."""
    
    def __init__(
        self,
        model: any,
        frames: List[np.ndarray],
        device: str = "cuda",
        target_skip_rate: float = 40.0,  # Target % of frames to skip
        min_skip_rate: float = 20.0,  # Minimum acceptable skip rate
        max_iterations: int = 50,
        patience: int = 10,  # Stop if no improvement for N iterations
        processor=None,
        text_queries=None,
    ):
        """
        Initialize optimizer.
        
        Args:
            model: PyTorch model
            frames: List of BGR frames to optimize on
            device: Device for inference
            target_skip_rate: Target percentage of frames to skip
            min_skip_rate: Minimum acceptable skip rate
            max_iterations: Maximum optimization iterations
            patience: Early stopping patience
            processor: Optional processor for detector models
            text_queries: Optional text queries for detector models
        """
        self.model = model
        self.frames = frames
        self.device = device
        self.target_skip_rate = target_skip_rate
        self.min_skip_rate = min_skip_rate
        self.max_iterations = max_iterations
        self.patience = patience
        self.processor = processor
        self.text_queries = text_queries
    
    def evaluate_parameters(
        self,
        motion_threshold: float,
        min_frames_between_calls: int,
        processor=None,
        text_queries=None,
    ) -> Tuple[Dict[str, float], float]:
        """
        Evaluate a set of parameters.
        
        Returns:
            (stats_dict, score) where score is higher for better configurations
        """
        engine = MotionGatedInference(
            model=self.model,
            device=self.device,
            motion_threshold=motion_threshold,
            min_frames_between_calls=min_frames_between_calls,
        )
        
        # Run inference on all frames
        for frame in self.frames:
            engine.infer(frame, processor=processor, text_queries=text_queries)
        
        stats = engine.get_stats()
        
        # Compute score: balance skip rate, FPS, and stability
        skip_rate = stats.get("skip_rate_pct", 0.0)
        effective_fps = stats.get("effective_fps", 0.0)
        
        # Score components
        # 1. Skip rate should be close to target (but not too low)
        skip_score = 0.0
        if skip_rate >= self.min_skip_rate:
            # Reward being close to target
            skip_diff = abs(skip_rate - self.target_skip_rate)
            skip_score = max(0.0, 100.0 - skip_diff * 2.0)  # Max 100 points
        else:
            # Penalize if below minimum
            skip_score = max(0.0, skip_rate / self.min_skip_rate * 50.0)
        
        # 2. FPS should be high (normalized)
        fps_score = min(100.0, effective_fps * 2.0)  # 50 FPS = 100 points
        
        # 3. Model calls should be reasonable (not too many, not too few)
        model_calls = stats.get("model_calls", 0)
        total_frames = stats.get("total_frames", 1)
        call_ratio = model_calls / total_frames
        # Ideal ratio is around 0.4-0.6 (40-60% of frames)
        if 0.3 <= call_ratio <= 0.7:
            call_score = 100.0
        elif call_ratio < 0.3:
            call_score = call_ratio / 0.3 * 100.0
        else:
            call_score = max(0.0, (1.0 - call_ratio) / 0.3 * 100.0)
        
        # Combined score (weighted)
        total_score = (skip_score * 0.4 + fps_score * 0.3 + call_score * 0.3)
        
        stats["optimization_score"] = total_score
        return stats, total_score
    
    def optimize(
        self,
        initial_threshold_range: Tuple[float, float],
        initial_min_frames: int,
        video_chars: Optional[VideoCharacteristics] = None,
    ) -> OptimizationResult:
        """
        Optimize parameters using iterative search.
        
        Args:
            initial_threshold_range: (min, max) threshold to search
            initial_min_frames: Starting min_frames value
            video_chars: Video characteristics (optional, for informed search)
            
        Returns:
            OptimizationResult with best parameters
        """
        threshold_min, threshold_max = initial_threshold_range
        
        # Use video characteristics to narrow search if available
        if video_chars:
            threshold_min = max(threshold_min, video_chars.recommended_threshold_range[0])
            threshold_max = min(threshold_max, video_chars.recommended_threshold_range[1])
            initial_min_frames = video_chars.recommended_min_frames
        
        best_score = -1.0
        best_threshold = (threshold_min + threshold_max) / 2.0
        best_min_frames = initial_min_frames
        best_stats = {}
        
        history = []
        no_improvement_count = 0
        
        # Grid search with refinement
        threshold_candidates = np.linspace(threshold_min, threshold_max, 10)
        min_frames_candidates = [1, 2, 3, 4, 5]
        
        print(f"🔍 Starting optimization (max {self.max_iterations} iterations)...")
        print(f"   Threshold range: [{threshold_min:.2f}, {threshold_max:.2f}]")
        print(f"   Min frames candidates: {min_frames_candidates}")
        
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Try different combinations
            improved = False
            
            for threshold in threshold_candidates:
                for min_frames in min_frames_candidates:
                    if iteration > 1 and no_improvement_count > self.patience // 2:
                        # Focus search around best so far
                        if abs(threshold - best_threshold) > (threshold_max - threshold_min) * 0.3:
                            continue
                    
                    try:
                        stats, score = self.evaluate_parameters(
                            threshold, min_frames,
                            processor=self.processor,
                            text_queries=self.text_queries,
                        )
                        
                        history.append({
                            "iteration": iteration,
                            "threshold": threshold,
                            "min_frames": min_frames,
                            "score": score,
                            "skip_rate": stats.get("skip_rate_pct", 0.0),
                            "fps": stats.get("effective_fps", 0.0),
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_threshold = threshold
                            best_min_frames = min_frames
                            best_stats = stats
                            improved = True
                            no_improvement_count = 0
                            
                            print(
                                f"   ✓ Iter {iteration}: threshold={threshold:.2f}, "
                                f"min_frames={min_frames}, score={score:.2f}, "
                                f"skip={stats.get('skip_rate_pct', 0):.1f}%, "
                                f"fps={stats.get('effective_fps', 0):.2f}"
                            )
                    except Exception as e:
                        print(f"   ⚠ Error evaluating params: {e}")
                        continue
            
            if not improved:
                no_improvement_count += 1
                if no_improvement_count >= self.patience:
                    print(f"   ⏹ Early stopping (no improvement for {self.patience} iterations)")
                    break
            
            # Refine search around best
            if iteration < self.max_iterations and best_score > 0:
                threshold_range = threshold_max - threshold_min
                threshold_min = max(threshold_min, best_threshold - threshold_range * 0.2)
                threshold_max = min(threshold_max, best_threshold + threshold_range * 0.2)
                threshold_candidates = np.linspace(threshold_min, threshold_max, 8)
        
        print(f"\n✅ Optimization complete!")
        print(f"   Best threshold: {best_threshold:.2f}")
        print(f"   Best min_frames: {best_min_frames}")
        print(f"   Best score: {best_score:.2f}")
        print(f"   Final skip rate: {best_stats.get('skip_rate_pct', 0):.1f}%")
        print(f"   Final FPS: {best_stats.get('effective_fps', 0):.2f}")
        
        return OptimizationResult(
            best_threshold=best_threshold,
            best_min_frames=best_min_frames,
            best_score=best_score,
            optimization_history=history,
            final_stats=best_stats,
        )

