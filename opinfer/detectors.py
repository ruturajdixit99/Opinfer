"""
Video characteristic detectors for adaptive motion gating.
Detects motion patterns, lighting conditions, and scene dynamics.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class VideoCharacteristics:
    """Detected characteristics of a video feed."""
    avg_motion_score: float
    motion_variance: float
    lighting_level: float  # 0-1, higher = brighter
    contrast_level: float  # 0-1, higher = more contrast
    scene_stability: float  # 0-1, higher = more stable
    motion_pattern: str  # "static", "slow", "moderate", "fast", "very_fast"
    lighting_condition: str  # "bright", "normal", "low", "very_low"
    recommended_threshold_range: Tuple[float, float]
    recommended_min_frames: int


class VideoCharacteristicDetector:
    """Detects video characteristics to inform adaptive motion gating."""
    
    def __init__(self, sample_frames: int = 100):
        """
        Args:
            sample_frames: Number of frames to sample for initial analysis
        """
        self.sample_frames = sample_frames
        self.motion_downsample = (64, 64)
    
    def detect_lighting(self, frame: np.ndarray) -> float:
        """Detect lighting level (0-1)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean() / 255.0
        return float(mean_brightness)
    
    def detect_contrast(self, frame: np.ndarray) -> float:
        """Detect contrast level (0-1)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Use standard deviation as contrast measure
        contrast = gray.std() / 255.0
        return float(contrast)
    
    def compute_motion_scores(self, frames: List[np.ndarray]) -> List[float]:
        """Compute motion scores for a sequence of frames."""
        scores = [0.0]
        prev_gray_small = None
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, self.motion_downsample, interpolation=cv2.INTER_AREA)
            
            if prev_gray_small is not None:
                diff = cv2.absdiff(gray_small, prev_gray_small)
                motion_score = float(diff.mean())
                scores.append(motion_score)
            else:
                scores.append(0.0)
            
            prev_gray_small = gray_small
        
        return scores[1:]  # Remove first 0.0
    
    def analyze_video(self, frames: List[np.ndarray]) -> VideoCharacteristics:
        """
        Analyze video characteristics from a sample of frames.
        
        Args:
            frames: List of BGR frames to analyze
            
        Returns:
            VideoCharacteristics object with detected properties
        """
        if not frames:
            raise ValueError("No frames provided for analysis")
        
        # Sample frames if too many
        if len(frames) > self.sample_frames:
            step = len(frames) // self.sample_frames
            sampled_frames = frames[::step][:self.sample_frames]
        else:
            sampled_frames = frames
        
        # Compute motion scores
        motion_scores = self.compute_motion_scores(sampled_frames)
        avg_motion = np.mean(motion_scores)
        motion_variance = np.var(motion_scores)
        
        # Detect lighting and contrast
        lighting_scores = [self.detect_lighting(f) for f in sampled_frames]
        contrast_scores = [self.detect_contrast(f) for f in sampled_frames]
        
        avg_lighting = np.mean(lighting_scores)
        avg_contrast = np.mean(contrast_scores)
        
        # Scene stability: inverse of motion variance (normalized)
        # Lower variance = more stable
        max_expected_variance = 100.0  # heuristic
        scene_stability = max(0.0, 1.0 - min(1.0, motion_variance / max_expected_variance))
        
        # Classify motion pattern
        if avg_motion < 2.0:
            motion_pattern = "static"
        elif avg_motion < 5.0:
            motion_pattern = "slow"
        elif avg_motion < 10.0:
            motion_pattern = "moderate"
        elif avg_motion < 20.0:
            motion_pattern = "fast"
        else:
            motion_pattern = "very_fast"
        
        # Classify lighting
        if avg_lighting > 0.7:
            lighting_condition = "bright"
        elif avg_lighting > 0.4:
            lighting_condition = "normal"
        elif avg_lighting > 0.2:
            lighting_condition = "low"
        else:
            lighting_condition = "very_low"
        
        # Recommend threshold range based on characteristics
        # Base threshold on motion level, adjust for lighting/contrast
        base_threshold = avg_motion * 0.8  # Start slightly below average
        
        # Adjust for low contrast (night scenes need lower threshold)
        if avg_contrast < 0.2:
            base_threshold *= 0.5  # Reduce threshold by 50% for low contrast
        elif avg_contrast < 0.3:
            base_threshold *= 0.7  # Reduce threshold by 30%
        
        # Adjust for low lighting
        if avg_lighting < 0.3:
            base_threshold *= 0.6  # Reduce threshold for dark scenes
        
        # For fast motion, we need higher threshold to avoid too many calls
        if motion_pattern in ["fast", "very_fast"]:
            base_threshold = max(base_threshold, avg_motion * 1.2)
        
        # Set reasonable bounds - ensure Python float types
        min_threshold = float(max(1.0, base_threshold * 0.5))
        max_threshold = float(min(50.0, base_threshold * 2.0))
        
        # Recommend min_frames_between_calls
        if motion_pattern == "very_fast":
            min_frames = 1  # Can't skip much
        elif motion_pattern == "fast":
            min_frames = 2
        elif motion_pattern in ["moderate", "slow"]:
            min_frames = 2
        else:  # static
            min_frames = 3
        
        return VideoCharacteristics(
            avg_motion_score=float(avg_motion),
            motion_variance=float(motion_variance),
            lighting_level=float(avg_lighting),
            contrast_level=float(avg_contrast),
            scene_stability=float(scene_stability),
            motion_pattern=motion_pattern,
            lighting_condition=lighting_condition,
            recommended_threshold_range=(min_threshold, max_threshold),
            recommended_min_frames=min_frames,
        )

