"""
Opinfer: Optimized Inference with Adaptive Motion Gating
A package for efficient real-time video inference using adaptive motion-gated Vision Transformers.
"""

__version__ = "0.1.0"
__author__ = "Opinfer Team"

from opinfer.core import MotionGatedInference
from opinfer.adaptive import AdaptiveMotionGater
from opinfer.models import ModelLoader
from opinfer.optimizer import ParameterOptimizer
from opinfer.api import OptimizedInference, load_video_frames
from opinfer.queue import QueuedInference

__all__ = [
    "MotionGatedInference",
    "AdaptiveMotionGater",
    "ModelLoader",
    "ParameterOptimizer",
    "OptimizedInference",
    "load_video_frames",
    "QueuedInference",
]

