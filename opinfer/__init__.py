"""
Opinfer: Optimized Inference with Adaptive Motion Gating

High-level API:
    from opinfer import OptimizedInference, load_video_frames
    
    # Quick start
    infer = OptimizedInference(model_name='vit_base_patch16_224')
    result = infer.process_video_file('video.mp4')
"""

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
