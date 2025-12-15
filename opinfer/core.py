"""
Core motion-gated inference engine.
"""

import time
from typing import Optional, Tuple, Dict, Any, List
import cv2
import numpy as np
import torch
import torch.nn as nn


class MotionGatedInference:
    """Core motion-gated inference engine."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        img_size: int = 224,
        motion_threshold: float = 4.0,
        min_frames_between_calls: int = 2,
        motion_downsample: Tuple[int, int] = (64, 64),
    ):
        """
        Initialize motion-gated inference engine.
        
        Args:
            model: PyTorch model (ViT, OWL-ViT, etc.)
            device: Device to run inference on
            img_size: Input image size for model
            motion_threshold: Motion score threshold for triggering inference
            min_frames_between_calls: Minimum frames between model calls
            motion_downsample: Size for motion computation (smaller = faster)
        """
        self.model = model
        self.device = device
        self.img_size = img_size
        self.motion_threshold = motion_threshold
        self.min_frames_between_calls = min_frames_between_calls
        self.motion_downsample = motion_downsample
        
        # ImageNet normalization - ensure they're on the correct device
        # Convert device string to torch device object for consistency
        if isinstance(device, str):
            torch_device = torch.device(device)
        else:
            torch_device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=torch_device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=torch_device).view(1, 3, 1, 1)
        
        # State
        self.prev_gray_small: Optional[np.ndarray] = None
        self.last_output: Optional[Any] = None
        self.frames_since_last_call = 0
        
        # Statistics
        self.total_frames = 0
        self.model_calls = 0
        self.total_inference_time = 0.0
        
        self.model.eval()
    
    def preprocess_frame(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Preprocess BGR frame for model input."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.img_size, self.img_size))
        img = torch.from_numpy(frame_resized).float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)
        # Move to device BEFORE normalization to ensure all tensors are on same device
        img = img.to(self.device)
        # Normalize - mean and std are already on device from __init__
        img = (img - self.mean) / self.std
        return img
    
    def compute_motion_score(self, frame_bgr: np.ndarray) -> float:
        """Compute motion score for current frame."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, self.motion_downsample, interpolation=cv2.INTER_AREA)
        
        if self.prev_gray_small is None:
            motion_score = 0.0
        else:
            diff = cv2.absdiff(gray_small, self.prev_gray_small)
            motion_score = float(diff.mean())
        
        self.prev_gray_small = gray_small
        return motion_score
    
    def should_call_model(self, motion_score: float) -> bool:
        """Determine if model should be called based on motion and state."""
        if self.prev_gray_small is None:
            return True  # First frame
        if motion_score >= self.motion_threshold:
            return True  # Significant motion
        if self.frames_since_last_call >= self.min_frames_between_calls:
            return True  # Force refresh
        return False  # Reuse previous output
    
    def infer(self, frame_bgr: np.ndarray, processor=None, text_queries=None) -> Tuple[Any, Dict[str, float]]:
        """
        Run motion-gated inference on a frame.
        
        Args:
            frame_bgr: BGR frame from video
            processor: Optional processor for detector models (OWL-ViT)
            text_queries: Optional text queries for detector models
            
        Returns:
            (output, stats_dict) where output is model prediction and stats contains metrics
        """
        self.total_frames += 1
        motion_score = self.compute_motion_score(frame_bgr)
        should_call = self.should_call_model(motion_score)
        
        stats = {
            "motion_score": motion_score,
            "did_call": False,
            "inference_ms": 0.0,
        }
        
        if should_call:
            t0 = time.time()
            with torch.no_grad():
                if processor is not None and text_queries is not None:
                    # Detector model (OWL-ViT)
                    from PIL import Image
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    inputs = processor(
                        text=[", ".join(text_queries)],
                        images=pil_img,
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs)
                    target_sizes = torch.tensor([pil_img.size[::-1]], device=self.device)
                    results = processor.post_process_object_detection(
                        outputs=outputs,
                        target_sizes=target_sizes,
                        threshold=0.1
                    )[0]
                    output = {
                        "boxes": results["boxes"].detach().cpu().numpy(),
                        "scores": results["scores"].detach().cpu().numpy(),
                        "labels": results["labels"].detach().cpu().numpy(),
                    }
                else:
                    # Classifier model
                    img = self.preprocess_frame(frame_bgr)
                    output = self.model(img)
                    if isinstance(output, (list, tuple)):
                        output = output[0]
                    output = output.cpu() if isinstance(output, torch.Tensor) else output
            
            infer_ms = (time.time() - t0) * 1000.0
            
            self.last_output = output
            self.frames_since_last_call = 0
            self.model_calls += 1
            self.total_inference_time += infer_ms / 1000.0
            
            stats["did_call"] = True
            stats["inference_ms"] = infer_ms
        else:
            self.frames_since_last_call += 1
            output = self.last_output
        
        return output, stats
    
    def update_parameters(
        self,
        motion_threshold: Optional[float] = None,
        min_frames_between_calls: Optional[int] = None,
    ):
        """Update motion gating parameters dynamically."""
        if motion_threshold is not None:
            self.motion_threshold = motion_threshold
        if min_frames_between_calls is not None:
            self.min_frames_between_calls = min_frames_between_calls
    
    def reset(self):
        """Reset state for new video sequence."""
        self.prev_gray_small = None
        self.last_output = None
        self.frames_since_last_call = 0
        self.total_frames = 0
        self.model_calls = 0
        self.total_inference_time = 0.0
    
    def get_stats(self) -> Dict[str, float]:
        """Get accumulated statistics."""
        if self.total_frames == 0:
            return {}
        
        skip_rate = 100.0 * (self.total_frames - self.model_calls) / self.total_frames
        avg_inference_ms = (self.total_inference_time / self.model_calls * 1000.0) if self.model_calls > 0 else 0.0
        effective_fps = 1000.0 / (avg_inference_ms * (self.total_frames / self.model_calls)) if self.model_calls > 0 else 0.0
        
        return {
            "total_frames": self.total_frames,
            "model_calls": self.model_calls,
            "skipped_frames": self.total_frames - self.model_calls,
            "skip_rate_pct": skip_rate,
            "avg_inference_ms": avg_inference_ms,
            "total_inference_time": self.total_inference_time,
            "effective_fps": effective_fps,
        }

