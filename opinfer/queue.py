"""
Frame queuing-based inference engine.
Processes frames in batches/queues for efficient inference.
"""

import time
from typing import List, Optional, Tuple, Dict, Any
from collections import deque
import cv2
import numpy as np
import torch
import torch.nn as nn


class QueuedInference:
    """Frame queuing-based inference engine."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        img_size: int = 224,
        queue_size: int = 4,
        batch_size: int = 4,
        max_queue_wait_ms: float = 33.0,  # ~30 FPS
    ):
        """
        Initialize queued inference engine.
        
        Args:
            model: PyTorch model
            device: Device to run inference on
            img_size: Input image size for model
            queue_size: Maximum number of frames to queue before processing
            batch_size: Number of frames to process in one batch
            max_queue_wait_ms: Maximum time to wait before processing queue (ms)
        """
        self.model = model
        self.device = device
        self.img_size = img_size
        self.queue_size = queue_size
        self.batch_size = batch_size
        self.max_queue_wait_ms = max_queue_wait_ms
        
        # ImageNet normalization
        if isinstance(device, str):
            torch_device = torch.device(device)
        else:
            torch_device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=torch_device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=torch_device).view(1, 3, 1, 1)
        
        # Queue state
        self.frame_queue: deque = deque(maxlen=queue_size)
        self.timestamp_queue: deque = deque(maxlen=queue_size)
        self.last_process_time = time.time()
        
        # Output cache (for frames not yet processed)
        self.output_cache: Dict[int, Any] = {}
        self.frame_counter = 0
        
        # Statistics
        self.total_frames = 0
        self.batches_processed = 0
        self.total_inference_time = 0.0
        self.total_queue_wait_time = 0.0
        
        self.model.eval()
    
    def preprocess_frame(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Preprocess BGR frame for model input."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.img_size, self.img_size))
        img = torch.from_numpy(frame_resized).float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)
        img = img.to(self.device)
        img = (img - self.mean) / self.std
        return img
    
    def preprocess_batch(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Preprocess a batch of frames."""
        batch_tensors = []
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self.img_size, self.img_size))
            img = torch.from_numpy(frame_resized).float() / 255.0
            img = img.permute(2, 0, 1)
            batch_tensors.append(img)
        
        # Stack into batch
        batch = torch.stack(batch_tensors).to(self.device)
        batch = (batch - self.mean) / self.std
        return batch
    
    def process_queue(self, force: bool = False) -> List[Tuple[int, Any]]:
        """
        Process queued frames in batches.
        
        Args:
            force: Force processing even if queue is not full
            
        Returns:
            List of (frame_id, output) tuples
        """
        current_time = time.time()
        elapsed_ms = (current_time - self.last_process_time) * 1000.0
        
        # Decide whether to process
        should_process = False
        if force:
            should_process = True
        elif len(self.frame_queue) >= self.batch_size:
            should_process = True
        elif len(self.frame_queue) > 0 and elapsed_ms >= self.max_queue_wait_ms:
            should_process = True
        
        if not should_process or len(self.frame_queue) == 0:
            return []
        
        # Get frames to process
        num_to_process = min(len(self.frame_queue), self.batch_size)
        frames_to_process = []
        frame_ids = []
        timestamps = []
        
        for _ in range(num_to_process):
            frame_id, frame, timestamp = self.frame_queue.popleft()
            frames_to_process.append(frame)
            frame_ids.append(frame_id)
            timestamps.append(timestamp)
        
        # Process batch
        queue_wait_time = (current_time - timestamps[0]) * 1000.0 if timestamps else 0.0
        self.total_queue_wait_time += queue_wait_time
        
        t0 = time.time()
        with torch.no_grad():
            batch = self.preprocess_batch(frames_to_process)
            outputs = self.model(batch)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            
            # Handle batch outputs
            if outputs.dim() > 2:
                # If output is not already batched properly, split it
                outputs = outputs if outputs.dim() == 2 else outputs.view(outputs.size(0), -1)
            
            # Convert to CPU and split
            outputs_cpu = outputs.cpu()
            batch_outputs = [outputs_cpu[i] for i in range(len(frames_to_process))]
        
        infer_ms = (time.time() - t0) * 1000.0
        self.total_inference_time += infer_ms / 1000.0
        self.batches_processed += 1
        
        # Store outputs
        results = []
        for frame_id, output in zip(frame_ids, batch_outputs):
            self.output_cache[frame_id] = output
            results.append((frame_id, output))
        
        self.last_process_time = time.time()
        return results
    
    def infer(self, frame_bgr: np.ndarray, processor=None, text_queries=None) -> Tuple[Any, Dict[str, float]]:
        """
        Run queued inference on a frame.
        
        Args:
            frame_bgr: BGR frame from video
            processor: Optional processor for detector models (not used in queuing)
            text_queries: Optional text queries for detector models (not used in queuing)
            
        Returns:
            (output, stats_dict) where output is model prediction and stats contains metrics
        """
        self.total_frames += 1
        frame_id = self.frame_counter
        self.frame_counter += 1
        
        current_time = time.time()
        
        # Add frame to queue
        self.frame_queue.append((frame_id, frame_bgr, current_time))
        self.timestamp_queue.append(current_time)
        
        # Try to process queue
        processed = self.process_queue(force=False)
        
        # Get output for current frame (might be from cache if already processed)
        if frame_id in self.output_cache:
            output = self.output_cache.pop(frame_id)
        else:
            # Frame is still in queue, try to process if queue is full
            if len(self.frame_queue) >= self.queue_size:
                self.process_queue(force=True)
                # Check again after processing
                if frame_id in self.output_cache:
                    output = self.output_cache.pop(frame_id)
                else:
                    # Frame still not processed, wait for next batch or process single
                    # For now, return None (caller should handle)
                    output = None
            else:
                # Queue not full yet, return None (frame will be processed in next batch)
                output = None
        
        # Calculate stats
        queue_size = len(self.frame_queue)
        avg_queue_wait = self.total_queue_wait_time / max(1, self.batches_processed)
        avg_batch_latency = (self.total_inference_time / max(1, self.batches_processed)) * 1000.0
        
        stats = {
            "queue_size": queue_size,
            "did_process": len(processed) > 0,
            "frames_processed": len(processed),
            "avg_queue_wait_ms": avg_queue_wait,
            "avg_batch_latency_ms": avg_batch_latency,
        }
        
        return output, stats
    
    def flush(self):
        """Process all remaining frames in queue."""
        results = []
        while len(self.frame_queue) > 0:
            batch_results = self.process_queue(force=True)
            results.extend(batch_results)
        return results
    
    def reset(self):
        """Reset state for new video sequence."""
        self.frame_queue.clear()
        self.timestamp_queue.clear()
        self.output_cache.clear()
        self.frame_counter = 0
        self.total_frames = 0
        self.batches_processed = 0
        self.total_inference_time = 0.0
        self.total_queue_wait_time = 0.0
        self.last_process_time = time.time()
    
    def get_stats(self) -> Dict[str, float]:
        """Get accumulated statistics."""
        if self.total_frames == 0:
            return {}
        
        avg_batch_latency_ms = (self.total_inference_time / max(1, self.batches_processed)) * 1000.0
        avg_queue_wait_ms = self.total_queue_wait_time / max(1, self.batches_processed)
        frames_per_batch = self.total_frames / max(1, self.batches_processed)
        effective_fps = 1000.0 / (avg_batch_latency_ms / frames_per_batch) if frames_per_batch > 0 else 0.0
        
        return {
            "total_frames": self.total_frames,
            "batches_processed": self.batches_processed,
            "avg_batch_latency_ms": avg_batch_latency_ms,
            "avg_queue_wait_ms": avg_queue_wait_ms,
            "frames_per_batch": frames_per_batch,
            "effective_fps": effective_fps,
            "total_inference_time": self.total_inference_time,
        }

