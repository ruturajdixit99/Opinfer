"""
Model loading utilities for various Vision Transformer architectures.
"""

import torch
import timm
from typing import Union, Tuple, Optional, Dict


class ModelLoader:
    """Load and manage various model types."""
    
    # Available classifier models
    CLASSIFIER_MODELS = [
        "vit_tiny_patch16_224",
        "vit_small_patch16_224",
        "vit_base_patch16_224",
        "vit_large_patch16_224",
        "deit_tiny_patch16_224",
        "deit_small_patch16_224",
        "deit_base_patch16_224",
    ]
    
    # Available detector models
    DETECTOR_MODELS = [
        ("owlvit-base", "google/owlvit-base-patch16"),
        ("owlvit-large", "google/owlvit-large-patch14"),
    ]
    
    @staticmethod
    def load_classifier(
        model_name: str,
        device: str = "cuda",
        pretrained: bool = True,
    ) -> torch.nn.Module:
        """
        Load a classifier model from timm.
        
        Args:
            model_name: Name of the model (e.g., "vit_base_patch16_224")
            device: Device to load model on
            pretrained: Whether to load pretrained weights
            
        Returns:
            Loaded model in eval mode
        """
        if model_name not in ModelLoader.CLASSIFIER_MODELS:
            raise ValueError(
                f"Unknown classifier model: {model_name}. "
                f"Available: {ModelLoader.CLASSIFIER_MODELS}"
            )
        
        model = timm.create_model(model_name, pretrained=pretrained)
        model = model.to(device).eval()
        return model
    
    @staticmethod
    def load_detector(
        model_name: str,
        device: str = "cuda",
    ) -> Tuple[any, any]:
        """
        Load an OWL-ViT detector model.
        
        Args:
            model_name: Short name ("owlvit-base" or "owlvit-large")
            device: Device to load model on
            
        Returns:
            (model, processor) tuple
        """
        # Lazy import to avoid NumPy compatibility issues if not using detectors
        try:
            from transformers import OwlViTProcessor, OwlViTForObjectDetection
        except ImportError as e:
            raise ImportError(
                f"transformers library required for detector models. "
                f"Install with: pip install transformers. Original error: {e}"
            )
        
        model_dict = dict(ModelLoader.DETECTOR_MODELS)
        if model_name not in model_dict:
            raise ValueError(
                f"Unknown detector model: {model_name}. "
                f"Available: {list(model_dict.keys())}"
            )
        
        model_id = model_dict[model_name]
        processor = OwlViTProcessor.from_pretrained(model_id)
        model = OwlViTForObjectDetection.from_pretrained(model_id).to(device).eval()
        
        return model, processor
    
    @staticmethod
    def list_models() -> Dict[str, list]:
        """List all available models."""
        return {
            "classifiers": ModelLoader.CLASSIFIER_MODELS,
            "detectors": [name for name, _ in ModelLoader.DETECTOR_MODELS],
        }

