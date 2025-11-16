"""
Modular model factory for loading different CV models.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_model_config


class ModelFactory:
    """Factory class for creating CV models."""
    
    @staticmethod
    def create_model(model_name: str, num_classes: int, device: str = "cuda"):
        """
        Create a model based on the model name and number of classes.
        
        Args:
            model_name: Name of the model (from config.py)
            num_classes: Number of output classes
            device: Device to load model on
            
        Returns:
            model: The created model
        """
        model_config = get_model_config(model_name)
        framework = model_config["framework"]
        task = model_config["task"]
        
        if framework == "torchvision":
            return ModelFactory._create_torchvision_model(
                model_config, num_classes, device
            )
        elif framework == "ultralytics":
            return ModelFactory._create_ultralytics_model(
                model_config, num_classes, device
            )
        elif framework == "detectron2":
            return ModelFactory._create_detectron2_model(
                model_config, num_classes, device
            )
        elif framework == "segmentation_models_pytorch":
            return ModelFactory._create_smp_model(
                model_config, num_classes, device
            )
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    @staticmethod
    def _create_torchvision_model(model_config: Dict, num_classes: int, device: str):
        """Create torchvision models (ResNet, EfficientNet, ViT, etc.)"""
        model_name = model_config["model_name"]
        task = model_config["task"]
        pretrained = model_config.get("pretrained", True)
        
        # Get the model class
        if hasattr(models, model_name):
            model_class = getattr(models, model_name)
        else:
            raise ValueError(f"Model {model_name} not found in torchvision.models")
        
        # Load pretrained weights
        if pretrained:
            if task == "classification":
                weights_name = f"{model_name.upper()}_Weights"
                if hasattr(models, weights_name):
                    weights_class = getattr(models, weights_name)
                    model = model_class(weights=weights_class.DEFAULT)
                else:
                    model = model_class(pretrained=True)
            elif task == "semantic_segmentation":
                weights_name = f"{model_name.upper()}_Weights"
                if hasattr(models.segmentation, weights_name):
                    weights_class = getattr(models.segmentation, weights_name)
                    model = getattr(models.segmentation, model_name)(weights=weights_class.DEFAULT)
                else:
                    model = getattr(models.segmentation, model_name)(pretrained=True)
        else:
            if task == "semantic_segmentation":
                model = getattr(models.segmentation, model_name)(pretrained=False, num_classes=num_classes)
            else:
                model = model_class(pretrained=False, num_classes=num_classes)
        
        # Modify final layer for custom number of classes
        if task == "classification":
            # Replace final layer
            if hasattr(model, 'fc'):
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
            elif hasattr(model, 'classifier'):
                if isinstance(model.classifier, nn.Sequential):
                    in_features = model.classifier[-1].in_features
                    model.classifier[-1] = nn.Linear(in_features, num_classes)
                else:
                    in_features = model.classifier.in_features
                    model.classifier = nn.Linear(in_features, num_classes)
            elif hasattr(model, 'head'):
                # For ViT models
                in_features = model.head.in_features
                model.head = nn.Linear(in_features, num_classes)
        
        elif task == "semantic_segmentation":
            # Modify output channels for segmentation models
            if hasattr(model, 'classifier'):
                model.classifier[-1] = nn.Conv2d(
                    model.classifier[-1].in_channels,
                    num_classes,
                    kernel_size=1
                )
            if hasattr(model, 'aux_classifier'):
                model.aux_classifier[-1] = nn.Conv2d(
                    model.aux_classifier[-1].in_channels,
                    num_classes,
                    kernel_size=1
                )
        
        return model.to(device)
    
    @staticmethod
    def _create_ultralytics_model(model_config: Dict, num_classes: int, device: str):
        """Create Ultralytics YOLO models."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Please install ultralytics: pip install ultralytics")
        
        model_name = model_config["model_name"]
        
        # Load pretrained YOLO model
        model = YOLO(model_name)
        
        # Models will be fine-tuned with their own training API
        return model
    
    @staticmethod
    def _create_detectron2_model(model_config: Dict, num_classes: int, device: str):
        """Create Detectron2 models (Mask R-CNN, etc.)"""
        try:
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            from detectron2.modeling import build_model
            import detectron2.data.transforms as T
        except ImportError:
            raise ImportError("Please install detectron2")
        
        model_name = model_config["model_name"]
        
        # Setup config
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_name))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.DEVICE = device
        
        # Build model
        model = build_model(cfg)
        
        return model, cfg
    
    @staticmethod
    def _create_smp_model(model_config: Dict, num_classes: int, device: str):
        """Create segmentation_models_pytorch models (U-Net, etc.)"""
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            raise ImportError("Please install segmentation_models_pytorch: pip install segmentation-models-pytorch")
        
        model_name = model_config["model_name"]
        encoder = model_config.get("encoder", "resnet34")
        pretrained = model_config.get("pretrained", True)
        
        # Create model
        if model_name == "unet":
            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights="imagenet" if pretrained else None,
                in_channels=3,
                classes=num_classes,
            )
        elif model_name == "unetplusplus":
            model = smp.UnetPlusPlus(
                encoder_name=encoder,
                encoder_weights="imagenet" if pretrained else None,
                in_channels=3,
                classes=num_classes,
            )
        elif model_name == "deeplabv3plus":
            model = smp.DeepLabV3Plus(
                encoder_name=encoder,
                encoder_weights="imagenet" if pretrained else None,
                in_channels=3,
                classes=num_classes,
            )
        else:
            raise ValueError(f"Unknown smp model: {model_name}")
        
        return model.to(device)


def get_model(model_name: str, num_classes: int, device: str = "cuda"):
    """
    Convenience function to create a model.
    
    Args:
        model_name: Name of the model (from config.py)
        num_classes: Number of output classes
        device: Device to load model on
        
    Returns:
        model: The created model
    """
    return ModelFactory.create_model(model_name, num_classes, device)
