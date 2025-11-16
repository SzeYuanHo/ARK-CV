"""
Configuration file for CV model fine-tuning pipeline.
Easily modify this file to switch between datasets, tasks, and models.
"""

import os
from typing import Dict, List

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

DATASETS = {
    "BP": {
        "task": "object_detection",  # Black phosphorus flake detection and classification
        "num_classes": 2,  # 0: suitable flakes, 1: further review (unsuitable not segmented)
        "class_names": ["suitable", "further_review"],
        "data_dir": os.path.join(os.path.dirname(__file__), "datasets", "data", "BP", "BP_data"),
        "image_dir": "images",
        "label_dir": "Labels",
        "label_format": "yolo",  # class_id x_center y_center width height (normalized)
        "image_ext": ".jpg",
        "label_ext": ".txt",
    },
    "Microplastics": {
        "task": "classification",
        "num_classes": 4,
        "class_names": ["algae I", "filament", "fragment", "pellet"],
        "data_dir": os.path.join(os.path.dirname(__file__), "datasets", "data", "Microplastics", "Microplastics_data"),
        "image_dir": None,  # ImageFolder structure: data_dir contains class subdirs
        "label_dir": None,
        "label_format": "imagefolder",
        "image_ext": ".jpg",
        "label_ext": None,
    },
    "PanNuke": {
        "task": "instance_segmentation",  # Cell nuclei instance segmentation
        "num_classes": 5,  # Neoplastic, Inflammatory, Connective, Dead, Epithelial
        "class_names": ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"],
        "data_dir": os.path.join(os.path.dirname(__file__), "datasets", "data", "PanNuke", "pannuke_data"),
        "image_dir": "images",
        "label_dir": "masks_class",  # semantic segmentation masks
        "instance_dir": "masks_instance",  # instance segmentation masks (.npy)
        "label_format": "mask",  # PNG masks with class IDs as pixel values
        "image_ext": ".png",
        "label_ext": ".png",
    },
    "TBM": {
        "task": "semantic_segmentation",
        "num_classes": 2,  # Binary: background or texture boundary
        "class_names": ["background", "boundary"],
        "data_dir": os.path.join(os.path.dirname(__file__), "datasets", "data", "TBM", "TBM_data"),
        "image_dir": "input_image",
        "label_dir": "expert_label",  # Use expert annotations
        "label_format": "mask",  # Binary masks for texture boundary detection (Metallography)
        "image_ext": ".png",
        "label_ext": ".png",
        "description": "Texture Boundary in Metallography - semantic-less boundary detection between texture regions",
    },
}

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODELS = {
    # Object Detection Models
    "yolov8n": {
        "task": "object_detection",
        "framework": "ultralytics",
        "model_name": "yolov8n.pt",
        "input_size": 640,
    },
    "yolov8s": {
        "task": "object_detection",
        "framework": "ultralytics",
        "model_name": "yolov8s.pt",
        "input_size": 640,
    },
    "yolov8m": {
        "task": "object_detection",
        "framework": "ultralytics",
        "model_name": "yolov8m.pt",
        "input_size": 640,
    },
    
    # Instance Segmentation Models
    "yolov8n-seg": {
        "task": "instance_segmentation",
        "framework": "ultralytics",
        "model_name": "yolov8n-seg.pt",
        "input_size": 640,
    },
    "yolov8s-seg": {
        "task": "instance_segmentation",
        "framework": "ultralytics",
        "model_name": "yolov8s-seg.pt",
        "input_size": 640,
    },
    "mask_rcnn_r50": {
        "task": "instance_segmentation",
        "framework": "detectron2",
        "model_name": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "input_size": 800,
    },
    "mask_rcnn_r101": {
        "task": "instance_segmentation",
        "framework": "detectron2",
        "model_name": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
        "input_size": 800,
    },
    
    # Semantic Segmentation Models
    "deeplabv3_resnet50": {
        "task": "semantic_segmentation",
        "framework": "torchvision",
        "model_name": "deeplabv3_resnet50",
        "input_size": 512,
        "pretrained": True,
    },
    "deeplabv3_resnet101": {
        "task": "semantic_segmentation",
        "framework": "torchvision",
        "model_name": "deeplabv3_resnet101",
        "input_size": 512,
        "pretrained": True,
    },
    "fcn_resnet50": {
        "task": "semantic_segmentation",
        "framework": "torchvision",
        "model_name": "fcn_resnet50",
        "input_size": 512,
        "pretrained": True,
    },
    "unet": {
        "task": "semantic_segmentation",
        "framework": "segmentation_models_pytorch",
        "model_name": "unet",
        "encoder": "resnet34",
        "input_size": 256,
        "pretrained": True,
    },
    "unet_lightweight": {
        "task": "semantic_segmentation",
        "framework": "segmentation_models_pytorch",
        "model_name": "unet",
        "encoder": "mobilenet_v2",  # Lightweight encoder
        "input_size": 256,
        "pretrained": True,
    },
    
    # Classification Models
    "resnet50": {
        "task": "classification",
        "framework": "torchvision",
        "model_name": "resnet50",
        "input_size": 224,
        "pretrained": True,
    },
    "resnet101": {
        "task": "classification",
        "framework": "torchvision",
        "model_name": "resnet101",
        "input_size": 224,
        "pretrained": True,
    },
    "efficientnet_b0": {
        "task": "classification",
        "framework": "torchvision",
        "model_name": "efficientnet_b0",
        "input_size": 224,
        "pretrained": True,
    },
    "vit_b_16": {
        "task": "classification",
        "framework": "torchvision",
        "model_name": "vit_b_16",
        "input_size": 224,
        "pretrained": True,
    },
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    "batch_size": 16,
    "epochs": 20,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "optimizer": "adam",  # adam, sgd, adamw
    "scheduler": "cosine",  # cosine, step, plateau
    "early_stopping_patience": 15,
    "save_dir": os.path.join(os.path.dirname(__file__), "runs"),
    "device": "cuda",  # cuda or cpu
    "num_workers": 4,
    "pin_memory": True,
    "mixed_precision": True,  # Use automatic mixed precision
    "val_split": 0.2,  # Validation split ratio
    "seed": 42,
}

# ============================================================================
# DATA AUGMENTATION CONFIGURATION
# ============================================================================

AUGMENTATION_CONFIG = {
    "classification": {
        "train": [
            "resize",
            "random_horizontal_flip",
            "random_vertical_flip",
            "random_rotation",
            "color_jitter",
            "normalize",
        ],
        "val": [
            "resize",
            "normalize",
        ],
    },
    "semantic_segmentation": {
        "train": [
            "resize",
            "random_horizontal_flip",
            "random_vertical_flip",
            "random_rotation",
            "random_crop",
            "normalize",
        ],
        "val": [
            "resize",
            "normalize",
        ],
    },
    "instance_segmentation": {
        "train": [
            "resize",
            "random_horizontal_flip",
            "random_rotation",
            "normalize",
        ],
        "val": [
            "resize",
            "normalize",
        ],
    },
    "object_detection": {
        "train": [
            "resize",
            "random_horizontal_flip",
            "random_rotation",
            "color_jitter",
            "normalize",
        ],
        "val": [
            "resize",
            "normalize",
        ],
    },
}

# ============================================================================
# EVALUATION METRICS CONFIGURATION
# ============================================================================

METRICS_CONFIG = {
    "classification": ["accuracy", "precision", "recall", "f1", "confusion_matrix"],
    "semantic_segmentation": ["iou", "dice", "pixel_accuracy"],
    "instance_segmentation": ["ap", "ap50", "ap75", "ar"],
    "object_detection": ["map", "map50", "map75", "precision", "recall"],
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_dataset_config(dataset_name: str) -> Dict:
    """Get configuration for a specific dataset."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {list(DATASETS.keys())}")
    return DATASETS[dataset_name]


def get_model_config(model_name: str) -> Dict:
    """Get configuration for a specific model."""
    if model_name not in MODELS:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODELS.keys())}")
    return MODELS[model_name]


def validate_model_task_compatibility(model_name: str, dataset_name: str) -> bool:
    """Check if model and dataset tasks are compatible."""
    model_task = MODELS[model_name]["task"]
    dataset_task = DATASETS[dataset_name]["task"]
    return model_task == dataset_task


def get_compatible_models(dataset_name: str) -> List[str]:
    """Get list of models compatible with a dataset."""
    dataset_task = DATASETS[dataset_name]["task"]
    return [model for model, config in MODELS.items() if config["task"] == dataset_task]


def get_compatible_datasets(model_name: str) -> List[str]:
    """Get list of datasets compatible with a model."""
    model_task = MODELS[model_name]["task"]
    return [dataset for dataset, config in DATASETS.items() if config["task"] == model_task]


# ============================================================================
# QUICK CONFIGURATION PRESETS
# ============================================================================

PRESETS = {
    "bp_detection": {
        "dataset": "BP",
        "model": "yolov8n",
        "task": "object_detection",
    },
    "microplastics_classification": {
        "dataset": "Microplastics",
        "model": "resnet50",
        "task": "classification",
    },
    "pannuke_instance_seg": {
        "dataset": "PanNuke",
        "model": "mask_rcnn_r50",
        "task": "instance_segmentation",
    },
    "tbm_semantic_seg": {
        "dataset": "TBM",
        "model": "unet",
        "task": "semantic_segmentation",
    },
}


def get_preset(preset_name: str) -> Dict:
    """Get a quick configuration preset."""
    if preset_name not in PRESETS:
        raise ValueError(f"Preset '{preset_name}' not found. Available presets: {list(PRESETS.keys())}")
    return PRESETS[preset_name]
