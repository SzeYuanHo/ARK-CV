"""
Dataset factory for creating datasets based on configuration.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_dataset_config, TRAINING_CONFIG
from datasets.classification_dataset import (
    ClassificationDataset,
    get_default_classification_transforms,
)
from datasets.segmentation_dataset import (
    SemanticSegmentationDataset,
    InstanceSegmentationDataset,
    get_default_segmentation_transforms,
)
from datasets.detection_dataset import (
    ObjectDetectionDataset,
    get_default_detection_transforms,
)
from torch.utils.data import DataLoader, random_split
import torch


def get_dataset(
    dataset_name: str,
    input_size: int = None,
    val_split: float = None,
    batch_size: int = None,
    num_workers: int = None,
):
    """
    Factory function to create train and val datasets and dataloaders.
    
    Args:
        dataset_name: Name of the dataset (from config.py)
        input_size: Input size for the model (default from config)
        val_split: Validation split ratio (default from config)
        batch_size: Batch size (default from config)
        num_workers: Number of workers (default from config)
        
    Returns:
        dict with keys: 'train_dataset', 'val_dataset', 'train_loader', 'val_loader', 'num_classes', 'task'
    """
    # Get dataset configuration
    dataset_config = get_dataset_config(dataset_name)
    task = dataset_config["task"]
    num_classes = dataset_config["num_classes"]
    
    # Use defaults from training config if not provided
    if val_split is None:
        val_split = TRAINING_CONFIG["val_split"]
    if batch_size is None:
        batch_size = TRAINING_CONFIG["batch_size"]
    if num_workers is None:
        num_workers = TRAINING_CONFIG["num_workers"]
    if input_size is None:
        input_size = 224  # Default
    
    # Create datasets based on task type
    if task == "classification":
        train_dataset, val_dataset = _create_classification_dataset(
            dataset_config, input_size, val_split
        )
        collate_fn = None
        
    elif task == "semantic_segmentation":
        train_dataset, val_dataset = _create_semantic_segmentation_dataset(
            dataset_config, input_size, val_split
        )
        collate_fn = None
        
    elif task == "instance_segmentation":
        train_dataset, val_dataset = _create_instance_segmentation_dataset(
            dataset_config, input_size, val_split
        )
        collate_fn = None
        
    elif task == "object_detection":
        train_dataset, val_dataset = _create_object_detection_dataset(
            dataset_config, input_size, val_split
        )
        # Use custom collate function for detection
        collate_fn = train_dataset.collate_fn
        
    else:
        raise ValueError(f"Unknown task type: {task}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=TRAINING_CONFIG["pin_memory"],
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=TRAINING_CONFIG["pin_memory"],
        collate_fn=collate_fn,
    )
    
    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "num_classes": num_classes,
        "task": task,
        "dataset_config": dataset_config,
    }


def _create_classification_dataset(dataset_config, input_size, val_split):
    """Create classification datasets."""
    data_dir = dataset_config["data_dir"]
    class_names = dataset_config["class_names"]
    image_ext = dataset_config["image_ext"]
    
    # Get transforms
    train_transform = get_default_classification_transforms(input_size, is_training=True)
    val_transform = get_default_classification_transforms(input_size, is_training=False)
    
    # Create full dataset
    full_dataset = ClassificationDataset(
        data_dir=data_dir,
        class_names=class_names,
        transform=None,  # We'll apply after split
        image_ext=image_ext,
    )
    
    # Split into train and val
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(TRAINING_CONFIG["seed"]),
    )
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    return train_dataset, val_dataset


def _create_semantic_segmentation_dataset(dataset_config, input_size, val_split):
    """Create semantic segmentation datasets."""
    data_dir = dataset_config["data_dir"]
    image_dir = dataset_config["image_dir"]
    label_dir = dataset_config["label_dir"]
    num_classes = dataset_config["num_classes"]
    image_ext = dataset_config["image_ext"]
    label_ext = dataset_config["label_ext"]
    
    # Get transforms
    train_transform = get_default_segmentation_transforms(input_size, is_training=True)
    val_transform = get_default_segmentation_transforms(input_size, is_training=False)
    
    # Create full dataset
    full_dataset = SemanticSegmentationDataset(
        data_dir=data_dir,
        image_dir=image_dir,
        label_dir=label_dir,
        num_classes=num_classes,
        transform=None,
        image_ext=image_ext,
        label_ext=label_ext,
    )
    
    # Split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(TRAINING_CONFIG["seed"]),
    )
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    return train_dataset, val_dataset


def _create_instance_segmentation_dataset(dataset_config, input_size, val_split):
    """Create instance segmentation datasets."""
    data_dir = dataset_config["data_dir"]
    image_dir = dataset_config["image_dir"]
    label_dir = dataset_config["label_dir"]
    instance_dir = dataset_config["instance_dir"]
    num_classes = dataset_config["num_classes"]
    image_ext = dataset_config["image_ext"]
    label_ext = dataset_config["label_ext"]
    
    # Get transforms (instance seg uses image-only transforms)
    from datasets.classification_dataset import get_default_classification_transforms
    train_transform = get_default_classification_transforms(input_size, is_training=True)
    val_transform = get_default_classification_transforms(input_size, is_training=False)
    
    # Create full dataset
    full_dataset = InstanceSegmentationDataset(
        data_dir=data_dir,
        image_dir=image_dir,
        label_dir=label_dir,
        instance_dir=instance_dir,
        num_classes=num_classes,
        transform=None,
        image_ext=image_ext,
        label_ext=label_ext,
    )
    
    # Split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(TRAINING_CONFIG["seed"]),
    )
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    return train_dataset, val_dataset


def _create_object_detection_dataset(dataset_config, input_size, val_split):
    """Create object detection datasets."""
    data_dir = dataset_config["data_dir"]
    image_dir = dataset_config["image_dir"]
    label_dir = dataset_config["label_dir"]
    num_classes = dataset_config["num_classes"]
    image_ext = dataset_config["image_ext"]
    label_ext = dataset_config["label_ext"]
    
    # Get transforms
    train_transform = get_default_detection_transforms(input_size, is_training=True)
    val_transform = get_default_detection_transforms(input_size, is_training=False)
    
    # Create full dataset
    full_dataset = ObjectDetectionDataset(
        data_dir=data_dir,
        image_dir=image_dir,
        label_dir=label_dir,
        num_classes=num_classes,
        transform=None,
        image_ext=image_ext,
        label_ext=label_ext,
    )
    
    # Split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(TRAINING_CONFIG["seed"]),
    )
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    return train_dataset, val_dataset
