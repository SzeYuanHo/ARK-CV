"""
Modular dataset loaders for different CV tasks.
"""

from .classification_dataset import ClassificationDataset
from .segmentation_dataset import SemanticSegmentationDataset, InstanceSegmentationDataset
from .detection_dataset import ObjectDetectionDataset
from .dataset_factory import get_dataset

__all__ = [
    "ClassificationDataset",
    "SemanticSegmentationDataset",
    "InstanceSegmentationDataset",
    "ObjectDetectionDataset",
    "get_dataset",
]
