"""
Object detection dataset loader (YOLO format).
"""

import os
from typing import Optional, Callable, Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import glob


class ObjectDetectionDataset(Dataset):
    """
    Dataset for object detection tasks in YOLO format.
    Each .txt file contains: class_id x_center y_center width height (normalized 0-1)
    """
    
    def __init__(
        self,
        data_dir: str,
        image_dir: str,
        label_dir: str,
        num_classes: int,
        transform: Optional[Callable] = None,
        image_ext: str = ".jpg",
        label_ext: str = ".txt",
    ):
        """
        Args:
            data_dir: Root data directory
            image_dir: Subdirectory containing images
            label_dir: Subdirectory containing YOLO format labels
            num_classes: Number of classes
            transform: Optional transform
            image_ext: Image file extension
            label_ext: Label file extension (.txt for YOLO)
        """
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, image_dir)
        self.label_dir = os.path.join(data_dir, label_dir)
        self.num_classes = num_classes
        self.transform = transform
        self.image_ext = image_ext
        self.label_ext = label_ext
        
        # Load paths
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, f"*{image_ext}")))
        
        # Match with labels
        self.samples = self._match_samples()
        
        if len(self.samples) == 0:
            raise RuntimeError(f"Found 0 image-label pairs in {data_dir}")
        
        print(f"Loaded {len(self.samples)} samples for object detection")
    
    def _match_samples(self):
        """Match images with their YOLO label files."""
        samples = []
        for img_path in self.image_paths:
            img_name = os.path.basename(img_path).replace(self.image_ext, "")
            
            # Find label file
            label_path = os.path.join(self.label_dir, img_name + self.label_ext)
            
            if os.path.exists(label_path):
                samples.append((img_path, label_path))
            else:
                # Image without labels (background image)
                samples.append((img_path, None))
        
        return samples
    
    def _parse_yolo_label(self, label_path: Optional[str], img_width: int, img_height: int) -> dict:
        """
        Parse YOLO format label file.
        
        Args:
            label_path: Path to label file
            img_width: Image width
            img_height: Image height
            
        Returns:
            dict with 'boxes', 'labels' keys
        """
        if label_path is None or not os.path.exists(label_path):
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            }
        
        boxes = []
        labels = []
        
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert from YOLO format (normalized center x, y, w, h) to
                # [x_min, y_min, x_max, y_max] in absolute coordinates
                x_min = (x_center - width / 2) * img_width
                y_min = (y_center - height / 2) * img_height
                x_max = (x_center + width / 2) * img_width
                y_max = (y_center + height / 2) * img_height
                
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(class_id)
        
        if len(boxes) == 0:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            }
        
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            idx: Index
            
        Returns:
            tuple: (image, target) where target is a dict with 'boxes' and 'labels'
        """
        img_path, label_path = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size
        
        # Parse YOLO labels
        target = self._parse_yolo_label(label_path, img_width, img_height)
        target["image_id"] = torch.tensor([idx])
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        return image, target
    
    def collate_fn(self, batch):
        """
        Custom collate function for object detection.
        Required because each image can have different numbers of bounding boxes.
        """
        images = []
        targets = []
        
        for image, target in batch:
            images.append(image)
            targets.append(target)
        
        images = torch.stack(images, dim=0)
        
        return images, targets


def get_default_detection_transforms(
    input_size: int = 640,
    is_training: bool = True,
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225],
) -> transforms.Compose:
    """
    Get default transforms for object detection tasks.
    
    Args:
        input_size: Target image size
        is_training: Whether for training or validation
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Composed transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
