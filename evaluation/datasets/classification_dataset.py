"""
Classification dataset loader.
"""

import os
from typing import Optional, Callable, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob


class ClassificationDataset(Dataset):
    """
    Dataset for image classification tasks.
    Supports ImageFolder structure: root/class1/*.jpg, root/class2/*.jpg, etc.
    """
    
    def __init__(
        self,
        data_dir: str,
        class_names: list,
        transform: Optional[Callable] = None,
        image_ext: str = ".jpg",
    ):
        """
        Args:
            data_dir: Root directory with class subdirectories
            class_names: List of class names (also the subdirectory names)
            transform: Optional transform to be applied on images
            image_ext: Image file extension
        """
        self.data_dir = data_dir
        self.class_names = class_names
        self.transform = transform
        self.image_ext = image_ext
        
        # Create class to index mapping
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
        
        # Load all image paths and labels
        self.samples = []
        self._load_samples()
        
        if len(self.samples) == 0:
            raise RuntimeError(f"Found 0 images in {data_dir}")
        
    def _load_samples(self):
        """Load all image paths and their labels."""
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            
            # Find all images with the specified extension
            pattern = os.path.join(class_dir, f"*{self.image_ext}")
            image_paths = glob.glob(pattern)
            
            class_idx = self.class_to_idx[class_name]
            for img_path in image_paths:
                self.samples.append((img_path, class_idx))
        
        print(f"Loaded {len(self.samples)} images from {len(self.class_names)} classes")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            idx: Index
            
        Returns:
            tuple: (image, label) where label is the class index
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> dict:
        """Get the distribution of samples per class."""
        distribution = {cls_name: 0 for cls_name in self.class_names}
        for _, label in self.samples:
            class_name = self.class_names[label]
            distribution[class_name] += 1
        return distribution


def get_default_classification_transforms(
    input_size: int = 224,
    is_training: bool = True,
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225],
) -> transforms.Compose:
    """
    Get default transforms for classification tasks.
    
    Args:
        input_size: Target image size
        is_training: Whether for training (with augmentation) or validation
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Composed transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
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
