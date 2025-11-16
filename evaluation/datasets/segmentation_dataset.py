"""
Segmentation dataset loaders for semantic and instance segmentation tasks.
"""

import os
from typing import Optional, Callable, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import glob


class SemanticSegmentationDataset(Dataset):
    """
    Dataset for semantic segmentation tasks.
    Each image has a corresponding mask with pixel-wise class labels.
    """
    
    def __init__(
        self,
        data_dir: str,
        image_dir: str,
        label_dir: str,
        num_classes: int,
        transform: Optional[Callable] = None,
        image_ext: str = ".png",
        label_ext: str = ".png",
    ):
        """
        Args:
            data_dir: Root data directory
            image_dir: Subdirectory containing images
            label_dir: Subdirectory containing masks
            num_classes: Number of classes
            transform: Optional transform (applied to both image and mask)
            image_ext: Image file extension
            label_ext: Label file extension
        """
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, image_dir)
        self.label_dir = os.path.join(data_dir, label_dir)
        self.num_classes = num_classes
        self.transform = transform
        self.image_ext = image_ext
        self.label_ext = label_ext
        
        # Load image and label paths
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, f"*{image_ext}")))
        self.label_paths = sorted(glob.glob(os.path.join(self.label_dir, f"*{label_ext}")))
        
        # Match images with labels
        self.samples = self._match_samples()
        
        if len(self.samples) == 0:
            raise RuntimeError(f"Found 0 image-mask pairs in {data_dir}")
        
        print(f"Loaded {len(self.samples)} image-mask pairs for semantic segmentation")
    
    def _match_samples(self):
        """Match image files with their corresponding mask files."""
        samples = []
        for img_path in self.image_paths:
            img_name = os.path.basename(img_path).replace(self.image_ext, "")
            
            # Try to find matching label
            label_path = os.path.join(self.label_dir, img_name + self.label_ext)
            
            if os.path.exists(label_path):
                samples.append((img_path, label_path))
            else:
                print(f"Warning: No label found for {img_name}")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx: Index
            
        Returns:
            tuple: (image, mask) where mask is a tensor with class indices
        """
        img_path, label_path = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Load mask
        mask = Image.open(label_path).convert("L")  # Grayscale
        mask = np.array(mask, dtype=np.int64)
        
        # Apply transforms if provided
        if self.transform:
            # Transform expects PIL images, convert back after
            image, mask = self.transform(image, Image.fromarray(mask.astype(np.uint8)))
            mask = np.array(mask, dtype=np.int64)
        else:
            image = transforms.ToTensor()(image)
        
        mask = torch.from_numpy(mask).long()
        
        return image, mask


class InstanceSegmentationDataset(Dataset):
    """
    Dataset for instance segmentation tasks.
    Each image has both semantic masks and instance masks.
    """
    
    def __init__(
        self,
        data_dir: str,
        image_dir: str,
        label_dir: str,
        instance_dir: str,
        num_classes: int,
        transform: Optional[Callable] = None,
        image_ext: str = ".png",
        label_ext: str = ".png",
    ):
        """
        Args:
            data_dir: Root data directory
            image_dir: Subdirectory containing images
            label_dir: Subdirectory containing semantic masks
            instance_dir: Subdirectory containing instance masks (.npy files)
            num_classes: Number of classes
            transform: Optional transform
            image_ext: Image file extension
            label_ext: Label file extension
        """
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, image_dir)
        self.label_dir = os.path.join(data_dir, label_dir)
        self.instance_dir = os.path.join(data_dir, instance_dir)
        self.num_classes = num_classes
        self.transform = transform
        self.image_ext = image_ext
        self.label_ext = label_ext
        
        # Load paths
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, f"*{image_ext}")))
        
        # Match with labels and instances
        self.samples = self._match_samples()
        
        if len(self.samples) == 0:
            raise RuntimeError(f"Found 0 image-mask pairs in {data_dir}")
        
        print(f"Loaded {len(self.samples)} samples for instance segmentation")
    
    def _match_samples(self):
        """Match images with semantic and instance masks."""
        samples = []
        for img_path in self.image_paths:
            img_name = os.path.basename(img_path).replace(self.image_ext, "")
            
            # Find semantic mask
            label_path = os.path.join(self.label_dir, img_name + self.label_ext)
            
            # Find instance mask (.npy)
            instance_path = os.path.join(self.instance_dir, img_name + ".npy")
            
            if os.path.exists(label_path) and os.path.exists(instance_path):
                samples.append((img_path, label_path, instance_path))
            else:
                if not os.path.exists(label_path):
                    print(f"Warning: No semantic mask found for {img_name}")
                if not os.path.exists(instance_path):
                    print(f"Warning: No instance mask found for {img_name}")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            idx: Index
            
        Returns:
            tuple: (image, semantic_mask, instance_mask)
        """
        img_path, label_path, instance_path = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Load semantic mask
        semantic_mask = Image.open(label_path).convert("L")
        semantic_mask = np.array(semantic_mask, dtype=np.int64)
        
        # Load instance mask
        instance_mask = np.load(instance_path)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        semantic_mask = torch.from_numpy(semantic_mask).long()
        instance_mask = torch.from_numpy(instance_mask).long()
        
        return image, semantic_mask, instance_mask


def get_default_segmentation_transforms(
    input_size: int = 512,
    is_training: bool = True,
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225],
):
    """
    Get default transforms for segmentation tasks.
    Note: This returns a custom transform that applies to both image and mask.
    
    Args:
        input_size: Target image size
        is_training: Whether for training or validation
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Custom transform function
    """
    def transform(image, mask):
        # Resize both
        resize = transforms.Resize((input_size, input_size))
        image = resize(image)
        mask = resize(mask)
        
        if is_training:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            
            # Random vertical flip
            if torch.rand(1) > 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)
        
        # Convert image to tensor and normalize
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=mean, std=std)(image)
        
        return image, mask
    
    return transform
