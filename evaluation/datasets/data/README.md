# Datasets Directory

This directory contains all the datasets used for fine-tuning computer vision models.

## Dataset Structure

### 1. BP - Black Phosphorus Flake Detection & Classification

**Task**: Object Detection + Classification  
**Format**: YOLO format (normalized bounding boxes)  
**Classes**: 2 classes
- Class 0: `suitable` - Suitable black phosphorus flakes
- Class 1: `further_review` - Flakes requiring further review

**Note**: Unsuitable flakes are NOT segmented or labeled - they are treated as background.

**Directory Structure**:
```
BP/
└── BP_data/
    ├── images/           # Input microscope images (.jpg)
    └── Labels/          # YOLO format labels (.txt)
```

**Label Format** (YOLO):
```
class_id x_center y_center width height
```
All coordinates are normalized to [0, 1].

**Example**: `001_50x_x0.76_y5.989_aug_1.txt`
```
0 0.447502 0.468567 0.060534 0.099947
```

---

### 2. Microplastics - Microplastic Type Classification

**Task**: Classification  
**Format**: ImageFolder (directory-based classification)  
**Classes**: 4 classes
- `algae I` - Algae type I microplastics
- `filament` - Filament-shaped microplastics
- `fragment` - Fragment microplastics
- `pellet` - Pellet-shaped microplastics

**Directory Structure**:
```
Microplastics/
└── Microplastics_data/
    ├── algae I/         # Class 0 images
    ├── filament/        # Class 1 images
    ├── fragment/        # Class 2 images
    └── pellet/          # Class 3 images
```

Each subdirectory contains `.jpg` images of that particular microplastic type.

---

### 3. PanNuke - Cell Nuclei Instance Segmentation

**Task**: Instance Segmentation (can also be used for semantic segmentation)  
**Format**: PNG images + PNG class masks + .npy instance masks  
**Classes**: 5 nuclei types
- Class 0: `Neoplastic` - Neoplastic cells
- Class 1: `Inflammatory` - Inflammatory cells
- Class 2: `Connective` - Connective tissue cells
- Class 3: `Dead` - Dead cells
- Class 4: `Epithelial` - Epithelial cells

**Directory Structure**:
```
PanNuke/
└── pannuke_data/
    ├── images/              # Input histopathology images (.png)
    ├── masks_class/         # Semantic segmentation masks (.png)
    └── masks_instance/      # Instance segmentation masks (.npy)
```

**Mask Format**:
- **masks_class**: PNG images where pixel value = class ID (0-5, with 0=background)
- **masks_instance**: Numpy arrays (.npy) where each instance has a unique integer ID

**Important Note**: Class masks may appear black when visualized because pixel values are small integers (0-5). Use proper colorization or unique value extraction for visualization.

---

### 4. TBM - Texture Boundary in Metallography

**Task**: Semantic Segmentation  
**Format**: PNG images + PNG binary masks  
**Classes**: 2 classes (binary segmentation)
- Class 0: `background` - Non-boundary regions
- Class 1: `boundary` - Texture boundaries

**Purpose**: Semantic-less texture boundary detection. The goal is to identify boundaries between regions of different textures in metallographic images, without classifying what the textures represent.

**Directory Structure**:
```
TBM/
└── TBM_data/
    ├── input_image/         # Input metallography images (.png)
    └── expert_label/        # Binary boundary masks (.png)
```

**Mask Format**: Binary PNG masks where:
- 0 (black) = background/non-boundary
- 255 (white) = boundary pixels

---

## Data Loading

The datasets are automatically loaded through the factory pattern in `datasets/dataset_factory.py`. The configuration in `config.py` specifies all paths and parameters.

### Usage Example:

```python
from datasets.dataset_factory import get_dataset

# Load BP dataset
dataset_dict = get_dataset(
    dataset_name="BP",
    input_size=640,
    batch_size=16,
)

train_loader = dataset_dict['train_loader']
val_loader = dataset_dict['val_loader']
num_classes = dataset_dict['num_classes']
```

## Adding New Datasets

To add a new dataset:

1. **Place data** in this `datasets/data/` directory
2. **Update `config.py`** with dataset configuration:
   ```python
   DATASETS = {
       "YourDataset": {
           "task": "classification",  # or other task
           "num_classes": 3,
           "class_names": ["class1", "class2", "class3"],
           "data_dir": os.path.join(os.path.dirname(__file__), "datasets", "data", "YourDataset"),
           "image_dir": "images",
           "label_dir": "labels",
           "label_format": "imagefolder",  # or yolo, mask, etc.
           "image_ext": ".jpg",
           "label_ext": ".txt",
       },
   }
   ```
3. **Create dataset loader** if using a custom format (see `datasets/` directory)

## Dataset Statistics

| Dataset | Task | Images | Classes | Input Size | Format |
|---------|------|--------|---------|------------|--------|
| BP | Object Detection | ~3000+ | 2 | 640x640 | YOLO |
| Microplastics | Classification | ~300+ | 4 | 224x224 | ImageFolder |
| PanNuke | Instance Seg | ~500+ | 5 | 256x256 | PNG+NPY |
| TBM | Semantic Seg | ~200+ | 2 | 256x256 | Binary Masks |

## Notes

- All datasets support automatic train/val splitting (default 80/20)
- Data augmentation is configured per-task in `config.py`
- Images and labels are automatically matched by filename
- YOLO coordinates are normalized to [0,1] range
- PNG masks should have integer class IDs as pixel values
