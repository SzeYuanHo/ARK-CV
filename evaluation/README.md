# CV Model Evaluation Pipeline

A streamlined, universal training pipeline for evaluating computer vision models across multiple tasks and datasets. This codebase provides a modular framework for quickly benchmarking different architectures on your custom datasets.

## Features

✅ **Universal Training Script** - One script to train any model on any compatible dataset  
✅ **Multiple Tasks** - Classification, Object Detection, Semantic Segmentation, Instance Segmentation  
✅ **Multiple Architectures** - ResNet, EfficientNet, ViT, YOLO, Mask R-CNN, U-Net, DeepLabV3  
✅ **4 Ready-to-Use Datasets** - BP, Microplastics, PanNuke, TBM  
✅ **Modular Design** - Easy to add new models or datasets  
✅ **Configuration-Driven** - All settings in one config file  
✅ **Training Features** - Mixed precision, learning rate scheduling, early stopping, checkpointing

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Train a Model

Use presets for quick training:

```bash
# Black Phosphorus flake detection with YOLO
python trainCVmodel.py --preset bp_detection

# Microplastics classification with ResNet
python trainCVmodel.py --preset microplastics_classification

# PanNuke cell nuclei segmentation with Mask R-CNN
python trainCVmodel.py --preset pannuke_instance_seg

# Texture boundary detection with U-Net
python trainCVmodel.py --preset tbm_semantic_seg
```

Or specify dataset and model directly:

```bash
python trainCVmodel.py --dataset BP --model yolov8n --epochs 50
python trainCVmodel.py --dataset Microplastics --model efficientnet_b0
python trainCVmodel.py --dataset TBM --model unet --batch_size 8
```

### 3. View Results

Training outputs are saved to `runs/{dataset}_{model}_{timestamp}/`:

```
runs/BP_yolov8n_20251115_120000/
├── config.json                 # Training configuration
├── best_model.pth             # Best checkpoint
├── final_model.pth            # Final checkpoint
└── training_history.json      # Metrics history
```

## Project Structure

```
evaluation/
├── config.py                      # All configurations (datasets, models, training)
├── trainCVmodel.py               # Universal training script
├── requirements.txt              # Dependencies
├── README.md                     # This file
│
├── datasets/                      # Dataset loaders
│   ├── __init__.py
│   ├── classification_dataset.py
│   ├── segmentation_dataset.py
│   ├── detection_dataset.py
│   ├── dataset_factory.py        # Dataset factory
│   └── data/                      # Dataset storage
│       ├── BP/                    # Black Phosphorus detection
│       ├── Microplastics/        # Microplastics classification
│       ├── PanNuke/              # Cell nuclei segmentation
│       └── TBM/                  # Texture boundary detection
│
├── models/                        # Model loaders
│   ├── __init__.py
│   └── model_factory.py          # Model factory
│
└── runs/                          # Training outputs
```

## Available Datasets

| Dataset | Task | Classes | Description |
|---------|------|---------|-------------|
| **BP** | Object Detection | 2 | Black phosphorus flake detection (suitable, further review) |
| **Microplastics** | Classification | 4 | Microplastic type classification (algae, filament, fragment, pellet) |
| **PanNuke** | Instance Segmentation | 5 | Cell nuclei segmentation (Neoplastic, Inflammatory, Connective, Dead, Epithelial) |
| **TBM** | Semantic Segmentation | 2 | Texture boundary detection in metallography images |

See `datasets/data/README.md` for detailed dataset information.

### Dataset Acquisition

**Datasets are not included in this repository.** Download them separately:

- **BP (Black Phosphorus)**: Not publicly available. Contact the dataset authors or substitute with similar microscopy datasets like [DeepFlake](https://github.com/chrimerss/DeepFlake) or [2D Material Flakes](https://github.com/smlab-niser/2DMD).
- **Microplastics**: Available from [Kaggle - Microplastics Dataset](https://www.kaggle.com/datasets/farzadnekouei/microplastics-image-dataset) or similar microplastic classification datasets.
- **PanNuke**: Download from [PanNuke Official Website](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke) ([Paper](https://arxiv.org/abs/2003.10778)).
- **TBM (Texture Boundary Metallography)**: Request from dataset authors or use similar metallography boundary detection datasets.

Place downloaded datasets in `datasets/data/<DatasetName>/` following the structure in `datasets/data/README.md`.

## Available Models

### Classification
- `resnet50`, `resnet101` - Deep residual networks
- `efficientnet_b0` - Efficient convolutional network
- `vit_b_16` - Vision Transformer

### Object Detection
- `yolov8n`, `yolov8s`, `yolov8m` - YOLO v8 (nano, small, medium)

### Instance Segmentation
- `yolov8n-seg`, `yolov8s-seg` - YOLO v8 segmentation
- `mask_rcnn_r50`, `mask_rcnn_r101` - Mask R-CNN

### Semantic Segmentation
- `unet`, `unet_lightweight` - U-Net with ResNet34 or MobileNetV2 encoder
- `deeplabv3_resnet50`, `deeplabv3_resnet101` - DeepLabV3
- `fcn_resnet50` - Fully Convolutional Network

## Configuration

All settings are centralized in `config.py`:

### Training Parameters

```python
TRAINING_CONFIG = {
    "batch_size": 16,
    "epochs": 20,
    "learning_rate": 0.001,
    "optimiser": "adam",          # adam, sgd, adamw
    "scheduler": "cosine",        # cosine, step, plateau
    "early_stopping_patience": 15,
    "device": "cuda",             # cuda or cpu
    "mixed_precision": True,
    "val_split": 0.2,
    "seed": 42,
}
```

### Adding a New Dataset

Edit `DATASETS` in `config.py`:

```python
DATASETS = {
    "MyDataset": {
        "task": "classification",
        "num_classes": 10,
        "class_names": ["class1", "class2", ...],
        "data_dir": "path/to/data",
        "image_dir": "images",
        "label_dir": "labels",
        "label_format": "imagefolder",
        "image_ext": ".jpg",
    },
}
```

### Adding a New Model

Edit `MODELS` in `config.py`:

```python
MODELS = {
    "my_model": {
        "task": "classification",
        "framework": "torchvision",
        "model_name": "my_model_name",
        "input_size": 224,
        "pretrained": True,
    },
}
```

## Command-Line Arguments

```bash
python trainCVmodel.py [OPTIONS]

Options:
  --dataset TEXT       Dataset name (BP, Microplastics, PanNuke, TBM)
  --model TEXT         Model name (see Available Models)
  --preset TEXT        Use a preset configuration
  --epochs INT         Number of training epochs
  --batch_size INT     Batch size
  --lr FLOAT          Learning rate
  --device TEXT        Device (cuda/cpu)
  --save_dir TEXT      Custom save directory
```

## Usage Examples

### Basic Training

```bash
# Train YOLO on BP dataset
python trainCVmodel.py --dataset BP --model yolov8n

# Train ResNet on Microplastics with custom epochs
python trainCVmodel.py --dataset Microplastics --model resnet50 --epochs 100

# Train U-Net on TBM with custom batch size
python trainCVmodel.py --dataset TBM --model unet --batch_size 8
```

### Using Presets

```bash
# Use predefined configurations
python trainCVmodel.py --preset bp_detection
python trainCVmodel.py --preset microplastics_classification
python trainCVmodel.py --preset pannuke_instance_seg
python trainCVmodel.py --preset tbm_semantic_seg
```

### Custom Hyperparameters

```bash
# Fine-tune all hyperparameters
python trainCVmodel.py \
  --dataset BP \
  --model yolov8s \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.001 \
  --device cuda
```

## Compatibility Matrix

Check model-dataset compatibility before training:

```python
from config import validate_model_task_compatibility, get_compatible_models

# Check compatibility
is_compatible = validate_model_task_compatibility("resnet50", "Microplastics")

# Get all compatible models for a dataset
models = get_compatible_models("BP")  # Returns: ['yolov8n', 'yolov8s', 'yolov8m']
```

## Dataset Formats

### Classification (ImageFolder)
```
data_dir/
├── class1/
│   ├── image1.jpg
│   └── image2.jpg
└── class2/
    └── image3.jpg
```

### Object Detection (YOLO)
```
data_dir/
├── images/
│   └── image1.jpg
└── labels/
    └── image1.txt  # class_id x_center y_center width height (normalized)
```

### Semantic Segmentation
```
data_dir/
├── images/
│   └── image1.png
└── masks/
    └── image1.png  # Grayscale mask with class IDs
```

### Instance Segmentation
```
data_dir/
├── images/
│   └── image1.png
├── masks_class/
│   └── image1.png  # Semantic mask
└── masks_instance/
    └── image1.npy  # Instance IDs (numpy array)
```

## Advanced Features

### Custom Data Augmentation

Modify augmentation in `config.py`:

```python
AUGMENTATION_CONFIG = {
    "classification": {
        "train": [
            "resize",
            "random_horizontal_flip",
            "color_jitter",
            "random_rotation",
        ],
    },
}
```

### Direct API Usage

Use the dataset and model factories directly:

```python
from datasets.dataset_factory import get_dataset
from models.model_factory import get_model

# Load dataset
dataset_dict = get_dataset(
    dataset_name="Microplastics",
    input_size=224,
    batch_size=32,
)

# Create model
model = get_model(
    model_name="resnet50",
    num_classes=4,
    device="cuda"
)
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size`: `--batch_size 8`
- Use smaller model: `yolov8n` instead of `yolov8m`
- Disable mixed precision in `config.py`

### Slow Training on Windows
- Set `num_workers: 0` in `config.py`
- Or use fewer workers: set to 2-4

### Import Errors
- Install all dependencies: `pip install -r requirements.txt`
- For optional frameworks (detectron2): see Installation section

### Model-Dataset Incompatibility
```bash
Error: Model resnet50 is not compatible with dataset BP
Compatible models for BP: ['yolov8n', 'yolov8s', 'yolov8m']
```
Use a compatible model or change the dataset.

## Performance Tips

1. **Use mixed precision** - Enabled by default, speeds up training 2-3x
2. **Adjust batch size** - Larger batches = faster training (if GPU memory allows)
3. **Use lightweight encoders** - `unet_lightweight` with MobileNetV2 is faster than ResNet
4. **Enable pin_memory** - Already enabled by default for CUDA
5. **Use cosine scheduler** - Often converges faster than step scheduler

## Extending the Pipeline

### Add a New Task

1. Create dataset loader in `datasets/`
2. Add task to `DATASETS` in `config.py`
3. Add compatible models to `MODELS` in `config.py`
4. Update `Trainer` class in `trainCVmodel.py` if needed

### Add a New Framework

1. Add framework support in `models/model_factory.py`
2. Implement `_create_<framework>_model()` method
3. Add models to `MODELS` in `config.py`

## Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision
- CUDA 11.x (recommended for GPU training)

See `requirements.txt` for full dependencies.

## Dataset Citations

If you use these datasets, please cite the original papers:

**PanNuke:**
```bibtex
@article{gamper2019pannuke,
  title={PanNuke: an open pan-cancer histology dataset for nuclei instance segmentation and classification},
  author={Gamper, Jevgenij and Koohbanani, Navid Alemi and Benet, Ksenija and Khuram, Ali and Rajpoot, Nasir},
  journal={European Congress on Digital Pathology},
  pages={11--19},
  year={2019}
}
```
**Microplastics:** Cite the source where you obtained the dataset (e.g., Kaggle uploader or original research paper).


## Citation

If you use this pipeline in your research, please cite the relevant dataset papers (above) and model papers.

## License

This project is for educational and research purposes.

---

**Ready to train!** Start with a preset and customize as needed:

```bash
python trainCVmodel.py --preset bp_detection
```
