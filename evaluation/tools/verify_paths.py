"""
Simple path verification script - checks if all dataset paths exist.
This doesn't require any external libraries.

Usage:
    python verify_paths.py
"""

import os

# Dataset configurations (copied from config.py for standalone verification)
DATASETS = {
    "BP": {
        "task": "object_detection",
        "num_classes": 2,
        "class_names": ["suitable", "further_review"],
        "data_dir": os.path.join(os.path.dirname(__file__), "datasets", "data", "BP", "BP_data"),
        "image_dir": "images",
        "label_dir": "Labels",
    },
    "Microplastics": {
        "task": "classification",
        "num_classes": 4,
        "class_names": ["algae I", "filament", "fragment", "pellet"],
        "data_dir": os.path.join(os.path.dirname(__file__), "datasets", "data", "Microplastics", "Microplastics_data"),
        "image_dir": None,
    },
    "PanNuke": {
        "task": "instance_segmentation",
        "num_classes": 5,
        "class_names": ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"],
        "data_dir": os.path.join(os.path.dirname(__file__), "datasets", "data", "PanNuke", "pannuke_data"),
        "image_dir": "images",
        "label_dir": "masks_class",
        "instance_dir": "masks_instance",
    },
    "TBM": {
        "task": "semantic_segmentation",
        "num_classes": 2,
        "class_names": ["background", "boundary"],
        "data_dir": os.path.join(os.path.dirname(__file__), "datasets", "data", "TBM", "TBM_data"),
        "image_dir": "input_image",
        "label_dir": "expert_label",
    },
}


def verify_directory(path, name):
    """Verify a directory exists and count files."""
    if not os.path.exists(path):
        print(f"  ❌ {name}: NOT FOUND")
        print(f"     Path: {path}")
        return False, 0
    
    if not os.path.isdir(path):
        print(f"  ❌ {name}: EXISTS but is not a directory")
        return False, 0
    
    # Count files
    try:
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        num_files = len(files)
        print(f"  ✓ {name}: {num_files} files")
        return True, num_files
    except Exception as e:
        print(f"  ❌ {name}: ERROR reading directory - {str(e)}")
        return False, 0


def verify_dataset(dataset_name):
    """Verify a dataset's paths."""
    print(f"\n{'='*60}")
    print(f"{dataset_name} Dataset")
    print(f"{'='*60}")
    
    config = DATASETS[dataset_name]
    print(f"Task: {config['task']}")
    print(f"Classes: {config['num_classes']} ({', '.join(config['class_names'])})")
    
    all_ok = True
    total_files = 0
    
    # Check data_dir
    print(f"\nData Directory:")
    exists, _ = verify_directory(config['data_dir'], "Root")
    all_ok = all_ok and exists
    
    if not exists:
        return False
    
    # Check image_dir
    if config['image_dir']:
        print(f"\nImage Directory:")
        img_path = os.path.join(config['data_dir'], config['image_dir'])
        exists, num_files = verify_directory(img_path, "Images")
        all_ok = all_ok and exists
        total_files += num_files
    else:
        # For ImageFolder, check class subdirectories
        print(f"\nClass Subdirectories (ImageFolder):")
        for class_name in config['class_names']:
            class_path = os.path.join(config['data_dir'], class_name)
            exists, num_files = verify_directory(class_path, f"  {class_name}")
            all_ok = all_ok and exists
            total_files += num_files
    
    # Check label_dir
    if config.get('label_dir'):
        print(f"\nLabel Directory:")
        label_path = os.path.join(config['data_dir'], config['label_dir'])
        exists, num_files = verify_directory(label_path, "Labels")
        all_ok = all_ok and exists
    
    # Check instance_dir (for PanNuke)
    if config.get('instance_dir'):
        print(f"\nInstance Directory:")
        inst_path = os.path.join(config['data_dir'], config['instance_dir'])
        exists, num_files = verify_directory(inst_path, "Instances")
        all_ok = all_ok and exists
    
    print(f"\nTotal data files: {total_files}")
    
    if all_ok:
        print(f"\n✅ {dataset_name} paths verified successfully")
    else:
        print(f"\n❌ {dataset_name} has missing paths")
    
    return all_ok


def main():
    """Verify all dataset paths."""
    print("="*60)
    print("CV Model Fine-Tuning Pipeline - Path Verification")
    print("="*60)
    print("\nThis script verifies that all dataset directories exist.")
    print("It does NOT require PyTorch or other dependencies.\n")
    
    results = {}
    for dataset_name in DATASETS.keys():
        results[dataset_name] = verify_dataset(dataset_name)
    
    # Summary
    print(f"\n{'='*60}")
    print("Verification Summary")
    print(f"{'='*60}")
    
    for dataset_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{dataset_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✅ All dataset paths verified!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Test dataset loading: python test_datasets.py")
        print("3. Start training: python trainCVmodel.py --preset <preset_name>")
    else:
        print("❌ Some dataset paths are missing")
        print("\nPlease ensure all datasets are in the correct location:")
        print("  datasets/data/<DatasetName>/")
    print(f"{'='*60}\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
