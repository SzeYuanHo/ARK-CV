"""
Quick test script to verify all datasets can be loaded correctly.
Run this to ensure dataset paths and formats are correct.

Usage:
    python test_datasets.py
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset_factory import get_dataset
from config import DATASETS


def test_dataset(dataset_name):
    """Test loading a dataset."""
    print(f"\n{'='*60}")
    print(f"Testing {dataset_name} dataset")
    print(f"{'='*60}")
    
    try:
        config = DATASETS[dataset_name]
        print(f"Task: {config['task']}")
        print(f"Number of classes: {config['num_classes']}")
        print(f"Class names: {config['class_names']}")
        print(f"Data directory: {config['data_dir']}")
        print(f"Format: {config['label_format']}")
        
        # Check if directory exists
        if not os.path.exists(config['data_dir']):
            print(f"❌ ERROR: Data directory does not exist: {config['data_dir']}")
            return False
        
        print(f"✓ Data directory exists")
        
        # Try to load dataset
        print(f"\nAttempting to load dataset...")
        dataset_dict = get_dataset(
            dataset_name=dataset_name,
            input_size=224,  # Small size for quick testing
            batch_size=2,
            num_workers=0,  # Avoid multiprocessing issues on Windows
        )
        
        train_dataset = dataset_dict['train_dataset']
        val_dataset = dataset_dict['val_dataset']
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        print(f"  Total samples: {len(train_dataset) + len(val_dataset)}")
        
        # Try to load one sample
        print(f"\nTesting data loading...")
        sample = train_dataset[0]
        
        if config['task'] == 'object_detection':
            image, target = sample
            print(f"  Image shape: {image.shape}")
            print(f"  Number of boxes: {len(target['boxes'])}")
            if len(target['boxes']) > 0:
                print(f"  Box labels: {target['labels'].tolist()}")
                print(f"  ✓ Sample contains annotations")
            else:
                print(f"  ⚠ Sample has no annotations (background image)")
        elif config['task'] == 'classification':
            image, label = sample
            print(f"  Image shape: {image.shape}")
            print(f"  Label: {label} ({config['class_names'][label]})")
            print(f"  ✓ Sample loaded successfully")
        elif config['task'] in ['semantic_segmentation', 'instance_segmentation']:
            image, mask = sample
            print(f"  Image shape: {image.shape}")
            if isinstance(mask, dict):
                print(f"  Mask type: instance segmentation")
                print(f"  Number of instances: {len(mask.get('masks', []))}")
            else:
                print(f"  Mask shape: {mask.shape}")
                unique_classes = torch.unique(mask)
                print(f"  Unique classes in mask: {unique_classes.tolist()}")
            print(f"  ✓ Sample loaded successfully")
        
        print(f"\n✅ {dataset_name} dataset test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ {dataset_name} dataset test FAILED")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all datasets."""
    print("\n" + "="*60)
    print("CV Model Fine-Tuning Pipeline - Dataset Verification")
    print("="*60)
    
    import torch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    dataset_names = list(DATASETS.keys())
    results = {}
    
    for dataset_name in dataset_names:
        results[dataset_name] = test_dataset(dataset_name)
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    
    for dataset_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{dataset_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✅ All dataset tests PASSED!")
        print("You can now proceed to train models using trainCVmodel.py")
    else:
        print("❌ Some dataset tests FAILED")
        print("Please check the errors above and fix the issues.")
    print(f"{'='*60}\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    import torch  # Import here for the test
    sys.exit(main())
