"""
Comprehensive fine-tuning pipeline for CV models.
This is the main entry point for training.

Usage:
    python trainCVmodel.py --dataset BP --model yolov8n --epochs 50
    python trainCVmodel.py --dataset Microplastics --model resnet50
    python trainCVmodel.py --dataset PanNuke --model mask_rcnn_r50
    python trainCVmodel.py --dataset TBM --model unet
    python trainCVmodel.py --preset bp_detection
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import json
from datetime import datetime

# Import configurations
from config import (
    get_dataset_config,
    get_model_config,
    validate_model_task_compatibility,
    get_compatible_models,
    TRAINING_CONFIG,
    PRESETS,
    get_preset,
)

# Import dataset and model factories
from datasets.dataset_factory import get_dataset
from models.model_factory import get_model


class Trainer:
    """Main trainer class for CV model fine-tuning."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        num_classes,
        task,
        save_dir,
        config,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        self.task = task
        self.save_dir = save_dir
        self.config = config
        
        # Setup criterion
        self.criterion = self._get_criterion()
        
        # Setup optimizer
        self.optimizer = self._get_optimizer()
        
        # Setup scheduler
        self.scheduler = self._get_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.early_stopping_counter = 0
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config["mixed_precision"] else None
    
    def _get_criterion(self):
        """Get loss function based on task."""
        if self.task == "classification":
            return nn.CrossEntropyLoss()
        elif self.task in ["semantic_segmentation", "instance_segmentation"]:
            return nn.CrossEntropyLoss()
        elif self.task == "object_detection":
            # Detection models usually have built-in loss
            return None
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def _get_optimizer(self):
        """Get optimizer from config."""
        opt_name = self.config["optimizer"].lower()
        lr = self.config["learning_rate"]
        weight_decay = self.config["weight_decay"]
        
        if opt_name == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "sgd":
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
    
    def _get_scheduler(self):
        """Get learning rate scheduler from config."""
        sched_name = self.config["scheduler"].lower()
        epochs = self.config["epochs"]
        
        if sched_name == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=epochs)
        elif sched_name == "step":
            return StepLR(self.optimizer, step_size=epochs//3, gamma=0.1)
        elif sched_name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5)
        else:
            return None
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            
            if self.task == "classification":
                targets = targets.to(self.device)
                
                # Forward pass
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{total_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
                
            elif self.task == "semantic_segmentation":
                targets = targets.to(self.device)
                
                # Forward pass
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        if isinstance(outputs, dict):
                            outputs = outputs['out']
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    if isinstance(outputs, dict):
                        outputs = outputs['out']
                    loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{total_loss/(batch_idx+1):.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total if total > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
        }
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")
            
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(self.device)
                
                if self.task == "classification":
                    targets = targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    pbar.set_postfix({
                        'loss': f'{total_loss/(batch_idx+1):.4f}',
                        'acc': f'{100.*correct/total:.2f}%'
                    })
                    
                elif self.task == "semantic_segmentation":
                    targets = targets.to(self.device)
                    outputs = self.model(inputs)
                    if isinstance(outputs, dict):
                        outputs = outputs['out']
                    loss = self.criterion(outputs, targets)
                    
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': f'{total_loss/(batch_idx+1):.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total if total > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
        }
    
    def train(self, epochs):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Starting training for {epochs} epochs")
        print(f"Task: {self.task}")
        print(f"Device: {self.device}")
        print(f"Save directory: {self.save_dir}")
        print(f"{'='*60}\n")
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            
            # Validate
            val_metrics = self.validate()
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_val_acc = val_metrics['accuracy']
                self.early_stopping_counter = 0
                self.save_checkpoint('best_model.pth')
                print(f"✓ New best model saved (val_loss: {val_metrics['loss']:.4f})")
            else:
                self.early_stopping_counter += 1
            
            # Early stopping
            if self.early_stopping_counter >= self.config["early_stopping_patience"]:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        # Save final model and history
        self.save_checkpoint('final_model.pth')
        self.save_history(history)
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"{'='*60}\n")
        
        return history
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)
    
    def save_history(self, history):
        """Save training history."""
        path = os.path.join(self.save_dir, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(history, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description='CV Model Fine-tuning Pipeline')
    parser.add_argument('--dataset', type=str, help='Dataset name (BP, Microplastics, PanNuke, TBM)')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--preset', type=str, help='Use a preset configuration')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default=None, help='Save directory')
    
    args = parser.parse_args()
    
    # Load preset if provided
    if args.preset:
        preset = get_preset(args.preset)
        dataset_name = preset['dataset']
        model_name = preset['model']
        print(f"Using preset: {args.preset}")
    else:
        if not args.dataset or not args.model:
            parser.error("Either --preset or both --dataset and --model must be provided")
        dataset_name = args.dataset
        model_name = args.model
    
    # Validate compatibility
    if not validate_model_task_compatibility(model_name, dataset_name):
        compatible = get_compatible_models(dataset_name)
        print(f"Error: Model {model_name} is not compatible with dataset {dataset_name}")
        print(f"Compatible models for {dataset_name}: {compatible}")
        sys.exit(1)
    
    # Get configurations
    dataset_config = get_dataset_config(dataset_name)
    model_config = get_model_config(model_name)
    
    # Override training config if provided
    training_config = TRAINING_CONFIG.copy()
    if args.epochs:
        training_config['epochs'] = args.epochs
    if args.batch_size:
        training_config['batch_size'] = args.batch_size
    if args.lr:
        training_config['learning_rate'] = args.lr
    if args.device:
        training_config['device'] = args.device
    
    # Setup device
    device = training_config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = args.save_dir or os.path.join(
        training_config['save_dir'],
        f"{dataset_name}_{model_name}_{timestamp}"
    )
    os.makedirs(save_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'dataset': dataset_name,
            'model': model_name,
            'dataset_config': dataset_config,
            'model_config': model_config,
            'training_config': training_config,
        }, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"Configuration")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Task: {dataset_config['task']}")
    print(f"Number of classes: {dataset_config['num_classes']}")
    print(f"Epochs: {training_config['epochs']}")
    print(f"Batch size: {training_config['batch_size']}")
    print(f"Learning rate: {training_config['learning_rate']}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset_dict = get_dataset(
        dataset_name=dataset_name,
        input_size=model_config['input_size'],
        batch_size=training_config['batch_size'],
        num_workers=training_config['num_workers'],
    )
    
    train_loader = dataset_dict['train_loader']
    val_loader = dataset_dict['val_loader']
    num_classes = dataset_dict['num_classes']
    
    print(f"Train samples: {len(dataset_dict['train_dataset'])}")
    print(f"Val samples: {len(dataset_dict['val_dataset'])}")
    
    # Create model
    print(f"\nCreating model: {model_name}")
    model = get_model(model_name, num_classes, device)
    print(f"Model created successfully")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_classes=num_classes,
        task=dataset_config['task'],
        save_dir=save_dir,
        config=training_config,
    )
    
    # Train
    history = trainer.train(training_config['epochs'])
    
    print(f"\nTraining completed. Results saved to: {save_dir}")


if __name__ == "__main__":
    main()
