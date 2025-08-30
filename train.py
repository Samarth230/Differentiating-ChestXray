import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import DataPreprocessor
from model import create_model, get_model_summary
from utils import EarlyStopping, save_checkpoint, load_checkpoint

class MedicalImageTrainer:
    """
    Comprehensive trainer for medical image classification with ethical considerations
    """
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.dataloaders = None
        self.writer = None
        
        self.current_epoch = 0
        self.best_val_score = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        self.fairness_metrics = {}
        self.bias_detection_results = {}
        
        self._setup_model()
        self._setup_training_components()
        self._setup_logging()
    
    def _setup_model(self):
        """Initialize the model"""
        print("Setting up model...")
        self.model = create_model(self.config['model'])
        self.model = self.model.to(self.device)
        
        print(get_model_summary(self.model))
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def _setup_training_components(self):
        """Setup loss function, optimizer, and scheduler"""
        if self.config['training'].get('use_class_weights', True):
            class_weights = self._calculate_class_weights()
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        optimizer_config = (
            self.config.get('training', {}).get('optimizer')
            or self.config.get('optimizer')
        )
        if optimizer_config is None:
            raise KeyError("optimizer")
        if optimizer_config['type'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        elif optimizer_config['type'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        elif optimizer_config['type'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        
        scheduler_config = (
            self.config.get('training', {}).get('scheduler')
            or self.config.get('scheduler')
        )
        if scheduler_config is None:
            self.scheduler = None
            return
        if scheduler_config['type'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=scheduler_config['epochs'],
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_config['type'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config['gamma']
            )
        elif scheduler_config['type'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_config['factor'],
                patience=scheduler_config['patience'],
                verbose=True
            )
    
    def _setup_logging(self):
        """Setup TensorBoard logging"""
        log_dir = os.path.join(self.config['training']['output_dir'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
    
    def _calculate_class_weights(self):
        """Calculate class weights for imbalanced data"""
        # This would typically be calculated from the actual dataset
        # For now, we'll use a reasonable default
        class_weights = torch.tensor([1.0, 2.0, 2.5])  # Normal, Pneumonia, Tuberculosis
        return class_weights
    
    def prepare_data(self, data_dir: str, metadata_file: str = None):
        """Prepare data loaders"""
        print("Preparing data...")
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(
            img_size=self.config['data']['img_size'],
            batch_size=self.config['training']['batch_size']
        )
        
        # Load and preprocess data
        image_paths, labels, metadata = preprocessor.load_dataset(data_dir, metadata_file)
        
        if len(image_paths) == 0:
            raise ValueError("No images found in the specified directory")
        
        print(f"Found {len(image_paths)} images")
        print(f"Labels: {set(labels)}")
        
        # Analyze data distribution
        distribution = preprocessor.analyze_data_distribution(labels, metadata)
        print("Data distribution:")
        for label, info in distribution['label_distribution'].items():
            print(f"  {label}: {info['count']} ({info['percentage']:.1f}%)")
        
        # Split data
        data_splits = preprocessor.split_data(
            image_paths, labels, metadata,
            test_size=self.config['data']['test_size'],
            val_size=self.config['data']['val_size']
        )
        
        # Create dataloaders
        self.dataloaders = preprocessor.create_dataloaders(data_splits)
        
        # Detect bias
        self.bias_detection_results = preprocessor.detect_bias(data_splits)
        
        print("Data preparation completed!")
        for split_name, dataloader in self.dataloaders.items():
            print(f"  {split_name.capitalize()}: {len(dataloader.dataset)} images")
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(self.dataloaders['train'], desc=f'Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            labels = batch['label']
            
            # Convert string labels to indices
            label_indices = torch.tensor([
                ['normal', 'pneumonia', 'tuberculosis'].index(label.lower())
                for label in labels
            ]).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            logits = outputs['logits']
            
            # Calculate loss
            loss = self.criterion(logits, label_indices)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clipping', True):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training'].get('max_grad_norm', 1.0)
                )
            
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total_predictions += label_indices.size(0)
            correct_predictions += (predicted == label_indices).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_predictions / total_predictions:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.dataloaders['train'])
        epoch_accuracy = 100 * correct_predictions / total_predictions
        
        return epoch_loss, epoch_accuracy
    
    def validate_epoch(self, epoch: int):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in self.dataloaders['validation']:
                images = batch['image'].to(self.device)
                labels = batch['label']
                
                # Convert string labels to indices
                label_indices = torch.tensor([
                    ['normal', 'pneumonia', 'tuberculosis'].index(label.lower())
                    for label in labels
                ]).to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                logits = outputs['logits']
                
                # Calculate loss
                loss = self.criterion(logits, label_indices)
                
                # Statistics
                running_loss += loss.item()
                probabilities = torch.softmax(logits, dim=1)
                _, predicted = torch.max(logits, 1)
                total_predictions += label_indices.size(0)
                correct_predictions += (predicted == label_indices).sum().item()
                
                # Store predictions for metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(label_indices.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        epoch_loss = running_loss / len(self.dataloaders['validation'])
        epoch_accuracy = 100 * correct_predictions / total_predictions
        
        # Calculate additional metrics
        metrics = self._calculate_metrics(all_labels, all_predictions, all_probabilities)
        
        return epoch_loss, epoch_accuracy, metrics
    
    def _calculate_metrics(self, true_labels, predictions, probabilities):
        """Calculate comprehensive metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = 100 * (np.array(true_labels) == np.array(predictions)).mean()
        
        # Per-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average=None, labels=[0, 1, 2]
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
        # Macro averages
        metrics['macro_precision'] = np.mean(precision)
        metrics['macro_recall'] = np.mean(recall)
        metrics['macro_f1'] = np.mean(f1)
        
        # AUC-ROC for each class
        try:
            auc_scores = []
            for i in range(3):  # 3 classes
                if len(np.unique([label == i for label in true_labels])) > 1:
                    auc = roc_auc_score(
                        [label == i for label in true_labels],
                        [prob[i] for prob in probabilities]
                    )
                    auc_scores.append(auc)
                else:
                    auc_scores.append(0.0)
            metrics['auc_per_class'] = auc_scores
            metrics['macro_auc'] = np.mean(auc_scores)
        except:
            metrics['auc_per_class'] = [0.0, 0.0, 0.0]
            metrics['macro_auc'] = 0.0
        
        return metrics
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Total epochs: {self.config['training']['epochs']}")
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config['training'].get('patience', 10),
            min_delta=self.config['training'].get('min_delta', 1e-4)
        )
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Training phase
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation phase
            val_loss, val_acc, val_metrics = self.validate_epoch(epoch)
            
            # Learning rate scheduling
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['macro_f1'])
            else:
                self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            self.writer.add_scalar('Metrics/Macro_F1', val_metrics['macro_f1'], epoch)
            self.writer.add_scalar('Metrics/Macro_AUC', val_metrics['macro_auc'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.config['training']['epochs']}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Val F1: {val_metrics['macro_f1']:.4f}, Val AUC: {val_metrics['macro_auc']:.4f}")
            
            # Save best model
            if val_metrics['macro_f1'] > self.best_val_score:
                self.best_val_score = val_metrics['macro_f1']
                self._save_best_model(val_metrics)
            
            # Check early stopping
            if early_stopping(val_metrics['macro_f1']):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Training completed
        print("Training completed!")
        self._save_training_history()
        self.writer.close()
    
    def _save_best_model(self, metrics):
        """Save the best model"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_score': self.best_val_score,
            'metrics': metrics,
            'config': self.config
        }
        
        save_path = os.path.join(
            self.config['training']['output_dir'], 
            'best_model.pth'
        )
        save_checkpoint(checkpoint, save_path)
        print(f"Best model saved with F1 score: {self.best_val_score:.4f}")
    
    def _save_training_history(self):
        """Save training history and plots"""
        # Save metrics
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_score': self.best_val_score,
            'bias_detection': self.bias_detection_results
        }
        
        history_path = os.path.join(
            self.config['training']['output_dir'], 
            'training_history.json'
        )
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Create plots
        self._create_training_plots()
    
    def _create_training_plots(self):
        """Create and save training plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.train_accuracies, label='Train Accuracy')
        axes[0, 1].plot(self.val_accuracies, label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Bias detection plot
        if self.bias_detection_results:
            train_bias = self.bias_detection_results.get('train', {}).get('label_balance', {})
            if train_bias:
                labels = list(train_bias.keys())
                counts = [train_bias[label]['count'] for label in labels]
                axes[1, 0].bar(labels, counts)
                axes[1, 0].set_title('Training Data Distribution')
                axes[1, 0].set_ylabel('Number of Images')
                axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Learning rate plot
        if self.scheduler:
            lr_history = []
            for epoch in range(len(self.train_losses)):
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    lr_history.append(self.optimizer.param_groups[0]['lr'])
                else:
                    lr_history.append(self.scheduler.get_last_lr()[0])
            
            axes[1, 1].plot(lr_history)
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(
            self.config['training']['output_dir'], 
            'training_plots.png'
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to {plot_path}")

def main():
    """Main training script"""
    # Configuration
    config = {
        'data': {
            'img_size': 224,
            'test_size': 0.15,
            'val_size': 0.15
        },
        'model': {
            'model_type': 'efficientnet_b0',
            'num_classes': 3,
            'pretrained': True,
            'dropout_rate': 0.3,
            'use_ensemble': False,
            'use_fairness': True,
            'sensitive_attributes': ['age', 'gender']
        },
        'training': {
            'epochs': 50,
            'batch_size': 32,
            'optimizer': {
                'type': 'adamw',
                'lr': 1e-4,
                'weight_decay': 1e-4
            },
            'scheduler': {
                'type': 'cosine',
                'epochs': 50,
                'min_lr': 1e-6
            },
            'use_class_weights': True,
            'gradient_clipping': True,
            'max_grad_norm': 1.0,
            'patience': 15,
            'min_delta': 1e-4,
            'output_dir': 'outputs'
        }
    }
    
    # Create output directory
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    
    # Initialize trainer
    trainer = MedicalImageTrainer(config)
    
    # Prepare data (you need to specify your data directory)
    data_dir = "data/"  # Change this to your data directory
    if os.path.exists(data_dir):
        trainer.prepare_data(data_dir)
        
        # Start training
        trainer.train()
    else:
        print(f"Data directory {data_dir} not found!")
        print("Please create the data directory with the following structure:")
        print("data/")
        print("├── normal/")
        print("├── pneumonia/")
        print("└── tuberculosis/")

if __name__ == "__main__":
    main()
