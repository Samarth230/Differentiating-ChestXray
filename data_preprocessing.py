import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class MedicalImageDataset(Dataset):
    """
    Custom dataset class for medical images with ethical considerations
    """
    def __init__(self, image_paths: List[str], labels: List[str], 
                 transform=None, metadata: Optional[Dict] = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.metadata = metadata or {}
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        return {
            'image': image,
            'label': label,
            'path': image_path,
            'metadata': {k: v[idx] if isinstance(v, (list, np.ndarray)) else v 
                        for k, v in self.metadata.items()}
        }

class DataPreprocessor:
    """
    Comprehensive data preprocessing with ethical considerations
    """
    def __init__(self, img_size: int = 224, batch_size: int = 32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.label_encoder = LabelEncoder()
        
        self.train_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
                A.Blur(blur_limit=3),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.3),
                A.ElasticTransform(p=0.3),
            ], p=0.3),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        self.val_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        self.test_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def load_dataset(self, data_dir: str, metadata_file: Optional[str] = None) -> Tuple[List, List, Dict]:
        """
        Load dataset with metadata for ethical analysis
        """
        image_paths = []
        labels = []
        metadata = {}
        
        # Walk through data directory
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    # Extract label from directory structure
                    label = os.path.basename(root)
                    if label.lower() in ['normal', 'pneumonia', 'tuberculosis']:
                        image_paths.append(os.path.join(root, file))
                        labels.append(label)
        
        # Load metadata if available
        if metadata_file and os.path.exists(metadata_file):
            try:
                meta_df = pd.read_csv(metadata_file)
                # Extract relevant metadata columns
                for col in ['age', 'gender', 'ethnicity', 'hospital']:
                    if col in meta_df.columns:
                        metadata[col] = meta_df[col].values
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")
        
        return image_paths, labels, metadata
    
    def analyze_data_distribution(self, labels: List[str], metadata: Dict) -> Dict:
        """
        Analyze data distribution for bias detection
        """
        analysis = {}
        
        # Label distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        analysis['label_distribution'] = dict(zip(unique_labels, counts))
        
        # Metadata analysis
        for key, values in metadata.items():
            if isinstance(values, (list, np.ndarray)):
                unique_values, value_counts = np.unique(values, return_counts=True)
                analysis[f'{key}_distribution'] = dict(zip(unique_values, value_counts))
        
        # Calculate imbalance ratios
        total_samples = len(labels)
        for label, count in analysis['label_distribution'].items():
            analysis['label_distribution'][label] = {
                'count': count,
                'percentage': (count / total_samples) * 100
            }
        
        return analysis
    
    def split_data(self, image_paths: List[str], labels: List[str], 
                   metadata: Dict, test_size: float = 0.15, val_size: float = 0.15) -> Dict:
        """
        Split data into train/validation/test sets with stratification
        """
        # First split: separate test set
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            image_paths, labels, test_size=test_size, 
            stratify=labels, random_state=42
        )
        
        # Second split: separate validation set from training
        val_size_adjusted = val_size / (1 - test_size)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels, test_size=val_size_adjusted,
            stratify=train_val_labels, random_state=42
        )
        
        # Split metadata accordingly
        train_metadata = self._split_metadata(metadata, train_paths, image_paths)
        val_metadata = self._split_metadata(metadata, val_paths, image_paths)
        test_metadata = self._split_metadata(metadata, test_paths, image_paths)
        
        return {
            'train': {'paths': train_paths, 'labels': train_labels, 'metadata': train_metadata},
            'validation': {'paths': val_paths, 'labels': val_labels, 'metadata': val_metadata},
            'test': {'paths': test_paths, 'labels': test_labels, 'metadata': test_metadata}
        }
    
    def _split_metadata(self, metadata: Dict, selected_paths: List[str], all_paths: List[str]) -> Dict:
        """Helper function to split metadata based on selected paths"""
        if not metadata:
            return {}
        
        # Create a mapping from path to index
        path_to_idx = {path: idx for idx, path in enumerate(all_paths)}
        selected_indices = [path_to_idx[path] for path in selected_paths]
        
        split_metadata = {}
        for key, values in metadata.items():
            if isinstance(values, (list, np.ndarray)):
                split_metadata[key] = [values[idx] for idx in selected_indices]
            else:
                split_metadata[key] = values
        
        return split_metadata
    
    def create_dataloaders(self, data_splits: Dict) -> Dict[str, DataLoader]:
        """
        Create DataLoaders for all splits
        """
        dataloaders = {}
        
        # Create datasets
        train_dataset = MedicalImageDataset(
            data_splits['train']['paths'],
            data_splits['train']['labels'],
            transform=self.train_transform,
            metadata=data_splits['train']['metadata']
        )
        
        val_dataset = MedicalImageDataset(
            data_splits['validation']['paths'],
            data_splits['validation']['labels'],
            transform=self.val_transform,
            metadata=data_splits['validation']['metadata']
        )
        
        test_dataset = MedicalImageDataset(
            data_splits['test']['paths'],
            data_splits['test']['labels'],
            transform=self.test_transform,
            metadata=data_splits['test']['metadata']
        )
        
        # Create dataloaders
        dataloaders['train'] = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )
        dataloaders['validation'] = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )
        dataloaders['test'] = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )
        
        return dataloaders
    
    def detect_bias(self, data_splits: Dict) -> Dict:
        """
        Detect potential biases in the dataset
        """
        bias_report = {}
        
        for split_name, split_data in data_splits.items():
            bias_report[split_name] = {}
            
            # Analyze label distribution
            labels = split_data['labels']
            unique_labels, counts = np.unique(labels, return_counts=True)
            total = len(labels)
            
            bias_report[split_name]['label_balance'] = {
                label: {'count': int(count), 'percentage': float((count/total)*100)}
                for label, count in zip(unique_labels, counts)
            }
            
            # Analyze metadata for bias
            metadata = split_data['metadata']
            for key, values in metadata.items():
                if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                    unique_values, value_counts = np.unique(values, return_counts=True)
                    bias_report[split_name][f'{key}_distribution'] = {
                        str(val): {'count': int(count), 'percentage': float((count/len(values))*100)}
                        for val, count in zip(unique_values, value_counts)
                    }
        
        return bias_report

def main():
    """
    Example usage of the data preprocessing pipeline
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor(img_size=224, batch_size=32)
    
    # Example data directory structure:
    # data/
    # ├── normal/
    # ├── pneumonia/
    # └── tuberculosis/
    
    print("Medical Image Classification - Data Preprocessing")
    print("=" * 50)
    
    # Load dataset
    print("Loading dataset...")
    image_paths, labels, metadata = preprocessor.load_dataset("data/")
    
    print(f"Total images found: {len(image_paths)}")
    print(f"Labels: {set(labels)}")
    
    # Analyze data distribution
    print("\nAnalyzing data distribution...")
    distribution = preprocessor.analyze_data_distribution(labels, metadata)
    print("Label distribution:")
    for label, info in distribution['label_distribution'].items():
        print(f"  {label}: {info['count']} ({info['percentage']:.1f}%)")
    
    # Split data
    print("\nSplitting data...")
    data_splits = preprocessor.split_data(image_paths, labels, metadata)
    
    for split_name, split_data in data_splits.items():
        print(f"{split_name.capitalize()}: {len(split_data['paths'])} images")
    
    # Detect bias
    print("\nDetecting potential biases...")
    bias_report = preprocessor.detect_bias(data_splits)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    dataloaders = preprocessor.create_dataloaders(data_splits)
    
    print("Data preprocessing completed successfully!")
    print(f"Train batches: {len(dataloaders['train'])}")
    print(f"Validation batches: {len(dataloaders['validation'])}")
    print(f"Test batches: {len(dataloaders['test'])}")

if __name__ == "__main__":
    main()
