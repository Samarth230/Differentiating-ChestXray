import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

from model import create_model
from utils import load_checkpoint
from utils import create_confusion_matrix

class MedicalImagePredictor:
    """
    Medical image predictor with uncertainty estimation and interpretability
    """
    def __init__(self, config: dict, model_path: str):
        self.config = config
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = None
        self.class_names = ['Normal', 'Pneumonia', 'Tuberculosis']
        self.class_colors = ['green', 'orange', 'red']
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model"""
        print("Loading trained model...")
        
        self.model = create_model(self.config['model'])
        self.model = self.model.to(self.device)
        
        checkpoint = load_checkpoint(self.model_path, self.model)
        
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Best validation score: {checkpoint.get('best_val_score', 'N/A')}")
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess a single image for prediction"""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        image = np.array(image)
        
        target_size = self.config['data']['img_size']
        image = cv2.resize(image, (target_size, target_size))
        
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        image = image.float()
        
        return image.to(self.device)
    
    def predict_single_image(self, image_path: str, return_attention: bool = False, 
                           return_uncertainty: bool = False) -> Dict:
        """Predict on a single image"""
        image_tensor = self.preprocess_image(image_path)
        
        with torch.no_grad():
            if return_attention or return_uncertainty:
                output = self.model(image_tensor, return_attention=return_attention, 
                                  return_uncertainty=return_uncertainty)
            else:
                output = self.model(image_tensor)
            
            logits = output['logits']
            probabilities = F.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
        
        result = {
            'predicted_class': self.class_names[predicted.item()],
            'predicted_class_idx': predicted.item(),
            'confidence': probabilities.max().item(),
            'probabilities': probabilities.cpu().numpy()[0].tolist(),
            'class_probabilities': dict(zip(self.class_names, probabilities.cpu().numpy()[0]))
        }
        
        if return_attention and 'attention_maps' in output:
            result['attention_maps'] = output['attention_maps'].cpu().numpy()
        
        if return_uncertainty and 'uncertainty' in output:
            result['uncertainty'] = output['uncertainty'].cpu().numpy()[0].item()
        
        return result
    
    def predict_with_uncertainty(self, image_path: str, num_samples: int = 10) -> Dict:
        """Predict with uncertainty estimation using Monte Carlo dropout"""
        image_tensor = self.preprocess_image(image_path)
        
        predictions = []
        uncertainties = []
        
        self.model.train()
        
        with torch.no_grad():
            for _ in range(num_samples):
                output = self.model(image_tensor, return_uncertainty=True)
                pred = F.softmax(output['logits'], dim=1)
                predictions.append(pred)
                uncertainties.append(output['uncertainty'])
        
        predictions = torch.stack(predictions)
        uncertainties = torch.stack(uncertainties)
        
        mean_pred = predictions.mean(dim=0)
        pred_uncertainty = predictions.var(dim=0).mean(dim=1, keepdim=True)
        model_uncertainty = uncertainties.mean(dim=0)
        
        _, predicted = torch.max(mean_pred, 1)
        
        result = {
            'predicted_class': self.class_names[predicted.item()],
            'predicted_class_idx': predicted.item(),
            'confidence': mean_pred.max().item(),
            'probabilities': mean_pred.cpu().numpy().tolist(),
            'class_probabilities': dict(zip(self.class_names, mean_pred.cpu().numpy())),
            'prediction_uncertainty': pred_uncertainty.cpu().numpy()[0].item(),
            'model_uncertainty': model_uncertainty.cpu().numpy()[0].item(),
            'total_uncertainty': (pred_uncertainty + model_uncertainty).cpu().numpy()[0].item()
        }
        
        self.model.eval()
        return result
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Predict on multiple images"""
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_single_image(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'predicted_class': 'Error',
                    'confidence': 0.0
                })
        
        return results
    
    def visualize_prediction(self, image_path: str, prediction_result: Dict, 
                           save_path: Optional[str] = None):
        """Visualize prediction with attention maps and confidence"""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        image = np.array(image)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image)
        axes[0].set_title(f'Original Image\nPredicted: {prediction_result["predicted_class"]}')
        axes[0].axis('off')
        
        classes = list(prediction_result['class_probabilities'].keys())
        probabilities = list(prediction_result['class_probabilities'].values())
        colors = [self.class_colors[self.class_names.index(cls)] for cls in classes]
        
        bars = axes[1].bar(classes, probabilities, color=colors, alpha=0.7)
        axes[1].set_title('Class Probabilities')
        axes[1].set_ylabel('Probability')
        axes[1].set_ylim(0, 1)
        
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{prob:.3f}', ha='center', va='bottom')
        
        if 'attention_maps' in prediction_result:
            attention = prediction_result['attention_maps'][0, 0].cpu().numpy()
            attention_resized = cv2.resize(attention, (image.shape[1], image.shape[0]))
            
            attention_colored = plt.cm.jet(attention_resized)[:, :, :3]
            overlay = 0.7 * image.astype(np.float32) / 255.0 + 0.3 * attention_colored
            overlay = np.clip(overlay, 0, 1)
            
            axes[2].imshow(overlay)
            axes[2].set_title('Attention Map Overlay')
            axes[2].axis('off')
        else:
            axes[2].text(0.5, 0.5, f'Confidence: {prediction_result["confidence"]:.3f}',
                        ha='center', va='center', transform=axes[2].transAxes,
                        fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            axes[2].set_title('Confidence Score')
            axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def create_prediction_report(self, predictions: List[Dict], save_path: str):
        """Create a comprehensive prediction report"""
        import pandas as pd
        
        report_data = []
        for pred in predictions:
            if 'error' not in pred:
                report_data.append({
                    'image_path': pred.get('image_path', 'N/A'),
                    'predicted_class': pred['predicted_class'],
                    'confidence': pred['confidence'],
                    'normal_prob': pred['class_probabilities'].get('Normal', 0),
                    'pneumonia_prob': pred['class_probabilities'].get('Pneumonia', 0),
                    'tuberculosis_prob': pred['class_probabilities'].get('Tuberculosis', 0)
                })
        
        if report_data:
            df = pd.DataFrame(report_data)
            
            df.to_csv(save_path, index=False)
            
            print(f"\nPrediction Report Summary:")
            print(f"Total images processed: {len(predictions)}")
            print(f"Successful predictions: {len(report_data)}")
            print(f"Errors: {len(predictions) - len(report_data)}")
            
            if len(report_data) > 0:
                print(f"\nClass distribution:")
                class_counts = df['predicted_class'].value_counts()
                for class_name, count in class_counts.items():
                    print(f"  {class_name}: {count}")
                
                print(f"\nConfidence statistics:")
                print(f"  Mean confidence: {df['confidence'].mean():.3f}")
                print(f"  Min confidence: {df['confidence'].min():.3f}")
                print(f"  Max confidence: {df['confidence'].max():.3f}")
            
            print(f"\nDetailed report saved to {save_path}")
        else:
            print("No successful predictions to report")
    
    def analyze_prediction_confidence(self, predictions: List[Dict], save_dir: str):
        """Analyze prediction confidence patterns"""
        os.makedirs(save_dir, exist_ok=True)
        
        valid_predictions = [p for p in predictions if 'error' not in p]
        
        if not valid_predictions:
            print("No valid predictions to analyze")
            return
        
        confidence_by_class = {}
        for pred in valid_predictions:
            class_name = pred['predicted_class']
            if class_name not in confidence_by_class:
                confidence_by_class[class_name] = []
            confidence_by_class[class_name].append(pred['confidence'])
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        all_confidences = [p['confidence'] for p in valid_predictions]
        axes[0].hist(all_confidences, bins=20, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Confidence Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Overall Confidence Distribution')
        axes[0].grid(True, alpha=0.3)
        
        for class_name, confidences in confidence_by_class.items():
            color = self.class_colors[self.class_names.index(class_name)]
            axes[1].hist(confidences, bins=15, alpha=0.7, label=class_name, 
                        color=color, edgecolor='black')
        
        axes[1].set_xlabel('Confidence Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Confidence Distribution by Class')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(save_dir, 'confidence_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confidence analysis saved to {plot_path}")
        
        confidence_stats = {}
        for class_name, confidences in confidence_by_class.items():
            confidence_stats[class_name] = {
                'count': len(confidences),
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
        
        stats_path = os.path.join(save_dir, 'confidence_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(confidence_stats, f, indent=2)
        
        print(f"Confidence statistics saved to {stats_path}")

def main():
    """Main prediction script"""
    config = {
        'data': {
            'img_size': 224
        },
        'model': {
            'model_type': 'efficientnet_b0',
            'num_classes': 3,
            'pretrained': True,
            'dropout_rate': 0.3,
            'use_ensemble': False,
            'use_fairness': True,
            'sensitive_attributes': ['age', 'gender']
        }
    }
    
    model_path = "outputs/best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please train the model first or specify the correct model path")
        return
    
    predictor = MedicalImagePredictor(config, model_path)
    
    print("Medical Image Classification - Prediction")
    print("=" * 50)
    
    image_path = "sample_image.jpg"
    
    if os.path.exists(image_path):
        print(f"Predicting on: {image_path}")
        
        result = predictor.predict_single_image(image_path)
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Class probabilities: {result['class_probabilities']}")
        
        uncertainty_result = predictor.predict_with_uncertainty(image_path, num_samples=10)
        print(f"\nUncertainty Analysis:")
        print(f"  Prediction uncertainty: {uncertainty_result['prediction_uncertainty']:.4f}")
        print(f"  Model uncertainty: {uncertainty_result['model_uncertainty']:.4f}")
        print(f"  Total uncertainty: {uncertainty_result['total_uncertainty']:.4f}")
        
        output_dir = "prediction_results"
        os.makedirs(output_dir, exist_ok=True)
        
        viz_path = os.path.join(output_dir, 'prediction_visualization.png')
        predictor.visualize_prediction(image_path, result, viz_path)
        
        report_path = os.path.join(output_dir, 'prediction_report.csv')
        predictor.create_prediction_report([result], report_path)
        
        predictor.analyze_prediction_confidence([result], output_dir)
        
    else:
        print(f"Image file not found: {image_path}")
        print("Please specify a valid image path for prediction")
        
        print("\nExample batch prediction:")
        sample_images = ["image1.jpg", "image2.jpg", "image3.jpg"]
        print(f"Sample image paths: {sample_images}")
        print("To use batch prediction, call:")
        print("  results = predictor.predict_batch(image_paths)")
        print("  predictor.create_prediction_report(results, 'batch_report.csv')")

if __name__ == "__main__":
    main()
