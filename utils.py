import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class EarlyStopping:
    """
    Early stopping utility to prevent overfitting
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_score: float, model: torch.nn.Module = None) -> bool:
        """
        Check if training should stop early
        Returns True if training should stop
        """
        if self.best_score is None:
            self.best_score = val_score
            if self.restore_best_weights and model is not None:
                self.best_weights = model.state_dict().copy()
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and model is not None and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            if self.restore_best_weights and model is not None:
                self.best_weights = model.state_dict().copy()
        
        return False

def save_checkpoint(checkpoint: Dict, filepath: str):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, 
                   scheduler: torch.optim.lr_scheduler._LRScheduler = None) -> Dict:
    """Load model checkpoint with PyTorch 2.6+ compatibility"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    try:
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=True)
    except Exception as e:
        if "WeightsUnpickler error" in str(e) or "numpy._core.multiarray.scalar" in str(e):
            print("⚠️  Checkpoint contains numpy objects. Loading with weights_only=False...")
            print("   This is safe for trusted checkpoints from your own training.")
            try:
                checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
            except Exception as e2:
                raise RuntimeError(f"Failed to load checkpoint even with weights_only=False: {e2}")
        else:
            raise e
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"✅ Checkpoint loaded from {filepath}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Best validation score: {checkpoint.get('best_val_score', 'N/A')}")
    
    return checkpoint

def create_confusion_matrix(y_true: List, y_pred: List, class_names: List[str], 
                           save_path: Optional[str] = None) -> np.ndarray:
    """Create and optionally save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    return cm

def plot_roc_curves(y_true: List, y_prob: np.ndarray, class_names: List[str], 
                    save_path: Optional[str] = None):
    """Plot ROC curves for all classes"""
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(class_names):
        # Convert to binary classification for each class
        y_true_binary = [1 if label == i else 0 for label in y_true]
        y_prob_binary = y_prob[:, i]
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob_binary)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    plt.show()

def plot_precision_recall_curves(y_true: List, y_prob: np.ndarray, class_names: List[str], 
                                save_path: Optional[str] = None):
    """Plot precision-recall curves for all classes"""
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(class_names):
        # Convert to binary classification for each class
        y_true_binary = [1 if label == i else 0 for label in y_true]
        y_prob_binary = y_prob[:, i]
        
        precision, recall, _ = precision_recall_curve(y_true_binary, y_prob_binary)
        avg_precision = average_precision_score(y_true_binary, y_prob_binary)
        
        plt.plot(recall, precision, label=f'{class_name} (AP = {avg_precision:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-recall curves saved to {save_path}")
    
    plt.show()

def calculate_fairness_metrics(predictions: List, true_labels: List, 
                             sensitive_attributes: Dict[str, List]) -> Dict:
    """Calculate fairness metrics across sensitive attributes"""
    fairness_metrics = {}
    
    for attr_name, attr_values in sensitive_attributes.items():
        fairness_metrics[attr_name] = {}
        
        # Get unique values for this attribute
        unique_values = list(set(attr_values))
        
        for value in unique_values:
            # Find indices where this attribute has this value
            indices = [i for i, v in enumerate(attr_values) if v == value]
            
            if len(indices) == 0:
                continue
            
            # Calculate metrics for this group
            group_predictions = [predictions[i] for i in indices]
            group_true_labels = [true_labels[i] for i in indices]
            
            # Accuracy for this group
            group_accuracy = sum(1 for p, t in zip(group_predictions, group_true_labels) if p == t) / len(indices)
            
            # F1 score for this group
            from sklearn.metrics import f1_score
            try:
                group_f1 = f1_score(group_true_labels, group_predictions, average='macro')
            except:
                group_f1 = 0.0
            
            fairness_metrics[attr_name][str(value)] = {
                'count': len(indices),
                'accuracy': group_accuracy,
                'f1_score': group_f1
            }
    
    return fairness_metrics

def detect_bias_in_predictions(predictions: List, true_labels: List, 
                              sensitive_attributes: Dict[str, List]) -> Dict:
    """Detect potential bias in model predictions"""
    bias_report = {}
    
    # Calculate overall metrics
    overall_accuracy = sum(1 for p, t in zip(predictions, true_labels) if p == t) / len(predictions)
    
    for attr_name, attr_values in sensitive_attributes.items():
        bias_report[attr_name] = {}
        
        # Get unique values for this attribute
        unique_values = list(set(attr_values))
        
        for value in unique_values:
            # Find indices where this attribute has this value
            indices = [i for i, v in enumerate(attr_values) if v == value]
            
            if len(indices) == 0:
                continue
            
            # Calculate metrics for this group
            group_predictions = [predictions[i] for i in indices]
            group_true_labels = [true_labels[i] for i in indices]
            
            group_accuracy = sum(1 for p, t in zip(group_predictions, group_true_labels) if p == t) / len(indices)
            
            # Calculate bias (difference from overall performance)
            bias = group_accuracy - overall_accuracy
            
            bias_report[attr_name][str(value)] = {
                'count': len(indices),
                'accuracy': group_accuracy,
                'bias': bias,
                'bias_percentage': (bias / overall_accuracy) * 100 if overall_accuracy > 0 else 0
            }
    
    return bias_report

def create_ethical_analysis_report(predictions: List, true_labels: List, 
                                  sensitive_attributes: Dict[str, List], 
                                  save_path: Optional[str] = None) -> str:
    """Create a comprehensive ethical analysis report"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ETHICAL ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Overall performance
    overall_accuracy = sum(1 for p, t in zip(predictions, true_labels) if p == t) / len(predictions)
    report_lines.append(f"Overall Model Performance:")
    report_lines.append(f"  Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    report_lines.append("")
    
    # Fairness metrics
    fairness_metrics = calculate_fairness_metrics(predictions, true_labels, sensitive_attributes)
    bias_report = detect_bias_in_predictions(predictions, true_labels, sensitive_attributes)
    
    report_lines.append("Fairness Analysis Across Sensitive Attributes:")
    report_lines.append("-" * 50)
    
    for attr_name in fairness_metrics:
        report_lines.append(f"\n{attr_name.upper()}:")
        report_lines.append(f"  {'Value':<15} {'Count':<8} {'Accuracy':<10} {'F1-Score':<10} {'Bias':<10}")
        report_lines.append(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
        
        for value, metrics in fairness_metrics[attr_name].items():
            bias_info = bias_report[attr_name].get(value, {})
            bias = bias_info.get('bias', 0)
            bias_str = f"{bias:+.3f}"
            
            report_lines.append(
                f"  {str(value):<15} {metrics['count']:<8} "
                f"{metrics['accuracy']:<10.3f} {metrics['f1_score']:<10.3f} {bias_str:<10}"
            )
    
    # Bias summary
    report_lines.append("\n" + "=" * 80)
    report_lines.append("BIAS SUMMARY")
    report_lines.append("=" * 80)
    
    for attr_name in bias_report:
        report_lines.append(f"\n{attr_name.upper()}:")
        max_bias = max(abs(metrics['bias']) for metrics in bias_report[attr_name].values())
        max_bias_group = max(bias_report[attr_name].items(), key=lambda x: abs(x[1]['bias']))
        
        report_lines.append(f"  Maximum bias: {max_bias:.4f}")
        report_lines.append(f"  Most biased group: {max_bias_group[0]} (bias: {max_bias_group[1]['bias']:+.4f})")
        
        if max_bias > 0.05:  # 5% threshold
            report_lines.append(f"  ⚠️  HIGH BIAS DETECTED - Consider bias mitigation strategies")
        elif max_bias > 0.02:  # 2% threshold
            report_lines.append(f"  ⚠️  MODERATE BIAS DETECTED - Monitor closely")
        else:
            report_lines.append(f"  ✅  LOW BIAS - Performance appears fair across groups")
    
    # Recommendations
    report_lines.append("\n" + "=" * 80)
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("=" * 80)
    
    report_lines.append("\n1. Data Collection:")
    report_lines.append("   - Ensure balanced representation across all sensitive attributes")
    report_lines.append("   - Collect metadata on potential sources of bias")
    
    report_lines.append("\n2. Model Training:")
    report_lines.append("   - Use fairness-aware training techniques")
    report_lines.append("   - Implement adversarial debiasing")
    report_lines.append("   - Consider ensemble methods for robustness")
    
    report_lines.append("\n3. Evaluation:")
    report_lines.append("   - Regularly monitor fairness metrics")
    report_lines.append("   - Test on diverse demographic groups")
    report_lines.append("   - Validate with domain experts")
    
    report_lines.append("\n4. Deployment:")
    report_lines.append("   - Implement continuous monitoring")
    report_lines.append("   - Provide explanations for predictions")
    report_lines.append("   - Establish feedback mechanisms")
    
    report = "\n".join(report_lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Ethical analysis report saved to {save_path}")
    
    return report

def save_metrics_to_csv(metrics: Dict, save_path: str):
    """Save metrics to CSV format"""
    # Flatten nested metrics
    flattened_metrics = {}
    
    def flatten_dict(d, prefix=''):
        for k, v in d.items():
            if isinstance(v, dict):
                flatten_dict(v, prefix + k + '_')
            else:
                flattened_metrics[prefix + k] = v
    
    flatten_dict(metrics)
    
    # Convert to DataFrame and save
    df = pd.DataFrame([flattened_metrics])
    df.to_csv(save_path, index=False)
    print(f"Metrics saved to {save_path}")

def create_performance_summary(predictions: List, true_labels: List, 
                              probabilities: np.ndarray, class_names: List[str],
                              save_dir: str):
    """Create comprehensive performance summary"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Confusion matrix
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    create_confusion_matrix(true_labels, predictions, class_names, cm_path)
    
    # ROC curves
    roc_path = os.path.join(save_dir, 'roc_curves.png')
    plot_roc_curves(true_labels, probabilities, class_names, roc_path)
    
    # Precision-recall curves
    pr_path = os.path.join(save_dir, 'precision_recall_curves.png')
    plot_precision_recall_curves(true_labels, probabilities, class_names, pr_path)
    
    # Classification report
    report = classification_report(true_labels, predictions, target_names=class_names, output_dict=True)
    report_path = os.path.join(save_dir, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Performance summary saved to {save_dir}")

if __name__ == "__main__":
    # Example usage
    print("Utility functions for medical image classification")
    print("This module provides:")
    print("- Early stopping functionality")
    print("- Checkpoint management")
    print("- Performance visualization")
    print("- Fairness and bias analysis")
    print("- Ethical reporting tools")
