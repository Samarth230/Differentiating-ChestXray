import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import DataPreprocessor
from model import create_model
from utils import load_checkpoint
from utils import (create_confusion_matrix, plot_roc_curves, plot_precision_recall_curves,
                   create_ethical_analysis_report, create_performance_summary,
                   calculate_fairness_metrics, detect_bias_in_predictions)

class ModelEvaluator:
    """
    Comprehensive model evaluator with ethical considerations
    """
    def __init__(self, config: dict, model_path: str):
        self.config = config
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = None
        self.dataloaders = None
        self.predictions = []
        self.true_labels = []
        self.probabilities = []
        self.image_paths = []
        self.metadata = {}
        
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
        print(f"Training completed at epoch: {checkpoint.get('epoch', 'N/A')}")
    
    def prepare_test_data(self, data_dir: str, metadata_file: str = None):
        """Prepare test data"""
        print("Preparing test data...")
        
        preprocessor = DataPreprocessor(
            img_size=self.config['data']['img_size'],
            batch_size=self.config['training']['batch_size']
        )
        
        image_paths, labels, metadata = preprocessor.load_dataset(data_dir, metadata_file)
        
        if len(image_paths) == 0:
            raise ValueError("No images found in the specified directory")
        
        print(f"Found {len(image_paths)} images")
        print(f"Labels: {set(labels)}")
        
        data_splits = preprocessor.split_data(
            image_paths, labels, metadata,
            test_size=self.config['data']['test_size'],
            val_size=self.config['data']['val_size']
        )
        
        self.dataloaders = preprocessor.create_dataloaders(data_splits)
        
        self.metadata = metadata
        
        print("Test data preparation completed!")
    
    def evaluate_model(self):
        """Evaluate the model on test data"""
        print("Evaluating model on test data...")
        
        if self.dataloaders is None:
            raise ValueError("Test data not prepared. Call prepare_test_data() first.")
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(self.dataloaders['test'], desc="Evaluating"):
                images = batch['image'].to(self.device)
                labels = batch['label']
                paths = batch['path']
                
                outputs = self.model(images)
                logits = outputs['logits']
                
                probabilities = F.softmax(logits, dim=1)
                _, predicted = torch.max(logits, 1)
                
                label_indices = [
                    ['normal', 'pneumonia', 'tuberculosis'].index(label.lower())
                    for label in labels
                ]
                
                self.predictions.extend(predicted.cpu().numpy())
                self.true_labels.extend(label_indices)
                self.probabilities.extend(probabilities.cpu().numpy())
                self.image_paths.extend(paths)
        
        print(f"Evaluation completed on {len(self.predictions)} samples")
    
    def calculate_metrics(self) -> dict:
        """Calculate comprehensive evaluation metrics"""
        print("Calculating evaluation metrics...")
        
        metrics = {}
        
        metrics['accuracy'] = accuracy_score(self.true_labels, self.predictions)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            self.true_labels, self.predictions, average=None, labels=[0, 1, 2]
        )
        
        metrics['precision_per_class'] = precision.tolist()
        metrics['recall_per_class'] = recall.tolist()
        metrics['f1_per_class'] = f1.tolist()
        metrics['support_per_class'] = support.tolist()
        
        metrics['macro_precision'] = np.mean(precision)
        metrics['macro_recall'] = np.mean(recall)
        metrics['macro_f1'] = np.mean(f1)
        
        metrics['weighted_precision'] = np.average(precision, weights=support)
        metrics['weighted_recall'] = np.average(recall, weights=support)
        metrics['weighted_f1'] = np.average(f1, weights=support)
        
        class_names = ['Normal', 'Pneumonia', 'Tuberculosis']
        auc_scores = []
        
        for i in range(3):
            try:
                if len(np.unique([label == i for label in self.true_labels])) > 1:
                    auc = roc_auc_score(
                        [label == i for label in self.true_labels],
                        [prob[i] for prob in self.probabilities]
                    )
                    auc_scores.append(auc)
                else:
                    auc_scores.append(0.0)
            except:
                auc_scores.append(0.0)
        
        metrics['auc_per_class'] = auc_scores
        metrics['macro_auc'] = np.mean(auc_scores)
        metrics['weighted_auc'] = np.average(auc_scores, weights=support)
        
        cm = confusion_matrix(self.true_labels, self.predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        metrics['per_class_accuracy'] = per_class_accuracy.tolist()
        
        return metrics
    
    def analyze_fairness(self) -> dict:
        """Analyze fairness across sensitive attributes"""
        print("Analyzing fairness and bias...")
        
        if not self.metadata:
            print("No metadata available for fairness analysis")
            return {}
        
        fairness_metrics = calculate_fairness_metrics(
            self.predictions, self.true_labels, self.metadata
        )
        
        bias_report = detect_bias_in_predictions(
            self.predictions, self.true_labels, self.metadata
        )
        
        ethical_report = create_ethical_analysis_report(
            self.predictions, self.true_labels, self.metadata
        )
        
        return {
            'fairness_metrics': fairness_metrics,
            'bias_report': bias_report,
            'ethical_report': ethical_report
        }
    
    def create_visualizations(self, save_dir: str):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        class_names = ['Normal', 'Pneumonia', 'Tuberculosis']
        
        y_true = np.array(self.true_labels)
        y_pred = np.array(self.predictions)
        y_prob = np.array(self.probabilities)
        
        cm_path = os.path.join(save_dir, 'confusion_matrix.png')
        create_confusion_matrix(y_true, y_pred, class_names, cm_path)
        
        roc_path = os.path.join(save_dir, 'roc_curves.png')
        plot_roc_curves(y_true, y_prob, class_names, roc_path)
        
        pr_path = os.path.join(save_dir, 'precision_recall_curves.png')
        plot_precision_recall_curves(y_true, y_prob, class_names, pr_path)
        
        create_performance_summary(y_true, y_pred, y_prob, class_names, save_dir)
        
        print(f"Visualizations saved to {save_dir}")
    
    def generate_detailed_report(self, save_dir: str):
        """Generate comprehensive evaluation report"""
        print("Generating detailed evaluation report...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        metrics = self.calculate_metrics()
        
        fairness_results = self.analyze_fairness()
        
        metrics_path = os.path.join(save_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        fairness_path = os.path.join(save_dir, 'fairness_analysis.json')
        with open(fairness_path, 'w') as f:
            json.dump(fairness_results, f, indent=2)
        
        if 'ethical_report' in fairness_results:
            ethical_path = os.path.join(save_dir, 'ethical_analysis_report.txt')
            with open(ethical_path, 'w') as f:
                f.write(fairness_results['ethical_report'])
        
        self._create_summary_report(metrics, fairness_results, save_dir)
        
        print(f"Detailed report saved to {save_dir}")
    
    def _create_summary_report(self, metrics: dict, fairness_results: dict, save_dir: str):
        """Create a human-readable summary report"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MODEL EVALUATION SUMMARY REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        report_lines.append("OVERALL PERFORMANCE")
        report_lines.append("-" * 30)
        report_lines.append(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        report_lines.append(f"Macro F1: {metrics['macro_f1']:.4f}")
        report_lines.append(f"Macro AUC: {metrics['macro_auc']:.4f}")
        report_lines.append("")
        
        class_names = ['Normal', 'Pneumonia', 'Tuberculosis']
        report_lines.append("PER-CLASS PERFORMANCE")
        report_lines.append("-" * 30)
        report_lines.append(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
        report_lines.append(f"{'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        
        for i, class_name in enumerate(class_names):
            precision = metrics['precision_per_class'][i]
            recall = metrics['recall_per_class'][i]
            f1 = metrics['f1_per_class'][i]
            auc = metrics['auc_per_class'][i]
            
            report_lines.append(
                f"{class_name:<15} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {auc:<10.3f}"
            )
        
        report_lines.append("")
        
        if fairness_results and 'bias_report' in fairness_results:
            report_lines.append("FAIRNESS SUMMARY")
            report_lines.append("-" * 30)
            
            for attr_name, attr_bias in fairness_results['bias_report'].items():
                max_bias = max(abs(metrics['bias']) for metrics in attr_bias.values())
                report_lines.append(f"{attr_name}: Maximum bias = {max_bias:.4f}")
                
                if max_bias > 0.05:
                    report_lines.append(f"  ⚠️  HIGH BIAS DETECTED")
                elif max_bias > 0.02:
                    report_lines.append(f"  ⚠️  MODERATE BIAS DETECTED")
                else:
                    report_lines.append(f"  ✅  LOW BIAS")
        
        summary_path = os.path.join(save_dir, 'evaluation_summary.txt')
        with open(summary_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print('\n'.join(report_lines))
    
    def analyze_predictions(self, save_dir: str):
        """Analyze individual predictions and errors"""
        print("Analyzing individual predictions...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        analysis_data = []
        
        for i, (pred, true, prob, path) in enumerate(zip(
            self.predictions, self.true_labels, self.probabilities, self.image_paths
        )):
            class_names = ['Normal', 'Pneumonia', 'Tuberculosis']
            
            analysis_data.append({
                'image_path': path,
                'true_label': class_names[true],
                'predicted_label': class_names[pred],
                'confidence': max(prob),
                'is_correct': pred == true,
                'true_probability': prob[true],
                'predicted_probability': prob[pred]
            })
        
        df = pd.DataFrame(analysis_data)
        
        analysis_path = os.path.join(save_dir, 'prediction_analysis.csv')
        df.to_csv(analysis_path, index=False)
        
        errors = df[~df['is_correct']]
        if len(errors) > 0:
            error_path = os.path.join(save_dir, 'error_analysis.csv')
            errors.to_csv(error_path, index=False)
            
            print(f"Error analysis saved to {error_path}")
            print(f"Total errors: {len(errors)}")
            print(f"Error rate: {len(errors)/len(df)*100:.2f}%")
        
        print(f"Prediction analysis saved to {analysis_path}")

def main():
    """Main evaluation script"""
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
            'batch_size': 32
        }
    }
    
    model_path = "outputs/best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please train the model first or specify the correct model path")
        return
    
    evaluator = ModelEvaluator(config, model_path)
    
    data_dir = "data/"
    if os.path.exists(data_dir):
        evaluator.prepare_test_data(data_dir)
        
        evaluator.evaluate_model()
        
        output_dir = "evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        evaluator.create_visualizations(output_dir)
        evaluator.generate_detailed_report(output_dir)
        evaluator.analyze_predictions(output_dir)
        
        print(f"\nEvaluation completed! Results saved to {output_dir}")
        
    else:
        print(f"Data directory {data_dir} not found!")
        print("Please specify the correct data directory path")

if __name__ == "__main__":
    main()
