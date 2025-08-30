import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from typing import Dict, Tuple, Optional
import numpy as np

class AttentionModule(nn.Module):
    """
    Attention mechanism to help the model focus on relevant regions
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super(AttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return self.sigmoid(out).view(x.size(0), x.size(1), 1, 1)

class MedicalImageClassifier(nn.Module):
    """
    Main model architecture for medical image classification
    Uses EfficientNet with attention mechanisms and ethical considerations
    """
    def __init__(self, num_classes: int = 3, model_name: str = 'efficientnet_b0', 
                 pretrained: bool = True, dropout_rate: float = 0.3):
        super(MedicalImageClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pretrained backbone
        if model_name.startswith('efficientnet'):
            self.backbone = timm.create_model(
                model_name, 
                pretrained=pretrained, 
                num_classes=0,  # Remove classifier
                global_pool=''
            )
            # Get feature dimensions
            if model_name == 'efficientnet_b0':
                feature_dim = 1280
            elif model_name == 'efficientnet_b1':
                feature_dim = 1280
            elif model_name == 'efficientnet_b2':
                feature_dim = 1408
            else:
                feature_dim = 1280  # Default
        else:
            # Fallback to ResNet
            self.backbone = models.resnet50(pretrained=pretrained)
            # Remove the final classification layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            feature_dim = 2048
        
        # Attention mechanism
        self.attention = AttentionModule(feature_dim)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head with regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_attention: bool = False, return_uncertainty: bool = False):
        """
        Forward pass with optional attention maps and uncertainty
        """
        # Ensure input is float32 for consistency
        x = x.float()
        
        # Extract features
        features = self.backbone(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Global pooling
        pooled_features = self.global_pool(attended_features).squeeze(-1).squeeze(-1)
        
        # Classification
        logits = self.classifier(pooled_features)
        
        # Uncertainty estimation
        uncertainty = None
        if return_uncertainty:
            uncertainty = self.uncertainty_head(pooled_features)
        
        # Prepare output
        output = {'logits': logits}
        
        if return_attention:
            output['attention_maps'] = attention_weights
        
        if return_uncertainty:
            output['uncertainty'] = uncertainty
        
        return output
    
    def get_attention_maps(self, x):
        """Extract attention maps for interpretability"""
        with torch.no_grad():
            features = self.backbone(x)
            attention_weights = self.attention(features)
            return attention_weights
    
    def predict_with_uncertainty(self, x, num_samples: int = 10):
        """
        Monte Carlo dropout for uncertainty estimation
        """
        self.train()  # Enable dropout
        predictions = []
        uncertainties = []
        
        for _ in range(num_samples):
            with torch.no_grad():
                output = self.forward(x, return_uncertainty=True)
                pred = F.softmax(output['logits'], dim=1)
                predictions.append(pred)
                uncertainties.append(output['uncertainty'])
        
        # Calculate mean and variance
        predictions = torch.stack(predictions)
        uncertainties = torch.stack(uncertainties)
        
        mean_pred = predictions.mean(dim=0)
        pred_uncertainty = predictions.var(dim=0).mean(dim=1, keepdim=True)
        model_uncertainty = uncertainties.mean(dim=0)
        
        return {
            'prediction': mean_pred,
            'prediction_uncertainty': pred_uncertainty,
            'model_uncertainty': model_uncertainty,
            'total_uncertainty': pred_uncertainty + model_uncertainty
        }

class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for improved robustness and fairness
    """
    def __init__(self, models: list, weights: Optional[list] = None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            # Equal weights
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Weights must match number of models"
            self.weights = weights
    
    def forward(self, x):
        outputs = []
        for i, model in enumerate(self.models):
            output = model(x)
            if isinstance(output, dict):
                outputs.append(output['logits'] * self.weights[i])
            else:
                outputs.append(output * self.weights[i])
        
        # Weighted average
        ensemble_output = torch.stack(outputs).sum(dim=0)
        return {'logits': ensemble_output}

class FairnessAwareModel(nn.Module):
    """
    Model wrapper that includes fairness constraints and bias mitigation
    """
    def __init__(self, base_model: nn.Module, sensitive_attributes: list):
        super(FairnessAwareModel, self).__init__()
        self.base_model = base_model
        self.sensitive_attributes = sensitive_attributes
        
        # Adversarial heads for fairness
        self.adversarial_heads = nn.ModuleDict({
            attr: nn.Sequential(
                nn.Linear(base_model.classifier[-1].out_features, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ) for attr in sensitive_attributes
        })
    
    def forward(self, x, metadata=None, return_attention=False, return_uncertainty=False):
        # Base prediction with optional parameters
        if hasattr(self.base_model, 'forward') and callable(getattr(self.base_model, 'forward', None)):
            # Check if base model supports these parameters
            import inspect
            sig = inspect.signature(self.base_model.forward)
            if 'return_attention' in sig.parameters and 'return_uncertainty' in sig.parameters:
                base_output = self.base_model(x, return_attention=return_attention, return_uncertainty=return_uncertainty)
            else:
                base_output = self.base_model(x)
        else:
            base_output = self.base_model(x)
        
        # Adversarial predictions for fairness
        adversarial_outputs = {}
        if metadata is not None:
            for attr in self.sensitive_attributes:
                if attr in metadata:
                    adv_logits = self.adversarial_heads[attr](base_output['logits'])
                    adversarial_outputs[f'adversarial_{attr}'] = adv_logits
        
        output = base_output.copy()
        output.update(adversarial_outputs)
        
        return output

def create_model(config: Dict) -> nn.Module:
    """
    Factory function to create models with different configurations
    """
    model_type = config.get('model_type', 'efficientnet_b0')
    num_classes = config.get('num_classes', 3)
    pretrained = config.get('pretrained', True)
    dropout_rate = config.get('dropout_rate', 0.3)
    use_ensemble = config.get('use_ensemble', False)
    use_fairness = config.get('use_fairness', False)
    sensitive_attributes = config.get('sensitive_attributes', [])
    
    if use_ensemble:
        # Create multiple models with different architectures
        models_list = []
        for arch in ['efficientnet_b0', 'efficientnet_b1', 'resnet50']:
            model = MedicalImageClassifier(
                num_classes=num_classes,
                model_name=arch,
                pretrained=pretrained,
                dropout_rate=dropout_rate
            )
            models_list.append(model)
        
        base_model = EnsembleModel(models_list)
    else:
        base_model = MedicalImageClassifier(
            num_classes=num_classes,
            model_name=model_type,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    
    if use_fairness:
        model = FairnessAwareModel(base_model, sensitive_attributes)
    else:
        model = base_model
    
    return model

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }

def get_model_summary(model: nn.Module) -> str:
    """Get a summary of the model architecture"""
    summary = []
    summary.append("=" * 60)
    summary.append("MODEL ARCHITECTURE SUMMARY")
    summary.append("=" * 60)
    
    # Model info
    summary.append(f"Model Type: {type(model).__name__}")
    
    # Parameter count
    param_info = count_parameters(model)
    summary.append(f"Total Parameters: {param_info['total_parameters']:,}")
    summary.append(f"Trainable Parameters: {param_info['trainable_parameters']:,}")
    summary.append(f"Non-trainable Parameters: {param_info['non_trainable_parameters']:,}")
    
    # Model structure
    summary.append("\nModel Structure:")
    summary.append("-" * 40)
    
    def add_module_info(module, prefix=""):
        for name, child in module.named_children():
            summary.append(f"{prefix}{name}: {type(child).__name__}")
            if len(list(child.children())) > 0:
                add_module_info(child, prefix + "  ")
    
    add_module_info(model)
    
    summary.append("=" * 60)
    return "\n".join(summary)

if __name__ == "__main__":
    # Example usage
    config = {
        'model_type': 'efficientnet_b0',
        'num_classes': 3,
        'pretrained': True,
        'dropout_rate': 0.3,
        'use_ensemble': False,
        'use_fairness': True,
        'sensitive_attributes': ['age', 'gender']
    }
    
    model = create_model(config)
    
    # Print model summary
    print(get_model_summary(model))
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output logits shape: {output['logits'].shape}")
    
    if 'adversarial_age' in output:
        print(f"Adversarial age output shape: {output['adversarial_age'].shape}")
