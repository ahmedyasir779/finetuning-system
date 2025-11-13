import torch
from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
    AutoConfig
)
from typing import Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Load and prepare models for fine-tuning
    """
    
    SUPPORTED_MODELS = {
        'vit-base': 'google/vit-base-patch16-224',
        'resnet-50': 'microsoft/resnet-50',
        'convnext-tiny': 'facebook/convnext-tiny-224',
        'swin-tiny': 'microsoft/swin-tiny-patch4-window7-224',
    }
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize model loader
        
        Args:
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"🔧 ModelLoader initialized (device: {self.device})")
    
    def load_model(
        self,
        model_name: str,
        num_labels: int,
        pretrained: bool = True
    ) -> Tuple[AutoModelForImageClassification, AutoImageProcessor]:
        """
        Load model and processor
        
        Args:
            model_name: Model identifier (key from SUPPORTED_MODELS or HF model ID)
            num_labels: Number of output classes
            pretrained: Whether to load pretrained weights
            
        Returns:
            Tuple of (model, processor)
        """
        # Get full model name
        if model_name in self.SUPPORTED_MODELS:
            full_model_name = self.SUPPORTED_MODELS[model_name]
        else:
            full_model_name = model_name
        
        logger.info(f"📥 Loading model: {full_model_name}")
        
        # Load processor (for image preprocessing)
        processor = AutoImageProcessor.from_pretrained(full_model_name)
        
        # Load model
        if pretrained:
            model = AutoModelForImageClassification.from_pretrained(
                full_model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True
            )
        else:
            config = AutoConfig.from_pretrained(
                full_model_name,
                num_labels=num_labels
            )
            model = AutoModelForImageClassification.from_config(config)
        
        # Move to device
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
        
        return model, processor
    
    def freeze_base_model(self, model: AutoModelForImageClassification) -> AutoModelForImageClassification:
        """
        Freeze all parameters except classifier head
        
        Args:
            model: Model to freeze
            
        Returns:
            Model with frozen base
        """
        logger.info("🔒 Freezing base model parameters...")
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier head
        if hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, 'head'):
            for param in model.head.parameters():
                param.requires_grad = True
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        logger.info(f"✅ Base model frozen!")
        logger.info(f"   Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        
        return model
    
    def get_model_info(self, model: AutoModelForImageClassification) -> Dict:
        """
        Get detailed model information
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary with model info
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params,
            'trainable_percentage': (trainable_params / total_params) * 100,
            'model_size_mb': (total_params * 4) / 1024 / 1024,
            'device': str(next(model.parameters()).device)
        }
    
    @staticmethod
    def list_supported_models() -> None:
        """Print all supported models"""
        logger.info("📋 Supported Models:")
        for key, value in ModelLoader.SUPPORTED_MODELS.items():
            logger.info(f"   {key}: {value}")


def print_model_summary(model: AutoModelForImageClassification):
    """
    Print detailed model summary
    
    Args:
        model: Model to summarize
    """
    print("\n" + "="*60)
    print("🔍 MODEL SUMMARY")
    print("="*60)
    
    total_params = 0
    trainable_params = 0
    
    print(f"\n{'Layer Name':<40} {'Parameters':>15} {'Trainable':>10}")
    print("-" * 70)
    
    for name, param in model.named_parameters():
        layer_params = param.numel()
        total_params += layer_params
        
        if param.requires_grad:
            trainable_params += layer_params
            trainable_str = "✅"
        else:
            trainable_str = "❌"
        
        # Only show main layers (not every sub-parameter)
        if '.' in name:
            layer_name = name.split('.')[0]
        else:
            layer_name = name
        
        # Print only first occurrence of each main layer
        print(f"{name[:40]:<40} {layer_params:>15,} {trainable_str:>10}")
    
    print("-" * 70)
    print(f"{'TOTAL':<40} {total_params:>15,}")
    print(f"{'TRAINABLE':<40} {trainable_params:>15,}")
    print(f"{'FROZEN':<40} {total_params - trainable_params:>15,}")
    print(f"\nTrainable: {trainable_params/total_params*100:.2f}%")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    print("="*60 + "\n")