"""
LoRA Configuration
Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning
"""

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftConfig
)
from transformers import AutoModelForImageClassification
import torch
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoRAConfigManager:
    """
    Manage LoRA configurations for different scenarios
    """
    
    # Predefined configurations
    CONFIGS = {
        'minimal': {
            'r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'description': 'Minimal LoRA (fastest, least memory)'
        },
        'balanced': {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'description': 'Balanced performance and efficiency'
        },
        'quality': {
            'r': 32,
            'lora_alpha': 64,
            'lora_dropout': 0.05,
            'description': 'Higher quality (more parameters)'
        },
        'aggressive': {
            'r': 64,
            'lora_alpha': 128,
            'lora_dropout': 0.05,
            'description': 'Aggressive fine-tuning (most parameters)'
        }
    }
    
    # Common target modules for different architectures
    TARGET_MODULES = {
        'vit': ['query', 'value'],  # Vision Transformer
        'resnet': ['conv1', 'conv2'],  # ResNet
        'convnext': ['dwconv', 'pwconv1', 'pwconv2'],  # ConvNeXt
        'swin': ['query', 'value']  # Swin Transformer
    }
    
    def __init__(self):
        """Initialize LoRA config manager"""
        logger.info("🎯 LoRAConfigManager initialized")
    
    def create_lora_config(
        self,
        preset: str = 'balanced',
        target_modules: Optional[list] = None,
        **kwargs
    ) -> LoraConfig:
        """
        Create LoRA configuration
        
        Args:
            preset: Configuration preset ('minimal', 'balanced', 'quality', 'aggressive')
            target_modules: Specific modules to apply LoRA (None = auto-detect)
            **kwargs: Override default parameters
            
        Returns:
            LoraConfig object
        """
        if preset not in self.CONFIGS:
            raise ValueError(f"Unknown preset: {preset}. Choose from {list(self.CONFIGS.keys())}")
        
        # Get preset config
        config_dict = self.CONFIGS[preset].copy()
        description = config_dict.pop('description')
        
        # Override with kwargs
        config_dict.update(kwargs)
        
        # Auto-detect target modules if not specified
        if target_modules is None:
            # Default: attention layers for vision transformers
            target_modules = ["query", "value"]
            logger.info(f"   Auto-detected target modules: {target_modules}")
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=config_dict['r'],
            lora_alpha=config_dict['lora_alpha'],
            lora_dropout=config_dict['lora_dropout'],
            bias='none',
            target_modules=target_modules
        )
        
        logger.info(f"✅ LoRA config created: {preset}")
        logger.info(f"   Description: {description}")
        logger.info(f"   Rank (r): {config_dict['r']}")
        logger.info(f"   Alpha: {config_dict['lora_alpha']}")
        logger.info(f"   Dropout: {config_dict['lora_dropout']}")
        logger.info(f"   Target modules: {target_modules}")
        
        return lora_config
    
    def apply_lora(
        self,
        model: AutoModelForImageClassification,
        lora_config: LoraConfig
    ) -> PeftModel:
        """
        Apply LoRA to model
        
        Args:
            model: Base model
            lora_config: LoRA configuration
            
        Returns:
            PEFT model with LoRA adapters
        """
        logger.info("🔧 Applying LoRA to model...")
        
        # Get original parameter count
        original_params = sum(p.numel() for p in model.parameters())
        
        # Apply LoRA
        peft_model = get_peft_model(model, lora_config)
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        
        logger.info(f"✅ LoRA applied successfully!")
        logger.info(f"   Original parameters: {original_params:,}")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Trainable percentage: {trainable_params/total_params*100:.2f}%")
        logger.info(f"   Parameter reduction: {(1 - trainable_params/total_params)*100:.2f}%")
        
        return peft_model
    
    @staticmethod
    def print_trainable_parameters(model: PeftModel):
        """
        Print trainable parameters info
        
        Args:
            model: PEFT model to analyze
        """
        trainable_params = 0
        all_params = 0
        
        for name, param in model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"\n{'='*60}")
        print(f"📊 TRAINABLE PARAMETERS")
        print(f"{'='*60}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"All params: {all_params:,}")
        print(f"Trainable %: {100 * trainable_params / all_params:.4f}%")
        print(f"{'='*60}\n")
    
    @staticmethod
    def list_presets():
        """List all available presets"""
        print("\n" + "="*60)
        print("📋 AVAILABLE LORA PRESETS")
        print("="*60)
        
        for name, config in LoRAConfigManager.CONFIGS.items():
            print(f"\n🎯 {name.upper()}")
            print(f"   Description: {config['description']}")
            print(f"   Rank (r): {config['r']}")
            print(f"   Alpha: {config['lora_alpha']}")
            print(f"   Dropout: {config['lora_dropout']}")
        
        print("\n" + "="*60 + "\n")


class LoRAComparator:
    """
    Compare full fine-tuning vs LoRA fine-tuning
    """
    
    def __init__(self):
        """Initialize comparator"""
        self.results = {}
    
    def compare_parameter_efficiency(
        self,
        full_model: AutoModelForImageClassification,
        lora_model: PeftModel
    ) -> Dict:
        """
        Compare parameter efficiency
        
        Args:
            full_model: Full fine-tuning model
            lora_model: LoRA model
            
        Returns:
            Comparison dictionary
        """
        # Full model stats
        full_total = sum(p.numel() for p in full_model.parameters())
        full_trainable = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
        
        # LoRA model stats
        lora_total = sum(p.numel() for p in lora_model.parameters())
        lora_trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        
        comparison = {
            'full_finetune': {
                'total_params': full_total,
                'trainable_params': full_trainable,
                'trainable_percentage': (full_trainable / full_total) * 100,
                'memory_mb': (full_total * 4) / 1024 / 1024
            },
            'lora_finetune': {
                'total_params': lora_total,
                'trainable_params': lora_trainable,
                'trainable_percentage': (lora_trainable / lora_total) * 100,
                'memory_mb': (lora_trainable * 4) / 1024 / 1024
            },
            'efficiency_gain': {
                'parameter_reduction': ((full_trainable - lora_trainable) / full_trainable) * 100,
                'memory_savings': ((full_total - lora_trainable) * 4 / 1024 / 1024)
            }
        }
        
        self._print_comparison(comparison)
        
        return comparison
    
    def _print_comparison(self, comparison: Dict):
        """Print comparison results"""
        print("\n" + "="*70)
        print("⚖️  FULL FINE-TUNING vs LoRA COMPARISON")
        print("="*70)
        
        print(f"\n{'Metric':<30} {'Full Fine-tune':>18} {'LoRA':>18}")
        print("-" * 70)
        
        full = comparison['full_finetune']
        lora = comparison['lora_finetune']
        
        print(f"{'Total Parameters':<30} {full['total_params']:>18,} {lora['total_params']:>18,}")
        print(f"{'Trainable Parameters':<30} {full['trainable_params']:>18,} {lora['trainable_params']:>18,}")
        print(f"{'Trainable %':<30} {full['trainable_percentage']:>17.2f}% {lora['trainable_percentage']:>17.2f}%")
        print(f"{'Memory (MB)':<30} {full['memory_mb']:>17.2f} {lora['memory_mb']:>17.2f}")
        
        print("\n" + "-" * 70)
        print(f"{'EFFICIENCY GAINS':<30}")
        print("-" * 70)
        
        gains = comparison['efficiency_gain']
        print(f"{'Parameter Reduction':<30} {gains['parameter_reduction']:>17.2f}%")
        print(f"{'Memory Savings (MB)':<30} {gains['memory_savings']:>17.2f}")
        
        print("="*70 + "\n")