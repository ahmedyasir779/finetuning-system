import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArabicFoodDataset(Dataset):
    """
    PyTorch dataset for Arabic food images
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str,
        processor: AutoImageProcessor
    ):
        """
        Initialize dataset
        
        Args:
            data_dir: Root data directory
            split: 'train', 'val', or 'test'
            processor: Image processor
        """
        self.data_dir = Path(data_dir) / split
        self.processor = processor
        
        # Get all classes
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Get all image paths
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.images.append(img_path)
                    self.labels.append(class_idx)
        
        logger.info(f"📊 {split.upper()} dataset loaded")
        logger.info(f"   Images: {len(self.images)}")
        logger.info(f"   Classes: {len(self.classes)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.images[idx]).convert('RGB')
        
        # Process image
        inputs = self.processor(image, return_tensors='pt')
        
        # Remove batch dimension
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(self.labels[idx])
        }


class ArabicFoodFineTuner:
    """
    Fine-tune vision model on Arabic food
    """
    
    def __init__(
        self,
        model_name: str = 'google/vit-base-patch16-224',
        data_dir: str = 'data/arabic_food',
        output_dir: str = 'models/arabic_food_lora',
        use_lora: bool = True
    ):
        """
        Initialize fine-tuner
        
        Args:
            model_name: Base model to fine-tune
            data_dir: Dataset directory
            output_dir: Output directory for checkpoints
            use_lora: Whether to use LoRA
        """
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.use_lora = use_lora
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"🔥 Arabic Food Fine-Tuner")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   LoRA: {use_lora}")
        
        # Will be initialized in setup()
        self.model = None
        self.processor = None
        self.train_dataset = None
        self.val_dataset = None
        self.num_labels = None
    
    def setup(self, lora_config = None):
        """
        Setup model, processor, and datasets
        
        Args:
            lora_config: LoRA configuration (if use_lora=True)
        """
        logger.info("\n🔧 Setting up fine-tuning pipeline...")
        
        # Load processor
        logger.info("   Loading processor...")
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        
        # Create datasets
        logger.info("   Creating datasets...")
        self.train_dataset = ArabicFoodDataset(
            self.data_dir, 'train', self.processor
        )
        self.val_dataset = ArabicFoodDataset(
            self.data_dir, 'val', self.processor
        )
        
        self.num_labels = len(self.train_dataset.classes)
        
        # Load model
        logger.info(f"   Loading model ({self.num_labels} classes)...")
        self.model = AutoModelForImageClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True
        )
        
        # Apply LoRA if requested
        if self.use_lora:
            if lora_config is None:
                from lora_config import LoRAConfigManager
                manager = LoRAConfigManager()
                lora_config = manager.create_lora_config(preset='balanced')
            
            logger.info("   Applying LoRA...")
            self.model = get_peft_model(self.model, lora_config)
            
            # Print trainable parameters
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            logger.info(f"   Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
        
        self.model.to(self.device)
        
        logger.info("✅ Setup complete!\n")
    
    def train(
        self,
        num_epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 2e-4,
        save_steps: int = 50
    ):
        """
        Train the model
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            save_steps: Save checkpoint every N steps
        """
        logger.info("🚀 Starting training...")
        logger.info(f"   Epochs: {num_epochs}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Learning rate: {learning_rate}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=10,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="none",
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self._compute_metrics,
        )
        
        # Train
        logger.info("\n" + "="*60)
        logger.info("🔥 TRAINING STARTED")
        logger.info("="*60 + "\n")
        
        train_result = trainer.train()
        
        logger.info("\n" + "="*60)
        logger.info("✅ TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"   Train loss: {train_result.training_loss:.4f}")
        logger.info(f"   Time: {train_result.metrics['train_runtime']:.2f}s")
        
        # Save final model
        trainer.save_model(f"{self.output_dir}/final")
        
        logger.info(f"\n💾 Model saved to: {self.output_dir}/final")
        
        return trainer
    
    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = (predictions == labels).mean()
        
        return {'accuracy': accuracy}
    
    def get_class_names(self):
        """Get class names"""
        return self.train_dataset.classes if self.train_dataset else None


def main():
    """Quick test of fine-tuning pipeline"""
    print("\n🔥 ARABIC FOOD FINE-TUNING TEST")
    print("Setting up pipeline...\n")
    
    # Create fine-tuner
    finetuner = ArabicFoodFineTuner(
        model_name='google/vit-base-patch16-224',
        data_dir='data/arabic_food',
        output_dir='models/arabic_food_lora',
        use_lora=True
    )
    
    # Setup
    finetuner.setup()
    
    # Print class names
    print("\n📋 Class Names:")
    for idx, name in enumerate(finetuner.get_class_names()):
        print(f"   {idx}: {name}")
    
    print("\n✅ Fine-tuning pipeline ready!")
    print("Run actual training with appropriate parameters.")


if __name__ == "__main__":
    main()