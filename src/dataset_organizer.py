from pathlib import Path
import shutil
import random
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetOrganizer:
    """
    Organize dataset into train/val/test splits
    """
    
    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ):
        """
        Initialize organizer
        
        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_seed: Random seed for reproducibility
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        random.seed(random_seed)
        
        logger.info(f"📊 Dataset Organizer initialized")
        logger.info(f"   Train: {train_ratio*100:.0f}%")
        logger.info(f"   Val: {val_ratio*100:.0f}%")
        logger.info(f"   Test: {test_ratio*100:.0f}%")
    
    def organize(
        self,
        source_dir: str,
        output_dir: str
    ) -> Dict:
        """
        Organize dataset into splits
        
        Args:
            source_dir: Source directory with raw images
            output_dir: Output directory for organized dataset
            
        Returns:
            Statistics dictionary
        """
        source_dir = Path(source_dir)
        output_dir = Path(output_dir)
        
        logger.info(f"\n🔧 Organizing dataset...")
        logger.info(f"   Source: {source_dir}")
        logger.info(f"   Output: {output_dir}")
        
        # Get all classes
        classes = [d for d in source_dir.iterdir() if d.is_dir()]
        
        logger.info(f"\n📂 Found {len(classes)} classes")
        
        stats = {
            'train': {},
            'val': {},
            'test': {}
        }
        
        for class_dir in classes:
            class_name = class_dir.name
            logger.info(f"\n   Processing: {class_name}")
            
            # Get all images
            images = list(class_dir.glob('*.jpg')) + \
                    list(class_dir.glob('*.png')) + \
                    list(class_dir.glob('*.jpeg'))
            
            # Filter out corrupt images (size < 1KB)
            images = [img for img in images if img.stat().st_size > 1000]
            
            logger.info(f"      Valid images: {len(images)}")
            
            # Shuffle
            random.shuffle(images)
            
            # Split
            n_train = int(len(images) * self.train_ratio)
            n_val = int(len(images) * self.val_ratio)
            
            train_images = images[:n_train]
            val_images = images[n_train:n_train + n_val]
            test_images = images[n_train + n_val:]
            
            # Copy to splits
            for split, split_images in [
                ('train', train_images),
                ('val', val_images),
                ('test', test_images)
            ]:
                split_dir = output_dir / split / class_name
                split_dir.mkdir(parents=True, exist_ok=True)
                
                for img in split_images:
                    shutil.copy2(img, split_dir / img.name)
                
                stats[split][class_name] = len(split_images)
                logger.info(f"      {split}: {len(split_images)} images")
        
        self._print_summary(stats)
        
        return stats
    
    def _print_summary(self, stats: Dict):
        """Print organization summary"""
        logger.info("\n" + "="*60)
        logger.info("📊 ORGANIZATION SUMMARY")
        logger.info("="*60)
        
        for split in ['train', 'val', 'test']:
            total = sum(stats[split].values())
            logger.info(f"\n{split.upper()}:")
            
            for class_name, count in sorted(stats[split].items()):
                logger.info(f"   {class_name:<15}: {count:>4} images")
            
            logger.info(f"   {'TOTAL':<15}: {total:>4} images")
        
        grand_total = sum(
            sum(stats[split].values()) 
            for split in ['train', 'val', 'test']
        )
        
        logger.info("\n" + "-"*60)
        logger.info(f"GRAND TOTAL: {grand_total} images")
        logger.info("="*60 + "\n")


def main():
    """Organize Arabic food dataset"""
    print("\n📊 ARABIC FOOD DATASET ORGANIZER")
    print("Splitting into train/val/test\n")
    
    # Create organizer
    organizer = DatasetOrganizer(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Organize dataset
    stats = organizer.organize(
        source_dir='data/arabic_food_raw',
        output_dir='data/arabic_food'
    )
    
    print("\n✅ Dataset organized!")
    print("📁 Location: data/arabic_food/")
    print("\nNext: Fine-tune model with LoRA")


if __name__ == "__main__":
    main()