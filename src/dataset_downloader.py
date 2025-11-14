
from bing_image_downloader import downloader
from pathlib import Path
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArabicFoodDatasetDownloader:
    """
    Download Arabic food images from Bing
    """
    
    # Our 10 Arabic dishes
    DISHES = {
        'kabsa': 'Saudi Kabsa rice chicken',
        'mandi': 'Mandi rice lamb meat',
        'shawarma': 'Arabic shawarma wrap sandwich',
        'falafel': 'Falafel Arabic fried chickpea',
        'hummus': 'Hummus Arabic chickpea dip',
        'kunafa': 'Kunafa Arabic dessert cheese',
        'samboosa': 'Samosa samboosa Arabic fried',
        'grilled_meat': 'Arabic grilled meat tikka kebab',
        'arabic_coffee': 'Arabic coffee dallah pot',
        'dates': 'Arabic dates plate arrangement'
    }
    
    def __init__(self, output_dir: str = 'data/arabic_food_raw'):
        """
        Initialize downloader
        
        Args:
            output_dir: Directory to save images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🍽️ Arabic Food Dataset Downloader")
        logger.info(f"   Output: {self.output_dir}")
    
    def download_all(self, images_per_class: int = 100):
        """
        Download all dishes
        
        Args:
            images_per_class: Number of images per dish
        """
        logger.info(f"\n📥 Downloading {len(self.DISHES)} dishes...")
        logger.info(f"   Images per class: {images_per_class}")
        logger.info(f"   Total images: ~{len(self.DISHES) * images_per_class}")
        
        for dish_name, search_query in self.DISHES.items():
            self._download_dish(dish_name, search_query, images_per_class)
        
        self._print_summary()
    
    def _download_dish(self, dish_name: str, search_query: str, limit: int):
        """Download images for one dish"""
        logger.info(f"\n🔍 Downloading: {dish_name}")
        logger.info(f"   Query: {search_query}")
        
        try:
            # Download using Bing
            downloader.download(
                search_query,
                limit=limit,
                output_dir=str(self.output_dir),
                adult_filter_off=True,
                force_replace=False,
                timeout=15,
                verbose=False
            )
            
            # Rename folder to clean name
            downloaded_folder = self.output_dir / search_query
            target_folder = self.output_dir / dish_name
            
            if downloaded_folder.exists():
                if target_folder.exists():
                    shutil.rmtree(target_folder)
                downloaded_folder.rename(target_folder)
                
                # Count images
                image_count = len(list(target_folder.glob('*.jpg'))) + \
                             len(list(target_folder.glob('*.png')))
                
                logger.info(f"   ✅ Downloaded: {image_count} images")
            else:
                logger.warning(f"   ⚠️  No images found for {dish_name}")
                
        except Exception as e:
            logger.error(f"   ❌ Error downloading {dish_name}: {e}")
    
    def _print_summary(self):
        """Print download summary"""
        logger.info("\n" + "="*60)
        logger.info("📊 DOWNLOAD SUMMARY")
        logger.info("="*60)
        
        total_images = 0
        
        for dish_name in self.DISHES.keys():
            dish_folder = self.output_dir / dish_name
            
            if dish_folder.exists():
                image_count = len(list(dish_folder.glob('*.jpg'))) + \
                             len(list(dish_folder.glob('*.png')))
                total_images += image_count
                
                logger.info(f"   {dish_name:<15}: {image_count:>4} images")
            else:
                logger.info(f"   {dish_name:<15}: {0:>4} images ❌")
        
        logger.info("-" * 60)
        logger.info(f"   {'TOTAL':<15}: {total_images:>4} images")
        logger.info("="*60 + "\n")


def main():
    """Download Arabic food dataset"""
    print("\n🍽️ ARABIC FOOD DATASET DOWNLOADER")
    print("Downloading 10 dishes for fine-tuning\n")
    
    # Create downloader
    downloader_obj = ArabicFoodDatasetDownloader()
    
    # Download images (100 per class = ~1000 total)
    downloader_obj.download_all(images_per_class=100)
    
    print("\n✅ Dataset download complete!")
    print("📁 Location: data/arabic_food_raw/")
    print("\nNext: Clean and organize dataset")


if __name__ == "__main__":
    main()