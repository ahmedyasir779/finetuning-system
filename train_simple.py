import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForImageClassification, AutoImageProcessor
from pathlib import Path
import json
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.food_finetuner import ArabicFoodDataset

print("\n" + "="*70)
print("🍽️ SIMPLE ARABIC FOOD CLASSIFIER")
print("="*70)
print("\n📊 Configuration:")
print("   Model: ViT-Base (pure fine-tuning)")
print("   Dataset: Arabic Food (992 images)")
print("   Epochs: 3 (quick training)")
print("   Batch Size: 16")

# Config
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 5e-5

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"   Device: {device}")
print("\n" + "="*70)

# Load model
print("\n🔧 Loading model...")
model = AutoModelForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=10,
    ignore_mismatched_sizes=True
)

processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")

model.to(device)

# Load datasets
print("\n🔧 Loading datasets...")
train_dataset = ArabicFoodDataset('data/arabic_food', 'train', processor)
val_dataset = ArabicFoodDataset('data/arabic_food', 'val', processor)
test_dataset = ArabicFoodDataset('data/arabic_food', 'test', processor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

classes = train_dataset.classes

print(f"   Train images: {len(train_dataset)}")
print(f"   Val images: {len(val_dataset)}")
print(f"   Test images: {len(test_dataset)}")
print(f"   Classes: {classes}")

# Save class mapping
output_dir = Path('models/simple_vit_arabic_food')
output_dir.mkdir(exist_ok=True)

class_mapping = {
    'classes': classes,
    'class_to_idx': {cls: idx for idx, cls in enumerate(classes)},
    'idx_to_class': {str(idx): cls for idx, cls in enumerate(classes)}
}

with open(output_dir / 'class_mapping.json', 'w') as f:
    json.dump(class_mapping, f, indent=2, ensure_ascii=False)

print(f"\n💾 Class mapping saved to: {output_dir}/class_mapping.json")

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print("\n" + "="*70)
input("Press ENTER to start training...")
print("="*70)

best_val_acc = 0

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n📊 Epoch {epoch}/{NUM_EPOCHS}")
    print("-" * 70)
    
    # Train
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        images = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        _, preds = outputs.logits.max(1)
        train_correct += preds.eq(labels).sum().item()
        train_total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{train_loss/train_total:.4f}',
            'acc': f'{100.*train_correct/train_total:.1f}%'
        })
    
    train_acc = 100. * train_correct / train_total
    
    # Validate
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for batch in pbar:
            images = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            
            val_loss += loss.item() * images.size(0)
            _, preds = outputs.logits.max(1)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)
            
            pbar.set_postfix({
                'acc': f'{100.*val_correct/val_total:.1f}%'
            })
    
    val_acc = 100. * val_correct / val_total
    
    print(f"\n   Train Loss: {train_loss/train_total:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"   Val Loss:   {val_loss/val_total:.4f} | Val Acc:   {val_acc:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print(f"   💾 New best! Saving model...")
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)

# Final test
print("\n" + "="*70)
print("📊 TESTING ON TEST SET")
print("="*70)

model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    pbar = tqdm(test_loader, desc='Testing')
    for batch in pbar:
        images = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(images)
        _, preds = outputs.logits.max(1)
        test_correct += preds.eq(labels).sum().item()
        test_total += labels.size(0)
        
        pbar.set_postfix({
            'acc': f'{100.*test_correct/test_total:.1f}%'
        })

test_acc = 100. * test_correct / test_total

print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print("="*70)
print(f"\n📊 Final Results:")
print(f"   Best Val Accuracy:  {best_val_acc:.2f}%")
print(f"   Test Accuracy:      {test_acc:.2f}%")
print(f"\n💾 Model saved to: {output_dir}")
print(f"\n✅ Ready for inference!")
print("\n" + "="*70 + "\n")