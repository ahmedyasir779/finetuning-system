"""
Test the simple model
"""

import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
from pathlib import Path
import json

print("\n🔮 TESTING ARABIC FOOD CLASSIFIER\n")

# Load model
print("Loading model...")
model = AutoModelForImageClassification.from_pretrained('models/simple_vit_arabic_food')
processor = AutoImageProcessor.from_pretrained('models/simple_vit_arabic_food')

with open('models/simple_vit_arabic_food/class_mapping.json', 'r') as f:
    class_mapping = json.load(f)

classes = class_mapping['classes']
idx_to_class = {int(k): v for k, v in class_mapping['idx_to_class'].items()}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

print(f"✅ Model loaded")
print(f"   Device: {device}")
print(f"   Classes: {len(classes)}\n")

# Test
test_dir = Path('data/arabic_food/test')
correct = 0
total = 0

print("📊 Testing on each class:\n")

for class_dir in sorted(test_dir.iterdir()):
    if class_dir.is_dir():
        img = list(class_dir.glob('*.jpg'))[0]
        
        image = Image.open(img).convert('RGB')
        inputs = processor(image, return_tensors='pt')
        pixel_values = inputs['pixel_values'].to(device)
        
        with torch.no_grad():
            outputs = model(pixel_values)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()
        
        true_class = class_dir.name
        pred_class = idx_to_class[pred_idx]
        
        if pred_class == true_class:
            status = "✅"
            correct += 1
        else:
            status = "❌"
        
        total += 1
        print(f"{status} {true_class:<15} → {pred_class:<15} ({confidence*100:.1f}%)")

accuracy = (correct / total) * 100

print("\n" + "="*60)
print(f"📊 Test Accuracy: {accuracy:.1f}% ({correct}/{total})")
print("="*60)

if accuracy >= 80:
    print("\n🎉 EXCELLENT! Model working perfectly!")
elif accuracy >= 60:
    print("\n👍 GOOD! Model working!")

print()