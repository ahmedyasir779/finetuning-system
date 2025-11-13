import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.model_loader import ModelLoader, print_model_summary
from src.lora_config import LoRAConfigManager, LoRAComparator


def test_model_loading():
    """Test loading pretrained models"""
    print("\n" + "="*60)
    print("🧪 TEST 1: MODEL LOADING")
    print("="*60 + "\n")
    
    loader = ModelLoader()
    
    # List supported models
    ModelLoader.list_supported_models()
    
    # Load a small model for testing
    print("\n📥 Loading ViT-Base model...")
    model, processor = loader.load_model(
        model_name='vit-base',
        num_labels=5,  # 5 flower classes
        pretrained=True
    )
    
    # Get model info
    info = loader.get_model_info(model)
    
    print(f"\n📊 Model Information:")
    print(f"   Total parameters: {info['total_parameters']:,}")
    print(f"   Trainable: {info['trainable_parameters']:,} ({info['trainable_percentage']:.2f}%)")
    print(f"   Size: {info['model_size_mb']:.2f} MB")
    print(f"   Device: {info['device']}")
    
    print("\n✅ Model loading test passed!\n")
    
    return model, processor


def test_lora_configuration():
    """Test LoRA configuration"""
    print("\n" + "="*60)
    print("🧪 TEST 2: LORA CONFIGURATION")
    print("="*60 + "\n")
    
    manager = LoRAConfigManager()
    
    # List all presets
    manager.list_presets()
    
    # Create different LoRA configs
    print("\n🔧 Testing different LoRA presets...\n")
    
    for preset in ['minimal', 'balanced', 'quality']:
        print(f"\n--- {preset.upper()} ---")
        config = manager.create_lora_config(preset=preset)
        print(f"Config created: r={config.r}, alpha={config.lora_alpha}")
    
    print("\n✅ LoRA configuration test passed!\n")
    
    return manager


def test_lora_application():
    """Test applying LoRA to model"""
    print("\n" + "="*60)
    print("🧪 TEST 3: LORA APPLICATION")
    print("="*60 + "\n")
    
    # Load model
    loader = ModelLoader()
    model, _ = loader.load_model('vit-base', num_labels=5)
    
    # Create LoRA config
    manager = LoRAConfigManager()
    lora_config = manager.create_lora_config(preset='balanced')
    
    # Apply LoRA
    print("\n🔧 Applying LoRA to model...")
    lora_model = manager.apply_lora(model, lora_config)
    
    # Print trainable parameters
    manager.print_trainable_parameters(lora_model)
    
    print("✅ LoRA application test passed!\n")
    
    return lora_model


def test_full_vs_lora_comparison():
    """Test comparison between full fine-tuning and LoRA"""
    print("\n" + "="*60)
    print("🧪 TEST 4: FULL vs LORA COMPARISON")
    print("="*60 + "\n")
    
    # Load two separate models
    loader = ModelLoader()
    
    # Full fine-tuning model (all parameters trainable)
    print("📥 Loading model for full fine-tuning...")
    full_model, _ = loader.load_model('vit-base', num_labels=5)
    
    # LoRA model
    print("\n📥 Loading model for LoRA fine-tuning...")
    lora_base_model, _ = loader.load_model('vit-base', num_labels=5)
    
    # Apply LoRA
    manager = LoRAConfigManager()
    lora_config = manager.create_lora_config(preset='balanced')
    lora_model = manager.apply_lora(lora_base_model, lora_config)
    
    # Compare
    comparator = LoRAComparator()
    comparison = comparator.compare_parameter_efficiency(full_model, lora_model)
    
    print("✅ Comparison test passed!\n")
    
    return comparison


def test_forward_pass():
    """Test forward pass with LoRA model"""
    print("\n" + "="*60)
    print("🧪 TEST 5: FORWARD PASS")
    print("="*60 + "\n")
    
    # Load model
    loader = ModelLoader()
    model, processor = loader.load_model('vit-base', num_labels=5)
    
    # Apply LoRA
    manager = LoRAConfigManager()
    lora_config = manager.create_lora_config(preset='minimal')
    lora_model = manager.apply_lora(model, lora_config)
    
    # Create dummy input
    print("🔄 Testing forward pass...")
    dummy_input = torch.randn(2, 3, 224, 224).to(loader.device)
    
    # Forward pass
    with torch.no_grad():
        outputs = lora_model(dummy_input)
    
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {outputs.logits.shape}")
    print(f"   Output classes: {outputs.logits.shape[1]}")
    
    assert outputs.logits.shape == (2, 5), "Output shape mismatch!"
    
    print("\n✅ Forward pass test passed!\n")


def main():
    """Run all tests"""
    print("\n🚀 LORA BASICS TESTS")
    print("Testing LoRA setup and configuration\n")
    
    try:
        # Test 1: Model Loading
        model, processor = test_model_loading()
        
        # Test 2: LoRA Configuration
        manager = test_lora_configuration()
        
        # Test 3: LoRA Application
        lora_model = test_lora_application()
        
        # Test 4: Full vs LoRA
        comparison = test_full_vs_lora_comparison()
        
        # Test 5: Forward Pass
        test_forward_pass()
        
        # Final summary
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\n✅ Model Loading: PASSED")
        print("✅ LoRA Configuration: PASSED")
        print("✅ LoRA Application: PASSED")
        print("✅ Full vs LoRA Comparison: PASSED")
        print("✅ Forward Pass: PASSED")
        
        print("\n💡 Key Findings:")
        print(f"   - LoRA reduces trainable parameters by {comparison['efficiency_gain']['parameter_reduction']:.1f}%")
        print(f"   - Memory savings: {comparison['efficiency_gain']['memory_savings']:.1f} MB")
        print(f"   - Full model: {comparison['full_finetune']['trainable_params']:,} trainable params")
        print(f"   - LoRA model: {comparison['lora_finetune']['trainable_params']:,} trainable params")
        
        print("\n🎯 LoRA fundamentals complete!")
        print("Ready to fine-tune models efficiently! 🚀\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())