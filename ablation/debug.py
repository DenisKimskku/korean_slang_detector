#!/usr/bin/env python3
"""
Debug script to identify import issues in the ablation study.
This will help diagnose why the transformers library import is failing.
"""

import sys
import os
import traceback
import importlib.util

def test_transformers_import():
    """Test importing transformers step by step"""
    print("="*60)
    print("TESTING TRANSFORMERS IMPORTS")
    print("="*60)
    
    try:
        print("1. Testing basic transformers import...")
        import transformers
        print(f"   ✓ transformers version: {transformers.__version__}")
    except Exception as e:
        print(f"   ✗ Failed to import transformers: {e}")
        traceback.print_exc()
        return False
    
    try:
        print("2. Testing AutoTokenizer import...")
        from transformers import AutoTokenizer
        print("   ✓ AutoTokenizer imported successfully")
    except Exception as e:
        print(f"   ✗ Failed to import AutoTokenizer: {e}")
        traceback.print_exc()
        return False
    
    try:
        print("3. Testing AutoModel import...")
        from transformers import AutoModel
        print("   ✓ AutoModel imported successfully")
    except Exception as e:
        print(f"   ✗ Failed to import AutoModel: {e}")
        traceback.print_exc()
        return False
    
    try:
        print("4. Testing specific model loading...")
        tokenizer = AutoTokenizer.from_pretrained('klue/roberta-small')
        print("   ✓ Successfully loaded klue/roberta-small tokenizer")
    except Exception as e:
        print(f"   ✗ Failed to load tokenizer: {e}")
        traceback.print_exc()
        return False
    
    print("\n✓ All transformers imports successful!")
    return True

def test_torch_import():
    """Test PyTorch imports"""
    print("\n" + "="*60)
    print("TESTING PYTORCH IMPORTS")
    print("="*60)
    
    try:
        print("1. Testing torch import...")
        import torch
        print(f"   ✓ PyTorch version: {torch.__version__}")
        print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   ✓ CUDA version: {torch.version.cuda}")
    except Exception as e:
        print(f"   ✗ Failed to import torch: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_other_imports():
    """Test other required imports"""
    print("\n" + "="*60)
    print("TESTING OTHER IMPORTS")
    print("="*60)
    
    imports_to_test = [
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('matplotlib.pyplot', 'plt'),
        ('sklearn.utils.class_weight', None),
        ('tqdm', None)
    ]
    
    for import_name, alias in imports_to_test:
        try:
            if alias:
                exec(f"import {import_name} as {alias}")
                print(f"   ✓ {import_name} (as {alias}) imported successfully")
            else:
                exec(f"import {import_name}")
                print(f"   ✓ {import_name} imported successfully")
        except Exception as e:
            print(f"   ✗ Failed to import {import_name}: {e}")
    
    return True

def test_main_script_import(script_path):
    """Test importing the main training script"""
    print(f"\n" + "="*60)
    print(f"TESTING MAIN SCRIPT IMPORT: {script_path}")
    print("="*60)
    
    if not os.path.exists(script_path):
        print(f"   ✗ Script file not found: {script_path}")
        return False
    
    print(f"   ✓ Script file exists: {script_path}")
    
    try:
        # Try to import the script
        spec = importlib.util.spec_from_file_location("test_module", script_path)
        module = importlib.util.module_from_spec(spec)
        
        print("   - Attempting to execute module...")
        spec.loader.exec_module(module)
        print("   ✓ Main script imported successfully!")
        
        # Check for required components
        required_components = [
            'AutoTokenizer',
            'ConversationAugmenter', 
            'SlidingWindowConversationDataset',
            'ConversationAnomalyDetector',
            'Trainer',
            'FocalLoss'
        ]
        
        print("\n   Checking for required components:")
        for component in required_components:
            if hasattr(module, component):
                print(f"   ✓ {component} found")
            else:
                print(f"   ✗ {component} missing")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Failed to import main script: {e}")
        print("\n   Full traceback:")
        traceback.print_exc()
        return False

def check_environment():
    """Check Python environment details"""
    print("\n" + "="*60)
    print("ENVIRONMENT INFORMATION")
    print("="*60)
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ Virtual environment detected")
    else:
        print("⚠ No virtual environment detected")

def main():
    """Main debug function"""
    print("ABLATION STUDY DEBUG SCRIPT")
    print("="*60)
    
    # Check environment
    check_environment()
    
    # Test imports step by step
    torch_ok = test_torch_import()
    transformers_ok = test_transformers_import()
    test_other_imports()
    
    # Test main script imports
    possible_scripts = [
        'legacy_msg.py',
        'paste.txt',
        'main.py',
        'train.py'
    ]
    
    main_script_found = False
    for script in possible_scripts:
        if os.path.exists(script):
            print(f"\nFound potential main script: {script}")
            if test_main_script_import(script):
                main_script_found = True
                break
    
    if not main_script_found:
        print(f"\n{'='*60}")
        print("MAIN SCRIPT NOT FOUND OR IMPORT FAILED")
        print("="*60)
        print("Available files in current directory:")
        for file in os.listdir('.'):
            if file.endswith('.py'):
                print(f"  - {file}")
    
    # Summary and recommendations
    print(f"\n{'='*60}")
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*60)
    
    if not torch_ok:
        print("❌ PyTorch installation issue detected")
        print("   Recommendation: pip install torch torchvision torchaudio")
    
    if not transformers_ok:
        print("❌ Transformers installation issue detected")
        print("   Recommendations:")
        print("   1. pip install transformers")
        print("   2. If already installed, try: pip install --upgrade transformers")
        print("   3. Check for conflicting dependencies: pip check")
        print("   4. Try creating a fresh virtual environment")
    
    if not main_script_found:
        print("❌ Main training script not found or has import errors")
        print("   Recommendations:")
        print("   1. Ensure your main training script exists")
        print("   2. Check that all required classes are defined in the script")
        print("   3. Verify script dependencies are installed")
        print("   4. Use --script flag to specify correct script path")
    
    if torch_ok and transformers_ok and main_script_found:
        print("✅ All checks passed! The ablation study should work.")
    else:
        print("❌ Issues detected. Please fix the above problems before running ablation study.")

if __name__ == "__main__":
    main()