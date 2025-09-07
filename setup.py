#!/usr/bin/env python3
"""
Setup script for LLM-MoE project
Automates environment setup and validation
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and return success status"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            return True
        else:
            print(f"❌ {description} - Failed")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - Exception: {e}")
        return False

def check_python_version():
    """Check if Python version is adequate"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is adequate (3.8+)")
        return True
    else:
        print("❌ Python version too old. Need Python 3.8+")
        return False

def check_gpu():
    """Check GPU availability"""
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            print("GPU Information:")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GeForce' in line or 'Tesla' in line or 'Quadro' in line or 'RTX' in line:
                    print(f"   {line.strip()}")
            return True
        else:
            print("⚠️  No NVIDIA GPU detected (will use CPU)")
            return False
    except:
        print("⚠️  nvidia-smi not found (will use CPU)")
        return False

def install_dependencies():
    """Install required Python packages"""
    packages = [
        "torch>=2.0.0",
        "datasets",
        "transformers", 
        "torchtune",
        "torchao",
        "tqdm",
        "numpy"
    ]
    
    print("📦 Installing dependencies...")
    for package in packages:
        success = run_command(f"pip install {package}", f"Installing {package}")
        if not success:
            print(f"⚠️  Failed to install {package}, trying without version constraint...")
            package_name = package.split(">=")[0]
            run_command(f"pip install {package_name}", f"Installing {package_name} (no version)")

def verify_installation():
    """Verify that all imports work"""
    print("🔍 Verifying installation...")
    
    imports_to_test = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("torchtune", "TorchTune"),
        ("torchao", "TorchAO"),
        ("numpy", "NumPy"),
        ("tqdm", "TQDM")
    ]
    
    all_good = True
    for module, name in imports_to_test:
        try:
            __import__(module)
            print(f"✅ {name} imported successfully")
        except ImportError as e:
            print(f"❌ {name} import failed: {e}")
            all_good = False
    
    return all_good

def check_cuda():
    """Check CUDA availability in PyTorch"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"✅ CUDA available with {device_count} device(s)")
            print(f"   Primary device: {device_name}")
            
            # Check memory
            if device_count > 0:
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"   GPU Memory: {memory_gb:.1f} GB")
                
                if memory_gb < 4:
                    print("⚠️  Low GPU memory detected. Consider reducing batch size.")
                elif memory_gb >= 8:
                    print("✅ Sufficient GPU memory for default settings")
            
            return True
        else:
            print("⚠️  CUDA not available in PyTorch")
            return False
    except:
        print("❌ Could not check CUDA availability")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ["data_cache", "logs", "checkpoints"]
    
    for dir_name in directories:
        path = Path(dir_name)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {dir_name}")
        else:
            print(f"✅ Directory exists: {dir_name}")

def suggest_configuration():
    """Suggest optimal configuration based on hardware"""
    print("\n🎛️  Configuration Suggestions:")
    
    try:
        import torch
        if torch.cuda.is_available():
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print("Based on your GPU memory:")
            if memory_gb < 6:
                print("   • Use smaller model: d_model=256, batch_size=8")
                print("   • Reduce max_steps to 1500 for faster experiments")
            elif memory_gb < 12:
                print("   • Default settings should work well")
                print("   • Consider batch_size=16-24")
            else:
                print("   • You can use larger models: d_model=512, batch_size=32")
                print("   • Try more experts: num_experts=16")
        else:
            print("   • CPU-only detected")
            print("   • Use small model: d_model=128, batch_size=4")
            print("   • Reduce max_steps to 500 for testing")
            
    except:
        print("   • Could not determine optimal settings")
        print("   • Start with default configuration")

def run_quick_test():
    """Run a quick test to verify everything works"""
    print("\n🧪 Running quick functionality test...")
    
    test_code = '''
import torch
import torch.nn as nn
from transformers import AutoTokenizer

# Test basic PyTorch functionality
print("Testing PyTorch...")
x = torch.randn(2, 3, 4)
linear = nn.Linear(4, 2)
y = linear(x)
print(f"PyTorch test passed: {y.shape}")

# Test tokenizer
print("Testing tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    tokens = tokenizer.encode("Hello world!")
    print(f"Tokenizer test passed: {len(tokens)} tokens")
except Exception as e:
    print(f"Tokenizer test failed: {e}")

print("✅ Quick test completed successfully!")
'''
    
    try:
        exec(test_code)
        return True
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 LLM-MoE Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("Please upgrade Python to 3.8+ and try again")
        return
    
    # Check system info
    print(f"💻 System: {platform.system()} {platform.release()}")
    print(f"🏗️  Architecture: {platform.machine()}")
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Install dependencies
    install_dependencies()
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        return
    
    # Check CUDA in PyTorch
    if has_gpu:
        check_cuda()
    
    # Create directories
    create_directories()
    
    # Run quick test
    if run_quick_test():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Read TUTORIAL.md for comprehensive guide")
        print("2. Run: python llm.py")
        print("3. Explore the code and experiment!")
    else:
        print("\n⚠️  Setup completed with issues")
        print("Check the error messages above and refer to TUTORIAL.md")
    
    # Suggest configuration
    suggest_configuration()

if __name__ == "__main__":
    main()