"""
Quick Setup Script for Brain Tumor Segmentation Project
Run this to verify your environment is properly configured
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if all required packages are installed"""
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'scikit-learn',
        'cv2': 'OpenCV',
        'optuna': 'Optuna',
        'scipy': 'SciPy',
        'PIL': 'Pillow'
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ {name} not found")
            missing.append(name)
    
    return len(missing) == 0, missing

def check_directory_structure():
    """Create necessary directories"""
    dirs = [
        'src',
        'notebooks',
        'data',
        'models',
        'results/figures'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ {dir_path}/")
    
    return True

def check_data_availability():
    """Check if dataset is available"""
    possible_paths = [
        'data/lgg-mri-segmentation',
        'data/kaggle_3m',
        '../data/lgg-mri-segmentation'
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            print(f"✓ Dataset found at: {path}")
            return True
    
    print("⚠ Dataset not found. Please download from:")
    print("  https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation")
    return False

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✓ Found {gpu_count} GPU(s)")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  - GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            return True
        else:
            print("⚠ No GPU found - will use CPU (slower)")
            return False
    except Exception as e:
        print(f"⚠ Error checking GPU: {e}")
        return False

def verify_src_files():
    """Verify all source files exist"""
    required_files = {
        'src/dataset.py': 'Data loading module',
        'src/model_utils.py': 'Model architecture module',
        'src/paper_plots.py': 'Visualization module',
        'notebooks/Project_Walkthrough.ipynb': 'Main notebook'
    }
    
    all_exist = True
    for file_path, description in required_files.items():
        if Path(file_path).exists():
            print(f"✓ {description}: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            all_exist = False
    
    return all_exist

def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. If dataset is missing, download it:")
    print("   kaggle datasets download -d mateuszbuda/lgg-mri-segmentation")
    print("   unzip lgg-mri-segmentation.zip -d data/")
    
    print("\n2. Launch the project:")
    print("   jupyter notebook notebooks/Project_Walkthrough.ipynb")
    
    print("\n3. Or run from command line:")
    print("   python -m jupyter nbconvert --execute --to notebook \\")
    print("       notebooks/Project_Walkthrough.ipynb")
    
    print("\n4. Results will be saved to:")
    print("   - models/        (trained models)")
    print("   - results/       (figures and summaries)")
    print("\n" + "="*70)

def main():
    """Run all checks"""
    print("="*70)
    print("BRAIN TUMOR SEGMENTATION PROJECT - SETUP VERIFICATION")
    print("="*70)
    
    checks = []
    
    print("\n🔍 Checking Python version...")
    checks.append(check_python_version())
    
    print("\n🔍 Checking dependencies...")
    deps_ok, missing = check_dependencies()
    checks.append(deps_ok)
    if missing:
        print(f"\n⚠ Install missing packages:")
        print(f"   pip install {' '.join(missing.lower().replace('-', '_') for dep in missing)}")
    
    print("\n🔍 Checking directory structure...")
    checks.append(check_directory_structure())
    
    print("\n🔍 Verifying source files...")
    checks.append(verify_src_files())
    
    print("\n🔍 Checking dataset...")
    data_ok = check_data_availability()
    
    print("\n🔍 Checking GPU...")
    check_gpu()
    
    print("\n" + "="*70)
    if all(checks):
        print("✅ SETUP COMPLETE - Ready to run!")
        if not data_ok:
            print("⚠ Remember to download the dataset first")
    else:
        print("❌ SETUP INCOMPLETE - Please fix the issues above")
    print("="*70)
    
    print_next_steps()

if __name__ == "__main__":
    main()
