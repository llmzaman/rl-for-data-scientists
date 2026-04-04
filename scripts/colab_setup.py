"""
Google Colab Setup Script
Run this cell first in any Colab notebook:

  !wget -q https://raw.githubusercontent.com/yourusername/rl-for-data-scientists/main/scripts/colab_setup.py
  exec(open('colab_setup.py').read())
"""
import subprocess, sys, os

def install(packages):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + packages)

print("Setting up RL for Data Scientists environment...")

# Core packages
install([
    'trl>=0.8.0',
    'peft>=0.10.0',
    'datasets>=2.18.0',
    'accelerate>=0.28.0',
    'bitsandbytes>=0.43.0',
    'gymnasium>=0.29.0',
    'sympy>=1.12',
    'sqlparse>=0.4.4',
    'sqlglot>=23.0.0',
])

# Check GPU
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)")
    else:
        print("⚠ No GPU detected. Some notebooks require GPU. Enable via: Runtime → Change runtime type → GPU")
except ImportError:
    print("PyTorch not found. Installing...")
    install(['torch'])

print("✓ Setup complete! You can now run the notebook cells.")
