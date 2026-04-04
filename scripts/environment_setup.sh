#!/usr/bin/env bash
# RL for Data Scientists — One-Command Environment Setup
# Usage: bash scripts/environment_setup.sh

set -e
echo "========================================"
echo " RL for Data Scientists — Setup"
echo "========================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip --quiet

# Install core dependencies
echo "Installing dependencies (this may take 5-10 minutes)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet 2>/dev/null || \
pip install torch torchvision --quiet  # CPU fallback

pip install \
  transformers>=4.40.0 \
  trl>=0.8.0 \
  peft>=0.10.0 \
  datasets>=2.18.0 \
  accelerate>=0.28.0 \
  bitsandbytes>=0.43.0 \
  gymnasium>=0.29.0 \
  numpy>=1.24.0 \
  pandas>=2.0.0 \
  matplotlib>=3.7.0 \
  seaborn>=0.12.0 \
  scikit-learn>=1.3.0 \
  sympy>=1.12 \
  sqlparse>=0.4.4 \
  sqlglot>=23.0.0 \
  jupyterlab>=4.0.0 \
  ipywidgets \
  --quiet

echo ""
echo "========================================"
echo " Setup complete!"
echo ""
echo " To activate the environment:"
echo "   source .venv/bin/activate"
echo ""
echo " To start JupyterLab:"
echo "   jupyter lab"
echo "========================================"
