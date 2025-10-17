#!/bin/bash

# Rock-Paper-Scissors CNN Project Setup Script
# This script sets up the complete development environment

echo "🚀 Setting up Rock-Paper-Scissors CNN Project Environment..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p results/models
mkdir -p results/plots
mkdir -p results/logs

# Set up Kaggle API (optional)
echo "🔑 Setting up Kaggle API (optional)..."
if [ ! -d "$HOME/.kaggle" ]; then
    mkdir -p "$HOME/.kaggle"
    echo "📝 Please add your Kaggle API credentials to ~/.kaggle/kaggle.json"
    echo "   You can download this file from: https://www.kaggle.com/account"
fi

# Create .env file from template
if [ ! -f ".env" ]; then
    echo "⚙️ Creating .env file from template..."
    cp env.example .env
    echo "📝 Please edit .env file with your configuration"
fi

echo ""
echo "🎉 Environment setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Download the dataset from Kaggle and place it in data/raw/"
echo "3. Edit config/config.yaml if needed"
echo "4. Edit .env file with your settings"
echo "5. Start Jupyter Lab: jupyter lab"
echo ""
echo "📚 For more information, see README.md"
