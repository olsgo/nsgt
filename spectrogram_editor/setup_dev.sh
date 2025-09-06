#!/bin/bash

# Real-time Spectrogram Editor - Development Setup Script
# This script sets up the development environment and starts both services

set -e

echo "🎵 Real-time Spectrogram Editor - Development Setup"
echo "=================================================="

# Check Python environment
echo "Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Check for virtual environment
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "Consider running: python3 -m venv venv && source venv/bin/activate"
fi

# Install NSGT library first
echo "Installing NSGT library..."
cd ../..
pip install -e .
cd spectrogram_editor

# Install DSP service dependencies
echo "Installing DSP service dependencies..."
pip install -r requirements.txt

# Check GPU availability
echo "Checking GPU availability..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
else:
    print('⚠️  CUDA not available - will use CPU fallback')
"

# Check nnAudio installation
echo "Checking nnAudio installation..."
python3 -c "
try:
    import nnAudio
    print(f'✅ nnAudio version: {nnAudio.__version__}')
except ImportError:
    print('⚠️  nnAudio not available - some features will be limited')
"

# Check Flutter installation (if available)
echo "Checking Flutter installation..."
if command -v flutter &> /dev/null; then
    echo "✅ Flutter found:"
    flutter --version
    
    # Setup Flutter app dependencies
    echo "Setting up Flutter app..."
    cd flutter_app
    flutter pub get
    flutter config --enable-windows-desktop --enable-macos-desktop --enable-linux-desktop
    cd ..
else
    echo "⚠️  Flutter not installed - UI will not be available"
    echo "To install Flutter: https://docs.flutter.dev/get-started/install"
fi

echo ""
echo "🚀 Setup complete! To start the application:"
echo ""
echo "1. Start DSP service:"
echo "   cd dsp_service"
echo "   python main.py"
echo ""
echo "2. In another terminal, start Flutter app:"
echo "   cd flutter_app"
echo "   flutter run -d linux  # or windows/macos"
echo ""
echo "📚 Documentation available in docs/"
echo "🔧 GPU shaders available in shaders/"
echo "🔌 Platform channels in platform_channels/"
echo ""
echo "Happy editing! 🎶"