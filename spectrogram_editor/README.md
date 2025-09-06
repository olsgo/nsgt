# Real-time Spectrogram Editor

A desktop application for real-time spectrogram visualization and editing with GPU-accelerated processing.

## Architecture Overview

This project combines:
- **NSGT (Non-Stationary Gabor Transform)** for advanced time-frequency analysis
- **nnAudio** for GPU-accelerated spectrogram generation (STFT, Mel, CQT, VQT)
- **Flutter Desktop UI** with real-time GPU texture rendering
- **PyTorch GPU pipeline** for efficient tensor operations
- **Custom brush tools** for interactive spectrogram editing

### Components

1. **DSP Service** (`dsp_service/`)
   - Python service running NSGT and nnAudio
   - GPU-resident tensor operations with PyTorch
   - Real-time spectrogram generation and manipulation
   - IPC bridge for communication with Flutter UI

2. **Flutter Desktop App** (`flutter_app/`)
   - Cross-platform desktop UI (Windows/macOS/Linux)
   - Real-time GPU texture display
   - Fragment shaders for brush effects (blur, warp)
   - Platform channels for native integration

3. **GPU Shaders** (`shaders/`)
   - Fragment shaders for brush effects
   - Vertex shaders for texture mapping
   - Compute shaders for advanced effects

4. **Platform Channels** (`platform_channels/`)
   - Native plugins for texture management
   - Platform-specific GPU integration
   - Shared memory and texture streaming

## Key Features

### Real-time Processing
- GPU-accelerated spectrogram generation with nnAudio
- Non-stationary Gabor transforms with NSGT
- Zero-copy tensor operations where possible
- Tiled processing for large spectrograms

### Interactive Editing
- **Blur Brush**: Gaussian blur with adjustable radius
- **Warp Brush**: Displacement-based distortion effects
- **Magnitude/Phase editing**: Domain-aware brush modes
- **Pressure sensitivity**: Tablet/stylus support
- **Non-destructive workflow**: Undo/redo with delta stacks

### GPU Pipeline
- PyTorch tensors resident on GPU
- Fragment shader effects in Flutter
- Ping-pong texture buffers for persistent edits
- Efficient ROI-based updates

## Development Setup

### Prerequisites
- Python 3.8+ with CUDA support
- Flutter SDK 3.0+
- NVIDIA GPU with CUDA 11.0+
- Platform-specific development tools

### Installation

1. **Install NSGT library**:
   ```bash
   cd nsgt
   pip install -e .
   ```

2. **Install DSP service dependencies**:
   ```bash
   cd spectrogram_editor
   pip install -r requirements.txt
   ```

3. **Verify GPU setup**:
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   python -c "import nnAudio; print('nnAudio imported successfully')"
   ```

4. **Setup Flutter**:
   ```bash
   cd flutter_app
   flutter pub get
   flutter config --enable-windows-desktop  # or macos/linux
   ```

### Running the Application

1. **Start DSP service**:
   ```bash
   cd dsp_service
   python main.py
   ```

2. **Run Flutter app**:
   ```bash
   cd flutter_app
   flutter run -d windows  # or macos/linux
   ```

## Technical Details

### Memory Layout
- Spectrograms stored as `(time, frequency, channels)` tensors
- Magnitude in linear or log scale (R channel)
- Phase or instantaneous frequency (G channel)
- Consistent GPU memory layout for texture mapping

### IPC Architecture
- **Option A**: Platform Channels for direct texture sharing
- **Option B**: gRPC service for long-running DSP operations
- **Option C**: Hybrid approach with shared memory buffers

### Brush Implementation
- GPU-resident convolution kernels in PyTorch
- Fragment shader effects for real-time preview
- ROI-based processing for performance
- Pressure-sensitive falloff curves

### Performance Optimizations
- Tiled spectrogram processing (256×256 or 512×512 tiles)
- Delta-based updates to minimize texture uploads
- GPU-resident mathematical operations
- Asynchronous processing pipeline

## File Structure

```
spectrogram_editor/
├── dsp_service/           # Python DSP backend
│   ├── nsgt_processor.py  # NSGT integration
│   ├── nnaudio_gpu.py     # nnAudio GPU pipeline
│   ├── texture_bridge.py  # GPU texture interface
│   ├── brush_kernels.py   # Brush effect implementations
│   └── main.py           # Service entry point
├── flutter_app/          # Flutter desktop application
│   ├── lib/
│   │   ├── widgets/       # Custom UI widgets
│   │   ├── shaders/       # Shader management
│   │   └── platform/      # Platform channel integration
│   ├── shaders/          # Fragment shader assets
│   └── pubspec.yaml      # Flutter dependencies
├── platform_channels/    # Native platform integration
│   ├── windows/          # Windows-specific plugins
│   ├── macos/            # macOS-specific plugins
│   └── linux/            # Linux-specific plugins
├── shaders/              # GPU shader source files
│   ├── blur.frag         # Gaussian blur fragment shader
│   ├── warp.frag         # Displacement warp shader
│   └── texture.vert      # Basic vertex shader
└── docs/                 # Architecture documentation
    ├── architecture.md   # Detailed system design
    ├── gpu_pipeline.md   # GPU processing details
    └── brush_design.md   # Brush tool specifications
```

## References

- [NSGT Repository](https://github.com/grrrr/nsgt)
- [nnAudio Repository](https://github.com/KinWaiCheuk/nnAudio)
- [Flutter Fragment Shaders](https://docs.flutter.dev/ui/design/graphics/fragment-shaders)
- [Flutter Desktop Platform Integration](https://docs.flutter.dev/platform-integration/desktop)
- [PyTorch DLPack](https://docs.pytorch.org/docs/stable/dlpack.html)

## License

This project builds upon the NSGT library covered by the Artistic License 2.0.