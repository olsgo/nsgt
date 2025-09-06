# System Architecture

## Overview

The Real-time Spectrogram Editor follows a multi-component architecture designed for high-performance audio processing and interactive visualization:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Flutter UI    │    │  DSP Service    │    │  GPU Pipeline   │
│   Desktop App   │◄──►│   (Python)      │◄──►│   (PyTorch)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         │                        │                        │
    ┌────▼────┐              ┌────▼────┐              ┌────▼────┐
    │Fragment │              │  NSGT   │              │ nnAudio │
    │Shaders  │              │Processor│              │  GPU    │
    └─────────┘              └─────────┘              └─────────┘
```

## Component Details

### 1. Flutter Desktop UI
- **Purpose**: Cross-platform user interface with real-time GPU rendering
- **Technologies**: Flutter 3.x, Fragment Shaders, Platform Channels
- **Responsibilities**:
  - Interactive spectrogram display
  - Brush tool controls and parameter adjustment
  - Real-time shader-based effects preview
  - File I/O and user interaction
  - Communication with DSP service

### 2. DSP Service (Python)
- **Purpose**: High-performance audio processing backend
- **Technologies**: Python 3.8+, FastAPI, WebSockets, AsyncIO
- **Responsibilities**:
  - NSGT and nnAudio transform computation
  - GPU tensor management and memory optimization
  - Brush effect kernel execution
  - IPC bridge to Flutter application
  - Audio file loading and preprocessing

### 3. GPU Pipeline (PyTorch)
- **Purpose**: GPU-accelerated tensor operations and spectrogram processing
- **Technologies**: PyTorch 2.x, CUDA, nnAudio, CuPy
- **Responsibilities**:
  - GPU-resident spectrogram tensors
  - Real-time brush effect computation
  - Memory-efficient ROI processing
  - Texture format conversion

## Data Flow Architecture

### 1. Audio Loading and Analysis
```
Audio File → DSP Service → NSGT/nnAudio → GPU Tensors → Texture Bridge → Flutter UI
```

### 2. Interactive Editing
```
User Input → Flutter UI → WebSocket → DSP Service → GPU Kernels → Tensor Update → Texture Refresh
```

### 3. Real-time Preview
```
Brush Input → Fragment Shader → GPU Effect → Canvas Render → Visual Feedback
```

## Memory Management

### GPU Memory Layout
- **Magnitude Tensor**: `(frequency_bins, time_frames)` in float32
- **Phase Tensor**: `(frequency_bins, time_frames)` in float32  
- **Texture Format**: RGBA8 with magnitude in R, phase in G channels
- **Brush Kernels**: Pre-computed and cached on GPU

### CPU-GPU Communication
- **Zero-copy where possible**: DLPack for tensor sharing within Python
- **Minimal transfers**: Only final texture updates sent to Flutter
- **ROI-based updates**: Process only affected regions during brush operations
- **Ping-pong buffers**: Maintain working and display textures separately

## IPC Architecture

### WebSocket Protocol
```json
{
  "command": "generate_spectrogram",
  "audio_data": [float_array],
  "transform_type": "nsgt|stft|mel|cqt|vqt"
}

{
  "command": "apply_brush", 
  "brush_type": "blur|warp|enhance|filter",
  "roi": {"x": int, "y": int, "width": int, "height": int},
  "params": {"radius": float, "strength": float, ...}
}
```

### Platform Channels (Alternative)
For tighter integration, native platform channels can provide:
- Direct texture sharing between native GPU context and Flutter
- Lower latency for real-time brush operations
- Platform-specific optimizations (Metal on macOS, DirectX on Windows)

## Performance Optimizations

### 1. Tiled Processing
- Divide large spectrograms into 256×256 or 512×512 tiles
- Update only dirty tiles during brush operations
- Parallelize tile processing across GPU cores

### 2. Asynchronous Pipeline
- Overlap computation and rendering using async/await
- Background processing for non-critical operations
- Smooth UI updates independent of DSP processing

### 3. Shader Optimization
- Use separable convolution for blur effects
- Leverage GPU texture cache with proper memory access patterns
- Minimize texture fetches through linear sampling

### 4. Memory Optimization
- Keep working tensors resident on GPU
- Use half-precision (float16) where accuracy permits
- Implement texture atlasing for small regions

## Scalability Considerations

### Multi-GPU Support
- Distribute large spectrograms across multiple GPU devices
- Use NCCL or similar for multi-GPU tensor operations
- Load balance based on GPU memory and compute capability

### Streaming Audio
- Process audio in real-time chunks using NSGT_sliced
- Maintain sliding window of spectrogram data
- Implement efficient ring buffer for continuous processing

### Cloud Deployment
- Containerize DSP service with GPU runtime support
- Use GPU-enabled cloud instances (AWS P3, Google Cloud GPU)
- Implement WebRTC for low-latency audio streaming

## Security and Error Handling

### Input Validation
- Sanitize all user inputs and file uploads
- Validate audio file formats and sizes
- Limit GPU memory usage to prevent system overload

### Error Recovery
- Graceful degradation when GPU is unavailable
- Automatic reconnection for DSP service communication
- Undo/redo stack for brush operations

### Resource Management
- Implement texture garbage collection
- Monitor GPU memory usage and alert on limits
- Timeout protection for long-running operations