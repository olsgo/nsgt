# GPU Pipeline Design

## Overview

The GPU pipeline is the performance-critical component responsible for real-time spectrogram processing and brush effect application. It leverages PyTorch's CUDA integration and nnAudio's optimized transforms to achieve interactive frame rates.

## Pipeline Architecture

```
Audio Input → Transform → GPU Tensors → Brush Kernels → Texture Output
     ↓              ↓           ↓             ↓              ↓
   CPU/Disk    NSGT/nnAudio  VRAM Store   GPU Compute   Flutter UI
```

## Transform Pipeline

### 1. NSGT (Non-Stationary Gabor Transform)
```python
# Frequency-adaptive analysis
scale = LogScale(fmin=80, fmax=8000, bins=12)
nsgt = CQ_NSGT(fmin, fmax, bins, sr, length)
coeffs = nsgt.forward(audio)  # CPU computation
magnitude = torch.abs(torch.tensor(coeffs)).cuda()
phase = torch.angle(torch.tensor(coeffs)).cuda()
```

**Characteristics**:
- Variable time-frequency resolution
- Logarithmic frequency spacing
- Complex-valued coefficients
- CPU-based computation (legacy NSGT)

### 2. nnAudio GPU Transforms
```python
# GPU-accelerated STFT
stft = nnAudio.STFT(sr=44100, n_fft=2048, hop_length=512).cuda()
spec = stft(audio_tensor)  # Direct GPU computation
magnitude = torch.sqrt(spec[..., 0]**2 + spec[..., 1]**2)
phase = torch.atan2(spec[..., 1], spec[..., 0])
```

**Supported Transforms**:
- **STFT**: Uniform time-frequency resolution
- **Mel Spectrogram**: Perceptually-motivated frequency scaling
- **CQT**: Logarithmic frequency resolution
- **VQT**: Variable-Q for musical analysis

## Memory Layout

### Tensor Organization
```
Magnitude Tensor: [frequency_bins, time_frames]
Phase Tensor:     [frequency_bins, time_frames]
Working Buffer:   [frequency_bins, time_frames]
Original Backup:  [frequency_bins, time_frames]
```

### Texture Mapping
```
RGBA Texture Layout:
R Channel: Normalized magnitude [0, 1]
G Channel: Normalized phase [-π, π] → [0, 1]
B Channel: Reserved for future use
A Channel: Alpha = 1.0
```

### Memory Optimization
- **Unified Memory**: Use `torch.cuda.set_per_process_memory_fraction(0.8)`
- **Memory Pool**: Pre-allocate tensor pools to avoid fragmentation
- **ROI Processing**: Work on sub-regions to reduce memory bandwidth

## Brush Effect Kernels

### 1. Gaussian Blur
```python
def gaussian_blur_kernel(magnitude, center, radius, sigma):
    # Separable convolution for efficiency
    kernel_1d = gaussian_kernel_1d(radius, sigma)
    
    # Horizontal pass
    blurred_h = F.conv1d(magnitude.unsqueeze(0), 
                        kernel_1d.view(1, 1, -1), 
                        padding=radius)
    
    # Vertical pass  
    blurred = F.conv1d(blurred_h.transpose(-1, -2),
                      kernel_1d.view(1, 1, -1),
                      padding=radius).transpose(-1, -2)
    
    return blurred.squeeze(0)
```

**Performance Notes**:
- Separable implementation reduces O(r²) to O(r) complexity
- GPU memory access is optimized for coalesced reads
- Supports variable kernel size based on brush radius

### 2. Displacement Warp
```python
def displacement_warp_kernel(magnitude, center, displacement, direction):
    h, w = magnitude.shape
    
    # Create displacement field
    y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w))
    coords = torch.stack([x_coords, y_coords], dim=-1).float().cuda()
    
    # Apply radial falloff
    distance = torch.norm(coords - center, dim=-1)
    falloff = torch.exp(-distance**2 / (2 * radius**2))
    
    # Compute displacement vectors
    displacement_field = direction * displacement * falloff.unsqueeze(-1)
    
    # Sample with displacement
    new_coords = coords + displacement_field
    normalized_coords = 2 * new_coords / torch.tensor([w-1, h-1]) - 1
    
    return F.grid_sample(magnitude.unsqueeze(0).unsqueeze(0),
                        normalized_coords.unsqueeze(0),
                        mode='bilinear', 
                        padding_mode='border').squeeze()
```

### 3. Spectral Filtering
```python
def spectral_filter_kernel(magnitude, filter_type, cutoff):
    # Forward FFT
    magnitude_fft = torch.fft.fft2(magnitude)
    
    # Create frequency mask
    freqs = torch.fft.fftfreq(magnitude.shape[-1]).cuda()
    freq_magnitude = torch.abs(freqs)
    
    if filter_type == 'lowpass':
        mask = (freq_magnitude <= cutoff).float()
    elif filter_type == 'highpass':
        mask = (freq_magnitude >= cutoff).float()
    
    # Apply filter and inverse FFT
    filtered_fft = magnitude_fft * mask
    return torch.fft.ifft2(filtered_fft).real
```

## ROI (Region of Interest) Processing

### ROI Extraction
```python
def extract_roi(tensor, x, y, width, height):
    """Extract region with bounds checking"""
    x_start = max(0, x)
    y_start = max(0, y) 
    x_end = min(tensor.shape[1], x + width)
    y_end = min(tensor.shape[0], y + height)
    
    return tensor[y_start:y_end, x_start:x_end]

def update_roi(tensor, roi_data, x, y):
    """Update tensor region with new data"""
    h, w = roi_data.shape
    tensor[y:y+h, x:x+w] = roi_data
```

### Batch ROI Processing
```python
def process_stroke_batch(magnitude, stroke_points, brush_params):
    """Process multiple brush applications efficiently"""
    results = []
    
    for point in stroke_points:
        roi = extract_roi(magnitude, point.x, point.y, 
                         brush_params.width, brush_params.height)
        
        processed_roi = apply_brush_kernel(roi, brush_params)
        results.append((processed_roi, point.x, point.y))
    
    # Batch update
    for roi_data, x, y in results:
        update_roi(magnitude, roi_data, x, y)
```

## Texture Bridge Implementation

### Tensor to Texture Conversion
```python
def tensor_to_texture(magnitude, phase):
    # Normalize magnitude to [0, 1]
    mag_min, mag_max = magnitude.min(), magnitude.max()
    magnitude_norm = (magnitude - mag_min) / (mag_max - mag_min)
    
    # Normalize phase to [0, 1] from [-π, π]
    phase_norm = (phase + torch.pi) / (2 * torch.pi)
    
    # Pack into RGBA
    rgba = torch.zeros((*magnitude.shape, 4), dtype=torch.float32, device=magnitude.device)
    rgba[..., 0] = magnitude_norm  # R: magnitude
    rgba[..., 1] = phase_norm      # G: phase
    rgba[..., 2] = 0.0             # B: reserved
    rgba[..., 3] = 1.0             # A: alpha
    
    # Convert to 8-bit
    return (rgba * 255).clamp(0, 255).byte()
```

### Colormap Application
```python
def apply_viridis_colormap(magnitude):
    """Apply Viridis colormap for visualization"""
    # Viridis color control points
    colors = torch.tensor([
        [0.267004, 0.004874, 0.329415],  # Dark purple
        [0.229739, 0.322361, 0.545706],  # Blue
        [0.127568, 0.566949, 0.550556],  # Teal
        [0.369214, 0.788888, 0.382914],  # Green
        [0.993248, 0.906157, 0.143936],  # Yellow
    ]).cuda()
    
    # Interpolate colors based on magnitude
    indices = magnitude * (len(colors) - 1)
    indices_floor = indices.floor().long().clamp(0, len(colors) - 2)
    indices_frac = indices - indices_floor.float()
    
    color_low = colors[indices_floor]
    color_high = colors[indices_floor + 1]
    
    return color_low + indices_frac.unsqueeze(-1) * (color_high - color_low)
```

## Performance Metrics and Optimization

### Target Performance
- **Transform Generation**: <100ms for 2-second audio
- **Brush Application**: <16ms per stroke (60 FPS)
- **Texture Updates**: <5ms per frame
- **Memory Usage**: <2GB VRAM for typical spectrograms

### Optimization Strategies

1. **Kernel Fusion**: Combine multiple operations into single GPU kernels
2. **Memory Coalescing**: Ensure aligned memory access patterns
3. **Occupancy Optimization**: Balance thread blocks for GPU utilization
4. **Stream Overlap**: Use CUDA streams for computation/memory overlap

### Profiling Tools
```python
# PyTorch profiler integration
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, 
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    result = brush_kernel(magnitude, params)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Error Handling and Fallbacks

### GPU Memory Management
```python
def safe_gpu_operation(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        # Retry with smaller batch size or fallback to CPU
        return cpu_fallback_func(*args, **kwargs)
```

### Device Compatibility
```python
def get_optimal_device():
    if torch.cuda.is_available():
        # Prefer GPU with most memory
        gpu_memory = [torch.cuda.get_device_properties(i).total_memory 
                     for i in range(torch.cuda.device_count())]
        return f"cuda:{gpu_memory.index(max(gpu_memory))}"
    else:
        return "cpu"
```