# Brush Tool Design Specification

## Overview

The brush tools provide intuitive, real-time editing capabilities for spectrograms, allowing users to apply various effects with mouse or stylus input. Each brush implements both GPU-based processing for permanent edits and shader-based preview for immediate visual feedback.

## Brush Tool Architecture

### Base Brush Interface
```python
class BrushTool:
    def __init__(self, device: torch.device):
        self.device = device
        self.radius = 20.0
        self.strength = 0.5
        self.pressure_sensitivity = True
        
    async def apply(self, magnitude: torch.Tensor, 
                   roi: Dict[str, int], 
                   params: Dict[str, float]) -> torch.Tensor:
        raise NotImplementedError
        
    def get_preview_shader(self) -> str:
        raise NotImplementedError
```

## Individual Brush Specifications

### 1. Blur Brush

**Purpose**: Apply Gaussian blur to smooth spectral content and reduce noise.

**Parameters**:
- `blur_radius`: Kernel radius in pixels (1.0 - 20.0)
- `strength`: Effect intensity (0.0 - 1.0)
- `preserve_energy`: Maintain total spectral energy (boolean)

**Implementation**:
```python
class BlurBrush(BrushTool):
    async def apply(self, magnitude, roi, params):
        blur_radius = params.get('blur_radius', 5.0)
        strength = params.get('strength', 0.5)
        
        # Extract region
        region = magnitude[roi['y']:roi['y']+roi['height'], 
                          roi['x']:roi['x']+roi['width']]
        
        # Create Gaussian kernel
        kernel = self._create_gaussian_kernel(blur_radius)
        
        # Apply convolution with reflection padding
        blurred = F.conv2d(
            region.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding='same'
        ).squeeze()
        
        # Blend with original
        return (1 - strength) * region + strength * blurred
```

**Shader Preview** (`blur.frag`):
```glsl
vec4 gaussianBlur(sampler2D tex, vec2 coord, float radius) {
    vec4 color = vec4(0.0);
    float totalWeight = 0.0;
    vec2 texelSize = 1.0 / textureSize(tex, 0);
    
    int kernelSize = int(ceil(radius));
    for (int x = -kernelSize; x <= kernelSize; x++) {
        for (int y = -kernelSize; y <= kernelSize; y++) {
            vec2 offset = vec2(x, y) * texelSize;
            float weight = exp(-dot(offset, offset) / (2.0 * radius * radius));
            color += texture(tex, coord + offset) * weight;
            totalWeight += weight;
        }
    }
    return color / totalWeight;
}
```

### 2. Warp/Displacement Brush

**Purpose**: Create fluid distortion effects by displacing pixels based on vector fields.

**Parameters**:
- `displacement`: Maximum displacement in pixels (0.5 - 10.0)
- `direction`: Displacement direction vector (x, y)
- `falloff`: Edge falloff curve steepness (0.1 - 2.0)
- `vector_field`: Optional custom displacement field

**Implementation**:
```python
class WarpBrush(BrushTool):
    async def apply(self, magnitude, roi, params):
        displacement = params.get('displacement', 2.0)
        direction = params.get('direction', (1.0, 0.0))
        falloff = params.get('falloff', 0.5)
        
        h, w = roi['height'], roi['width']
        region = magnitude[roi['y']:roi['y']+h, roi['x']:roi['x']+w]
        
        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=self.device),
            torch.arange(w, device=self.device)
        )
        
        # Calculate displacement field
        center_x, center_y = w / 2, h / 2
        dist_x, dist_y = x_coords - center_x, y_coords - center_y
        distance = torch.sqrt(dist_x**2 + dist_y**2)
        
        # Radial falloff
        max_dist = min(w, h) / 2
        falloff_mask = torch.exp(-(distance / max_dist)**falloff)
        
        # Apply displacement
        dx, dy = direction
        displacement_x = dx * displacement * falloff_mask
        displacement_y = dy * displacement * falloff_mask
        
        # Sample with displacement using grid_sample
        new_x = (x_coords + displacement_x) / (w - 1) * 2 - 1
        new_y = (y_coords + displacement_y) / (h - 1) * 2 - 1
        grid = torch.stack([new_x, new_y], dim=-1).unsqueeze(0)
        
        warped = F.grid_sample(
            region.unsqueeze(0).unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='border'
        ).squeeze()
        
        return warped
```

### 3. Enhancement Brush

**Purpose**: Enhance spectral features by adjusting contrast and applying adaptive filtering.

**Parameters**:
- `enhancement_factor`: Contrast enhancement (0.5 - 3.0)
- `adaptive_threshold`: Percentile for feature detection (0.5 - 0.95)
- `preserve_dynamics`: Maintain relative amplitude relationships

**Implementation**:
```python
class EnhancementBrush(BrushTool):
    async def apply(self, magnitude, roi, params):
        enhancement = params.get('enhancement_factor', 1.5)
        threshold_pct = params.get('adaptive_threshold', 0.8)
        
        region = magnitude[roi['y']:roi['y']+roi['height'], 
                          roi['x']:roi['x']+roi['width']]
        
        # Calculate adaptive threshold
        threshold = torch.quantile(region, threshold_pct)
        
        # Apply power-law enhancement
        enhanced = torch.where(
            region > threshold,
            torch.pow(region, 1.0 / enhancement),
            region * 0.7  # Reduce below-threshold content
        )
        
        # Preserve energy if requested
        if params.get('preserve_energy', True):
            original_energy = region.sum()
            new_energy = enhanced.sum()
            if new_energy > 0:
                enhanced = enhanced * (original_energy / new_energy)
        
        return enhanced
```

### 4. Spectral Filter Brush

**Purpose**: Apply frequency-domain filtering (lowpass, highpass, bandpass, notch).

**Parameters**:
- `filter_type`: 'lowpass', 'highpass', 'bandpass', 'notch'
- `cutoff_frequency`: Normalized cutoff (0.0 - 1.0)
- `filter_order`: Filter sharpness (1 - 8)
- `bandwidth`: For bandpass/notch filters (0.05 - 0.5)

**Implementation**:
```python
class SpectralFilterBrush(BrushTool):
    async def apply(self, magnitude, roi, params):
        filter_type = params.get('filter_type', 'lowpass')
        cutoff = params.get('cutoff_frequency', 0.5)
        order = params.get('filter_order', 2)
        
        region = magnitude[roi['y']:roi['y']+roi['height'], 
                          roi['x']:roi['x']+roi['width']]
        
        # FFT to frequency domain
        region_fft = torch.fft.fft2(region)
        
        # Create frequency coordinates
        h, w = region.shape
        freq_y = torch.fft.fftfreq(h, device=self.device)
        freq_x = torch.fft.fftfreq(w, device=self.device)
        freq_mag = torch.sqrt(freq_x[None, :]**2 + freq_y[:, None]**2)
        
        # Design filter
        if filter_type == 'lowpass':
            filter_mask = 1.0 / (1.0 + (freq_mag / cutoff)**(2 * order))
        elif filter_type == 'highpass':
            filter_mask = 1.0 / (1.0 + (cutoff / freq_mag)**(2 * order))
        elif filter_type == 'bandpass':
            bandwidth = params.get('bandwidth', 0.1)
            low_freq = cutoff - bandwidth / 2
            high_freq = cutoff + bandwidth / 2
            lowpass = 1.0 / (1.0 + (freq_mag / high_freq)**(2 * order))
            highpass = 1.0 / (1.0 + (low_freq / freq_mag)**(2 * order))
            filter_mask = lowpass * highpass
        
        # Apply filter and return to spatial domain
        filtered_fft = region_fft * filter_mask
        filtered = torch.fft.ifft2(filtered_fft).real
        
        return filtered
```

## Brush Interaction Model

### Stroke Handling
```python
class StrokeProcessor:
    def __init__(self, brush: BrushTool):
        self.brush = brush
        self.stroke_points = []
        self.pressure_curve = []
        
    def add_point(self, x: float, y: float, pressure: float = 1.0):
        self.stroke_points.append((x, y))
        self.pressure_curve.append(pressure)
        
    async def apply_stroke(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Apply entire stroke with adaptive spacing"""
        result = magnitude.clone()
        
        for i, (x, y) in enumerate(self.stroke_points):
            pressure = self.pressure_curve[i]
            
            # Adapt brush parameters based on pressure
            brush_radius = self.brush.radius * pressure
            brush_strength = self.brush.strength * pressure
            
            # Calculate ROI
            roi = {
                'x': int(x - brush_radius),
                'y': int(y - brush_radius),
                'width': int(2 * brush_radius),
                'height': int(2 * brush_radius)
            }
            
            # Apply brush effect
            params = {
                'strength': brush_strength,
                'pressure': pressure,
                **self.brush.get_default_params()
            }
            
            processed_roi = await self.brush.apply(result, roi, params)
            
            # Update result
            self._update_roi(result, processed_roi, roi)
            
        return result
```

### Pressure Sensitivity
```python
def apply_pressure_curve(base_value: float, pressure: float, 
                        sensitivity: float = 1.0) -> float:
    """Apply pressure sensitivity curve"""
    # Smooth pressure curve using smoothstep
    smooth_pressure = pressure * pressure * (3 - 2 * pressure)
    
    # Apply sensitivity scaling
    scaled_pressure = 1.0 - sensitivity + sensitivity * smooth_pressure
    
    return base_value * scaled_pressure
```

## Performance Optimizations

### ROI Clipping and Bounds Checking
```python
def safe_roi_extract(tensor: torch.Tensor, x: int, y: int, 
                    width: int, height: int) -> Tuple[torch.Tensor, Dict]:
    """Extract ROI with proper bounds checking"""
    h, w = tensor.shape[-2:]
    
    # Clip coordinates to tensor bounds
    x_start = max(0, x)
    y_start = max(0, y)
    x_end = min(w, x + width)
    y_end = min(h, y + height)
    
    # Extract valid region
    roi_tensor = tensor[..., y_start:y_end, x_start:x_end]
    
    # Return region and mapping info
    mapping = {
        'original_x': x, 'original_y': y,
        'actual_x': x_start, 'actual_y': y_start,
        'actual_width': x_end - x_start,
        'actual_height': y_end - y_start
    }
    
    return roi_tensor, mapping
```

### Batch Processing for Continuous Strokes
```python
async def process_stroke_batch(magnitude: torch.Tensor, 
                              stroke_points: List[Tuple[float, float]],
                              brush: BrushTool) -> torch.Tensor:
    """Process multiple stroke points efficiently"""
    
    # Group nearby points to reduce redundant processing
    grouped_points = group_nearby_points(stroke_points, threshold=brush.radius * 0.5)
    
    # Process groups in parallel where possible
    tasks = []
    for group in grouped_points:
        task = process_point_group(magnitude, group, brush)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    # Merge results
    final_result = magnitude.clone()
    for result_roi, roi_info in results:
        update_roi(final_result, result_roi, roi_info)
    
    return final_result
```

## User Experience Considerations

### Visual Feedback
- **Real-time preview**: Shader-based effects show immediate feedback
- **Brush cursor**: Dynamic cursor showing brush size and type
- **Pressure visualization**: Opacity or size changes with stylus pressure
- **Undo/redo**: Non-destructive editing with operation history

### Customization Options
- **Brush presets**: Saved parameter combinations
- **Hotkeys**: Keyboard shortcuts for brush switching
- **Tablet support**: Pressure, tilt, and rotation sensitivity
- **Gesture recognition**: Multi-touch gestures for parameter adjustment

### Performance Monitoring
```python
@dataclass
class BrushPerformanceMetrics:
    avg_processing_time: float
    peak_memory_usage: int
    frames_per_second: float
    gpu_utilization: float
    
    def is_performance_acceptable(self) -> bool:
        return (self.avg_processing_time < 0.016 and  # 60 FPS
                self.frames_per_second > 30 and
                self.gpu_utilization < 0.9)
```