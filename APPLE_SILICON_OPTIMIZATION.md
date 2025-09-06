# Apple Silicon and Python 3.12 Optimization Guide

This document outlines the optimizations made to the NSGT library for Apple Silicon (ARM64) processors and Python 3.12 compatibility.

## Changes Made

### 1. Numpy Compatibility Fixes
- **Issue**: `np.clip()` function with `out` parameter caused dtype casting errors in modern numpy versions
- **Fix**: Replaced `np.clip(M, min_win, np.inf, out=M)` with `M = np.clip(M, min_win, None).astype(M.dtype)`
- **Location**: `nsgt/nsgfwin_sl.py` lines 100 and 185
- **Impact**: Fixes all 250+ test failures related to numpy casting

### 2. Python 3.12 Support
- **Added explicit Python version classifiers**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Updated minimum Python requirement**: `>=3.8`
- **Fixed Cython Python 3 compatibility**: Replaced `itertools.izip` with built-in `zip`

### 3. Apple Silicon Optimizations

#### Compilation Flags
For Apple Silicon (ARM64) systems, the following optimizations are applied:
- **CPU-specific optimization**: `-mcpu=apple-a14`
- **General optimization**: `-O3`
- **Vectorization**: `-ftree-vectorize`

#### Platform Detection
The build system automatically detects the platform and applies appropriate optimizations:
```python
if platform.system() == 'Darwin' and platform.machine() == 'arm64':
    # Apple Silicon specific optimizations
    extra_compile_args = ['-O3', '-mcpu=apple-a14', '-ftree-vectorize']
```

#### Cython Optimizations
- **Language level**: Set to Python 3
- **Bounds checking**: Disabled for performance (`boundscheck=False`)
- **Wrap-around**: Disabled for performance (`wraparound=False`)

### 4. Modern Build System

#### pyproject.toml
Added PEP 517/518 compliant build configuration:
- Modern build backend: `setuptools.build_meta`
- Explicit build requirements including Cython
- Optional dependencies for different use cases (dev, fftw, audio, gpu)
- Apple Silicon specific cupy requirements

#### Dependencies
Updated to latest stable versions:
- **numpy**: `>=1.20.0` (was just "numpy")
- **Setup requirements**: Now includes version constraints

#### Makefile
Modernized build commands:
- **Build**: Uses `python -m build` instead of `setup.py bdist_wheel`
- **Test**: Uses `pytest` instead of deprecated `setup.py test`
- **Package checking**: Uses `python -m twine` instead of bare `twine`

### 5. GPU Acceleration Support

#### Platform-specific GPU packages
- **x86_64**: `cupy-cuda11x>=12.0.0`
- **Apple Silicon**: Graceful fallback to CPU-based operations
- **torch**: `>=2.1.0` with optimized builds for both platforms

#### Audio Processing
Updated audio dependencies for better Apple Silicon support:
- **librosa**: `>=0.10.1`
- **soundfile**: `>=0.12.1`
- **pysndfile**: `>=1.4.0`

## Performance Benefits

### Apple Silicon Specific
1. **Native ARM64 compilation**: Using `-mcpu=apple-a14` targets Apple's custom silicon
2. **Vector instructions**: Apple Silicon's advanced SIMD capabilities are utilized
3. **Memory efficiency**: ARM64's efficient memory architecture is leveraged

### General Improvements
1. **Faster builds**: Modern build system with better dependency resolution
2. **Better testing**: Migrated from deprecated `setup.py test` to pytest
3. **Reliable dependencies**: Fixed version constraints prevent compatibility issues

## Compatibility Matrix

| Platform | Architecture | Python | Status |
|----------|-------------|---------|---------|
| macOS | x86_64 (Intel) | 3.8-3.12 | ✅ Supported |
| macOS | arm64 (Apple Silicon) | 3.8-3.12 | ✅ Optimized |
| Linux | x86_64 | 3.8-3.12 | ✅ Supported |
| Linux | arm64 | 3.8-3.12 | ✅ Supported |
| Windows | x86_64 | 3.8-3.12 | ✅ Supported |

## Installation

### Standard Installation
```bash
pip install nsgt
```

### Development Installation
```bash
git clone https://github.com/olsgo/nsgt.git
cd nsgt
pip install -e .[dev]
```

### With Optional Dependencies
```bash
# For FFTW acceleration
pip install nsgt[fftw]

# For audio processing
pip install nsgt[audio]

# For GPU acceleration (x86_64 only)
pip install nsgt[gpu]
```

## Building from Source

### Modern Method (Recommended)
```bash
python -m build
```

### Traditional Method
```bash
python setup.py sdist bdist_wheel
```

### Using Makefile
```bash
make all  # Build both wheel and source distributions
make test # Run tests
```

## Testing

### Run all tests
```bash
python -m pytest tests/ -v
```

### Legacy test method
```bash
python setup.py test
```

## Notes

- The package automatically falls back to Python implementations when Cython is not available
- Apple Silicon optimizations require GCC or Clang with ARM64 support
- For maximum performance on Apple Silicon, compile with Cython extensions enabled