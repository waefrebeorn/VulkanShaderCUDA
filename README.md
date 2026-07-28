# VulkanShaderCUDA

**VulkanShaderCUDA** is a high-performance tensor computation framework that implements PyTorch-like operations using Vulkan compute shaders. The project aims to provide a vendor-agnostic alternative to CUDA-based deep learning frameworks, enabling GPU acceleration across a wider range of hardware.

## Current Status

### ✅ Working Features
- **Core Operations**
  - Element-wise addition with near-zero overhead
  - Matrix multiplication (optimized with shared memory tiling)
  - ReLU activation function
  - Sigmoid activation function (numerically stable implementation)
  - All core operations validated against PyTorch with matching precision

- **Memory Management**
  - Zero-copy buffer pooling system
  - Efficient resource reuse
  - Automated cleanup

### 🚧 Under Development
- **Advanced Operations**
  - Softmax (numerical stability improvements in progress)
  - MaxPool2D (implementation refinements ongoing)
  - Conv2D (tensor reshape handling in progress)
  
- **Gradient Computations**
  - Element-wise operation gradients complete
  - Matrix multiplication gradients working
  - Advanced operation gradients in development

## Architecture

### Core Design
- Memory-first architecture with buffer pooling
- Vulkan compute shader-based operations
- PyBind11 integration for seamless NumPy interop
- SPIR-V shader compilation pipeline

### Performance Features
- Shared memory utilization in compute shaders
- Workgroup size optimization
- Asynchronous command buffer execution
- Minimal host-device transfers

## Prerequisites

1. **Vulkan SDK**:
   ```bash
   # Download and install from:
   https://vulkan.lunarg.com/sdk/home
   # Minimum version: 1.3.296.0
   ```

2. **Python Environment**:
   ```bash
   pip install numpy pybind11 torch torchvision torchaudio
   ```

## Quick Start

### Windows Setup
```batch
setup_vulkan_project.bat
```

The script handles:
- Vulkan SDK environment configuration
- Python virtual environment setup
- Dependency installation
- SPIR-V shader compilation
- Backend module building

## Usage Examples

### Basic Operations
```python
import numpy as np
from vulkan_backend import init_vulkan, vulkan_add, vulkan_matmul

# Initialize Vulkan
init_vulkan()

# Element-wise Addition
a = np.random.rand(1024).astype(np.float32)
b = np.random.rand(1024).astype(np.float32)
c = np.zeros_like(a)
vulkan_add(a, b, c)

# Matrix Multiplication
M, K, N = 128, 256, 128
a = np.random.rand(M, K).astype(np.float32)
b = np.random.rand(K, N).astype(np.float32)
c = np.zeros((M, N), dtype=np.float32)
vulkan_matmul(a.flatten(), b.flatten(), c.flatten(), M, K, N)

# Activation Functions
vulkan_relu(input_data.flatten(), output.flatten())
vulkan_sigmoid(input_data.flatten(), output.flatten())
```

## Development Roadmap

### Short-term Goals
1. Stabilize Softmax implementation
2. Complete Conv2D tensor handling
3. Optimize MaxPool2D implementation
4. Add BatchNorm support

### Medium-term Goals
1. Implement automatic differentiation
2. Add layer abstractions
3. Support model import/export
4. Optimize memory patterns for training

### Long-term Vision
1. Full PyTorch model compatibility
2. Custom model deployment pipeline
3. Mobile GPU optimization
4. Distributed computing support

## Technical Details

### Memory Management
- Smart buffer pooling system
- Automatic resource cleanup
- Zero-copy operations where possible
- Shared memory optimization

### Shader Implementation
- SPIR-V based compute shaders
- Workgroup optimization
- Local memory utilization
- Batched operation support

## Contributing

Contributions are welcome! We're particularly interested in:

1. Numerical stability improvements
2. Memory optimization techniques
3. New operation implementations
4. Testing and validation

## Support

For technical support:
- Discord: Contact **waefrebeorn**
- Submit issues through GitHub

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Vulkan SDK team for comprehensive documentation
- PyBind11 team for Python binding capabilities
- PyTorch team for architectural inspiration
- Open-source ML community for testing and feedback

---

## License

This project is licensed under the **Waefrebeorn Umbrella License v3.0**.
See the [LICENSE](LICENSE) file for the full license text.

The Waefrebeorn Umbrella License is a custom source-available license.
It is not OSI-approved and not FSF-approved.
