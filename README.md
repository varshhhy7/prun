# Prun

A minimal transformer inference engine built from scratch in C++ to understand deep learning systems at a fundamental level.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![CMake](https://img.shields.io/badge/CMake-3.10+-green.svg)](https://cmake.org/)

## What is This?

Most ML engineers interact with transformers through high-level APIs and pretrained models. But what actually happens under the hood? How does memory layout affect performance? Where are the real bottlenecks?

**Prun** is a ground-up implementation of a transformer inference pipeline in C++. It strips away the abstraction layers to expose exactly what's happening: tensor operations in memory, how data moves through the computation graph, and where efficiency really comes from.

This is built to explore:
- How inference engines work at the systems level
- Why memory layout and data movement matter more than model architecture
- Real performance bottlenecks in neural network computation
- Optimization strategies that actually move the needle (quantization, buffer reuse, sparse routing)
- What it takes to run models efficiently on real hardware

Not a framework. Not a library. Just a ground-up implementation to understand the fundamentals.

## What's Implemented

- **Minimal Tensor System** — Flat memory layout with manual indexing (row-major). Row-major keeps your memory access linear and cache-friendly
- **Core Operations** — MatMul, softmax, transpose. Hand-written to understand what's actually happening at the CPU level
- **Scaled Dot-Product Attention** — The heart of transformers: QKᵀ / √d → softmax → V. Implemented to profile each stage
- **Linear Layers** — Weights, biases, and operations. Manual computation to see where memory bottlenecks emerge
- **Feedforward Networks** — Two-layer MLPs with activation functions. Profile these separately to understand relative cost
- **Full Transformer Blocks** — Attention + feedforward + layer structure combined. Where the whole thing comes together
- **Mixture of Experts (MoE)** — Gating layer that routes inputs to expert networks. Sparse computation and load balancing challenges exposed

## Why This Way?

When you use PyTorch or TensorFlow, you don't see:
- How memory is actually laid out
- Where data copies happen
- What makes one MatMul implementation 10x faster than another
- The cost of allocations vs. reuse
- How routing decisions affect expert utilization
- What quantization actually does to your computation

Prun puts all of that front and center.

## Requirements

- **Compiler**: C++17 compatible (GCC 7+, Clang 5+, MSVC 2017+)
- **Build System**: CMake 3.10+
- **Platform**: Windows, macOS, Linux (no GPU dependencies currently)

## Building

```bash
# Clone and build
git clone https://github.com/yourusername/prun.git
cd prun
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

The Release build matters here—it enables optimizations that reveal what's actually fast vs slow.

## Quick Start

### Running an Inference Pass

```cpp
#include "tensor/Tensor.h"
#include "model/TransformerBlock.h"

int main() {
    // Single inference forward pass through a transformer block
    TransformerBlock block(512);  // 512-dim embedding

    Tensor input({1, 512});       // Batch size 1
    Tensor output = block.forward(input);

    return 0;
}
```

### Profiling Attention Cost

```cpp
#include "tensor/Tensor.h"
#include "model/Attention.h"
#include "utils/Timer.h"

int main() {
    Tensor Q({64, 64});
    Tensor K({64, 64});
    Tensor V({64, 64});

    // Initialize data...

    Timer timer;
    Tensor out = attention(Q, K, V);
    auto elapsed = timer.elapsed();  // See exactly how long attention takes

    return 0;
}
```

This is the whole point—measure everything. Understanding where time actually goes is where optimization begins.

## Architecture

### Project Structure

```
prun/
├── tensor/              # Core tensor data structure
│   └── Tensor.h
├── ops/                 # Primitive operations
│   └── MatMul.h
├── layers/              # Neural network layers
│   ├── Linear.h         # Fully connected layer
│   └── FeedForward.h    # Feed-forward network
├── model/               # High-level model components
│   ├── Attention.h      # Multi-head attention
│   ├── TransformerBlock.h
│   └── MoE.h            # Mixture of experts
├── core/                # Execution and model management
│   ├── Model.h
│   └── Executor.h
├── utils/               # Utility functions
│   ├── Logger.h
│   └── Timer.h
├── examples/            # Usage examples
├── benchmarks/          # Performance benchmarking
├── tests/               # Unit tests
└── CMakeLists.txt
```

### Core Components

**Tensor**  
The fundamental unit. Shape + flat data array. When you iterate through it, you're iterating through memory. No black boxes.

**MatMul**  
Three nested loops. O(n³) on paper, but everything else is about making those loops cache-efficient. This is where 80% of inference time lives.

**Attention**  
QKᵀ (matmul) → scale → softmax → V (matmul). Profile each stage. Softmax seems cheap until you realize it's a synchronization point that kills parallelism.

**Transformer Block**  
Attention + residual + LayerNorm + FFN + residual. See how much of the cost is in attention vs feedforward.

**MoE (Mixture of Experts)**  
Gating layer that routes each token to the highest-scoring expert. Sparse computation wins only if load is balanced. Routing collapse kills the speedup.

### Memory Model

- **Flat tensors**: No nested vectors. Single contiguous `float*` with shape info
- **Row-major layout**: Sequential memory access = cache hits
- **Manual indexing**: You see exactly how `[i,j]` becomes `data[i*cols+j]`
- **No dynamic allocation during inference**: Buffers preallocated upfront

## Optimization Path

Here's what we're working toward:

1. **Profile Everything** — Understand the current cost breakdown before touching anything
2. **MatMul First** — This is the bottleneck. Cache-friendly tiling, blocking, better memory access
3. **Buffer Reuse** — Stop allocating and deallocating during inference. Preallocate, reuse
4. **Quantization** — INT8 inference. Float tensors → quantized, see the speed/accuracy tradeoff
5. **MoE Optimization** — Fix routing collapse, better load balancing across experts
6. **Sparse Kernels** — Only compute what matters
7. **SIMD & Intrinsics** — When you've profiled down to the actual CPU bottlenecks

Right now this is a ground-truth implementation. The goal is to progressively optimize without losing clarity about what changed and why.

## Contributing

If you're interested in inference optimization, this is an open invitation to contribute:

- **New optimization techniques** — Novel memory layouts, quantization strategies, kernel designs
- **Profiling and analysis** — Identify bottlenecks on different hardware, platforms
- **Sparse computation** — Better routing, pruning, dynamic execution
- **SIMD implementations** — Hand-tuned kernels for specific operations
- **Experimental features** — Quantization variants, new MoE routing, gradient checkpointing

The bar is simple: show your work. Measure before and after. Explain why something is faster.

## Frequently Asked Questions

**Q: Should I use this in production?**  
A: No. This is a learning tool and sandbox for inference optimization research. For production, use optimized backends like TensorRT, CoreML, ONNX Runtime.

**Q: Why C++?**  
A: Because inference bottlenecks are memory and CPU-level concerns. C++ gives you the control to see and optimize those. Python doesn't.

**Q: Can this run actual models?**  
A: Not yet—we'd need weight loaders, quantization that maps real model weights, etc. This is currently a transformer building block library, not an inference framework that runs trained models.

**Q: Will you add GPU support?**  
A: GPU is a different beast. Right now CPU is interesting enough—most edge devices don't have GPUs. GPU optimization is its own project.

**Q: How does this compare to TensorRT / TVM / llama.cpp?**  
A: Not meaningfully yet. Those are production systems. Prun is "understand the fundamentals" stage. Eventually the interesting comparison will be on specific optimizations.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```
@software{prun2026,
  title = {Prun: A Minimal Transformer Inference Engine},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/prun}
}
```

## References

Understanding transformer inference at the systems level requires reading about:

- **Vaswani et al. (2017)** — Attention is All You Need: The foundational transformer paper
- **Shazeer et al. (2017)** — Outrageously Large Neural Networks for Efficient Conditional Computation: MoE fundamentals
- **Roark et al.** — Memory-Efficient Attention on GPUs: Understanding memory bottlenecks in attention
- **Bone et al.** — MatMul Optimization: The real bottleneck in neural network inference
- **Stock et al.** — And the Bit Goes Down: Post-training quantization and inference optimization

## Inspirations

Built with respect for systems-level optimization work in:
- llama.cpp (bringing model inference to CPU seriously)
- ONNX Runtime (understanding inference backend design)
- TVM (compiler approach to kernels)
- PyTorch C++ API (how modern inference APIs look)

## Feedback & Discussion

- **Have an optimization idea?** Open an issue with benchmarks
- **Found a bottleneck?** Document it. Measure it. Show us the profile
- **Want to contribute an optimization?** Include before/after numbers, not just code
- **Questions about how something works?** Read the header files, run the benchmarks, profile it yourself

Serious inquiries only. This is about understanding systems, not collecting features.
