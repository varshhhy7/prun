# Prun

A lightweight C++ library for neural network operations and attention mechanisms.

## Overview

Prun provides core implementations of tensor operations, matrix multiplication, and transformer attention mechanisms. It's designed as a minimal, efficient foundation for building and experimenting with neural network architectures.

## Features

- **Tensor Operations**: Multi-dimensional tensor support with basic operations
- **Matrix Multiplication**: Optimized matrix multiplication routines
- **Attention Mechanism**: Transformer-style attention implementation
- **Modern C++**: Built with C++17 standard

## Requirements

- C++17 compiler
- CMake 3.10 or later

## Building

```bash
mkdir Build
cd Build
cmake ..
cmake --build .
```

## Usage

```cpp
#include "tensor/Tensor.h"
#include "ops/MatMul.h"
#include "model/Attention.h"

// Create tensors
Tensor Q({2, 2});
Tensor K({2, 2});
Tensor V({2, 2});

// Set data
Q.data = {1, 0, 0, 1};
K.data = {1, 2, 3, 4};
V.data = {5, 6, 7, 8};

// Compute attention
Tensor output = attention(Q, K, V);
output.print();
```

## Project Structure

```
prun/
├── tensor/        # Tensor data structure and operations
├── ops/           # Basic operations (MatMul, etc.)
├── model/         # Higher-level models (Attention, etc.)
├── CMakeLists.txt # Build configuration
└── main.cpp       # Example usage
```

## License

See LICENSE file for details.
