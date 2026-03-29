#pragma once
#include "../tensor/Tensor.h"
#include "../ops/MatMul.h"
#include <cmath>

// Softmax
inline Tensor softmax(const Tensor& input) {
    Tensor output = input;

    int rows = input.shape[0];
    int cols = input.shape[1];

    for (int i = 0; i < rows; i++) {
        float max_val = -1e9;

        for (int j = 0; j < cols; j++) {
            max_val = std::max(max_val, input.data[i * cols + j]);
        }

        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            float val = std::exp(input.data[i * cols + j] - max_val);
            output.data[i * cols + j] = val;
            sum += val;
        }

        for (int j = 0; j < cols; j++) {
            output.data[i * cols + j] /= sum;
        }
    }

    return output;
}

inline Tensor attention(const Tensor& Q, const Tensor& K, const Tensor& V) {
    Tensor scores = matmul(Q, K);

    float scale = std::sqrt(K.shape[1]);
    for (auto& x : scores.data) x /= scale;

    Tensor probs = softmax(scores);

    Tensor output = matmul(probs, V);

    return output;
}