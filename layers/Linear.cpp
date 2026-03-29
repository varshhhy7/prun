#include "Linear.h"
#include "../ops/MatMul.h"

Linear::Linear(int in_f, int out_f) {
    in_features = in_f;
    out_features = out_f;

    weight = Tensor({in_features, out_features});
    bias = Tensor({1, out_features});

    for (auto &x : weight.data) x = 0.1f;
    for (auto &x : bias.data) x = 0.0f;
}

Tensor Linear::forward(const Tensor& input) {
    Tensor out = matmul(input, weight);

    int rows = out.shape[0];
    int cols = out.shape[1];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out.data[i * cols + j] += bias.data[j];
        }
    }

    return out;
}