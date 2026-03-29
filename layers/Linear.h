#pragma once
#include "../tensor/Tensor.h"

class Linear {
public:
    Tensor weight;
    Tensor bias;

    int in_features;
    int out_features;

    Linear(int in_features, int out_features);

    Tensor forward(const Tensor& input);
};