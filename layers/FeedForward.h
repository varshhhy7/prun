#pragma once
#include "Linear.h"

class FeedForward {
public:
    Linear fc1;
    Linear fc2;

    FeedForward(int dim, int hidden_dim);

    Tensor relu(const Tensor& x);
    Tensor forward(const Tensor& input);
};