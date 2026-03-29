#pragma once
#include "../layers/FeedForward.h"
#include "../layers/Linear.h"
#include "../tensor/Tensor.h"
#include <vector>

class MoE {
public:
    int num_experts;
    int dim;

    std::vector<FeedForward> experts;
    Linear gate;

    MoE(int dim, int num_experts);

    int argmax(const std::vector<float>& scores);

    Tensor forward(const Tensor& input);
};