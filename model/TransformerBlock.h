#pragma once
#include "../layers/Linear.h"
#include "../layers/FeedForward.h"
#include "Attention.h"

class TransformerBlock {
public:
    Linear Wq;
    Linear Wk;
    Linear Wv;
    FeedForward ffn;

    TransformerBlock(int dim);

    Tensor forward(const Tensor& input);
};