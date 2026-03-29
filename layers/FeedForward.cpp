#include "FeedForward.h"

FeedForward::FeedForward(int dim, int hidden_dim)
    : fc1(dim, hidden_dim), fc2(hidden_dim, dim) {}

Tensor FeedForward::relu(const Tensor& x) {
    Tensor out = x;
    for (auto &v : out.data) {
        if (v < 0) v = 0;
    }
    return out;
}

Tensor FeedForward::forward(const Tensor& input) {
    Tensor x = fc1.forward(input);
    x = relu(x);
    x = fc2.forward(x);
    return x;
}