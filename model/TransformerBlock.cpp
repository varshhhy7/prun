#include "TransformerBlock.h"

TransformerBlock::TransformerBlock(int dim)
    : Wq(dim, dim), Wk(dim, dim), Wv(dim, dim), ffn(dim, dim * 2) {}

Tensor TransformerBlock::forward(const Tensor& input) {
    Tensor Q = Wq.forward(input);
    Tensor K = Wk.forward(input);
    Tensor V = Wv.forward(input);

    Tensor attn = attention(Q, K, V);

    Tensor out = attn;

    out = ffn.forward(out);

    return out;
}