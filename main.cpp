#include "tensor/Tensor.h"
#include "ops/MatMul.h"
#include "model/Attention.h"
#include <iostream>

int main() {
    Tensor Q({2, 2});
    Tensor K({2, 2});
    Tensor V({2, 2});

    Q.data = {1, 0, 0, 1};
    K.data = {1, 2, 3, 4};
    V.data = {5, 6, 7, 8};

    Tensor out = attention(Q, K, V);

    std::cout << "Attention Output:\n";
    out.print();

    return 0;
}