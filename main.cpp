#include "tensor/Tensor.h"
#include "model/MoE.h"
#include <iostream>

int main() {
    Tensor input({2, 2});
    input.data = {1, 0, 0, 1};

    MoE moe(2, 3); // dim=2, 3 experts

    Tensor out = moe.forward(input);

    std::cout << "MoE Output:\n";
    out.print();

    return 0;
}