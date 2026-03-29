#include "MoE.h"
#include <algorithm>


MoE::MoE(int dim_, int num_experts_)
    : dim(dim_), num_experts(num_experts_), gate(dim_, num_experts_) {

    for (int i = 0; i < num_experts; i++) {
        experts.emplace_back(dim, dim * 2);
    }
}


int MoE::argmax(const std::vector<float>& scores) {
    return std::distance(scores.begin(),
                         std::max_element(scores.begin(), scores.end()));
}


Tensor MoE::forward(const Tensor& input) {
    int rows = input.shape[0];
    int cols = input.shape[1];

    Tensor output({rows, cols});

    
    Tensor gate_scores = gate.forward(input);

    for (int i = 0; i < rows; i++) {

        std::vector<float> scores(num_experts);
        for (int j = 0; j < num_experts; j++) {
            scores[j] = gate_scores.data[i * num_experts + j];
        }

        int expert_id = argmax(scores);

       
        Tensor row_input({1, cols});
        for (int j = 0; j < cols; j++) {
            row_input.data[j] = input.data[i * cols + j];
        }

        Tensor row_output = experts[expert_id].forward(row_input);

        
        for (int j = 0; j < cols; j++) {
            output.data[i * cols + j] = row_output.data[j];
        }
    }

    return output;
}