#pragma once
#include <vector>
#include <iostream>

class Tensor {
public:
    std::vector<float>data;
    std::vector<int>shape;

    Tensor() {}

    Tensor(std::vector<int>shape_) : shape(shape_){
        int size = 1;
        for(int dim: shape) size*=dim;
        data.resize(size, 0.0f);
    }
    float& operator[](int idx) {
        return data[idx];
    }
    const float& operator[](int idx) const{
        return data[idx];
    }
    void print() const {
        for (float v: data) {
            std:: cout<< v << " ";
        }
        std:: cout <<std::endl;
    }
};