#pragma once 
#include "../tensor/Tensor.h"

Tensor matmul(const Tensor& A, const Tensor& B){
    int m = A.shape[0];
    int n = A.shape[1];
    int p = B.shape[1];

    Tensor C({m,p});
    
    for(int i=0 ; i<m ; i++){
        for(int j=0 ; j<p ; j++){
            float sum = 0.0f;
            for(int k=0; k<n; k++){
                sum += A.data[i*n+k] * B.data[k*p+j];

            }
            C.data[i*p+j]= sum;
        }
    }

    return C;


}