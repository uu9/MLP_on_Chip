#include <stdio.h>
#include <stdlib.h>
#include <math.h>

inline double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

inline double relu(double x) {
    return x > 0 ? x : 0;
}

void linear_forward(float* input, float* output, int dim1, int dim2, int features_out) {
    // (dim1 * dim2) x (dim2 * output_dim) = (dim1 * output_dim)
    for(int i=0;i<dim1;i++){
        for (int j=0;j<features_out;j++){
            
        }
    }
}
