#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cassert>
using namespace std;

inline double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

inline double relu(double x) {
    return x > 0 ? x : 0;
}

inline float get_element(float* mat, int mat_cols, int row, int col){
    return mat[mat_cols * row + col];
}

// 按行存储
void matrix_mul(float* mat1, float* mat2, int mat1_rows, int mat1_cols, int mat2_rows, int mat2_cols){
    assert(mat1_cols == mat2_rows);
}

void linear_forward(float* input, float* output, int dim1, int dim2, int features_out) {
    // (dim1 * dim2) x (dim2 * output_dim) = (dim1 * output_dim)
    for(int i=0;i<dim1;i++){
        for (int j=0;j<features_out;j++){
            
        }
    }
}

int main(){
    const int rows = 2, cols = 3;
    float mat[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    cout << get_element(mat, cols, 0, 0) << endl;
}
