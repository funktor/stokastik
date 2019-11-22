#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <random>
#include <vector>
#include <chrono>

#define M_DIM 50

std::vector<std::vector<int>> random_matrix(const int num_rows, const int num_cols, const int min_val=0.0, const int max_val=1000.0) {
    std::vector<std::vector<int>> my_arr;
    static std::random_device rd;
    static std::mt19937 mte(rd());
    std::uniform_int_distribution<int> dist(min_val, max_val);
    
    for (int i = 0; i < num_rows; i++) {
        std::vector<int> my_arr_col;
        for (int j = 0; j < num_cols; j++) {
            my_arr_col.push_back(dist(mte));
        }
        my_arr.push_back(my_arr_col);
    }
    
    return my_arr;
}

void multiply(const int *mat_1, const int *mat_2, int *mat_prod, const int n, const int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            int sum = 0;
            for (int k = 0; k < M_DIM; k++) {
                sum += mat_1[i*M_DIM+k] * mat_2[k*p+j];
            }
            mat_prod[i*p+j] = sum;
        }
    }
}

int main(void) {
    int n = 5000;
    int p = 8000;
    
    int *mat_1, *mat_2, *mat_prod;

    mat_1 = (int *)malloc(n*M_DIM*sizeof(int));
    mat_2 = (int *)malloc(M_DIM*p*sizeof(int));
    mat_prod = (int *)malloc(n*p*sizeof(int));
    
    std::vector<std::vector<int>> my_arr_1 = random_matrix(n, M_DIM, 0, 10);
    std::vector<std::vector<int>> my_arr_2 = random_matrix(M_DIM, p, 0, 10);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < M_DIM; j++) {
            mat_1[M_DIM*i + j] = my_arr_1[i][j];
        }
    }
    
    for (int i = 0; i < M_DIM; i++) {
        for (int j = 0; j < p; j++) {
            mat_2[p*i + j] = my_arr_2[i][j];
        }
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    multiply(mat_1, mat_2, mat_prod, n, p);
    auto t2 = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

    std::cout << duration << std::endl;

    return 0;
}

