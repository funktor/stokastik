#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <random>
#include <vector>
#include <chrono>

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

void multiply(const int *mat_1, const int *mat_2, int *mat_prod, const int n, const int m, const int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            int sum = 0;
            for (int k = 0; k < m; k++) {
                sum += mat_1[i*m+k] * mat_2[k*p+j];
            }
            mat_prod[i*p+j] = sum;
        }
    }
}

int main(void) {
    int n = 2000;
    int m = 1000;
    int p = 5000;
    
    int *mat_1, *mat_2, *mat_prod;

    mat_1 = (int *)malloc(n*m*sizeof(int));
    mat_2 = (int *)malloc(m*p*sizeof(int));
    mat_prod = (int *)malloc(n*p*sizeof(int));
    
    std::vector<std::vector<int>> my_arr_1 = random_matrix(n, m, 0, 10);
    std::vector<std::vector<int>> my_arr_2 = random_matrix(m, p, 0, 10);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            mat_1[m*i + j] = my_arr_1[i][j];
        }
    }
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            mat_2[p*i + j] = my_arr_2[i][j];
        }
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    multiply(mat_1, mat_2, mat_prod, n, m, p);
    auto t2 = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

    std::cout << duration << std::endl;
    
    free(mat_1);
    free(mat_2);
    free(mat_prod);

    return 0;
}

