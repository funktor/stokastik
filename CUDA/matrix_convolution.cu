#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <random>
#include <vector>
#include <chrono>

#define TILE_DIM 32

__global__ void convolve(const int *mat_1, const int *mat_2, int *mat_conv, const int n, const int m, const int p, const int q, const int stride, const int u, const int v) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int start_row = stride*y;
    int start_col = stride*x;
    
    if (y < u && x < v && start_row+p <= n && start_col+q <= m) {
        int sum = 0;

        for (int i = start_row; i < start_row+p; i++) {
            int i1 = i-start_row;
            for (int j = start_col; j < start_col+q; j++) {
                int j1 = j-start_col;
                sum += mat_1[i*m+j]*mat_2[i1*q+j1];
            }
        }

        mat_conv[y*v+x] = sum;
    }
}

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

bool check_correctness(const int *mat_1, const int *mat_2, int *mat_conv, const int n, const int m, const int p, const int q, const int stride, const int u, const int v) {
    for (int i = 0; i < u*v; i++) {
        int r = i/v;
        int c = i % v;
        
        int start_row = stride*r;
        int start_col = stride*c;
        
        if (start_row+p <= n && start_col+q <= m) {
            int sum = 0;
            for (int i1 = start_row; i1 < start_row+p; i1++) {
                int i2 = i1-start_row;
                for (int j1 = start_col; j1 < start_col+q; j1++) {
                    int j2 = j1-start_col;
                    sum += mat_1[i1*m+j1]*mat_2[i2*q+j2];
                }
            }
            if (sum != mat_conv[i]) {
                return false;
            }
        }
        else {
            return false;
        }
    }
    return true;
}

int main(void) {
    int n = 5000;
    int m = 8000;
    
    int p = 7;
    int q = 11;
    
    int stride = 1;
    
    int u = (n-p)/stride + 1;
    int v = (m-q)/stride + 1;

    dim3 dimGrid((v + TILE_DIM - 1)/TILE_DIM, (u + TILE_DIM - 1)/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
    
    int *mat_1, *mat_2, *mat_conv; 

    cudaMallocManaged(&mat_1, n*m*sizeof(int));
    cudaMallocManaged(&mat_2, p*q*sizeof(int));
    cudaMallocManaged(&mat_conv, u*v*sizeof(int));
    
    std::vector<std::vector<int>> my_arr_1 = random_matrix(n, m, 0, 10);
    std::vector<std::vector<int>> my_arr_2 = random_matrix(p, q, 0, 10);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            mat_1[i*m + j] = my_arr_1[i][j];
        }
    }
    
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            mat_2[i*q + j] = my_arr_2[i][j];
        }
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    convolve<<<dimGrid, dimBlock>>>(mat_1, mat_2, mat_conv, n, m, p, q, stride, u, v);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

    std::cout << duration << std::endl;
    std::cout << check_correctness(mat_1, mat_2, mat_conv, n, m, p, q, stride, u, v) << std::endl;
    
    return 0;
}

