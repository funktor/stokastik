#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <random>
#include <vector>
#include <chrono>

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transposeNaive(int *odata, const int *idata, const int n, const int m) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    
    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS) {
        if (x < m && (y+j) < n) {
            odata[x*n + (y+j)] = idata[(y+j)*m + x];
        }
    }
}

__global__ void transposeSharedMem(int *odata, const int *idata, const int n, const int m) {
    __shared__ int tile[TILE_DIM][TILE_DIM+1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;


    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*m + x];
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS) {
        if (x < n && (y+j) < m) {
            odata[(y+j)*n + x] = tile[threadIdx.x][threadIdx.y+j];
        }
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

bool check_correctness(int *odata, const int *idata, const int n, const int m) {
    for (int i = 0; i < n*m; i++) {
        int y = i/m;
        int x = i % m;
        if ((n*x + y) >= n*m || odata[n*x + y] != idata[i]) {
            return false;
        }
    }
    return true;
}

int main(void) {
    int n = 2000;
    int m = 5000;

    dim3 dimGrid((m + TILE_DIM - 1)/TILE_DIM, (n + TILE_DIM - 1)/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    
    int *idata, *odata;

    cudaMallocManaged(&idata, n*m*sizeof(int));
    cudaMallocManaged(&odata, n*m*sizeof(int));
    
    std::vector<std::vector<int>> my_arr = random_matrix(n, m, 0.0, 100.0);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            idata[m*i + j] = my_arr[i][j];
        }
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    transposeSharedMem<<<dimGrid, dimBlock>>>(odata, idata, n, m);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

    std::cout << duration << std::endl;
    std::cout << check_correctness(odata, idata, n, m) << std::endl;

    return 0;
}

