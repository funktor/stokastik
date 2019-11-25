#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <random>
#include <vector>
#include <chrono>

#define TILE_DIM 32

__global__ void multiplyNaive(const int *mat_1, const int *mat_2, int *mat_prod, const int n, const int m, const int p) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < n && x < p) {
        int sum = 0;
    
        for (int i = 0; i < m; i++) {
            sum += mat_1[y*m+i] * mat_2[i*p+x];
        }
    
        mat_prod[y*p+x] = sum;
    }
}

__global__ void multiplySharedMem(const int *mat_1, const int *mat_2, int *mat_prod, const int n, const int m, const int p, const int start, const int end) {
    __shared__ int aTile[TILE_DIM][TILE_DIM], bTile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    aTile[threadIdx.y][threadIdx.x] = mat_1[y*m+threadIdx.x+start];
    bTile[threadIdx.y][threadIdx.x] = mat_2[(threadIdx.y+start)*p+x];
    
    __syncthreads();
    
    int dims_tile = end-start;
    
    if (y < n && x < p) {
        int sum = 0;
    
        for (int i = 0; i < dims_tile; i++) {
            sum += aTile[threadIdx.y][i] * bTile[i][threadIdx.x];
        }
    
        mat_prod[y*p+x] += sum;
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

bool check_correctness(const int *mat_1, const int *mat_2, int *mat_prod, const int n, const int m, const int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            int sum = 0;
            for (int k = 0; k < m; k++) {
                sum += mat_1[i*m+k] * mat_2[k*p+j];
            }
            if (sum != mat_prod[i*p+j]) {
                return false;
            }
        }
    }
    return true;
}

int main(void) {
    int n = 5000;
    int m = 500;
    int p = 8000;

    dim3 dimGrid((p + TILE_DIM - 1)/TILE_DIM, (n + TILE_DIM - 1)/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
    
    int *mat_1, *mat_2, *mat_prod, *mat_prod_d;

    mat_prod = (int *)malloc(n*p*sizeof(int));

    for (int j = 0; j < n*p; j++) {
        mat_prod[j] = 0;
    }

    cudaMallocManaged(&mat_1, n*m*sizeof(int));
    cudaMallocManaged(&mat_2, m*p*sizeof(int));
    
    cudaMalloc((void**)&mat_prod_d, n*p*sizeof(int));
    
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
    
    int num_parts = (m+TILE_DIM-1)/TILE_DIM;
    cudaStream_t stream[num_parts];

    auto t1 = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_parts; i++) {
        int start = i*TILE_DIM;
        int end = m < ((i+1)*TILE_DIM)?m:((i+1)*TILE_DIM);

        cudaStreamCreate(&stream[i]);

        cudaMemcpyAsync(mat_prod_d, mat_prod, n*p*sizeof(int), cudaMemcpyHostToDevice, stream[i]);
        multiplySharedMem<<<dimGrid, dimBlock, 0, stream[i]>>>(mat_1, mat_2, mat_prod_d, n, m, p, start, end);
        cudaMemcpyAsync(mat_prod, mat_prod_d, n*p*sizeof(int), cudaMemcpyDeviceToHost, stream[i]);
    }
    
    cudaDeviceSynchronize();

    auto t2 = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

    std::cout << duration << std::endl;
    std::cout << check_correctness(mat_1, mat_2, mat_prod, n, m, p) << std::endl;
    
    for (int i = 0; i < num_parts; i++) {
        cudaStreamDestroy(stream[i]);
    }
    
    cudaFree(mat_1);
    cudaFree(mat_2);
    cudaFree(mat_prod_d);
    free(mat_prod);
    
    return 0;
}
