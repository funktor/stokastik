#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <random>
#include <vector>
#include <chrono>
#include <deque>
#include <algorithm>
#include <iterator>
#include <curand.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 1024

__global__ void min_reduce(int *arr, const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        int j = n-i-1;
        int x = arr[i];
        int y = arr[j];
        arr[i] = x < y ? x:y;
    }
}

int get_min_val(int *min_arr, int n) {
    while (n > 1) {
        min_reduce<<<(n + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(min_arr, n);
        n = (n+1)/2;
    }
    cudaDeviceSynchronize();
    return min_arr[0];
}

void random_vector(int *arr, const int n, const int min_val=0.0, const int max_val=1000.0) {
    static std::random_device rd;
    static std::mt19937 mte(rd());
    std::uniform_int_distribution<int> dist(min_val, max_val);
    
    for (int i = 0; i < n; i++) {
        arr[i] = dist(mte);
    }
}

bool check_correctness(int *arr, int pred, int n) {
    int min_el = 1 << 30;
    for (int i = 0; i < n; i++) {
        if (arr[i] < min_el) { 
            min_el = arr[i];
        }
    }
    return pred == min_el;
}

int main(void) {
    int n = 1 << 25;
    
    int *arr, *temp;
    cudaMallocManaged(&arr, n*sizeof(int));
    
    random_vector(arr, n, 0, 10000);
    
    temp = new int[n];
    std::copy(arr, arr+n, temp);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    int min_el = get_min_val(arr, n);
    auto t2 = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

    std::cout << duration << std::endl;
    
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << check_correctness(temp, min_el, n) << std::endl;
    t2 = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

    std::cout << duration << std::endl;
    
    cudaFree(arr);

    return 0;
}
