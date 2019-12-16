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

__global__ void partition(int *arr, int *bit_arr, int *l, int *r, const int pivot, const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        if (bit_arr[i] == 1) {
            if (arr[i] < pivot) {
                l[i] = 1;
                r[i] = 0;
            }
            else {
                l[i] = 0;
                r[i] = 1;
            }
        }
        else {
            l[i] = 0;
            r[i] = 0;
        }
    }
}

void partition_array(int *arr, int *bit_arr, int *l, int *r, const int pivot, const int n) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    partition<<<(n + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(arr, bit_arr, l, r, pivot, n);
    cudaDeviceSynchronize();
    cudaStreamDestroy(stream);
}

__global__ void copy_arr(int *in_arr, int *out_arr, const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        out_arr[i] = in_arr[i];
    }
}

void copy_array(int *arr1, int *arr2, const int n) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    copy_arr<<<(n + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(arr1, arr2, n);
    cudaDeviceSynchronize();
    cudaStreamDestroy(stream);
}

__global__ void init_arr(int *arr, int val, const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        arr[i] = val;
    }
}

void init_array(int *arr, int val, const int n) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    init_arr<<<(n + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(arr, val, n);
    cudaDeviceSynchronize();
    cudaStreamDestroy(stream);
}

__global__ void max_reduce(int *arr, const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        int j = n-i-1;
        int x = arr[i];
        int y = arr[j];
        arr[i] = x > y ? x:y;
    }
}

__global__ void min_reduce(int *arr, const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        int j = n-i-1;
        int x = arr[i];
        int y = arr[j];
        arr[i] = x < y ? x:y;
    }
}

__global__ void cnt_reduce(int *arr, const int n, const int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < m) {
        int j = n-i-1;
        if (i != j) {
            arr[i] = arr[i] + arr[j];
        }
    }
}

__global__ void init_min_arr(int *arr, int *bit_arr, int *min_arr, const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        if (bit_arr[i] == 1) {
            min_arr[i] = arr[i];
        }
        else {
            min_arr[i] = 1 << 30;
        }
    }
}

void init_min_array(int *arr, int *bit_arr, int *min_arr, const int n) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    init_min_arr<<<(n + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(arr, bit_arr, min_arr, n);
    cudaDeviceSynchronize();
    cudaStreamDestroy(stream);
}

__global__ void init_max_arr(int *arr, int *bit_arr, int *max_arr, const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        if (bit_arr[i] == 1) {
            max_arr[i] = arr[i];
        }
        else {
            max_arr[i] = -(1 << 30);
        }
    }
}

void init_max_array(int *arr, int *bit_arr, int *max_arr, const int n) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    init_max_arr<<<(n + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(arr, bit_arr, max_arr, n);
    cudaDeviceSynchronize();
    cudaStreamDestroy(stream);
}

int count_size(int *cnt_arr, int n) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    int *temp_arr;
    cudaMallocManaged(&temp_arr, n*sizeof(int));
    
    copy_array(cnt_arr, temp_arr, n);
    
    int m = (n+1)/2;
    
    while (n > 1) {
        cnt_reduce<<<(m + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(temp_arr, n, m);
        n = m;
        m = (n+1)/2;
    }
    cudaDeviceSynchronize();
    
    int out = temp_arr[0];
    cudaFree(temp_arr);
    cudaStreamDestroy(stream);
    
    return out;
}

int get_max_val(int *max_arr, int n) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    while (n > 1) {
        max_reduce<<<(n + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(max_arr, n);
        n = (n+1)/2;
    }
    cudaDeviceSynchronize();
    cudaStreamDestroy(stream);
    
    return max_arr[0];
}

int get_min_val(int *min_arr, int n) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    while (n > 1) {
        min_reduce<<<(n + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(min_arr, n);
        n = (n+1)/2;
    }
    cudaDeviceSynchronize();
    cudaStreamDestroy(stream);
    
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

int quickselect(int *arr, int k, int n) {
    int *l, *r, *bit_arr, *min_arr, *max_arr;
    
    cudaMallocManaged(&l, n*sizeof(int));
    cudaMallocManaged(&r, n*sizeof(int));
    cudaMallocManaged(&bit_arr, n*sizeof(int));
    cudaMallocManaged(&min_arr, n*sizeof(int));
    cudaMallocManaged(&max_arr, n*sizeof(int));
    
    init_array(bit_arr, 1, n);
    init_array(l, 0, n);
    init_array(r, 0, n);
    
    int out = -1;
        
    while (k > 0 && n > 0) {
        init_min_array(arr, bit_arr, min_arr, n);
        init_max_array(arr, bit_arr, max_arr, n);
        
        int a = get_min_val(min_arr, n);
        int b = get_max_val(max_arr, n);
        
        if (a == b) {
            out = a;
            break;
        }
        
        int pivot = a+1 + rand() % static_cast<int>(b-a);
        
        partition_array(arr, bit_arr, l, r, pivot, n);
        
        int p = count_size(l, n);
        int q = count_size(r, n);
        
        if (p == k-1) {
            init_min_array(arr, r, min_arr, n);
            out = get_min_val(min_arr, n);
            break;
        }
        
        if (p == k) {
            init_max_array(arr, l, max_arr, n);
            out = get_max_val(max_arr, n);
            break;
        }
        
        if (p > k) {
            copy_array(l, bit_arr, n);
        }
        else {
            copy_array(r, bit_arr, n);
            k -= p;
        }
    }
    
    cudaFree(l);
    cudaFree(r);
    cudaFree(bit_arr);
    cudaFree(min_arr);
    cudaFree(max_arr);
    
    return out;
}

bool check_correctness(int *arr, int pred, int k, int n) {
    std::sort(arr, arr+n);
    return pred == arr[k-1];
}

int main(void) {
    int n = 1 << 20;
    int k = 1 << 19;
    
    int *arr, *temp;
    cudaMallocManaged(&arr, n*sizeof(int));
    
    random_vector(arr, n, 0, 10000);
    
    temp = new int[n];
    std::copy(arr, arr+n, temp);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    int pred = quickselect(arr, k, n);
    auto t2 = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

    std::cout << duration << std::endl;
    
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << check_correctness(temp, pred, k, n) << std::endl;
    t2 = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

    std::cout << duration << std::endl;
    
    cudaFree(arr);

    return 0;
}
