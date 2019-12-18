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
#include <set>

#define BLOCK_SIZE 1024

struct bstree {
    int *left_child;
    int *right_child;
    int *parent;
    bool *flag;
};

__global__ void populate_child_parent(float *arr, int *i_left_child, int *i_right_child, int *i_parent, int *o_left_child, int *o_right_child, int *o_parent, bool *flag, const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        int y = i_parent[i];

        if (i_left_child[i] == -1 && i_right_child[i] == -1 && i != y) {
            flag[0] = true;
            int x, p;

            if (arr[i] <= arr[y]) {
                p = i_left_child[y];
                x = (p != -1) ? p:y;
                o_parent[i] = x;
            }
            else {
                p = i_right_child[y];
                x = (p != -1) ? p:y;
                o_parent[i] = x;
            }

            if (i != x) {
                if (arr[i] <= arr[x]) {
                    if (o_left_child[x] == -1) {
                        o_left_child[x] = i;
                    }
                }
                else {
                    if (o_right_child[x] == -1) {
                        o_right_child[x] = i;
                    }
                }
            }
        }
    }
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

void random_vector(float *arr, const int n, const float min_val=0.0, const float max_val=1000.0) {
    static std::random_device rd;
    static std::mt19937 mte(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);
    
    for (int i = 0; i < n; i++) {
        arr[i] = dist(mte);
    }
}

bstree construct_binary_tree(float *arr, bstree g, bstree g1, const int n) {
    copy_array(g.left_child, g1.left_child, n);
    copy_array(g.right_child, g1.right_child, n);
    copy_array(g.parent, g1.parent, n);
    
    g1.flag[0] = false;
    
    populate_child_parent<<<(n + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(arr, g.left_child, g.right_child, g.parent, g1.left_child, g1.right_child, g1.parent, g1.flag, n);
    cudaDeviceSynchronize();
            
    return g1;
}

bstree bs_tree(float *arr, int root_index, bstree g, bstree g1, const int n) {
    g.flag[0] = true;
    
    while (g.flag[0]) {
        g = construct_binary_tree(arr, g, g1, n);
    }
    
    return g;
}

float *traversal(float *arr, int *left_child, int *right_child, int root_index, const int n) {
    int *stack = new int[n];
    float *out = new float[n];
    
    stack[0] = root_index;
    int p = 1;
    int i = 0;

    std::set<int> visited;

    while (p > 0) {
        int curr_root = stack[p-1];
        
        if (left_child[curr_root] != -1 && visited.find(left_child[curr_root]) == visited.end()) {
            stack[p++] = left_child[curr_root];
        }
        else {
            if (visited.find(curr_root) == visited.end()) {
                out[i++] = arr[curr_root];
                visited.insert(curr_root);
            }
            
            if (right_child[curr_root] != -1 && visited.find(right_child[curr_root]) == visited.end()) {
                stack[p++] = right_child[curr_root];
            }
            else {
                p -= 1;
            }
        }
    }
    
    return out;
}

float *inorder_traversal(float *arr, bstree g, bstree g1, const int n) {
    int root_index = rand() % static_cast<int>(n);
    std::fill(g.parent, g.parent+n, root_index);
    
    
    auto t1 = std::chrono::high_resolution_clock::now();
    g = bs_tree(arr, root_index, g, g1, n);
    auto t2 = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

    std::cout << duration << std::endl;
    return traversal(arr, g.left_child, g.right_child, root_index, n);
}

bool check_correctness(float *arr, float *pred_arr, const int n) {
    std::sort(arr, arr+n);
    for (int i = 0; i < n ; i++) {
        if (arr[i] != pred_arr[i]) {
            return false;
        }
    }
    return true;
}
    
int main(void) {
    int n = 1 << 25;
    
    float *arr, *temp;
    cudaMallocManaged(&arr, n*sizeof(float));
    
    random_vector(arr, n, 0, 10000);
    
    temp = new float[n];
    std::copy(arr, arr+n, temp);
    
    bstree g, g1;
    
    cudaMallocManaged(&g.left_child, n*sizeof(int));
    cudaMallocManaged(&g.right_child, n*sizeof(int));
    cudaMallocManaged(&g.parent, n*sizeof(int));
    cudaMallocManaged(&g.flag, sizeof(bool));
    
    cudaMallocManaged(&g1.left_child, n*sizeof(int));
    cudaMallocManaged(&g1.right_child, n*sizeof(int));
    cudaMallocManaged(&g1.parent, n*sizeof(int));
    cudaMallocManaged(&g1.flag, sizeof(bool));
    
    std::fill(g.left_child, g.left_child+n, -1);
    std::fill(g.right_child, g.right_child+n, -1);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    float *pred = inorder_traversal(arr, g, g1, n);
    auto t2 = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

    std::cout << duration << std::endl;
    
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << check_correctness(temp, pred, n) << std::endl;
    t2 = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

    std::cout << duration << std::endl;
    
    cudaFree(arr);
    
    cudaFree(g.left_child);
    cudaFree(g.right_child);
    cudaFree(g.parent);
    cudaFree(g.flag);
    
    cudaFree(g1.left_child);
    cudaFree(g1.right_child);
    cudaFree(g1.parent);
    cudaFree(g1.flag);

    return 0;
}
