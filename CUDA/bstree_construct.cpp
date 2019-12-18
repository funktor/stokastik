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
#include <tuple>
#include <deque>

void random_vector(float *arr, const int n, const float min_val=0.0, const float max_val=1000.0) {
    static std::random_device rd;
    static std::mt19937 mte(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);
    
    for (int i = 0; i < n; i++) {
        arr[i] = dist(mte);
    }
}

struct bstree {
    int *left_child;
    int *right_child;
    int *parent;
    bool flag;
};

bstree construct_binary_tree(float *arr, bstree g, bstree g1, const int n) {
    std::copy(g.left_child, g.left_child+n, g1.left_child);
    std::copy(g.right_child, g.right_child+n, g1.right_child);
    std::copy(g.parent, g.parent+n, g1.parent);
    
    g1.flag = false;
    
    for (int i = 0; i < n; i++) {
        int y = g.parent[i];
        
        if (g.left_child[i] == -1 && g.right_child[i] == -1 && i != g.parent[i]) {
            g1.flag = true;
            int x;
            
            if (arr[i] <= arr[y]) {
                x = (g.left_child[y] != -1) ? g.left_child[y]:y;
                g1.parent[i] = x;
            }
            else {
                x = (g.right_child[y] != -1) ? g.right_child[y]:y;
                g1.parent[i] = x;
            }
            
            if (i != x) {
                if (arr[i] <= arr[x]) {
                    if (g1.left_child[x] == -1) {
                        g1.left_child[x] = i;
                    }
                }
                else {
                    if (g1.right_child[x] == -1) {
                        g1.right_child[x] = i;
                    }
                }
            }
        }
    }
            
    return g1;
}

bstree bs_tree(float *arr, int root_index, bstree g, bstree g1, const int n) {
    g.flag = true;
    
    while (g.flag) {
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
    arr = new float[n];
    
    random_vector(arr, n, 0, 10000);
    
    temp = new float[n];
    std::copy(arr, arr+n, temp);
    
    bstree g, g1;
    
    g.left_child = new int[n];
    g.right_child = new int[n];
    g.parent = new int[n];
    
    g1.left_child = new int[n];
    g1.right_child = new int[n];
    g1.parent = new int[n];
    
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

    return 0;
}
