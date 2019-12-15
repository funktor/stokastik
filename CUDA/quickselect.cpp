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

void random_vector(int *arr, const int n, const int min_val=0.0, const int max_val=1000.0) {
    static std::random_device rd;
    static std::mt19937 mte(rd());
    std::uniform_int_distribution<int> dist(min_val, max_val);
    
    for (int i = 0; i < n; i++) {
        arr[i] = dist(mte);
    }
}

int quickselect(int *arr, int k, int n) {
    while (k > 0 && n > 0) {
        auto a = std::min_element(arr, arr+n);
        auto b = std::max_element(arr, arr+n);
        
        if (*a == *b) {
            return (int)*a;
        }
        
        int pivot = *a+1 + rand() % static_cast<int>(*b-*a);
        
        int *l, *r;
        
        l = new int[n];
        r = new int[n];
        
        int p, q;
        
        p = 0;
        q = 0;
        
        for (int i = 0; i < n; i++) {
            if (arr[i] < pivot) {
                l[p++] = arr[i];
            }
            else {
                r[q++] = arr[i];
            }
        }
        
        if (p == k-1) {
            return (int)*std::min_element(r, r+q);
        }
        
        if (p == k) {
            return (int)*std::max_element(l, l+p);
        }
        
        if (p > k-1) {
            arr = l;
            n = p;
        }
        else {
            arr = r;
            n = q;
            k -= p;
        }
    }
    return -1;
}

bool check_correctness(int *arr, int pred, int k, int n) {
    std::sort(arr, arr+n);
    return pred == arr[k-1];
}

int main(void) {
    int n = 1 << 25;
    int k = 1 << 24;
    
    int *arr;
    arr = new int[n];
    
    random_vector(arr, n, 0, 10000);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    int pred = quickselect(arr, k, n);
    auto t2 = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

    std::cout << duration << std::endl;
    
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << check_correctness(arr, pred, k, n) << std::endl;
    t2 = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

    std::cout << duration << std::endl;

    return 0;
}
