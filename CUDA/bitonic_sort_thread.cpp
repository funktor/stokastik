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
#include <thread>

bool check_correctness(int *orig_arr, int *arr_sorted, const int n) {
    std::sort(orig_arr, orig_arr+n);
    
    for (int i = 0; i < n; i++) {
        if (orig_arr[i] != arr_sorted[i]) {
            return false;
        }
    }
    return true;
}

void swap_thread(int *arr, const int skip, const int oflag, const int order, const int start, const int end) {
    for (int i = start; i < end; i++) {
        int j = i*2 - i%skip;
        int k = j + skip;
        
        int f = ((int) (i/oflag)) % 2 == 0 ? order:-order;
        
        int x, y;
        
        if (f == 1) {
            x = arr[j] < arr[k] ? arr[j]:arr[k];
            y = arr[j] > arr[k] ? arr[j]:arr[k];
        }
        else {
            x = arr[j] > arr[k] ? arr[j]:arr[k];
            y = arr[j] < arr[k] ? arr[j]:arr[k];
        }
        
        arr[j] = x;
        arr[k] = y;
    }
}


void swap(int* arr, const int skip, const int oflag, const int order, const int n) {
    int n_threads = 1 << 5;
    std::vector<std::thread> threads(n_threads);
    
    int batch_size = n/(2*n_threads);

    for (int i = 0; i < n_threads; i++) {
        int start = batch_size*i;
        int end = std::min(batch_size*(i+1), n/2);
        threads[i] = std::thread(swap_thread, arr, skip, oflag, order, start, end);
    }

    for (int i = 0; i < n_threads; i++) {
        threads[i].join();
    }
}

void bitonic_sort(int *arr, const int order, const int n) {
    int skip = 1;
    
    while (skip < n) {
        int k = skip;
        
        while (k > 0) {
            swap(arr, k, skip, order, n);
            k /= 2;
        }
            
        skip *= 2;
    }
}

void random_vector(int *arr, const int n, const int min_val=0.0, const int max_val=1000.0) {
    static std::random_device rd;
    static std::mt19937 mte(rd());
    std::uniform_int_distribution<int> dist(min_val, max_val);
    
    for (int i = 0; i < n; i++) {
        arr[i] = dist(mte);
    }
}

int main(void) {
    int n = 1 << 25;
    
    int *arr, *temp;
    arr = new int[n];
    
    random_vector(arr, n, 0, 1000);
    
    temp = new int[n];
    std::copy(arr, arr+n, temp);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    bitonic_sort(arr, 1, n);
    auto t2 = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

    std::cout << duration << std::endl;
    
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << check_correctness(temp, arr, n) << std::endl;
    t2 = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

    std::cout << duration << std::endl;
    
    return 0;
}
