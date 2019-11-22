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

void transpose(int *odata, const int *idata, const int n, const int m) {
    for (int i = 0; i < n*m; i++) {
        int y = i/m;
        int x = i % m;
        odata[n*x + y] = idata[i];
    }
}

int main(void) {
    int n = 2000;
    int m = 5000;
    
    int *idata, *odata;

    idata = (int *)malloc(n*m*sizeof(int));
    odata = (int *)malloc(n*m*sizeof(int));
    
    std::vector<std::vector<int>> my_arr = random_matrix(n, m, 0.0, 100.0);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            idata[m*i + j] = my_arr[i][j];
        }
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    transpose(odata, idata, n, m);
    auto t2 = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

    std::cout << duration << std::endl;

    return 0;
}

