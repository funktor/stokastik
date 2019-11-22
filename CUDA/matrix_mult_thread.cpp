#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <random>
#include <vector>
#include <chrono>
#include <thread>

#define M_DIM 50

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

void multiply(const int *mat_1, const int *mat_2, int *mat_prod, const int n, const int p, const int start, const int end) {
    for (int index = start; index < end; index++) {
	int row = index/p;
        int col = index % p;

	int sum = 0;
	for (int k = 0; k < M_DIM; k++) {
	    sum += mat_1[row*M_DIM+k] * mat_2[k*p+col];
	}	
	mat_prod[index] = sum;
    }
}

bool check_correctness(const int *mat_1, const int *mat_2, int *mat_prod, const int n, const int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            int sum = 0;
            for (int k = 0; k < M_DIM; k++) {
                sum += mat_1[i*M_DIM+k] * mat_2[k*p+j];
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
    int p = 8000;
    int n_threads = 100;
    
    int *mat_1, *mat_2, *mat_prod;
    std::vector<std::thread> threads(n_threads);

    mat_1 = (int *)malloc(n*M_DIM*sizeof(int));
    mat_2 = (int *)malloc(M_DIM*p*sizeof(int));
    mat_prod = (int *)malloc(n*p*sizeof(int));
    
    std::vector<std::vector<int>> my_arr_1 = random_matrix(n, M_DIM, 0, 10);
    std::vector<std::vector<int>> my_arr_2 = random_matrix(M_DIM, p, 0, 10);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < M_DIM; j++) {
            mat_1[M_DIM*i + j] = my_arr_1[i][j];
        }
    }
    
    for (int i = 0; i < M_DIM; i++) {
        for (int j = 0; j < p; j++) {
            mat_2[p*i + j] = my_arr_2[i][j];
        }
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    int k = (n*p + n_threads - 1)/n_threads;

    for (int i = 0; i < n_threads; i++) {
        int start = k*i;
        int end = std::min(k*(i+1), n*p);
        threads[i] = std::thread(multiply, mat_1, mat_2, mat_prod, n, p, start, end);
    }

    for (int i = 0; i < n_threads; i++) {
        threads[i].join();
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

    std::cout << duration << std::endl;
    std::cout << check_correctness(mat_1, mat_2, mat_prod, n, p) << std::endl;

    return 0;
}

