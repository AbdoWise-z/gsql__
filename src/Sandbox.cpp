//
// Created by xabdomo on 4/19/25.
//


#include <vector>

#include "query/gpu/gpu_function_interface.cuh"
#include "tensor/tensor.hpp"

#include <omp.h>
#include <iostream>

int main() {

    int sum = 0;
    std::vector<int> data(1000, 1);

    for (auto& i : data) {
        i = rand();
    }

    #pragma omp parallel for
    for (auto i : data) {
        std::cout << "Thread " << omp_get_thread_num() << " handles iteration " << i << std::endl;
        #pragma omp critical
        {
            sum += i;
        }
    }
    return 0;
}
