//
// Created by xabdomo on 4/19/25.
//


#include <vector>

#include "query/gpu/gpu_function_interface.cuh"
#include "tensor/tensor.hpp"

int main() {
    auto t = new tensor<char, Device::GPU>({10, 10});

    for (size_t i = 0; i < 10; i += 1) {
        char c = 'a';
        if (i % 2 == 0) c = 'b';
        GFI::fill(t, c, {i, 0}, {0, 1});
    }

    auto cpu = t->toCPU();
    for (size_t i = 0;i < 10;i++) {
        for (size_t j = 0;j < 10;j++) {
            std::cout << cpu[{i, j}] << " ";
        }
        std::cout << std::endl;
    }
}
