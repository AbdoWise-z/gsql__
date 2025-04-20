//
// Created by xabdomo on 4/13/25.
//

#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <memory>
#include <vector>
#include <numeric>

enum Device {
    CPU,
    GPU
};

template<typename T, Device d>
class tensor {};

#include "cpu_tensor.hpp"
#include "gpu_tensor.hpp"


static std::vector<size_t> unmap(std::vector<size_t> shape, const size_t i) {
    std::vector<size_t> indices;
    size_t remaining = i;
    for (size_t dim : shape) {
        indices.push_back(remaining % dim);
        remaining = remaining / dim;
    }
    return indices;
}

#endif //TENSOR_HPP
