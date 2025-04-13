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

#endif //TENSOR_HPP
