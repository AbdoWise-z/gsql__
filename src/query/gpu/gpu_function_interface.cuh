//
// Created by xabdomo on 4/20/25.
//

#ifndef GPU_FUNCTION_INTERFACE_CUH
#define GPU_FUNCTION_INTERFACE_CUH

#include "tensor/tensor.hpp"


// GFI for gpu function interface
namespace GFI {
    void fill(tensor<char, Device::GPU> *output_data, char value);
    void fill(tensor<char, Device::GPU> *output_data, char value, std::vector<size_t> position, std::vector<size_t> mask);
};



#endif //GPU_FUNCTION_INTERFACE_CUH
