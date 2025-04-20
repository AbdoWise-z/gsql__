//
// Created by xabdomo on 4/20/25.
//

#include "gpu_function_interface.cuh"

#include "store.hpp"
#include "kernels/tensor_kernels.cuh"

void GFI::fill(tensor<char, Device::GPU> *output_data, const char value) {
    const auto size = output_data->totalSize();
    dim3 grid((size + Cfg::BlockDimX - 1) / (Cfg::BlockDimX));

    TensorKernel::fill_kernel<<< grid, dim3(Cfg::BlockDimX) >>>(output_data->data, value, size);
}

void GFI::fill(
        tensor<char, Device::GPU> *output_data,
        char value,
        std::vector<size_t> position,
        std::vector<size_t> mask
    ) {

    auto _position = static_cast<size_t*>(cu::vectorToDevice(position));
    auto _mask     = static_cast<size_t*>(cu::vectorToDevice(mask));
    auto _shape    = static_cast<size_t*>(cu::vectorToDevice(output_data->shape));

    const auto size = output_data->totalSize();
    dim3 grid((size + Cfg::BlockDimX - 1) / (Cfg::BlockDimX));

    TensorKernel::fill_kernel<<< grid, dim3(Cfg::BlockDimX) >>>(
        output_data->data,
        output_data->totalSize(),
        value,
        _position,
        _mask,
        _shape,
        mask.size());

    cu::free(_position);
    cu::free(_mask);
    cu::free(_shape);
}
