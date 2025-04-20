//
// Created by xabdomo on 4/20/25.
//

#include "gpu_function_interface.cuh"

#include "gpu_buffer_pool.cuh"
#include "store.hpp"
#include "kernels/equality_kernel.cuh"
#include "kernels/inequality_kernel.cuh"
#include "kernels/tensor_kernels.cuh"
#include "utils/murmur_hash3_cuda.cuh"

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

static auto columnPoolAllocator = [] (void* ptr, BufferAllocator alloc) {
    auto col = static_cast<column*>(ptr);
    auto size = col->data.size();
    void* gpuPtr = 0;
    switch (col->type) {
        case INTEGER:
            gpuPtr = alloc(size * sizeof(int64_t));
            cu::toDevice(col->data.data(), gpuPtr, size * sizeof(int64_t));
            size = size * sizeof(int64_t);
            break;
        case FLOAT:
            gpuPtr = alloc(size * sizeof(double));
            cu::toDevice(col->data.data(), gpuPtr, size * sizeof(double));
            size = size * sizeof(double);
            break;
        case STRING:
            // TODO: Handle string data
        default:
            throw std::runtime_error("Unsupported column type");

    }
    return std::make_pair(size, gpuPtr);
};

static auto columnIndexPoolAllocator = [] (void* ptr, BufferAllocator alloc) {
    auto col = static_cast<column*>(ptr);
    auto index_table = col->sorted;
    auto size = index_table.size();
    void* gpuPtr = 0;
    if (index_table.empty()) {
        throw std::runtime_error("Empty index table");
    }

    gpuPtr = alloc(size * sizeof(size_t));
    cu::toDevice(index_table.data(), gpuPtr, size * sizeof(size_t));

    return std::make_pair(size * sizeof(size_t), gpuPtr);
};

static auto columnHashPoolAllocator = [] (void* ptr, BufferAllocator alloc) {
    auto col = static_cast<column*>(ptr);
    auto hash_table = col->hashed;
    auto size = hash_table.size();
    void* gpuPtr = 0;
    if (hash_table.empty()) {
        throw std::runtime_error("Empty hash table");
    }

    auto ex_size = col->hashExSize;

    auto pad = std::vector<size_t>{};
    for (int i = 0;i < ex_size;i++) {
        pad.push_back(static_cast<size_t>(-1)); // don't worry about the warning here .. it's intended
    }

    gpuPtr = alloc(size * ex_size * sizeof(size_t));

    for (int i = 0;i < size;i++) {
        auto data_size = hash_table[i].size();
        cu::toDevice(hash_table[i].data(), static_cast<char*>(gpuPtr) + i * ex_size * sizeof(size_t), data_size * sizeof(size_t));
        if (ex_size - data_size > 0) {
            cu::toDevice(pad.data(), static_cast<char*>(gpuPtr) + (i * ex_size + data_size) * sizeof(size_t), (ex_size - data_size) * sizeof(size_t));
        }
    }

    return std::make_pair(size * ex_size * sizeof(size_t), gpuPtr);
};

static GpuBufferPool pool(Cfg::maxGPUMemory);

void GFI::equality(
        tensor<char, Device::GPU> *result,
        column *col_1,
        column *col_2,

        std::vector<size_t> tileOffset,
        std::vector<size_t> tileSize,

        size_t table_1_index,
        size_t table_2_index,

        std::vector<size_t> mask
    ) {

    auto _col_1 = pool.getBufferOrCreate(static_cast<void*>(col_1), static_cast<void*>(col_1), columnPoolAllocator);
    auto _col_2 = pool.getBufferOrCreate(static_cast<void*>(col_2), static_cast<void*>(col_2), columnPoolAllocator);

    auto _hash_table = pool.getBufferOrCreate(static_cast<void*>(&(col_2->hashed)), static_cast<void*>(col_2), columnHashPoolAllocator);

    auto _mask = static_cast<size_t*>(cu::vectorToDevice(mask));
    auto _tileOffset = static_cast<size_t*>(cu::vectorToDevice(tileOffset));
    auto _tileSize = static_cast<size_t*>(cu::vectorToDevice(tileSize));


    const auto size = result->totalSize();
    dim3 grid((size + Cfg::BlockDimX - 1) / (Cfg::BlockDimX));

    switch (col_1->type) {
        case INTEGER:
            EqualityKernel::equality_kernel_int64_t<<<grid, dim3(Cfg::BlockDimX)>>>(
                result->data,
                result->totalSize(),
                tileOffset.size(),

                static_cast<int64_t*>(_col_1),
                static_cast<int64_t*>(_col_2),
                col_1->data.size(),
                col_2->data.size(),

                static_cast<size_t*>(_hash_table), // hash table
                col_2->hashExSize, // hash ext size

                _mask,
                table_1_index,
                table_2_index,

                _tileSize,
                _tileOffset
                );
            break;
        default:
            throw std::runtime_error("Unsupported column type");
    }

    TensorKernel::extend_plain_kernel<<<grid, dim3(Cfg::BlockDimX)>>>(result->data, result->totalSize(), _mask, _tileSize, mask.size());

    cu::free(_mask);
    cu::free(_tileOffset);
    cu::free(_tileSize);
}

void GFI::inequality(
        tensor<char, Device::GPU> *result,

        column *col_1,
        column *col_2,

        std::vector<size_t> tileOffset,
        std::vector<size_t> tileSize,

        size_t table_1_index,
        size_t table_2_index,

        std::vector<size_t> mask,

        column::SortedSearchType operation
    ) {

    auto _col_1 = pool.getBufferOrCreate(static_cast<void*>(col_1), static_cast<void*>(col_1), columnPoolAllocator);
    auto _col_2 = pool.getBufferOrCreate(static_cast<void*>(col_2), static_cast<void*>(col_2), columnPoolAllocator);

    auto _index_table = pool.getBufferOrCreate(static_cast<void*>(&(col_2->sorted)), static_cast<void*>(col_2), columnIndexPoolAllocator);

    auto _mask = static_cast<size_t*>(cu::vectorToDevice(mask));
    auto _tileOffset = static_cast<size_t*>(cu::vectorToDevice(tileOffset));
    auto _tileSize = static_cast<size_t*>(cu::vectorToDevice(tileSize));


    const auto size = result->totalSize();
    dim3 grid((size + Cfg::BlockDimX - 1) / (Cfg::BlockDimX));

    switch (col_1->type) {
        case INTEGER:
            InequalityKernel::Inequality_kernel_int64_t<<<grid, dim3(Cfg::BlockDimX)>>>(
                result->data,
                result->totalSize(),
                tileOffset.size(),

                static_cast<int64_t*>(_col_1),
                static_cast<int64_t*>(_col_2),
                col_1->data.size(),
                col_2->data.size(),

                static_cast<size_t*>(_index_table),

                _mask,
                table_1_index,
                table_2_index,

                _tileSize,
                _tileOffset,

                operation
                );
        break;
        default:
            throw std::runtime_error("Unsupported column type");
    }

    TensorKernel::extend_plain_kernel<<<grid, dim3(Cfg::BlockDimX)>>>(result->data, result->totalSize(), _mask, _tileSize, mask.size());

    cu::free(_mask);
    cu::free(_tileOffset);
    cu::free(_tileSize);
}

void GFI::logical_and(const tensor<char, Device::GPU> *a, const tensor<char, Device::GPU> *b, tensor<char, Device::GPU> *out) {
    const auto size = out->totalSize();
    dim3 grid((size + Cfg::BlockDimX - 1) / (Cfg::BlockDimX));

    TensorKernel::logical_and<<<grid, dim3(Cfg::BlockDimX)>>>(
        a->data,
        b->data,
        size,
        out->data
    );
}

void GFI::logical_or(const tensor<char, Device::GPU> *a, const tensor<char, Device::GPU> *b, tensor<char, Device::GPU> *out) {
    const auto size = out->totalSize();
    dim3 grid((size + Cfg::BlockDimX - 1) / (Cfg::BlockDimX));

    TensorKernel::logical_or<<<grid, dim3(Cfg::BlockDimX)>>>(
        a->data,
        b->data,
        size,
        out->data
    );
}

void GFI::logical_not(const tensor<char, Device::GPU> *a, tensor<char, Device::GPU> *out) {
    const auto size = out->totalSize();
    dim3 grid((size + Cfg::BlockDimX - 1) / (Cfg::BlockDimX));

    TensorKernel::logical_not<<<grid, dim3(Cfg::BlockDimX)>>>(
        a->data,
        size,
        out->data
    );
}