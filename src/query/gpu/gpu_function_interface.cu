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


// Macro to check the last CUDA error and abort if one occurred
#define CUDA_CHECK_LAST_ERROR(msg)                                  \
    do {                                                            \
        cudaError_t err__ = cudaGetLastError();                     \
        if (err__ != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s: %s\n",        \
                    __FILE__, __LINE__, msg,                         \
                    cudaGetErrorString(err__));                     \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Macro to wrap a kernel launch (or any CUDA API call) and then check
#define CUDA_LAUNCH_AND_CHECK(kernelLaunchCall) \
    do {                                        \
        kernelLaunchCall;                       \
        CUDA_CHECK_LAST_ERROR(#kernelLaunchCall); \
    } while (0)


void GFI::fill(tensor<char, Device::GPU> *output_data, const char value) {
    const auto size = output_data->totalSize();
    dim3 grid((size + Cfg::BlockDim - 1) / (Cfg::BlockDim));

    TensorKernel::fill_kernel<<< grid, dim3(Cfg::BlockDim) >>>(output_data->data, value, size);
    CUDA_CHECK_LAST_ERROR("TensorKernel::fill_kernel");
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
    dim3 grid((size + Cfg::BlockDim - 1) / (Cfg::BlockDim));

    TensorKernel::fill_kernel<<< grid, dim3(Cfg::BlockDim) >>>(
        output_data->data,
        output_data->totalSize(),
        value,
        _position,
        _mask,
        _shape,
        mask.size());

    CUDA_CHECK_LAST_ERROR("TensorKernel::fill_kernel");

    cu::free(_position);
    cu::free(_mask);
    cu::free(_shape);
}


static auto columnPoolAllocator = [] (void* ptr, BufferAllocator alloc) {
    auto col = static_cast<column*>(ptr);
    auto size = col->data.size();
    void* gpuPtr = 0;

    // for dt
    std::vector<dateTime> dateTimeData(0);

    // for strings
    std::vector<void*> ptrs;

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
        case DateTime:
            gpuPtr = alloc(size * sizeof(dateTime));
            dateTimeData = std::vector<dateTime>(size); // fk switch
            for (size_t i = 0; i < size; ++i) {
                dateTimeData[i] = *col->data[i].t;
            }
            cu::toDevice(dateTimeData.data(), gpuPtr, size * sizeof(dateTime));
            size = size * sizeof(dateTime);
            break;
        case STRING:
            for (size_t i = 0; i < size; ++i) {
                std::string* str = col->data[i].s;
                auto gpu_str = alloc(str->size() + 1);
                cu::toDevice(str->c_str(), gpu_str, str->size() + 1);
                ptrs.push_back(gpu_str);
            }
            gpuPtr = cu::vectorToDevice(ptrs);
            break;
        default:
            throw std::runtime_error("Unsupported column type");

    }

    auto result = std::make_tuple(size, gpuPtr, [=] () {
        // free function
        if (col->type == STRING) {
            auto vec = cu::vectorFromDevice<void*>(gpuPtr, size);
            for (size_t i = 0; i < size; ++i) {
                auto gpu_str = vec[i];
                if (gpu_str != nullptr) {
                    cu::free(gpu_str);
                }
            }
            cu::free(gpuPtr);
        } else {
            cu::free(gpuPtr);
        }
    });

    return result;
};

static auto columnIndexPoolAllocator = [] (void* ptr, const BufferAllocator& alloc) {
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

static auto columnHashPoolAllocator = [] (void* ptr, const BufferAllocator& alloc) {
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

    if (col_1->type != col_2->type) {
        throw std::runtime_error("Column types must be same");
    }

    auto _col_1 = pool.getBufferOrCreate(static_cast<void*>(col_1), static_cast<void*>(col_1), columnPoolAllocator);
    auto _col_2 = pool.getBufferOrCreate(static_cast<void*>(col_2), static_cast<void*>(col_2), columnPoolAllocator);
    CUDA_CHECK_LAST_ERROR("GFI::equality pool.getBufferOrCreate columnPoolAllocator");

    void* _hash_table = nullptr;
    if (col_2->isHashIndexed()) {
        _hash_table = pool.getBufferOrCreate(static_cast<void*>(&(col_2->hashed)), static_cast<void*>(col_2), columnHashPoolAllocator);
        CUDA_CHECK_LAST_ERROR("GFI::equality pool.getBufferOrCreate columnHashPoolAllocator");
    }

    auto _mask = static_cast<size_t*>(cu::vectorToDevice(mask));
    auto _tileOffset = static_cast<size_t*>(cu::vectorToDevice(tileOffset));
    auto _tileSize = static_cast<size_t*>(cu::vectorToDevice(tileSize));

    if (_hash_table) {
        const auto size = tileSize[table_1_index];
        dim3 grid((size + Cfg::BlockDim - 1) / (Cfg::BlockDim));

        switch (col_1->type) {
            case INTEGER:
                EqualityKernel::equality_kernel<<<grid, dim3(Cfg::BlockDim)>>>(
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
            case FLOAT:
                EqualityKernel::equality_kernel<<<grid, dim3(Cfg::BlockDim)>>>(
                    result->data,
                    result->totalSize(),
                    tileOffset.size(),

                    static_cast<double*>(_col_1),
                    static_cast<double*>(_col_2),
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
            case STRING:
                EqualityKernel::equality_kernel<<<grid, dim3(Cfg::BlockDim)>>>(
                    result->data,
                    result->totalSize(),
                    tileOffset.size(),

                    static_cast<const char**>(_col_1),
                    static_cast<const char**>(_col_2),
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
            case DateTime:
                EqualityKernel::equality_kernel<<<grid, dim3(Cfg::BlockDim)>>>(
                    result->data,
                    result->totalSize(),
                    tileOffset.size(),

                    static_cast<dateTime*>(_col_1),
                    static_cast<dateTime*>(_col_2),
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
    } else {
        dim3 grid((tileSize[table_1_index] + Cfg::BlockDim2D - 1) / (Cfg::BlockDim2D), (tileSize[table_2_index] + Cfg::BlockDim2D - 1) / (Cfg::BlockDim2D));

        switch (col_1->type) {
            case INTEGER:
                EqualityKernel::equality_kernel<<<grid, dim3(Cfg::BlockDim2D, Cfg::BlockDim2D)>>>(
                    result->data,
                    result->totalSize(),
                    tileOffset.size(),

                    static_cast<int64_t*>(_col_1),
                    static_cast<int64_t*>(_col_2),
                    col_1->data.size(),
                    col_2->data.size(),

                    _mask,
                    table_1_index,
                    table_2_index,

                    _tileSize,
                    _tileOffset
                    );
            break;
            case FLOAT:
                EqualityKernel::equality_kernel<<<grid, dim3(Cfg::BlockDim2D, Cfg::BlockDim2D)>>>(
                    result->data,
                    result->totalSize(),
                    tileOffset.size(),

                    static_cast<double*>(_col_1),
                    static_cast<double*>(_col_2),
                    col_1->data.size(),
                    col_2->data.size(),

                    _mask,
                    table_1_index,
                    table_2_index,

                    _tileSize,
                    _tileOffset
                    );
            break;
            case STRING:
                EqualityKernel::equality_kernel<<<grid, dim3(Cfg::BlockDim2D, Cfg::BlockDim2D)>>>(
                    result->data,
                    result->totalSize(),
                    tileOffset.size(),

                    static_cast<const char**>(_col_1),
                    static_cast<const char**>(_col_2),
                    col_1->data.size(),
                    col_2->data.size(),

                    _mask,
                    table_1_index,
                    table_2_index,

                    _tileSize,
                    _tileOffset
                    );
            break;
            case DateTime:
                EqualityKernel::equality_kernel<<<grid, dim3(Cfg::BlockDim2D, Cfg::BlockDim2D)>>>(
                    result->data,
                    result->totalSize(),
                    tileOffset.size(),

                    static_cast<dateTime*>(_col_1),
                    static_cast<dateTime*>(_col_2),
                    col_1->data.size(),
                    col_2->data.size(),

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
    }
    CUDA_CHECK_LAST_ERROR("EqualityKernel::equality_kernel");

    const auto size = result->totalSize();
    dim3 grid((size + Cfg::BlockDim - 1) / (Cfg::BlockDim));
    TensorKernel::extend_plain_kernel<<<grid, dim3(Cfg::BlockDim)>>>(result->data, result->totalSize(), _mask, _tileSize, mask.size());

    CUDA_CHECK_LAST_ERROR("TensorKernel::extend_plain_kernel");

    cu::free(_mask);
    cu::free(_tileOffset);
    cu::free(_tileSize);
}

void GFI::equality(tensor<char, Device::GPU> *result, column *col_1, tval value, std::vector<size_t> tileOffset,
    std::vector<size_t> tileSize, size_t table_1_index, std::vector<size_t> mask) {

    auto _col_1 = pool.getBufferOrCreate(static_cast<void*>(col_1), static_cast<void*>(col_1), columnPoolAllocator);
    CUDA_CHECK_LAST_ERROR("GFI::equality pool.getBufferOrCreate columnPoolAllocator");

    auto _mask       = static_cast<size_t*>(cu::vectorToDevice(mask));
    auto _tileOffset = static_cast<size_t*>(cu::vectorToDevice(tileOffset));
    auto _tileSize   = static_cast<size_t*>(cu::vectorToDevice(tileSize));

    char* _sVal       = nullptr;
    if (col_1->type == STRING) {
        _sVal = static_cast<char*>(cu::stringToDevice(*value.s));
    }

    {
        dim3 grid((tileSize[table_1_index] + Cfg::BlockDim - 1) / (Cfg::BlockDim));
        switch (col_1->type) {
            case INTEGER:
                EqualityKernel::equality_kernel<<<grid, dim3(Cfg::BlockDim)>>>(
                    result->data,
                    result->totalSize(),
                    tileOffset.size(),

                    static_cast<int64_t*>(_col_1),
                    value.i,
                    col_1->data.size(),

                    _mask,
                    table_1_index,

                    _tileSize,
                    _tileOffset
                );
            break;
            case FLOAT:
                EqualityKernel::equality_kernel<<<grid, dim3(Cfg::BlockDim)>>>(
                    result->data,
                    result->totalSize(),
                    tileOffset.size(),

                    static_cast<double*>(_col_1),
                    value.d,
                    col_1->data.size(),

                    _mask,
                    table_1_index,

                    _tileSize,
                    _tileOffset
                    );
            break;
            case STRING:
                EqualityKernel::equality_kernel<<<grid, dim3(Cfg::BlockDim)>>>(
                    result->data,
                    result->totalSize(),
                    tileOffset.size(),

                    static_cast<const char**>(_col_1),
                    static_cast<const char*>(_sVal),
                    col_1->data.size(),

                    _mask,
                    table_1_index,

                    _tileSize,
                    _tileOffset
                    );
            break;
            case DateTime:
                EqualityKernel::equality_kernel<<<grid, dim3(Cfg::BlockDim)>>>(
                    result->data,
                    result->totalSize(),
                    tileOffset.size(),

                    static_cast<dateTime*>(_col_1),
                    *value.t,
                    col_1->data.size(),

                    _mask,
                    table_1_index,

                    _tileSize,
                    _tileOffset
                    );
            break;
            default:
                throw std::runtime_error("Unsupported column type");
        }

    }
    CUDA_CHECK_LAST_ERROR("EqualityKernel::equality_kernel");

    const auto size = result->totalSize();
    dim3 grid((size + Cfg::BlockDim - 1) / (Cfg::BlockDim));
    TensorKernel::extend_plain_kernel<<<grid, dim3(Cfg::BlockDim)>>>(result->data, result->totalSize(), _mask, _tileSize, mask.size());

    CUDA_CHECK_LAST_ERROR("TensorKernel::extend_plain_kernel");

    cu::free(_mask);
    cu::free(_tileOffset);
    cu::free(_tileSize);
    cu::free(_sVal);
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
    CUDA_CHECK_LAST_ERROR("GFI::equality pool.getBufferOrCreate columnPoolAllocator");

    void* _index_table = nullptr;
    if (col_2->isSortIndexed()) {
        _index_table = pool.getBufferOrCreate(static_cast<void*>(&(col_2->sorted)), static_cast<void*>(col_2), columnIndexPoolAllocator);
        CUDA_CHECK_LAST_ERROR("GFI::equality pool.getBufferOrCreate columnIndexPoolAllocator");
    }

    auto _mask = static_cast<size_t*>(cu::vectorToDevice(mask));
    auto _tileOffset = static_cast<size_t*>(cu::vectorToDevice(tileOffset));
    auto _tileSize = static_cast<size_t*>(cu::vectorToDevice(tileSize));

    if (_index_table) {
        const auto size = tileSize[table_1_index];
        dim3 grid((size + Cfg::BlockDim - 1) / (Cfg::BlockDim));

        switch (col_1->type) {
            case INTEGER:
                InequalityKernel::inequality_kernel<<<grid, dim3(Cfg::BlockDim)>>>(
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
            case FLOAT:
                InequalityKernel::inequality_kernel<<<grid, dim3(Cfg::BlockDim)>>>(
                    result->data,
                    result->totalSize(),
                    tileOffset.size(),

                    static_cast<double*>(_col_1),
                    static_cast<double*>(_col_2),
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
            case STRING:
                InequalityKernel::inequality_kernel<<<grid, dim3(Cfg::BlockDim)>>>(
                    result->data,
                    result->totalSize(),
                    tileOffset.size(),

                    static_cast<char**>(_col_1),
                    static_cast<char**>(_col_2),
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
            case DateTime:
                InequalityKernel::inequality_kernel<<<grid, dim3(Cfg::BlockDim)>>>(
                    result->data,
                    result->totalSize(),
                    tileOffset.size(),

                    static_cast<dateTime*>(_col_1),
                    static_cast<dateTime*>(_col_2),
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
    } else {
        dim3 grid((tileSize[table_1_index] + Cfg::BlockDim2D - 1) / (Cfg::BlockDim2D), (tileSize[table_2_index] + Cfg::BlockDim2D - 1) / (Cfg::BlockDim2D));

        switch (col_1->type) {
            case INTEGER:
                InequalityKernel::inequality_kernel<<<grid, dim3(Cfg::BlockDim2D, Cfg::BlockDim2D)>>>(
                    result->data,
                    result->totalSize(),
                    tileOffset.size(),

                    static_cast<int64_t*>(_col_1),
                    static_cast<int64_t*>(_col_2),
                    col_1->data.size(),
                    col_2->data.size(),

                    _mask,
                    table_1_index,
                    table_2_index,

                    _tileSize,
                    _tileOffset,
                    operation
                    );
            break;
            case FLOAT:
                InequalityKernel::inequality_kernel<<<grid, dim3(Cfg::BlockDim2D, Cfg::BlockDim2D)>>>(
                    result->data,
                    result->totalSize(),
                    tileOffset.size(),

                    static_cast<double*>(_col_1),
                    static_cast<double*>(_col_2),
                    col_1->data.size(),
                    col_2->data.size(),

                    _mask,
                    table_1_index,
                    table_2_index,

                    _tileSize,
                    _tileOffset,
                    operation
                    );
            break;
            case STRING:
                InequalityKernel::inequality_kernel<<<grid, dim3(Cfg::BlockDim2D, Cfg::BlockDim2D)>>>(
                    result->data,
                    result->totalSize(),
                    tileOffset.size(),

                    static_cast<const char**>(_col_1),
                    static_cast<const char**>(_col_2),
                    col_1->data.size(),
                    col_2->data.size(),

                    _mask,
                    table_1_index,
                    table_2_index,

                    _tileSize,
                    _tileOffset,
                    operation
                    );
            break;
            case DateTime:
                InequalityKernel::inequality_kernel<<<grid, dim3(Cfg::BlockDim2D, Cfg::BlockDim2D)>>>(
                    result->data,
                    result->totalSize(),
                    tileOffset.size(),

                    static_cast<dateTime*>(_col_1),
                    static_cast<dateTime*>(_col_2),
                    col_1->data.size(),
                    col_2->data.size(),

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
    }
    CUDA_CHECK_LAST_ERROR("EqualityKernel::equality_kernel");

    const auto size = result->totalSize();
    dim3 grid((size + Cfg::BlockDim - 1) / (Cfg::BlockDim));
    TensorKernel::extend_plain_kernel<<<grid, dim3(Cfg::BlockDim)>>>(result->data, result->totalSize(), _mask, _tileSize, mask.size());

    CUDA_CHECK_LAST_ERROR("TensorKernel::extend_plain_kernel");

    cu::free(_mask);
    cu::free(_tileOffset);
    cu::free(_tileSize);
}

void GFI::inequality(tensor<char, Device::GPU> *result, column *col_1, tval value, std::vector<size_t> tileOffset,
    std::vector<size_t> tileSize, size_t table_1_index, std::vector<size_t> mask, column::SortedSearchType operation) {

    auto _col_1 = pool.getBufferOrCreate(static_cast<void*>(col_1), static_cast<void*>(col_1), columnPoolAllocator);
    CUDA_CHECK_LAST_ERROR("GFI::equality pool.getBufferOrCreate columnPoolAllocator");

    auto _mask       = static_cast<size_t*>(cu::vectorToDevice(mask));
    auto _tileOffset = static_cast<size_t*>(cu::vectorToDevice(tileOffset));
    auto _tileSize   = static_cast<size_t*>(cu::vectorToDevice(tileSize));

    char* _sVal       = nullptr;
    if (col_1->type == STRING) {
        _sVal = static_cast<char*>(cu::stringToDevice(*value.s));
    }

    {
        dim3 grid((tileSize[table_1_index] + Cfg::BlockDim - 1) / (Cfg::BlockDim));
        switch (col_1->type) {
            case INTEGER:
                InequalityKernel::inequality_kernel<<<grid, dim3(Cfg::BlockDim)>>>(
                    result->data,
                    result->totalSize(),
                    tileOffset.size(),

                    static_cast<int64_t*>(_col_1),
                    value.i,
                    col_1->data.size(),

                    _mask,
                    table_1_index,

                    _tileSize,
                    _tileOffset,
                    operation
                );
            break;
            case FLOAT:
                InequalityKernel::inequality_kernel<<<grid, dim3(Cfg::BlockDim)>>>(
                    result->data,
                    result->totalSize(),
                    tileOffset.size(),

                    static_cast<double*>(_col_1),
                    value.d,
                    col_1->data.size(),

                    _mask,
                    table_1_index,

                    _tileSize,
                    _tileOffset,
                    operation
                    );
            break;
            case STRING:
                InequalityKernel::inequality_kernel<<<grid, dim3(Cfg::BlockDim)>>>(
                    result->data,
                    result->totalSize(),
                    tileOffset.size(),

                    static_cast<const char**>(_col_1),
                    static_cast<const char*>(_sVal),
                    col_1->data.size(),

                    _mask,
                    table_1_index,

                    _tileSize,
                    _tileOffset,
                    operation
                    );
            break;
            case DateTime:
                InequalityKernel::inequality_kernel<<<grid, dim3(Cfg::BlockDim)>>>(
                    result->data,
                    result->totalSize(),
                    tileOffset.size(),

                    static_cast<dateTime*>(_col_1),
                    *value.t,
                    col_1->data.size(),

                    _mask,
                    table_1_index,

                    _tileSize,
                    _tileOffset,
                    operation
                    );
            break;
            default:
                throw std::runtime_error("Unsupported column type");
        }
    }

    CUDA_CHECK_LAST_ERROR("EqualityKernel::equality_kernel");

    const auto size = result->totalSize();
    dim3 grid((size + Cfg::BlockDim - 1) / (Cfg::BlockDim));
    TensorKernel::extend_plain_kernel<<<grid, dim3(Cfg::BlockDim)>>>(result->data, result->totalSize(), _mask, _tileSize, mask.size());

    CUDA_CHECK_LAST_ERROR("TensorKernel::extend_plain_kernel");

    cu::free(_mask);
    cu::free(_tileOffset);
    cu::free(_tileSize);
    cu::free(_sVal);
}

void GFI::logical_and(const tensor<char, Device::GPU> *a, const tensor<char, Device::GPU> *b, tensor<char, Device::GPU> *out) {
    const auto size = out->totalSize();
    dim3 grid((size + Cfg::BlockDim - 1) / (Cfg::BlockDim));

    TensorKernel::logical_and<<<grid, dim3(Cfg::BlockDim)>>>(
        a->data,
        b->data,
        size,
        out->data
    );

    CUDA_CHECK_LAST_ERROR("TensorKernel::logical_and");
}

void GFI::logical_or(const tensor<char, Device::GPU> *a, const tensor<char, Device::GPU> *b, tensor<char, Device::GPU> *out) {
    const auto size = out->totalSize();
    dim3 grid((size + Cfg::BlockDim - 1) / (Cfg::BlockDim));

    TensorKernel::logical_or<<<grid, dim3(Cfg::BlockDim)>>>(
        a->data,
        b->data,
        size,
        out->data
    );

    CUDA_CHECK_LAST_ERROR("TensorKernel::logical_or");
}

void GFI::logical_not(const tensor<char, Device::GPU> *a, tensor<char, Device::GPU> *out) {
    const auto size = out->totalSize();
    dim3 grid((size + Cfg::BlockDim - 1) / (Cfg::BlockDim));

    TensorKernel::logical_not<<<grid, dim3(Cfg::BlockDim)>>>(
        a->data,
        size,
        out->data
    );

    CUDA_CHECK_LAST_ERROR("TensorKernel::logical_not");
}

static void run_scan(size_t* input, size_t* output, size_t n) {
    if (n == 0) return;

    size_t blocks = (n + Cfg::BlockDim * 2 - 1) / (Cfg::BlockDim * 2);

    size_t *d_block_sums;
    cudaMalloc(&d_block_sums, blocks * sizeof(size_t));

    TensorKernel::efficient_prefix_sum<<<blocks, Cfg::BlockDim, (Cfg::BlockDim * 2 + 1) * sizeof(size_t)>>>(input, output, n, d_block_sums);
    CUDA_CHECK_LAST_ERROR("TensorKernel::efficient_prefix_sum");

    if (blocks > Cfg::BlockDim) {
        auto r_v = static_cast<size_t*>(cu::malloc(blocks * sizeof(size_t)));
        run_scan(d_block_sums, r_v, blocks);
        CUDA_CHECK_LAST_ERROR("run_scan::recursive");
        cu::free(d_block_sums);
        d_block_sums = r_v;
    } else {
        TensorKernel::efficient_prefix_sum<<<1, Cfg::BlockDim, (Cfg::BlockDim * 2 + 1) * sizeof(size_t)>>>(d_block_sums, d_block_sums, blocks, nullptr);
        CUDA_CHECK_LAST_ERROR("TensorKernel::efficient_prefix_sum");
    }

    if (blocks > 1) {
        CUDA_CHECK_LAST_ERROR("TensorKernel::add_aux before");
        TensorKernel::add_aux<<<blocks - 1, Cfg::BlockDim>>>(output, n, d_block_sums);
        CUDA_CHECK_LAST_ERROR("TensorKernel::add_aux");
    }

    cudaFree(d_block_sums);
}

static void run_scan(char* input, size_t* output, size_t n) {
    if (n == 0) return;

    size_t blocks = (n + Cfg::BlockDim * 2 - 1) / (Cfg::BlockDim * 2);

    auto d_block_sums = static_cast<size_t*>(cu::malloc(blocks * sizeof(size_t)));

    TensorKernel::efficient_prefix_sum<<<blocks, Cfg::BlockDim, (Cfg::BlockDim * 2) * sizeof(size_t)>>>(input, output, n, d_block_sums);
    CUDA_CHECK_LAST_ERROR("TensorKernel::efficient_prefix_sum");

    if (blocks > Cfg::BlockDim) {
        auto r_v = static_cast<size_t*>(cu::malloc(blocks * sizeof(size_t)));
        run_scan(d_block_sums, r_v, blocks);
        CUDA_CHECK_LAST_ERROR("run_scan::recursive");
        cu::free(d_block_sums);
        d_block_sums = r_v;
    } else {
        TensorKernel::efficient_prefix_sum<<<1, Cfg::BlockDim, (Cfg::BlockDim * 2) * sizeof(size_t)>>>(d_block_sums, d_block_sums, blocks, nullptr);
        CUDA_CHECK_LAST_ERROR("TensorKernel::efficient_prefix_sum");
    }

    if (blocks > 1) {

        TensorKernel::add_aux<<<blocks - 1, Cfg::BlockDim>>>(output, n, d_block_sums);
        CUDA_CHECK_LAST_ERROR("TensorKernel::add_aux");
    }

    cu::free(d_block_sums);
}


std::vector<size_t> GFI::iterator(tensor<char, Device::GPU> *a) {
    std::vector<size_t> result;
    const auto size = a->totalSize();

    auto out = static_cast<size_t*>(cu::malloc(sizeof(size_t) * size));

    run_scan(a->data, out, size);

    auto cpu_out = static_cast<size_t*>(malloc(sizeof(size_t) * size));

    cu::toHost(out, cpu_out, sizeof(size_t) * size);

    size_t prev = 0;

    for (size_t i = 0;i < size;i++) {
        auto val = cpu_out[i];
        if (val == prev) {
            auto low = i;
            auto high = size;
            while (low < high) {
                auto mid = low + (high - low) / 2;
                if (cpu_out[mid] == val) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }

            i = low - 1;
        } else {
            result.push_back(i);
            prev = val;
        }
    }

    cu::free(out);
    free(cpu_out);

    return result;
}
