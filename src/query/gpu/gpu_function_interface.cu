//
// Created by xabdomo on 4/20/25.
//

#include "gpu_function_interface.cuh"

#include "gpu_buffer_pool.cuh"
#include "store.hpp"
#include "kernels/equality_kernel.cuh"
#include "kernels/helper_kernels.cuh"
#include "kernels/inequality_kernel.cuh"
#include "kernels/order_by.cuh"
#include "kernels/reduce_kernels.cuh"
#include "kernels/tensor_kernels.cuh"
#include "query/errors.hpp"
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

#define CUDA_CHECK(kernelLaunchCall) CUDA_LAUNCH_AND_CHECK(kernelLaunchCall)


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

static auto columnNullsAllocator = [] (void* ptr, BufferAllocator alloc) {
    auto col = static_cast<column*>(ptr);
    auto size = col->nulls.size();
    void* gpuPtr = 0;

    if (size == 0) return std::make_tuple(size, static_cast<void*>(nullptr), static_cast<VoidFunction>(nullptr));

    gpuPtr = alloc(size * sizeof(char));
    cu::toDevice(col->nulls.data(), gpuPtr, size * sizeof(char));
    size = size * sizeof(int64_t);


    return std::make_tuple(size, gpuPtr, static_cast<VoidFunction>(nullptr));
};


static auto columnPoolAllocator = [] (void* ptr, BufferAllocator alloc) {
    auto col = static_cast<column*>(ptr);
    auto size = col->data.size();
    auto dataSize = size;
    void* gpuPtr = 0;

    // for dt
    std::vector<dateTime> dateTimeData(0);

    // for strings
    std::vector<void*> ptrs;

    switch (col->type) {
        case INTEGER:
            gpuPtr = alloc(size * sizeof(int64_t));
            cu::toDevice(col->data.data(), gpuPtr, size * sizeof(int64_t));
            dataSize = size * sizeof(int64_t);
            break;
        case FLOAT:
            gpuPtr = alloc(size * sizeof(double));
            cu::toDevice(col->data.data(), gpuPtr, size * sizeof(double));
            dataSize = size * sizeof(double);
            break;
        case DateTime:
            gpuPtr = alloc(size * sizeof(dateTime));
            dateTimeData = std::vector<dateTime>(size); // fk switch
            for (size_t i = 0; i < size; ++i) {
                dateTimeData[i] = *col->data[i].t;
            }
            cu::toDevice(dateTimeData.data(), gpuPtr, size * sizeof(dateTime));
            dataSize = size * sizeof(dateTime);
            break;
        case STRING:
            dataSize = size * sizeof(char*);
            for (size_t i = 0; i < size; ++i) {
                std::string* str = col->data[i].s;
                auto gpu_str = alloc(str->size() + 1);
                cu::toDevice(str->c_str(), gpu_str, str->size() + 1);
                ptrs.push_back(gpu_str);
                dataSize += str->size();
            }
            gpuPtr = cu::vectorToDevice(ptrs);
            break;
        default:
            throw std::runtime_error("Unsupported column type");

    }

    auto result = std::make_tuple(dataSize, gpuPtr, [=] () {
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
        throw std::runtime_error("Column types must be the same");
    }

    auto _col_1 = pool.getBufferOrCreate(static_cast<void*>(col_1), static_cast<void*>(col_1), columnPoolAllocator);
    auto _col_2 = pool.getBufferOrCreate(static_cast<void*>(col_2), static_cast<void*>(col_2), columnPoolAllocator);

    auto _col_1_nulls = static_cast<char*>(pool.getBufferOrCreate(static_cast<void*>(&col_1->nulls), static_cast<void*>(col_1), columnNullsAllocator));
    auto _col_2_nulls = static_cast<char*>(pool.getBufferOrCreate(static_cast<void*>(&col_2->nulls), static_cast<void*>(col_2), columnNullsAllocator));
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
                    _col_1_nulls,
                    static_cast<int64_t*>(_col_2),
                    _col_2_nulls,
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
                    _col_1_nulls,
                    static_cast<double*>(_col_2),
                    _col_2_nulls,
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
                    _col_1_nulls,
                    static_cast<const char**>(_col_2),
                    _col_2_nulls,
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
                    _col_1_nulls,
                    static_cast<dateTime*>(_col_2),
                    _col_2_nulls,
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
                    _col_1_nulls,
                    _col_2_nulls,
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
                    _col_1_nulls,
                    _col_2_nulls,
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
                    _col_1_nulls,
                    _col_2_nulls,
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
                    _col_1_nulls,
                    _col_2_nulls,
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
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto size = result->totalSize();
    dim3 grid((size + Cfg::BlockDim - 1) / (Cfg::BlockDim));
    TensorKernel::extend_plain_kernel<<<grid, dim3(Cfg::BlockDim)>>>(result->data, result->totalSize(), _mask, _tileSize, mask.size());

    CUDA_CHECK_LAST_ERROR("TensorKernel::extend_plain_kernel");

    cu::free(_mask);
    cu::free(_tileOffset);
    cu::free(_tileSize);
}

void GFI::equality(tensor<char, Device::GPU> *result, column *col_1, tval value, bool isNull, std::vector<size_t> tileOffset,
    std::vector<size_t> tileSize, size_t table_1_index, std::vector<size_t> mask) {

    auto _col_1 = pool.getBufferOrCreate(static_cast<void*>(col_1), static_cast<void*>(col_1), columnPoolAllocator);
    auto _col_1_nulls = static_cast<char*>(pool.getBufferOrCreate(static_cast<void*>(&col_1->nulls), static_cast<void*>(col_1), columnNullsAllocator));
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
                    _col_1_nulls,
                    isNull,
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
                    _col_1_nulls,
                    isNull,
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
                    _col_1_nulls,
                    isNull,
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
                    _col_1_nulls,
                    isNull,
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
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto size = result->totalSize();
    dim3 grid((size + Cfg::BlockDim - 1) / (Cfg::BlockDim));
    TensorKernel::extend_plain_kernel<<<grid, dim3(Cfg::BlockDim)>>>(result->data, result->totalSize(), _mask, _tileSize, mask.size());

    CUDA_CHECK_LAST_ERROR("TensorKernel::extend_plain_kernel");

    cu::free(_mask);
    cu::free(_tileOffset);
    cu::free(_tileSize);
    cu::free(_sVal);
}


void GFI::equality_date(tensor<char, Device::GPU> *result, column *col_1, tval value, std::vector<size_t> tileOffset,
    std::vector<size_t> tileSize, size_t table_1_index, std::vector<size_t> mask) {

    if (col_1->type != DateTime) {
        throw UnsupportedOperationError("Expected col to have a type of dateTime");
    }

    auto _col_1 = pool.getBufferOrCreate(static_cast<void*>(col_1), static_cast<void*>(col_1), columnPoolAllocator);
    CUDA_CHECK_LAST_ERROR("GFI::equality pool.getBufferOrCreate columnPoolAllocator");

    auto _mask       = static_cast<size_t*>(cu::vectorToDevice(mask));
    auto _tileOffset = static_cast<size_t*>(cu::vectorToDevice(tileOffset));
    auto _tileSize   = static_cast<size_t*>(cu::vectorToDevice(tileSize));


    {
        dim3 grid((tileSize[table_1_index] + Cfg::BlockDim - 1) / (Cfg::BlockDim));
        EqualityKernel::equality_kernel_date<<<grid, dim3(Cfg::BlockDim)>>>(
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

    }
    CUDA_CHECK_LAST_ERROR("EqualityKernel::equality_kernel");
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto size = result->totalSize();
    dim3 grid((size + Cfg::BlockDim - 1) / (Cfg::BlockDim));
    TensorKernel::extend_plain_kernel<<<grid, dim3(Cfg::BlockDim)>>>(result->data, result->totalSize(), _mask, _tileSize, mask.size());

    CUDA_CHECK_LAST_ERROR("TensorKernel::extend_plain_kernel");

    cu::free(_mask);
    cu::free(_tileOffset);
    cu::free(_tileSize);
}


void GFI::equality_time(tensor<char, Device::GPU> *result, column *col_1, tval value, std::vector<size_t> tileOffset,
    std::vector<size_t> tileSize, size_t table_1_index, std::vector<size_t> mask) {

    if (col_1->type != DateTime) {
        throw UnsupportedOperationError("Expected col to have a type of dateTime");
    }

    auto _col_1 = pool.getBufferOrCreate(static_cast<void*>(col_1), static_cast<void*>(col_1), columnPoolAllocator);
    CUDA_CHECK_LAST_ERROR("GFI::equality pool.getBufferOrCreate columnPoolAllocator");

    auto _mask       = static_cast<size_t*>(cu::vectorToDevice(mask));
    auto _tileOffset = static_cast<size_t*>(cu::vectorToDevice(tileOffset));
    auto _tileSize   = static_cast<size_t*>(cu::vectorToDevice(tileSize));


    {
        dim3 grid((tileSize[table_1_index] + Cfg::BlockDim - 1) / (Cfg::BlockDim));
        EqualityKernel::equality_kernel_time<<<grid, dim3(Cfg::BlockDim)>>>(
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

    }
    CUDA_CHECK_LAST_ERROR("EqualityKernel::equality_kernel");
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto size = result->totalSize();
    dim3 grid((size + Cfg::BlockDim - 1) / (Cfg::BlockDim));
    TensorKernel::extend_plain_kernel<<<grid, dim3(Cfg::BlockDim)>>>(result->data, result->totalSize(), _mask, _tileSize, mask.size());

    CUDA_CHECK_LAST_ERROR("TensorKernel::extend_plain_kernel");

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

    auto _col_1_nulls = static_cast<char*>(pool.getBufferOrCreate(static_cast<void*>(&col_1->nulls), static_cast<void*>(col_1), columnNullsAllocator));
    auto _col_2_nulls = static_cast<char*>(pool.getBufferOrCreate(static_cast<void*>(&col_2->nulls), static_cast<void*>(col_2), columnNullsAllocator));
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
                    _col_1_nulls,
                    _col_2_nulls,
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
                    _col_1_nulls,
                    _col_2_nulls,
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
                    _col_1_nulls,
                    _col_2_nulls,
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
                    _col_1_nulls,
                    _col_2_nulls,
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
                    _col_1_nulls,
                    _col_2_nulls,
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
                    _col_1_nulls,
                    _col_2_nulls,
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
                    _col_1_nulls,
                    _col_2_nulls,
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
                    _col_1_nulls,
                    _col_2_nulls,
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
    CUDA_CHECK(cudaDeviceSynchronize());

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
    auto _col_1_nulls = static_cast<char*>(pool.getBufferOrCreate(static_cast<void*>(&col_1->nulls), static_cast<void*>(col_1), columnNullsAllocator));

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
                    _col_1_nulls,
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
                    _col_1_nulls,
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
                    _col_1_nulls,
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
                    _col_1_nulls,
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
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto size = result->totalSize();
    dim3 grid((size + Cfg::BlockDim - 1) / (Cfg::BlockDim));
    TensorKernel::extend_plain_kernel<<<grid, dim3(Cfg::BlockDim)>>>(result->data, result->totalSize(), _mask, _tileSize, mask.size());

    CUDA_CHECK_LAST_ERROR("TensorKernel::extend_plain_kernel");

    cu::free(_mask);
    cu::free(_tileOffset);
    cu::free(_tileSize);
    cu::free(_sVal);
}

template<typename T, typename __rf>
static T run_reducer(T* input, size_t n, __rf func) {
    if (n == 0) return static_cast<T>(0);
    size_t blocks = (n + Cfg::BlockDim * 2 - 1) / (Cfg::BlockDim * 2);

    T *d_block_sums = static_cast<T*>(cu::malloc(blocks * sizeof(T)));
    func<<<blocks, Cfg::BlockDim, (Cfg::BlockDim * 2) * sizeof(T)>>>(input, n, d_block_sums);
    CUDA_CHECK_LAST_ERROR("run_reducer::Reducer_Func");

    T result{};

    if (blocks > Cfg::BlockDim * 2) {
        result = run_reducer<T, __rf>(d_block_sums, blocks, func);
        CUDA_CHECK_LAST_ERROR("run_reducer::recursive");
    } else {
        T* final = static_cast<T*>(cu::malloc(sizeof(T)));
        func<<<1, Cfg::BlockDim, (Cfg::BlockDim * 2) * sizeof(T)>>>(d_block_sums, blocks, final);
        CUDA_CHECK_LAST_ERROR("run_reducer::Reducer_Func (internal)");
        cu::toHost(final, &result, sizeof(T));
        cu::free(final);
    }

    cu::free(d_block_sums);
    return result;
}


template<typename T, typename __nrf>
static T run_reducer_nulls(T* input, char* nulls, size_t n, __nrf func) {
    if (n == 0) return static_cast<T>(0);
    size_t blocks = (n + Cfg::BlockDim * 2 - 1) / (Cfg::BlockDim * 2);

    T    *d_block_sums  = static_cast<T*>(cu::malloc(blocks * sizeof(T)));
    char *d_block_nulls = static_cast<char*>(cu::malloc(blocks * sizeof(char)));
    func<<<blocks, Cfg::BlockDim, (Cfg::BlockDim * 2) * sizeof(T)>>>(input, nulls, n, d_block_sums, d_block_nulls);
    CUDA_CHECK_LAST_ERROR("run_reducer_nulls::Reducer_Func");

    T result{};

    if (blocks > Cfg::BlockDim * 2) {
        result = run_reducer_nulls<T, __nrf>(d_block_sums, nulls, blocks, func);
        CUDA_CHECK_LAST_ERROR("run_reducer_nulls::recursive");
    } else {
        T* final = static_cast<T*>(cu::malloc(sizeof(T)));
        func<<<1, Cfg::BlockDim, (Cfg::BlockDim * 2) * sizeof(T)>>>(d_block_sums, d_block_nulls , blocks, final, nullptr);
        CUDA_CHECK_LAST_ERROR("run_reducer_nulls::Reducer_Func (internal)");
        cu::toHost(final, &result, sizeof(T));
        cu::free(final);
    }

    cu::free(d_block_sums);
    cu::free(d_block_nulls);
    return result;
}

tval GFI::max(column *col_1) {
    auto _col_1 = pool.getBufferOrCreate(static_cast<void*>(col_1), static_cast<void*>(col_1), columnPoolAllocator);
    auto _col_1_nulls = static_cast<char*>(pool.getBufferOrCreate(static_cast<void*>(&col_1->nulls), static_cast<void*>(col_1), columnNullsAllocator));
    CUDA_CHECK_LAST_ERROR("GFI::equality pool.getBufferOrCreate columnPoolAllocator");

    const char* devPtr = nullptr;

    size_t* devSizePtr = static_cast<size_t *> (cu::malloc(sizeof(size_t)));
    size_t* hostSizePtr = static_cast<size_t *> (malloc(sizeof(size_t)));
    char* hostPtr = nullptr;

    tval result;
    switch (col_1->type) {
        case INTEGER:
            result = ValuesHelper::create_from(run_reducer_nulls(static_cast<int64_t*>(_col_1), _col_1_nulls, col_1->data.size(), ReduceKernel::max_nulls<int64_t>));
            break;
        case FLOAT:
            result = ValuesHelper::create_from(run_reducer_nulls(static_cast<double*>(_col_1), _col_1_nulls, col_1->data.size(), ReduceKernel::max_nulls<double>));
            break;
        case DateTime:
            result = ValuesHelper::create_from(run_reducer_nulls(static_cast<dateTime*>(_col_1), _col_1_nulls, col_1->data.size(), ReduceKernel::max_nulls<dateTime>));
            break;
        case STRING:
            devPtr  = run_reducer_nulls(static_cast<const char**>(_col_1), _col_1_nulls, col_1->data.size(), ReduceKernel::max_nulls<const char*>);
            HelperKernels::strlen<<<1, 1>>>(devPtr, devSizePtr);
            cudaDeviceSynchronize();
            cu::toHost(devSizePtr, hostSizePtr, sizeof(size_t));
            hostPtr = static_cast<char*>(malloc(hostSizePtr[0] + 1));
            cu::toHost(devPtr, hostPtr, hostSizePtr[0]);
            hostPtr[hostSizePtr[0]] = '\0';
            result = ValuesHelper::create_from(hostPtr);
            break;
        default:
            throw std::runtime_error("Unsupported column type");
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cu::free(devSizePtr);
    free(hostSizePtr);
    free(hostPtr);
    return result;
}

tval GFI::min(column *col_1) {
    auto _col_1 = pool.getBufferOrCreate(static_cast<void*>(col_1), static_cast<void*>(col_1), columnPoolAllocator);
    auto _col_1_nulls = static_cast<char*>(pool.getBufferOrCreate(static_cast<void*>(&col_1->nulls), static_cast<void*>(col_1), columnNullsAllocator));
    CUDA_CHECK_LAST_ERROR("GFI::min pool.getBufferOrCreate columnPoolAllocator");

    const char* devPtr = nullptr;

    size_t* devSizePtr = static_cast<size_t *> (cu::malloc(sizeof(size_t)));
    size_t* hostSizePtr = static_cast<size_t *> (malloc(sizeof(size_t)));
    char* hostPtr = nullptr;

    tval result;
    switch (col_1->type) {
        case INTEGER:
            result = ValuesHelper::create_from(run_reducer_nulls(static_cast<int64_t*>(_col_1), _col_1_nulls, col_1->data.size(), ReduceKernel::min_nulls<int64_t>));
            break;
        case FLOAT:
            result = ValuesHelper::create_from(run_reducer_nulls(static_cast<double*>(_col_1), _col_1_nulls, col_1->data.size(), ReduceKernel::min_nulls<double>));
            break;
        case DateTime:
            result = ValuesHelper::create_from(run_reducer_nulls(static_cast<dateTime*>(_col_1), _col_1_nulls, col_1->data.size(), ReduceKernel::min_nulls<dateTime>));
            break;
        case STRING:
            devPtr  = run_reducer_nulls(static_cast<const char**>(_col_1), _col_1_nulls, col_1->data.size(), ReduceKernel::min_nulls<const char*>);
            HelperKernels::strlen<<<1, 1>>>(devPtr, devSizePtr);
            cudaDeviceSynchronize();
            cu::toHost(devSizePtr, hostSizePtr, sizeof(size_t));
            hostPtr = static_cast<char*>(malloc(hostSizePtr[0] + 1));
            cu::toHost(devPtr, hostPtr, hostSizePtr[0]);
            hostPtr[hostSizePtr[0]] = '\0';
            result = ValuesHelper::create_from(hostPtr);
            break;
        default:
            throw std::runtime_error("Unsupported column type");
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    cu::free(devSizePtr);
    free(hostSizePtr);
    free(hostPtr);
    return result;
}

tval GFI::sum(column *col_1) {
    if (col_1->type == STRING) throw std::runtime_error("Unsupported column type");

    auto _col_1 = pool.getBufferOrCreate(static_cast<void*>(col_1), static_cast<void*>(col_1), columnPoolAllocator);
    auto _col_1_nulls = static_cast<char*>(pool.getBufferOrCreate(static_cast<void*>(&col_1->nulls), static_cast<void*>(col_1), columnNullsAllocator));
    CUDA_CHECK_LAST_ERROR("GFI::sum pool.getBufferOrCreate columnPoolAllocator");

    CUDA_CHECK(cudaDeviceSynchronize());

    tval result;
    switch (col_1->type) {
        case INTEGER:
            result = ValuesHelper::create_from(run_reducer_nulls(static_cast<int64_t*>(_col_1), _col_1_nulls, col_1->data.size(), ReduceKernel::sum_nulls<int64_t>));
            break;
        case FLOAT:
            result = ValuesHelper::create_from(run_reducer_nulls(static_cast<double*>(_col_1), _col_1_nulls, col_1->data.size(), ReduceKernel::sum_nulls<double>));
            break;
        case DateTime:
            result = ValuesHelper::create_from(run_reducer_nulls(static_cast<dateTime*>(_col_1), _col_1_nulls, col_1->data.size(), ReduceKernel::sum_nulls<dateTime>));
            break;
        default:
            throw std::runtime_error("Unsupported column type");
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
}

tval GFI::avg(column *col_1) {
    auto _s = sum(col_1);
    switch (col_1->type) {
        case INTEGER:
            _s.d = static_cast<float>(_s.i) / (col_1->data.size() - col_1->nullsCount);
            break;
        case FLOAT:
            _s.d = _s.d / (col_1->data.size() - col_1->nullsCount);
            break;
        case DateTime:
            _s.t->day = _s.t->day / (col_1->data.size() - col_1->nullsCount);
            _s.t->month = _s.t->month / (col_1->data.size() - col_1->nullsCount);
            _s.t->year = _s.t->year / (col_1->data.size() - col_1->nullsCount);
            _s.t->hour = _s.t->hour / (col_1->data.size() - col_1->nullsCount);
            _s.t->minute = _s.t->minute / (col_1->data.size() - col_1->nullsCount);
            _s.t->second = _s.t->second / (col_1->data.size() - col_1->nullsCount);
            break;
        default:
            break;
    }

    return _s;
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
    CUDA_CHECK(cudaDeviceSynchronize());

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
    CUDA_CHECK(cudaDeviceSynchronize());
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
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK_LAST_ERROR("TensorKernel::logical_not");
}

static void run_prefix_sum(size_t* input, size_t* output, size_t n) {
    if (n == 0) return;

    size_t blocks = (n + Cfg::BlockDim * 2 - 1) / (Cfg::BlockDim * 2);

    size_t *d_block_sums;
    cudaMalloc(&d_block_sums, blocks * sizeof(size_t));

    TensorKernel::efficient_prefix_sum<<<blocks, Cfg::BlockDim, (Cfg::BlockDim * 2 + 1) * sizeof(size_t)>>>(input, output, n, d_block_sums);
    CUDA_CHECK_LAST_ERROR("TensorKernel::efficient_prefix_sum");

    if (blocks > Cfg::BlockDim) {
        auto r_v = static_cast<size_t*>(cu::malloc(blocks * sizeof(size_t)));
        run_prefix_sum(d_block_sums, r_v, blocks);
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

static void run_prefix_sum(char* input, size_t* output, size_t n) {
    if (n == 0) return;

    size_t blocks = (n + Cfg::BlockDim * 2 - 1) / (Cfg::BlockDim * 2);

    auto d_block_sums = static_cast<size_t*>(cu::malloc(blocks * sizeof(size_t)));

    TensorKernel::efficient_prefix_sum<<<blocks, Cfg::BlockDim, (Cfg::BlockDim * 2) * sizeof(size_t)>>>(input, output, n, d_block_sums);
    CUDA_CHECK_LAST_ERROR("TensorKernel::efficient_prefix_sum");

    if (blocks > Cfg::BlockDim) {
        auto r_v = static_cast<size_t*>(cu::malloc(blocks * sizeof(size_t)));
        run_prefix_sum(d_block_sums, r_v, blocks);
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

static void run_prefix_sum(index_t* input, index_t* output, size_t n) {
    if (n == 0) return;

    size_t blocks = (n + Cfg::BlockDim * 2 - 1) / (Cfg::BlockDim * 2);

    auto d_block_sums = static_cast<index_t*>(cu::malloc(blocks * sizeof(index_t)));

    TensorKernel::efficient_prefix_sum_index_t<<<blocks, Cfg::BlockDim, (Cfg::BlockDim * 2) * sizeof(index_t)>>>(input, output, n, d_block_sums);
    CUDA_CHECK_LAST_ERROR("TensorKernel::efficient_prefix_sum");

    if (blocks > Cfg::BlockDim) {
        auto r_v = static_cast<index_t*>(cu::malloc(blocks * sizeof(index_t)));
        run_prefix_sum(d_block_sums, r_v, blocks);
        CUDA_CHECK_LAST_ERROR("run_scan::recursive");
        cu::free(d_block_sums);
        d_block_sums = r_v;
    } else {
        TensorKernel::efficient_prefix_sum_index_t<<<1, Cfg::BlockDim, (Cfg::BlockDim * 2) * sizeof(index_t)>>>(d_block_sums, d_block_sums, blocks, nullptr);
        CUDA_CHECK_LAST_ERROR("TensorKernel::efficient_prefix_sum");
    }

    if (blocks > 1) {
        TensorKernel::add_aux<<<blocks - 1, Cfg::BlockDim>>>(output, n, d_block_sums);
        CUDA_CHECK_LAST_ERROR("TensorKernel::add_aux");
    }

    cu::free(d_block_sums);
}
//
// static void run_prefix_sum(uint32_t* input, uint32_t* output, size_t n, size_t pins) {
//     if (n == 0) return;
//
//     size_t blocks = (n + Cfg::BlockDim * 2 - 1) / (Cfg::BlockDim * 2);
//
//     auto d_block_sums = static_cast<uint32_t*>(cu::malloc(pins * blocks * sizeof(uint32_t)));
//
//     OrderBy::efficient_local_prefix_sum<<<blocks, Cfg::BlockDim, (Cfg::BlockDim * 2) * sizeof(uint32_t) * pins >>>(input, output, n, pins, d_block_sums);
//     CUDA_CHECK_LAST_ERROR("OrderBy::efficient_local_prefix_sum");
//
//     if (blocks > Cfg::BlockDim) {
//         auto r_v = static_cast<uint32_t*>(cu::malloc(pins * blocks * sizeof(uint32_t)));
//         run_prefix_sum(d_block_sums, r_v, blocks, pins);
//         CUDA_CHECK_LAST_ERROR("run_prefix_sum::recursive");
//         cu::free(d_block_sums);
//         d_block_sums = r_v;
//     } else {
//         OrderBy::efficient_local_prefix_sum<<<1, Cfg::BlockDim, (Cfg::BlockDim * 2) * sizeof(uint32_t) * pins>>>(d_block_sums, d_block_sums, blocks, pins, nullptr);
//         CUDA_CHECK_LAST_ERROR("OrderBy::efficient_local_prefix_sum internal");
//     }
//
//     if (blocks > 1) {
//         OrderBy::add_local_aux<<<blocks - 1, Cfg::BlockDim>>>(output, n, pins, d_block_sums);
//         CUDA_CHECK_LAST_ERROR("OrderBy::add_local_aux");
//     }
//
//     cu::free(d_block_sums);
// }


std::vector<size_t> GFI::iterator(tensor<char, Device::GPU> *a) {
    std::vector<size_t> result;
    const auto size = a->totalSize();

    auto out = static_cast<size_t*>(cu::malloc(sizeof(size_t) * size));

    run_prefix_sum(a->data, out, size);
    CUDA_CHECK(cudaDeviceSynchronize());

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

static std::vector<index_t> sort_int64_t(column* col_1) {
    if (col_1->type != INTEGER) throw UnsupportedOperationError("Sort of type int on non-integer");

    size_t size = col_1->data.size();
    std::vector<index_t> result(size, 0);

    dim3 grid((size + Cfg::BlockDim - 1) / (Cfg::BlockDim));

    auto indices_in  = static_cast<index_t*>(cu::malloc(sizeof(index_t) * size));
    auto indices_out = static_cast<index_t*>(cu::malloc(sizeof(index_t) * size));

    auto _col_1      = static_cast<int64_t*>(cu::malloc(sizeof(int64_t) * size));
    cu::toDevice(col_1->data.data(), _col_1, size * sizeof(int64_t));
    CUDA_CHECK_LAST_ERROR("GFI::sort move col");

    auto maskSize = Cfg::radixIntegerMaskSize;
    index_t numPins  = (1 << maskSize);
    index_t maskBits = (1 << maskSize) - 1;

    index_t shiftBits = 0;

    auto histogram     = static_cast<index_t*>(cu::malloc(sizeof(index_t) * numPins));
    auto pins_offset   = static_cast<index_t*>(cu::malloc(sizeof(index_t) * numPins));
    auto pins          = static_cast<index_t*>(cu::malloc(sizeof(index_t) * size * numPins));
    auto local_offset  = static_cast<index_t*>(cu::malloc(sizeof(index_t) * size * numPins));

    // std::vector<index_t> observer(size * numPins, 0);
    // std::vector<index_t> observer2(size * numPins, 0);

    // initialize
    for (size_t i = 0;i < size;i++) {
        result[i] = i;
    }

    // move initial data
    cu::toDevice(result.data(), indices_in, sizeof(index_t) * size);

    // now do sorting
    while (shiftBits < sizeof(int64_t) * 8) {
        // do histogram
        TensorKernel::fill_kernel<<<(numPins + Cfg::BlockDim - 1) / Cfg::BlockDim ,Cfg::BlockDim>>>(histogram, (index_t) 0, numPins);
        OrderBy::histogram_kernel_indexed<<<grid, Cfg::BlockDim, sizeof(index_t) * numPins>>>(
            static_cast<int64_t*>(_col_1),
            indices_in,
            histogram,
            pins,
            size,
            maskBits,
            shiftBits,
            numPins
        );
        CUDA_CHECK_LAST_ERROR("GFI::sort OrderBy::histogram_kernel_indexed<<<grid, Cfg::BlockDim, sizeof(index_t) * numPins>>>");

        // cu::toHost(histogram, result.data(), sizeof(index_t) * std::min(numPins, (index_t) size));
        // cu::toHost(pins, observer.data(), sizeof(index_t) * size * numPins);
        // for (int i = 0;i < size;i++) {
        //     int idx = 0;
        //     for (int j = 0 ;j < numPins;j++) {
        //         if (observer[i + size * j]) {
        //             idx = j; break;
        //         }
        //     }
        //
        //     observer[i] = idx;
        // }
        // CUDA_CHECK_LAST_ERROR("GFI::sort CPU Copy");

        // do prefix
        for (size_t i = 0;i < numPins;i++) {
            const auto offset = size * i;
            run_prefix_sum(pins + offset, local_offset + offset, size);
        }
        CUDA_CHECK_LAST_ERROR("GFI::sort run_prefix_sum local");
        run_prefix_sum(histogram, pins_offset, numPins);
        CUDA_CHECK_LAST_ERROR("GFI::sort run_prefix_sum");

        // cu::toHost(pins_offset, result.data(), sizeof(index_t) * std::min(numPins, (index_t) size));
        // cu::toHost(local_offset, observer2.data(), sizeof(index_t) * size * numPins);
        // CUDA_CHECK_LAST_ERROR("GFI::sort CPU Copy");

        // do radix
        OrderBy::radix_scatter_pass<<<grid, Cfg::BlockDim>>>(
            static_cast<int64_t*>(_col_1),
            indices_in,
            indices_out,
            pins_offset,
            local_offset,
            size,
            maskBits,
            shiftBits
        );
        CUDA_CHECK_LAST_ERROR("GFI::sort OrderBy::radix_scatter_pass<<<grid, Cfg::BlockDim>>>");
        // cu::toHost(indices_out, result.data(), sizeof(index_t) * size);
        // CUDA_CHECK_LAST_ERROR("GFI::sort CPU Copy");

        CUDA_CHECK(cudaDeviceSynchronize());

        auto temp = indices_in;
        indices_in = indices_out;
        indices_out = temp;

        shiftBits += maskSize;
    }

    cu::toHost(indices_in, result.data(), sizeof(index_t) * size);

    cu::free(indices_in);
    cu::free(indices_out);

    cu::free(histogram);
    cu::free(local_offset);
    cu::free(pins);
    cu::free(pins_offset);
    cu::free(_col_1);

    return result;
}

static std::vector<index_t> sort_int64_t_streaming(column* col_1) {
    if (col_1->type != INTEGER)
        throw UnsupportedOperationError("Sort of type int on non-integer");

    const size_t size = col_1->data.size();
    std::vector<index_t> result(size);

    // initialize result on host
    for (size_t i = 0; i < size; ++i)
        result[i] = i;

    // allocate device buffers
    index_t *d_col, *indices_in, *indices_out;
    CUDA_LAUNCH_AND_CHECK(cudaMalloc(&d_col, sizeof(int64_t) * size));
    CUDA_LAUNCH_AND_CHECK(cudaMalloc(&indices_in,  sizeof(index_t) * size));
    CUDA_LAUNCH_AND_CHECK(cudaMalloc(&indices_out, sizeof(index_t) * size));

    // copy full column and initial indices to device
    CUDA_LAUNCH_AND_CHECK(cudaMemcpy(d_col, col_1->data.data(),
                          sizeof(int64_t) * size,
                          cudaMemcpyHostToDevice));
    CUDA_LAUNCH_AND_CHECK(cudaMemcpy(indices_in, result.data(),
                          sizeof(index_t) * size,
                          cudaMemcpyHostToDevice));

    // radix parameters
    const int maskBits    = (1 << Cfg::radixIntegerMaskSize) - 1;
    const int numBins     = 1 << Cfg::radixIntegerMaskSize;
    int shiftBits         = 0;

    // Allocate per-pass histogram and offsets (small)
    index_t *d_histogram, *d_binOffsets;
    CUDA_LAUNCH_AND_CHECK(cudaMalloc(&d_histogram,  sizeof(index_t) * numBins));
    CUDA_LAUNCH_AND_CHECK(cudaMalloc(&d_binOffsets, sizeof(index_t) * numBins));

    // Choose a chunk size for streaming (in elements)
    const size_t CHUNK_SIZE = 1 << 20;

    // Create a pool of streams for overlapping histogram compute and memcopies
    const int NUM_STREAMS = Cfg::numStreams;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i)
        CUDA_LAUNCH_AND_CHECK(cudaStreamCreate(&streams[i]));

    // Temporary per-chunk buffers for pins and local offsets
    // Note: pins are size * numBins in total across all chunks
    index_t *d_pins, *d_local;
    CUDA_LAUNCH_AND_CHECK(cudaMalloc(&d_pins,  sizeof(index_t) * CHUNK_SIZE * numBins));
    CUDA_LAUNCH_AND_CHECK(cudaMalloc(&d_local, sizeof(index_t) * CHUNK_SIZE * numBins));

    // Launch parameters (constant)
    dim3 grid_full((CHUNK_SIZE + Cfg::BlockDim - 1) / Cfg::BlockDim);
    dim3 grid_hist((numBins + Cfg::BlockDim - 1) / Cfg::BlockDim);

    // Main radix loop
    while (shiftBits < 64) {
        // zero out global histogram
        CUDA_LAUNCH_AND_CHECK(cudaMemset(d_histogram, 0, sizeof(index_t) * numBins));

        // 1) compute chunked histograms into pins array
        size_t offset = 0, stream_idx = 0;
        while (offset < size) {
            size_t this_chunk = std::min(CHUNK_SIZE, size - offset);
            auto s = streams[stream_idx];

            OrderBy::histogram_kernel_indexed
                <<<grid_hist, Cfg::BlockDim, 0, s>>>(
                    (int64_t*)d_col + offset,
                    indices_in         + offset,
                    d_histogram,            // accumulate into global
                    d_pins    + offset*numBins,
                    this_chunk,
                    maskBits,
                    shiftBits,
                    numBins
                );
            CUDA_CHECK_LAST_ERROR("histogram_kernel_indexed chunk");

            offset     += this_chunk;
            stream_idx  = (stream_idx + 1) % NUM_STREAMS;
        }

        // wait all chunk histograms
        CUDA_LAUNCH_AND_CHECK(cudaDeviceSynchronize());

        // 2) prefix-sum on the global histogram
        run_prefix_sum(d_histogram, d_binOffsets, numBins);

        // 3) for each chunk, do local prefix and scatter
        offset     = 0;
        stream_idx = 0;
        while (offset < size) {
            size_t this_chunk = std::min(CHUNK_SIZE, size - offset);
            auto s = streams[stream_idx];

            // local prefix per-bin
            for (int b = 0; b < numBins; ++b) {
                auto bin_pins   = d_pins   + offset*numBins + b*this_chunk;
                auto bin_local  = d_local  + offset*numBins + b*this_chunk;
                run_prefix_sum(bin_pins, bin_local, this_chunk);
            }

            // scatter chunk
            OrderBy::radix_scatter_pass
                <<<grid_full, Cfg::BlockDim, 0, s>>>(
                    (int64_t*)d_col + offset,
                    indices_in         + offset,
                    indices_out        + offset,
                    d_binOffsets,    // global start
                    d_local   + offset*numBins,
                    this_chunk,
                    maskBits,
                    shiftBits
                );
            CUDA_CHECK_LAST_ERROR("radix_scatter_pass chunk");

            offset     += this_chunk;
            stream_idx  = (stream_idx + 1) % NUM_STREAMS;
        }

        // sync and swap buffers
        CUDA_LAUNCH_AND_CHECK(cudaDeviceSynchronize());
        std::swap(indices_in, indices_out);

        shiftBits += Cfg::radixIntegerMaskSize;
    }

    // copy back sorted indices
    CUDA_LAUNCH_AND_CHECK(cudaMemcpy(result.data(), indices_in,
                          sizeof(index_t) * size,
                          cudaMemcpyDeviceToHost));

    // cleanup
    CUDA_LAUNCH_AND_CHECK(cudaFree(d_col));
    CUDA_LAUNCH_AND_CHECK(cudaFree(indices_in));
    CUDA_LAUNCH_AND_CHECK(cudaFree(indices_out));
    CUDA_LAUNCH_AND_CHECK(cudaFree(d_histogram));
    CUDA_LAUNCH_AND_CHECK(cudaFree(d_binOffsets));
    CUDA_LAUNCH_AND_CHECK(cudaFree(d_pins));
    CUDA_LAUNCH_AND_CHECK(cudaFree(d_local));
    for (auto &s : streams) cudaStreamDestroy(s);

    return result;
}

static std::vector<index_t> sort_double_t(column* col_1) {
    if (col_1->type != FLOAT) throw UnsupportedOperationError("Sort of type float on non-float");

    size_t size = col_1->data.size();
    std::vector<index_t> result(size, 0);

    dim3 grid((size + Cfg::BlockDim - 1) / (Cfg::BlockDim));

    auto indices_in  = static_cast<index_t*>(cu::malloc(sizeof(index_t) * size));
    auto indices_out = static_cast<index_t*>(cu::malloc(sizeof(index_t) * size));

    auto _col_1      = static_cast<int64_t*>(cu::malloc(sizeof(int64_t) * size));
    std::vector<uint64_t> col_data(col_1->data.size(), 0);
    for (int i = 0;i < col_1->data.size();i++) {
        uint64_t bits = *reinterpret_cast<const uint64_t *>(&col_1->data[i].d);
        col_data[i] = (bits & 0x8000000000000000ULL) ? ~bits : (bits ^ 0x8000000000000000ULL);
    }
    cu::toDevice(col_data.data(), _col_1, size * sizeof(uint64_t));
    CUDA_CHECK_LAST_ERROR("GFI::sort move col");

    auto maskSize = Cfg::radixIntegerMaskSize;
    index_t numPins  = (1 << maskSize);
    index_t maskBits = (1 << maskSize) - 1;

    index_t shiftBits = 0;

    auto histogram     = static_cast<index_t*>(cu::malloc(sizeof(index_t) * numPins));
    auto pins_offset   = static_cast<index_t*>(cu::malloc(sizeof(index_t) * numPins));
    auto pins          = static_cast<index_t*>(cu::malloc(sizeof(index_t) * size * numPins));
    auto local_offset  = static_cast<index_t*>(cu::malloc(sizeof(index_t) * size * numPins));

    // std::vector<index_t> observer(size * numPins, 0);
    // std::vector<index_t> observer2(size * numPins, 0);

    // initialize
    for (size_t i = 0;i < size;i++) {
        result[i] = i;
    }

    // move initial data
    cu::toDevice(result.data(), indices_in, sizeof(index_t) * size);

    // now do sorting
    while (shiftBits < sizeof(int64_t) * 8) {
        // do histogram
        TensorKernel::fill_kernel<<<(numPins + Cfg::BlockDim - 1) / Cfg::BlockDim ,Cfg::BlockDim>>>(histogram, (index_t) 0, numPins);
        OrderBy::histogram_kernel_indexed<<<grid, Cfg::BlockDim, sizeof(index_t) * numPins>>>(
            static_cast<int64_t*>(_col_1),
            indices_in,
            histogram,
            pins,
            size,
            maskBits,
            shiftBits,
            numPins
        );
        CUDA_CHECK_LAST_ERROR("GFI::sort OrderBy::histogram_kernel_indexed<<<grid, Cfg::BlockDim, sizeof(index_t) * numPins>>>");

        // cu::toHost(histogram, result.data(), sizeof(index_t) * std::min(numPins, (index_t) size));
        // cu::toHost(pins, observer.data(), sizeof(index_t) * size * numPins);
        // for (int i = 0;i < size;i++) {
        //     int idx = 0;
        //     for (int j = 0 ;j < numPins;j++) {
        //         if (observer[i + size * j]) {
        //             idx = j; break;
        //         }
        //     }
        //
        //     observer[i] = idx;
        // }
        // CUDA_CHECK_LAST_ERROR("GFI::sort CPU Copy");

        // do prefix
        for (size_t i = 0;i < numPins;i++) {
            const auto offset = size * i;
            run_prefix_sum(pins + offset, local_offset + offset, size);
        }
        CUDA_CHECK_LAST_ERROR("GFI::sort run_prefix_sum local");
        run_prefix_sum(histogram, pins_offset, numPins);
        CUDA_CHECK_LAST_ERROR("GFI::sort run_prefix_sum");

        // cu::toHost(pins_offset, result.data(), sizeof(index_t) * std::min(numPins, (index_t) size));
        // cu::toHost(local_offset, observer2.data(), sizeof(index_t) * size * numPins);
        // CUDA_CHECK_LAST_ERROR("GFI::sort CPU Copy");

        // do radix
        OrderBy::radix_scatter_pass<<<grid, Cfg::BlockDim>>>(
            static_cast<int64_t*>(_col_1),
            indices_in,
            indices_out,
            pins_offset,
            local_offset,
            size,
            maskBits,
            shiftBits
        );
        CUDA_CHECK_LAST_ERROR("GFI::sort OrderBy::radix_scatter_pass<<<grid, Cfg::BlockDim>>>");
        // cu::toHost(indices_out, result.data(), sizeof(index_t) * size);
        // CUDA_CHECK_LAST_ERROR("GFI::sort CPU Copy");

        CUDA_CHECK(cudaDeviceSynchronize());

        auto temp = indices_in;
        indices_in = indices_out;
        indices_out = temp;

        shiftBits += maskSize;
    }

    cu::toHost(indices_in, result.data(), sizeof(index_t) * size);

    cu::free(indices_in);
    cu::free(indices_out);

    cu::free(histogram);
    cu::free(local_offset);
    cu::free(pins);
    cu::free(pins_offset);
    cu::free(_col_1);

    return result;
}


static std::vector<index_t> sort_double_t_streaming(column* col_1) {
    if (col_1->type != FLOAT)
        throw UnsupportedOperationError("Sort of type float on non-float");

    const size_t size = col_1->data.size();
    std::vector<index_t> result(size);
    for (size_t i = 0; i < size; ++i)
        result[i] = i;

    // --- Device buffers ---
    uint64_t *d_bits = nullptr;
    index_t  *d_in  = nullptr, *d_out = nullptr;
    CUDA_LAUNCH_AND_CHECK(cudaMalloc(&d_bits, sizeof(uint64_t) * size));
    CUDA_CHECK(cudaMalloc(&d_in,   sizeof(index_t)  * size));
    CUDA_CHECK(cudaMalloc(&d_out,  sizeof(index_t)  * size));

    // Prepare host bit‐mapped array:
    std::vector<uint64_t> host_bits(size);
    for (size_t i = 0; i < size; ++i) {
        uint64_t bits = *reinterpret_cast<const uint64_t*>(&col_1->data[i].d);
        // flip sign‐bit to get correct lexicographic ordering
        host_bits[i] = (bits & 0x8000000000000000ULL)
                       ? ~bits
                       : (bits ^ 0x8000000000000000ULL);
    }
    CUDA_CHECK(cudaMemcpy(d_bits, host_bits.data(),
                          sizeof(uint64_t) * size,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in,  result.data(),
                          sizeof(index_t) * size,
                          cudaMemcpyHostToDevice));

    // --- Radix params ---
    const int   maskBitsTotal = Cfg::radixIntegerMaskSize;
    const int   numBins       = 1 << maskBitsTotal;
    const int   maskBits      = numBins - 1;
    int         shiftBits     = 0;

    // Allocate small per-pass arrays
    index_t *d_hist    = nullptr, *d_binOffsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hist,       sizeof(index_t) * numBins));
    CUDA_CHECK(cudaMalloc(&d_binOffsets, sizeof(index_t) * numBins));

    // Chunking parameters
    const size_t CHUNK_SIZE = 1 << 20;            // elements per chunk
    const int    NUM_STREAMS = Cfg::numStreams;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i)
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

    // Temp per-chunk pin arrays
    index_t *d_pins   = nullptr, *d_local = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pins,   sizeof(index_t) * CHUNK_SIZE * numBins));
    CUDA_CHECK(cudaMalloc(&d_local,  sizeof(index_t) * CHUNK_SIZE * numBins));

    // Grid dims
    dim3 grid_hist((numBins + Cfg::BlockDim - 1) / Cfg::BlockDim);
    dim3 grid_chunk((CHUNK_SIZE + Cfg::BlockDim - 1) / Cfg::BlockDim);

    // --- Main radix loop ---
    while (shiftBits < 64) {
        // zero‐histogram
        CUDA_CHECK(cudaMemset(d_hist, 0, sizeof(index_t) * numBins));

        // 1) Chunked histogram accumulation
        size_t offset = 0, sidx = 0;
        while (offset < size) {
            size_t chunk = std::min(CHUNK_SIZE, size - offset);
            auto   s     = streams[sidx];
            OrderBy::histogram_kernel_indexed
                <<<grid_hist, Cfg::BlockDim, 0, s>>>(
                    (const int64_t*)(d_bits + offset),
                    d_in  + offset,
                    d_hist,
                    d_pins  + offset * numBins,
                    chunk,
                    maskBits,
                    shiftBits,
                    numBins
                );
            CUDA_CHECK_LAST_ERROR("float histogram chunk");

            offset += chunk;
            sidx    = (sidx + 1) % NUM_STREAMS;
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // 2) Global prefix‐sum on d_hist → d_binOffsets
        run_prefix_sum(d_hist, d_binOffsets, numBins);

        // 3) Per‐chunk local prefix + scatter
        offset = 0; sidx = 0;
        while (offset < size) {
            size_t chunk = std::min(CHUNK_SIZE, size - offset);
            auto   s     = streams[sidx];

            // local per‐bin prefix
            for (int b = 0; b < numBins; ++b) {
                auto pins_ptr  = d_pins   + offset * numBins + b * chunk;
                auto local_ptr = d_local  + offset * numBins + b * chunk;
                run_prefix_sum(pins_ptr, local_ptr, chunk);
            }

            // scatter pass
            OrderBy::radix_scatter_pass
                <<<grid_chunk, Cfg::BlockDim, 0, s>>>(
                    (const int64_t*)(d_bits + offset),
                    d_in   + offset,
                    d_out  + offset,
                    d_binOffsets,
                    d_local + offset * numBins,
                    chunk,
                    maskBits,
                    shiftBits
                );
            CUDA_CHECK_LAST_ERROR("float scatter chunk");

            offset += chunk;
            sidx    = (sidx + 1) % NUM_STREAMS;
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // swap buffers
        std::swap(d_in, d_out);
        shiftBits += maskBitsTotal;
    }

    // copy back
    CUDA_CHECK(cudaMemcpy(result.data(), d_in,
                          sizeof(index_t) * size,
                          cudaMemcpyDeviceToHost));

    // cleanup
    cudaFree(d_bits);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_hist);
    cudaFree(d_binOffsets);
    cudaFree(d_pins);
    cudaFree(d_local);
    for (int i = 0; i < NUM_STREAMS; ++i)
        cudaStreamDestroy(streams[i]);

    return result;
}


static std::vector<index_t> sort_dt_t(column* col_1) {
    if (col_1->type != DateTime) throw UnsupportedOperationError("Sort of type DateTime on non-DateTime");

    size_t size = col_1->data.size();
    std::vector<index_t> result(size, 0);

    dim3 grid((size + Cfg::BlockDim - 1) / (Cfg::BlockDim));

    auto indices_in  = static_cast<index_t*>(cu::malloc(sizeof(index_t) * size));
    auto indices_out = static_cast<index_t*>(cu::malloc(sizeof(index_t) * size));

    auto _col_1      = static_cast<int64_t*>(cu::malloc(sizeof(int64_t) * size));
    std::vector<uint64_t> col_data(col_1->data.size(), 0);
    for (int i = 0;i < col_1->data.size();i++) {
        const auto dt = *(col_1->data[i].t);
        const uint64_t bits = ValuesHelper::dateTimeToInt(dt);
        col_data[i] = bits;
    }
    cu::toDevice(col_data.data(), _col_1, size * sizeof(uint64_t));
    CUDA_CHECK_LAST_ERROR("GFI::sort move col");

    auto maskSize = Cfg::radixIntegerMaskSize;
    index_t numPins  = (1 << maskSize);
    index_t maskBits = (1 << maskSize) - 1;

    index_t shiftBits = 0;

    auto histogram     = static_cast<index_t*>(cu::malloc(sizeof(index_t) * numPins));
    auto pins_offset   = static_cast<index_t*>(cu::malloc(sizeof(index_t) * numPins));
    auto pins          = static_cast<index_t*>(cu::malloc(sizeof(index_t) * size * numPins));
    auto local_offset  = static_cast<index_t*>(cu::malloc(sizeof(index_t) * size * numPins));

    // std::vector<index_t> observer(size * numPins, 0);
    // std::vector<index_t> observer2(size * numPins, 0);

    // initialize
    for (size_t i = 0;i < size;i++) {
        result[i] = i;
    }

    // move initial data
    cu::toDevice(result.data(), indices_in, sizeof(index_t) * size);

    // now do sorting
    while (shiftBits < sizeof(int64_t) * 8) {
        // do histogram
        TensorKernel::fill_kernel<<<(numPins + Cfg::BlockDim - 1) / Cfg::BlockDim ,Cfg::BlockDim>>>(histogram, (index_t) 0, numPins);
        OrderBy::histogram_kernel_indexed<<<grid, Cfg::BlockDim, sizeof(index_t) * numPins>>>(
            static_cast<int64_t*>(_col_1),
            indices_in,
            histogram,
            pins,
            size,
            maskBits,
            shiftBits,
            numPins
        );
        CUDA_CHECK_LAST_ERROR("GFI::sort OrderBy::histogram_kernel_indexed<<<grid, Cfg::BlockDim, sizeof(index_t) * numPins>>>");

        // cu::toHost(histogram, result.data(), sizeof(index_t) * std::min(numPins, (index_t) size));
        // cu::toHost(pins, observer.data(), sizeof(index_t) * size * numPins);
        // for (int i = 0;i < size;i++) {
        //     int idx = 0;
        //     for (int j = 0 ;j < numPins;j++) {
        //         if (observer[i + size * j]) {
        //             idx = j; break;
        //         }
        //     }
        //
        //     observer[i] = idx;
        // }
        // CUDA_CHECK_LAST_ERROR("GFI::sort CPU Copy");

        // do prefix
        for (size_t i = 0;i < numPins;i++) {
            const auto offset = size * i;
            run_prefix_sum(pins + offset, local_offset + offset, size);
        }
        CUDA_CHECK_LAST_ERROR("GFI::sort run_prefix_sum local");
        run_prefix_sum(histogram, pins_offset, numPins);
        CUDA_CHECK_LAST_ERROR("GFI::sort run_prefix_sum");

        // cu::toHost(pins_offset, result.data(), sizeof(index_t) * std::min(numPins, (index_t) size));
        // cu::toHost(local_offset, observer2.data(), sizeof(index_t) * size * numPins);
        // CUDA_CHECK_LAST_ERROR("GFI::sort CPU Copy");

        // do radix
        OrderBy::radix_scatter_pass<<<grid, Cfg::BlockDim>>>(
            static_cast<int64_t*>(_col_1),
            indices_in,
            indices_out,
            pins_offset,
            local_offset,
            size,
            maskBits,
            shiftBits
        );
        CUDA_CHECK_LAST_ERROR("GFI::sort OrderBy::radix_scatter_pass<<<grid, Cfg::BlockDim>>>");
        // cu::toHost(indices_out, result.data(), sizeof(index_t) * size);
        // CUDA_CHECK_LAST_ERROR("GFI::sort CPU Copy");

        CUDA_CHECK(cudaDeviceSynchronize());

        auto temp = indices_in;
        indices_in = indices_out;
        indices_out = temp;

        shiftBits += maskSize;
    }

    cu::toHost(indices_in, result.data(), sizeof(index_t) * size);

    cu::free(indices_in);
    cu::free(indices_out);

    cu::free(histogram);
    cu::free(local_offset);
    cu::free(pins);
    cu::free(pins_offset);
    cu::free(_col_1);

    return result;
}

static std::vector<index_t> sort_dt_t_streaming(column* col_1) {
    if (col_1->type != DateTime)
        throw UnsupportedOperationError("Sort of type DateTime on non-DateTime");

    const size_t size = col_1->data.size();
    std::vector<index_t> result(size);
    for (size_t i = 0; i < size; ++i)
        result[i] = i;

    // --- Device buffers ---
    uint64_t *d_bits = nullptr;
    index_t  *d_in   = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bits, sizeof(uint64_t) * size));
    CUDA_CHECK(cudaMalloc(&d_in,   sizeof(index_t)  * size));
    CUDA_CHECK(cudaMalloc(&d_out,  sizeof(index_t)  * size));

    // Host→device: pack DateTime into uint64_t via your helper
    std::vector<uint64_t> host_bits(size);
    for (size_t i = 0; i < size; ++i) {
        const auto dt = *(col_1->data[i].t);
        host_bits[i]  = ValuesHelper::dateTimeToInt(dt);
    }
    CUDA_CHECK(cudaMemcpy(d_bits, host_bits.data(),
                          sizeof(uint64_t) * size,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in, result.data(),
                          sizeof(index_t) * size,
                          cudaMemcpyHostToDevice));

    // --- Radix parameters ---
    const int   maskSizeTotal = Cfg::radixIntegerMaskSize;
    const int   numBins       = 1 << maskSizeTotal;
    const int   maskBits      = numBins - 1;
    int         shiftBits     = 0;

    // small per-pass arrays
    index_t *d_hist = nullptr, *d_binOffsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hist,       sizeof(index_t) * numBins));
    CUDA_CHECK(cudaMalloc(&d_binOffsets, sizeof(index_t) * numBins));

    // chunking
    const size_t CHUNK_SIZE = 1 << 20;
    const int    NUM_STREAMS = Cfg::numStreams;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i)
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

    // per-chunk pin/local arrays
    index_t *d_pins  = nullptr, *d_local = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pins,  sizeof(index_t) * CHUNK_SIZE * numBins));
    CUDA_CHECK(cudaMalloc(&d_local, sizeof(index_t) * CHUNK_SIZE * numBins));

    // grid dims
    dim3 grid_hist((numBins + Cfg::BlockDim - 1) / Cfg::BlockDim);
    dim3 grid_chunk((CHUNK_SIZE + Cfg::BlockDim - 1) / Cfg::BlockDim);

    // --- Main radix loop ---
    while (shiftBits < 64) {
        // zero histogram
        CUDA_CHECK(cudaMemset(d_hist, 0, sizeof(index_t) * numBins));

        // 1) chunked histogram into d_hist
        size_t offset = 0, sidx = 0;
        while (offset < size) {
            size_t chunk = std::min(CHUNK_SIZE, size - offset);
            auto   s     = streams[sidx];
            OrderBy::histogram_kernel_indexed
                <<<grid_hist, Cfg::BlockDim, 0, s>>>(
                    (const int64_t*)(d_bits + offset),
                    d_in  + offset,
                    d_hist,
                    d_pins  + offset * numBins,
                    chunk,
                    maskBits,
                    shiftBits,
                    numBins
                );
            CUDA_CHECK_LAST_ERROR("dt histogram chunk");

            offset += chunk;
            sidx    = (sidx + 1) % NUM_STREAMS;
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // 2) prefix-sum on d_hist → d_binOffsets
        run_prefix_sum(d_hist, d_binOffsets, numBins);

        // 3) per-chunk local prefix + scatter
        offset = 0; sidx = 0;
        while (offset < size) {
            size_t chunk = std::min(CHUNK_SIZE, size - offset);
            auto   s     = streams[sidx];

            // local prefix per bin
            for (int b = 0; b < numBins; ++b) {
                auto pins_ptr  = d_pins   + offset * numBins + b * chunk;
                auto local_ptr = d_local  + offset * numBins + b * chunk;
                run_prefix_sum(pins_ptr, local_ptr, chunk);
            }

            // scatter
            OrderBy::radix_scatter_pass
                <<<grid_chunk, Cfg::BlockDim, 0, s>>>(
                    (const int64_t*)(d_bits + offset),
                    d_in   + offset,
                    d_out  + offset,
                    d_binOffsets,
                    d_local + offset * numBins,
                    chunk,
                    maskBits,
                    shiftBits
                );
            CUDA_CHECK_LAST_ERROR("dt scatter chunk");

            offset += chunk;
            sidx    = (sidx + 1) % NUM_STREAMS;
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // swap and next digit
        std::swap(d_in, d_out);
        shiftBits += maskSizeTotal;
    }

    // copy back
    CUDA_CHECK(cudaMemcpy(result.data(), d_in,
                          sizeof(index_t) * size,
                          cudaMemcpyDeviceToHost));

    // cleanup
    cudaFree(d_bits);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_hist);
    cudaFree(d_binOffsets);
    cudaFree(d_pins);
    cudaFree(d_local);
    for (int i = 0; i < NUM_STREAMS; ++i)
        cudaStreamDestroy(streams[i]);

    return result;
}

static std::vector<index_t> sort_string_t(column* col) {
    if (col->type != STRING) throw UnsupportedOperationError("Sort of type String on non-String");

    size_t size = col->data.size();
    std::vector<index_t> result(size, 0);

    dim3 grid((size + Cfg::BlockDim - 1) / (Cfg::BlockDim));

    auto indices_in  = static_cast<index_t*>(cu::malloc(sizeof(index_t) * size));
    auto indices_out = static_cast<index_t*>(cu::malloc(sizeof(index_t) * size));

    std::vector<void*>  ptrs(col->data.size(), nullptr);
    std::vector<size_t> sizes(col->data.size(), 0);

    index_t max_string_size = 0;
    for (size_t i = 0; i < size; ++i) {
        std::string* str = col->data[i].s;
        auto gpu_str = cu::malloc(str->size() + 1);
        cu::toDevice(str->c_str(), gpu_str, str->size() + 1);
        ptrs[i]  = gpu_str;
        sizes[i] = str->size();
        if (str->size() > max_string_size) {
            max_string_size = str->size();
        }
    }
    auto _col_data  = static_cast<const char**>(cu::malloc(sizeof(char**) * size));
    auto _col_sizes = static_cast<size_t*>(cu::malloc(sizeof(size_t) * size));
    cu::toDevice(ptrs.data(), _col_data, sizeof(char**) * size);
    cu::toDevice(sizes.data(), _col_sizes, sizeof(size_t) * size);
    CUDA_CHECK_LAST_ERROR("GFI::sort move col");

    auto maskSize = 8;
    index_t numPins  = (1 << maskSize);
    index_t maskBits = (1 << maskSize) - 1;

    index_t curr_char = 0;

    auto histogram     = static_cast<index_t*>(cu::malloc(sizeof(index_t) * numPins));
    auto pins_offset   = static_cast<index_t*>(cu::malloc(sizeof(index_t) * numPins));
    auto pins          = static_cast<index_t*>(cu::malloc(sizeof(index_t) * size * numPins));
    auto local_offset  = static_cast<index_t*>(cu::malloc(sizeof(index_t) * size * numPins));

    // std::vector<index_t> observer(size * numPins, 0);
    // std::vector<index_t> observer2(size * numPins, 0);

    // initialize
    for (size_t i = 0;i < size;i++) {
        result[i] = i;
    }

    // move initial data
    cu::toDevice(result.data(), indices_in, sizeof(index_t) * size);

    // now do sorting
    while (curr_char < max_string_size) {
        // do histogram
        TensorKernel::fill_kernel<<<(numPins + Cfg::BlockDim - 1) / Cfg::BlockDim ,Cfg::BlockDim>>>(histogram, (index_t) 0, numPins);
        OrderBy::histogram_kernel_indexed<<<grid, Cfg::BlockDim>>>(
            _col_data,
            _col_sizes,
            indices_in,
            histogram,
            pins,
            size,
            curr_char,
            max_string_size
        );
        CUDA_CHECK_LAST_ERROR("GFI::sort OrderBy::histogram_kernel_indexed<<<grid, Cfg::BlockDim>>>");

        // cu::toHost(histogram, result.data(), sizeof(index_t) * std::min(numPins, (index_t) size));
        // cu::toHost(pins, observer.data(), sizeof(index_t) * size * numPins);
        // for (int i = 0;i < size;i++) {
        //     int idx = 0;
        //     for (int j = 0 ;j < numPins;j++) {
        //         if (observer[i + size * j]) {
        //             idx = j; break;
        //         }
        //     }
        //
        //     observer[i] = idx;
        // }
        // CUDA_CHECK_LAST_ERROR("GFI::sort CPU Copy");

        // do prefix
        for (size_t i = 0;i < numPins;i++) {
            const auto offset = size * i;
            run_prefix_sum(pins + offset, local_offset + offset, size);
        }
        CUDA_CHECK_LAST_ERROR("GFI::sort run_prefix_sum local");
        run_prefix_sum(histogram, pins_offset, numPins);
        CUDA_CHECK_LAST_ERROR("GFI::sort run_prefix_sum");

        // cu::toHost(pins_offset, result.data(), sizeof(index_t) * std::min(numPins, (index_t) size));
        // cu::toHost(local_offset, observer2.data(), sizeof(index_t) * size * numPins);
        // CUDA_CHECK_LAST_ERROR("GFI::sort CPU Copy");

        // do radix
        OrderBy::radix_scatter_pass<<<grid, Cfg::BlockDim>>>(
            _col_data,
             _col_sizes,
            indices_in,
            indices_out,
            pins_offset,
            local_offset,
            size,
            curr_char,
            max_string_size
        );
        CUDA_CHECK_LAST_ERROR("GFI::sort OrderBy::radix_scatter_pass<<<grid, Cfg::BlockDim>>>");
        // cu::toHost(indices_out, result.data(), sizeof(index_t) * size);
        // CUDA_CHECK_LAST_ERROR("GFI::sort CPU Copy");
        CUDA_CHECK(cudaDeviceSynchronize());

        auto temp = indices_in;
        indices_in = indices_out;
        indices_out = temp;

        curr_char += 1;
    }

    cu::toHost(indices_in, result.data(), sizeof(index_t) * size);

    cu::free(indices_in);
    cu::free(indices_out);

    cu::free(histogram);
    cu::free(local_offset);
    cu::free(pins);
    cu::free(pins_offset);

    for (const auto item: ptrs) {
        cu::free(item);
    }

    cu::free(_col_data);
    cu::free(_col_sizes);

    return result;
}


static std::vector<index_t> sort_string_t_streaming(column* col) {
    if (col->type != STRING)
        throw UnsupportedOperationError("Sort of type String on non-String");

    const size_t size = col->data.size();
    std::vector<index_t> result(size);
    for (size_t i = 0; i < size; ++i)
        result[i] = i;

    // --- Transfer strings to GPU buffers ---
    std::vector<const char*> host_ptrs(size);
    std::vector<size_t>      host_sizes(size);
    size_t max_len = 0;
    for (size_t i = 0; i < size; ++i) {
        auto s = col->data[i].s;
        auto gpu_str = cu::malloc(s->size() + 1);
        cu::toDevice(s->c_str(), gpu_str, s->size() + 1);
        host_ptrs[i]  = (const char*)gpu_str;
        host_sizes[i] = s->size();
        max_len       = std::max(max_len, s->size());
    }

    const size_t PTR_BUF  = sizeof(const char*) * size;
    const size_t SZ_BUF   = sizeof(size_t)     * size;
    const size_t IDX_BUF  = sizeof(index_t)    * size;

    const char** d_ptrs   = nullptr;
    size_t*      d_sizes  = nullptr;
    index_t*     d_in     = nullptr;
    index_t*     d_out    = nullptr;

    CUDA_CHECK(cudaMalloc(&d_ptrs,  PTR_BUF));
    CUDA_CHECK(cudaMalloc(&d_sizes, SZ_BUF));
    CUDA_CHECK(cudaMalloc(&d_in,    IDX_BUF));
    CUDA_CHECK(cudaMalloc(&d_out,   IDX_BUF));

    CUDA_CHECK(cudaMemcpy(d_ptrs,  host_ptrs.data(), PTR_BUF, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sizes, host_sizes.data(), SZ_BUF, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in,    result.data(),     IDX_BUF, cudaMemcpyHostToDevice));

    // --- Radix parameters ---
    const int   maskSize   = 8;
    const int   numBins    = 1 << maskSize;
    const size_t CHUNK_SIZE = 1 << 20;
    const int   NUM_STREAMS = Cfg::numStreams;

    // small per-pass arrays
    index_t *d_hist       = nullptr, *d_binOffsets = nullptr;
    index_t *d_pins       = nullptr, *d_local      = nullptr;

    CUDA_CHECK(cudaMalloc(&d_hist,      sizeof(index_t) * numBins));
    CUDA_CHECK(cudaMalloc(&d_binOffsets, sizeof(index_t) * numBins));
    CUDA_CHECK(cudaMalloc(&d_pins,      sizeof(index_t) * CHUNK_SIZE * numBins));
    CUDA_CHECK(cudaMalloc(&d_local,     sizeof(index_t) * CHUNK_SIZE * numBins));

    // create streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i)
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

    dim3 grid_bins( (numBins + Cfg::BlockDim - 1) / Cfg::BlockDim );
    dim3 grid_chunk( (CHUNK_SIZE + Cfg::BlockDim - 1) / Cfg::BlockDim );

    size_t curr_char = 0;
    while (curr_char < max_len) {
        // zero global histogram
        CUDA_CHECK(cudaMemset(d_hist, 0, sizeof(index_t) * numBins));

        // 1) chunked histogram accumulation
        size_t offset = 0, sidx = 0;
        while (offset < size) {
            size_t chunk = std::min(CHUNK_SIZE, size - offset);
            auto   s     = streams[sidx];

            OrderBy::histogram_kernel_indexed
              <<<grid_bins, Cfg::BlockDim, 0, s>>>(
                d_ptrs,
                d_sizes,
                d_in   + offset,
                d_hist,
                d_pins + offset * numBins,
                chunk,
                curr_char,
                max_len
            );
            CUDA_CHECK_LAST_ERROR("string histogram chunk");

            offset += chunk;
            sidx    = (sidx + 1) % NUM_STREAMS;
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // 2) prefix-sum on global histogram → binOffsets
        run_prefix_sum(d_hist, d_binOffsets, numBins);

        // 3) per-chunk local prefix & scatter
        offset = 0; sidx = 0;
        while (offset < size) {
            size_t chunk = std::min(CHUNK_SIZE, size - offset);
            auto   s     = streams[sidx];

            // local prefix for each bin within this chunk
            for (int b = 0; b < numBins; ++b) {
                auto pins_ptr  = d_pins   + offset * numBins + b * chunk;
                auto local_ptr = d_local  + offset * numBins + b * chunk;
                run_prefix_sum(pins_ptr, local_ptr, chunk);
            }

            OrderBy::radix_scatter_pass
              <<<grid_chunk, Cfg::BlockDim, 0, s>>>(
                d_ptrs,
                d_sizes,
                d_in    + offset,
                d_out   + offset,
                d_binOffsets,
                d_local + offset * numBins,
                chunk,
                curr_char,
                max_len
            );
            CUDA_CHECK_LAST_ERROR("string scatter chunk");

            offset += chunk;
            sidx    = (sidx + 1) % NUM_STREAMS;
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // swap buffers & next character
        std::swap(d_in, d_out);
        curr_char += 1;
    }

    // copy back final ordering
    CUDA_CHECK(cudaMemcpy(result.data(), d_in,
                          sizeof(index_t) * size,
                          cudaMemcpyDeviceToHost));

    // cleanup
    cudaFree(d_ptrs);
    cudaFree(d_sizes);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_hist);
    cudaFree(d_binOffsets);
    cudaFree(d_pins);
    cudaFree(d_local);
    for (int i = 0; i < NUM_STREAMS; ++i)
        cudaStreamDestroy(streams[i]);
    for (auto p : host_ptrs)
        cu::free((void*)p);

    return result;
}



std::vector<index_t> GFI::sort(column *col_1) {
    switch (col_1->type) {
        case INTEGER:
            if (Cfg::numStreams > 0)
                return sort_int64_t_streaming(col_1);
            return sort_int64_t(col_1);
        case FLOAT:
            if (Cfg::numStreams > 0)
                return sort_double_t_streaming(col_1);
            return sort_double_t(col_1);
        case STRING:
            if (Cfg::numStreams > 0)
                return sort_string_t_streaming(col_1);
            return sort_string_t(col_1);
        case DateTime:
            if (Cfg::numStreams > 0)
                return sort_dt_t_streaming(col_1);
            return sort_dt_t(col_1);
    }

    throw UnsupportedOperationError("Sort failed, unknown col type.");
}

void GFI::clearCache(void *col) {
    CUDA_CHECK(cudaDeviceSynchronize());
    pool.releaseBuffer(col);
    CUDA_CHECK(cudaDeviceSynchronize());
}