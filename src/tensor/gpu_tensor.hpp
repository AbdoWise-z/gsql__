//
// Created by xabdomo on 4/13/25.
//

#ifndef GPU_TENSOR_HPP
#define GPU_TENSOR_HPP

#include "query/gpu/gpu_function_interface.cuh"
#include "query/gpu/gpu_function_interface.cuh"
#include "utils/memory.cuh"

namespace GFI {
    void fill(tensor<char, Device::GPU>* output_data, char value);
    void fill(tensor<char, Device::GPU>* output_data, char value, 
             std::vector<size_t> position, std::vector<size_t> mask);

    void logical_and(const tensor<char, Device::GPU> *a, const tensor<char, Device::GPU> *b, tensor<char, Device::GPU> *out);
    void logical_or (const tensor<char, Device::GPU> *a, const tensor<char, Device::GPU> *b, tensor<char, Device::GPU> *out);
    void logical_not(const tensor<char, Device::GPU> *a, tensor<char, Device::GPU> *out);

}

template<typename T>
class tensor<T, Device::GPU> {
public:
    T* data;
    std::vector<size_t> shape;

    explicit tensor(const std::vector<size_t>& shape) : shape(shape) {
        //std::cout << "GPU create from shape\n";
        data = static_cast<T *>(cu::malloc(sizeof(T) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>())));
    }

    tensor(T* data, const std::vector<size_t>& shape) : data(data), shape(shape) {
        //std::cout << "GPU create from shape & data\n";
    }

    tensor(const tensor& other) : shape(other.shape) {
        //std::cout << "CPU create from copy\n";
        data = static_cast<T *>(cu::malloc(sizeof(T) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>())));
        std::copy(other.data, other.data + std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()), data);
    }

    tensor(tensor&& other) noexcept : data(other.data), shape(std::move(other.shape)) {
        //std::cout << "CPU create from move\n";
        other.data = nullptr;
    }

    Device getDevice() {
        return GPU;
    }

    T set(const std::vector <size_t> &indices, T value) {
        if (indices.size() != shape.size()) {
            std::cout << indices.size() << " " << shape.size() << std::endl;
            throw std::invalid_argument("Tensor::operator[]: indices size mismatch (!= shape)");
        }

        size_t index = 0;
        size_t acc = 1;
        for (size_t i = 0;i < shape.size();i++) {
            index += indices[i] * acc;
            acc *= shape[i];
        }

        return set(index, value);
    }

    T set(size_t index, T value) {
        if (index >= totalSize()) {
            throw std::invalid_argument("Tensor::operator[]: index out of range");
        }

        T ret;
        cu::toHost(data + index * sizeof(T), &ret, sizeof(T));
        cu::toDevice(&value, data + index * sizeof(T), sizeof(T));
        return ret;
    }

    T operator[](const std::vector <size_t> &indices) {
        if (indices.size() != shape.size()) {
            std::cout << indices.size() << " " << shape.size() << std::endl;
            throw std::invalid_argument("Tensor::operator[]: indices size mismatch (!= shape)");
        }

        size_t index = 0;
        size_t acc = 1;
        for (size_t i = 0;i < shape.size();i++) {
            index += indices[i] * acc;
            acc *= shape[i];
        }

        T val;
        cu::toHost(data + index * sizeof(T), &val, sizeof(T));
        return val;
    }

    template<typename... Args>
    T operator()(Args&&... args) {
        const std::vector <size_t> indices = {std::forward<Args>(args)...};
        return this->operator[](indices);
    }

    tensor<T, CPU> toCPU() {
        T* cpuData = static_cast<T *>(malloc(sizeof(T) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>())));
        cu::toHost(data, cpuData, sizeof(T) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()));
        return tensor<T, CPU>(cpuData, shape);
    }

    virtual ~tensor() {
        cu::free(data);
        data = nullptr;
    }

    virtual size_t totalSize() {
        size_t acc = 1;
        for (const size_t i : shape) {
            acc *= i;
        }

        return acc;
    }

    virtual std::vector<size_t> unmap(const size_t i) {
        std::vector<size_t> indices;
        size_t remaining = i;
        for (size_t dim : shape) {
            indices.push_back(remaining % dim);
            remaining = remaining / dim;
        }
        return indices;
    }

    virtual void setAll(T t) {
        GFI::fill(this, t);
    }

    virtual void fill(T t, std::vector<uint64_t> pos, std::vector<size_t> dims) {
        if (pos.size() != dims.size() || pos.size() != shape.size()) {
            throw std::invalid_argument("fill: pos and dims size mismatch");
        }

        GFI::fill(this, t, pos, dims);
    }

    virtual tensor<T, Device::GPU> operator&&(const tensor<T, Device::GPU>& other) {
        if (other.shape != this->shape) {
            throw std::runtime_error("[AND] Tensor size mismatch");
        }

        tensor result(this->shape);

        GFI::logical_and(this, &other, &result);

        return result;
    }

    virtual tensor<T, Device::GPU> operator||(const tensor<T, Device::GPU>& other) {
        if (other.shape != this->shape) {
            throw std::runtime_error("[AND] Tensor size mismatch");
        }

        tensor result(this->shape);

        GFI::logical_or(this, &other, &result);

        return result;
    }

    virtual tensor<T, Device::GPU> operator!() {

        tensor result(this->shape);

        GFI::logical_not(this, &result);

        return result;
    }
};


#endif //GPU_TENSOR_HPP
