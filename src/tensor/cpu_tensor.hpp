//
// Created by xabdomo on 4/13/25.
//

#ifndef CPU_TENSOR_HPP
#define CPU_TENSOR_HPP

#include "utils/memory.cuh"
#include <iostream>


template<typename T>
class tensor<T, Device::CPU> {
private:
    void fill_hyperplane(T t, std::vector<uint64_t>& pos, const std::vector<size_t>& dims, size_t current_dim) {
        if (current_dim == dims.size()) {
            uint64_t index = 0;
            uint64_t acc = 1;

            for (size_t i = 0; i < dims.size(); i++) {
                index += pos[i] * acc;
                acc *= shape[i];
            }

            this->operator[](index) = t;
            return;
        }

        if (dims[current_dim] == 1) {
            for (uint64_t i = 0; i < shape[current_dim]; i++) {
                pos[current_dim] = i;
                fill_hyperplane(t, pos, dims, current_dim + 1);
            }
            pos[current_dim] = (current_dim < pos.size()) ? pos[current_dim] : 0;
        } else {
            fill_hyperplane(t, pos, dims, current_dim + 1);
        }
    }

public:
    T* data;
    std::vector<size_t> shape;

    explicit tensor(const std::vector<size_t>& shape) : shape(shape) {
        //std::cout << "CPU create from shape\n";
        //std::cout << "Alloc: " << sizeof(T) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()) << std::endl;
        data = static_cast<T *>(malloc(sizeof(T) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>())));
    }

    tensor(T* data, const std::vector<size_t>& shape) : shape(shape), data(data) {
        //std::cout << "CPU create from shape & data\n";

    }

    tensor(const tensor& other) : shape(other.shape) {
        //std::cout << "CPU create from copy\n";
        //std::cout << "Alloc: " << sizeof(T) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()) << std::endl;
        data = static_cast<T *>(malloc(sizeof(T) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>())));
        std::copy(other.data, other.data + std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()), data);
    }

    tensor(tensor&& other) noexcept : shape(std::move(other.shape)), data(other.data) {
        //std::cout << "CPU create from data\n";
        other.data = nullptr;
    }

    Device getDevice() {
        return CPU;
    }

    virtual T& operator[](const std::vector <size_t> &indices) {
        if (indices.size() != shape.size()) {
            std::cout << indices.size() << " " << shape.size() << std::endl;
            throw std::invalid_argument("Tensor::operator[]: indices size mismatch (!= shape)");
        }

        uint64_t index = 0;
        uint64_t acc = 1;
        for (size_t i = 0;i < shape.size();i++) {
            index += indices[i] * acc;
            acc *= shape[i];
        }
        return data[index];
    }

    virtual T& operator[](size_t index) {
        uint64_t acc = 1;
        for (unsigned long i : shape) {
            acc *= i;
        }

        if (index >= acc)
            throw std::invalid_argument("Tensor::operator[]: index out of range");

        return data[index];
    }

    [[nodiscard]] virtual size_t map(const std::vector<size_t>& indices) const {
        if (indices.size() != shape.size()) {
            throw std::invalid_argument("Tensor::map: indices size mismatch (!= shape)");
        }

        size_t index = 0;
        size_t acc = 1;
        for (size_t i = 0;i < shape.size();i++) {
            index += indices[i] * acc;
            acc *= shape[i];
        }
        return index;
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

    template<typename... Args>
    T& operator()(Args&&... args) {
        const std::vector <size_t> indices = {std::forward<Args>(args)...};
        return this->operator[](indices);
    }

    virtual T* getData() {
        return data;
    }

    virtual tensor<T, Device::GPU> toGPU() {
        T* gpuData = static_cast<T *>(cu::malloc(sizeof(T) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>())));
        cu::toDevice(data, gpuData, sizeof(T) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()));
        return tensor<T, Device::GPU>(gpuData, shape);
    }

    virtual size_t totalSize() {
        size_t acc = 1;
        for (const size_t i : shape) {
            acc *= i;
        }

        return acc;
    }

    virtual void setAll(T t) {
        const auto size = totalSize();
        for (size_t i = 0;i < size;i++) {
            data[i] = t;
        }
    }

    virtual void fill(T t, std::vector<uint64_t> pos, std::vector<size_t> dims) {
        if (pos.size() != dims.size() || pos.size() != shape.size()) {
            throw std::invalid_argument("fill: pos and dims size mismatch");
        }

        // Recursively fill all positions where dims[i] == 0 (free variables)
        fill_hyperplane(t, pos, dims, 0);
    }

    virtual ~tensor() {
        //std::cout << "Free: " << sizeof(T) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()) << std::endl;
        free(data);
        data = nullptr;
    }

    virtual tensor<T, Device::CPU> operator&&(const tensor<T, Device::CPU>& other) {
        if (other.shape != this->shape) {
            throw std::runtime_error("[AND] Tensor size mismatch");
        }

        tensor result(this->shape);
        size_t acc = 1;
        for (const size_t i : shape) {
            acc *= i;
        }

        for (size_t i = 0;i < acc;i++) {
            result.data[i] = this->data[i] && other.data[i];
        }

        return result;
    }

    virtual tensor<T, Device::CPU> operator||(const tensor<T, Device::CPU>& other) {
        if (other.shape != this->shape) {
            throw std::runtime_error("[AND] Tensor size mismatch");
        }

        tensor result(this->shape);
        size_t acc = 1;
        for (const size_t i : shape) {
            acc *= i;
        }

        for (size_t i = 0;i < acc;i++) {
            result.data[i] = this->data[i] || other.data[i];
        }

        return result;
    }

    virtual tensor<T, Device::CPU> operator!() {

        tensor result(this->shape);

        size_t acc = 1;
        for (const size_t i : shape) {
            acc *= i;
        }

        for (size_t i = 0;i < acc;i++) {
            result.data[i] = !this->data[i];
        }

        return result;
    }
};


#endif //CPU_TENSOR_HPP
