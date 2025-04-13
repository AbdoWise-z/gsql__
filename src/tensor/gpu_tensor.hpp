//
// Created by xabdomo on 4/13/25.
//

#ifndef GPU_TENSOR_HPP
#define GPU_TENSOR_HPP

#include "utils/memory.cuh"

template<typename T>
class tensor<T, GPU> {
private:
    T* data;
    std::vector<size_t> shape;

public:

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

    T operator[](const std::vector <int> &indices) {
        if (indices.size() != shape.size()) {
            std::cout << indices.size() << " " << shape.size() << std::endl;
            throw std::invalid_argument("Tensor::operator[]: indices size mismatch (!= shape)");
        }

        int index = 0;
        int acc = 1;
        for (int i = 0;i < shape.size();i++) {
            index += indices[i] * acc;
            acc *= shape[i];
        }

        T val;
        cu::toHost(data, &val, sizeof(T));

        return val;
    }

    template<typename... Args>
    T operator()(Args&&... args) {
        const std::vector <int> indices = {std::forward<Args>(args)...};
        return this->operator[](indices);
    }

    T* getData() {
        return data;
    }

    tensor<T, CPU> toCPU() {
        T* cpuData = static_cast<T *>(malloc(sizeof(T) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>())));
        cu::toHost(data, cpuData, sizeof(T) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()));
        return tensor<T, CPU>(cpuData, shape);
    }

    ~tensor() {
        cu::free(data);
        data = nullptr;
    }
};


#endif //GPU_TENSOR_HPP
