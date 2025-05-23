//
// Created by xabdomo on 4/13/25.
//

#include "memory.cuh"

#include <stdexcept>


void * cu::malloc(size_t size) {
    void* ptr;
    auto err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA malloc failed: " + std::string(cudaGetErrorString(err)));
    }
    return ptr;
}

void cu::free(void *ptr) {
    cudaFree(ptr);
}

void cu::toDevice(const void *src, void *dst, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void cu::toHost(const void *src, void *dst, size_t size) {
    cudaDeviceSynchronize();
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}
