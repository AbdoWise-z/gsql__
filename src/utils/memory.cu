//
// Created by xabdomo on 4/13/25.
//

#include "memory.cuh"


void * cu::malloc(size_t size) {
    void* ptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

void cu::free(void *ptr) {
    cudaFree(ptr);
}

void cu::toDevice(void *src, void *dst, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void cu::toHost(void *src, void *dst, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}
