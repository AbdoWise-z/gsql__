//
// Created by xabdomo on 4/20/25.
//

#include "gpu_buffer_pool.cuh"

#include <cuda_runtime.h>
#include <iostream>
#include <list>
#include <unordered_map>
#include <stdexcept>

#include "utils/memory.cuh"

//----------------------------------------------------------------------------------------------------------------------
// Constructor + Destructor
//----------------------------------------------------------------------------------------------------------------------
GpuBufferPool::GpuBufferPool(size_t maxMemory)
  : m_maxMemory(maxMemory)
{}

GpuBufferPool::~GpuBufferPool() {
    // Free all remaining GPU allocations
    for (auto &kv : m_map) {
        cudaFree(kv.second.gpuPtr);
    }
}



//----------------------------------------------------------------------------------------------------------------------
// setMaxMemory
//----------------------------------------------------------------------------------------------------------------------
void GpuBufferPool::setMaxMemory(size_t bytes) {
    m_maxMemory = bytes;
    // Optionally, immediately evict if currentMemory > new limit:
    while (m_currentMemory > m_maxMemory && !m_lruList.empty()) {
        void* evictKey = m_lruList.back();
        auto info = m_map[evictKey];
        m_lruList.pop_back();
        m_currentMemory -= info.size;

        if (!info.freeFunc) cu::free(info.gpuPtr);
        else info.freeFunc();

        m_map.erase(evictKey);
    }
}

//----------------------------------------------------------------------------------------------------------------------
// getBufferOrCreate
//----------------------------------------------------------------------------------------------------------------------
void * GpuBufferPool::getBufferOrCreate(void *ptr, void *data,
    std::function<std::pair<size_t, void *>(void *, BufferAllocator)> alloc) {
    auto it = m_map.find(ptr);
    if (it != m_map.end()) {
        // Move to front of LRU
        m_lruList.erase(it->second.lruIt);
        m_lruList.push_front(ptr);
        it->second.lruIt = m_lruList.begin();
        return it->second.gpuPtr;
    }

    // --- CACHE MISS: ALLOCATE ---
    // User‐provided alloc returns (bufferSize, gpuPtr)
    auto [bufSize, gpuPtr] = alloc(data, cu::malloc);

    // --- EVICT AS NEEDED ---
    // Pop back until we have room
    while (m_currentMemory + bufSize > m_maxMemory && !m_lruList.empty()) {
        void* evictKey = m_lruList.back();
        auto info = m_map[evictKey];
        m_lruList.pop_back();
        m_currentMemory -= info.size;

        if (!info.freeFunc) cu::free(info.gpuPtr);
        else info.freeFunc();

        m_map.erase(evictKey);
    }

    // If the single buffer is larger than the entire cache, bail out
    if (bufSize > m_maxMemory) {
        cu::free(gpuPtr);
        throw std::runtime_error("Requested buffer exceeds maximum cache size");
    }

    // --- INSERT NEW ENTRY ---
    m_lruList.push_front(ptr);
    BufferInfo bi{ bufSize, gpuPtr, nullptr, m_lruList.begin() };
    m_map[ptr] = bi;
    m_currentMemory += bufSize;

    return gpuPtr;
}

void* GpuBufferPool::getBufferOrCreate(
    void* ptr,
    void* data,
    std::function<std::tuple<size_t, void*, VoidFunction>(void*, BufferAllocator)> alloc
) {
    // --- CACHE HIT? ---
    auto it = m_map.find(ptr);
    if (it != m_map.end()) {
        // Move to front of LRU
        m_lruList.erase(it->second.lruIt);
        m_lruList.push_front(ptr);
        it->second.lruIt = m_lruList.begin();
        return it->second.gpuPtr;
    }

    // --- CACHE MISS: ALLOCATE ---
    // User‐provided alloc returns (bufferSize, gpuPtr)
    auto [bufSize, gpuPtr, freeFunc] = alloc(data, cu::malloc);

    // --- EVICT AS NEEDED ---
    // Pop back until we have room
    while (m_currentMemory + bufSize > m_maxMemory && !m_lruList.empty()) {
        void* evictKey = m_lruList.back();
        auto info = m_map[evictKey];
        m_lruList.pop_back();
        m_currentMemory -= info.size;
        if (!info.freeFunc) cu::free(info.gpuPtr);
        else info.freeFunc();
        m_map.erase(evictKey);
    }

    // If the single buffer is larger than the entire cache, bail out
    if (bufSize > m_maxMemory) {
        cu::free(gpuPtr);
        throw std::runtime_error("Requested buffer exceeds maximum cache size");
    }

    // --- INSERT NEW ENTRY ---
    m_lruList.push_front(ptr);
    BufferInfo bi{ bufSize, gpuPtr, freeFunc, m_lruList.begin() };
    m_map[ptr] = bi;
    m_currentMemory += bufSize;

    return gpuPtr;
}

//----------------------------------------------------------------------------------------------------------------------
// releaseBuffer
//----------------------------------------------------------------------------------------------------------------------
void GpuBufferPool::releaseBuffer(void* ptr) {
    auto it = m_map.find(ptr);
    if (it == m_map.end()) return;

    // Free GPU memory and remove from LRU + map
    if (!it->second.freeFunc) cu::free(it->second.gpuPtr);
    else it->second.freeFunc();

    m_currentMemory -= it->second.size;
    m_lruList.erase(it->second.lruIt);
    m_map.erase(it);
}