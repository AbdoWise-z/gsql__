//
// Created by xabdomo on 4/20/25.
//

#ifndef GPU_BUFFER_POOL_CUH
#define GPU_BUFFER_POOL_CUH

#include <functional>
#include <cstddef>
#include <list>

typedef std::function<void*(size_t)> BufferAllocator;
typedef std::function<void()> VoidFunction;

// A simple LRU GPU buffer cache & pool
class GpuBufferPool {
public:
    // maxMemory = maximum total bytes to keep allocated on the GPU
    explicit GpuBufferPool(size_t maxMemory = (1ULL<<30));  // default 1 GiB
    ~GpuBufferPool();

    // Lookup or create a GPU buffer for key “ptr”.
    // If missing, calls `alloc(internalAllocator)` to get {size, gpuPtr}.
    // Evicts least‐recently‐used entries if needed.
    void* getBufferOrCreate(
        void* ptr,
        void* data,
        std::function<std::pair<size_t, void*>(void*, BufferAllocator)> alloc
    );

    void* getBufferOrCreate(
        void* ptr,
        void* data,
        std::function<std::tuple<size_t, void*, VoidFunction>(void*, BufferAllocator)> alloc
    );


    // Immediately free & remove the buffer for key “ptr”.
    void  releaseBuffer(void* ptr);

    // (Optional) Change the cache’s memory‐limit on the fly.
    void  setMaxMemory(size_t bytes);

private:
    struct BufferInfo {
        size_t                        size{};
        void*                       gpuPtr{};
        VoidFunction              freeFunc{};
        std::list<void*>::iterator     lruIt;
    };

    size_t                                 m_maxMemory;
    size_t                                 m_currentMemory = 0;
    std::unordered_map<void*,BufferInfo>   m_map;
    std::list<void*>                       m_lruList;
};

#endif // GPU_BUFFER_POOL_CUH
