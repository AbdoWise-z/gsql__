//
// Created by xabdomo on 4/13/25.
//

#ifndef CU_MEM_INTERFACE_CUH
#define CU_MEM_INTERFACE_CUH


#include <vector>


namespace cu {
    void* malloc(size_t size);
    void free(void* ptr);
    void toDevice(void* src, void* dst, size_t size);
    void toHost(void* src, void* dst, size_t size);

    template<typename T>
    void* vectorToDevice(std::vector<T> &vec) {
        void* ptr = malloc(vec.size() * sizeof(T));
        toDevice(vec.data(), ptr, vec.size() * sizeof(T));
        return ptr;
    }
}

#endif //CU_MEM_INTERFACE_CUH
