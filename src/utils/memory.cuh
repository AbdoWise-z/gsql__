//
// Created by xabdomo on 4/13/25.
//

#ifndef CU_MEM_INTERFACE_CUH
#define CU_MEM_INTERFACE_CUH


#include <string>
#include <vector>


namespace cu {
    void* malloc(size_t size);
    void free(void* ptr);
    void toDevice(const void* src, void* dst, size_t size);
    void toHost(const void* src, void* dst, size_t size);

    template<typename T>
    void* vectorToDevice(std::vector<T> &vec) {
        void* ptr = malloc(vec.size() * sizeof(T));
        toDevice(vec.data(), ptr, vec.size() * sizeof(T));
        return ptr;
    }

    template<typename T>
    std::vector<T> vectorFromDevice(void* ptr, size_t size) {
        size_t totalSize = size * sizeof(T);
        void* thisDev    = ::malloc(totalSize);
        toHost(ptr, thisDev, totalSize);
        std::vector<T> result;
        for (size_t i = 0; i < size; i++) {
            result.push_back(static_cast<T*>(thisDev)[i]);
        }
        return result;
    }

    inline void* stringToDevice(const std::string& str) {
        auto size = str.size() + 1;
        void* ptr = malloc(size * sizeof(char));
        toDevice(str.c_str(), ptr, size * sizeof(char));
        return ptr;
    }
}

#endif //CU_MEM_INTERFACE_CUH
