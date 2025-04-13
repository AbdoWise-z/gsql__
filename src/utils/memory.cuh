//
// Created by xabdomo on 4/13/25.
//

#ifndef CU_MEM_INTERFACE_CUH
#define CU_MEM_INTERFACE_CUH


namespace cu {
    void* malloc(size_t size);
    void free(void* ptr);
    void toDevice(void* src, void* dst, size_t size);
    void toHost(void* src, void* dst, size_t size);
}

#endif //CU_MEM_INTERFACE_CUH
