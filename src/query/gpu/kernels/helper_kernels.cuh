//
// Created by xabdomo on 4/27/25.
//

#ifndef HELPER_KERNELS_CUH
#define HELPER_KERNELS_CUH



namespace HelperKernels {
    __global__ void strlen(const char* c, size_t* result);
}



#endif //HELPER_KERNELS_CUH
