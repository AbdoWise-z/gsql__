//
// Created by xabdomo on 4/13/25.
//

#include <iostream>
#include <cuda_runtime.h>

void printCudaDeviceInfo() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found.\n";
        return;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        std::cout << "=== Device #" << device << " ===\n";
        std::cout << "Name: " << prop.name << "\n";
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "Total global memory: " << (prop.totalGlobalMem >> 20) << " MB\n";
        std::cout << "Shared memory per block: " << prop.sharedMemPerBlock << " bytes\n";
        std::cout << "Registers per block: " << prop.regsPerBlock << "\n";
        std::cout << "Warp size: " << prop.warpSize << "\n";
        std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "Multiprocessor count: " << prop.multiProcessorCount << "\n";
        std::cout << "Clock rate: " << (prop.clockRate / 1000) << " MHz\n";
        std::cout << "Memory clock rate: " << (prop.memoryClockRate / 1000) << " MHz\n";
        std::cout << "Memory bus width: " << prop.memoryBusWidth << " bits\n";
        std::cout << "L2 cache size: " << prop.l2CacheSize << " bytes\n";
        std::cout << "Max grid dimensions: [" << prop.maxGridSize[0] << ", "
                                              << prop.maxGridSize[1] << ", "
                                              << prop.maxGridSize[2] << "]\n";
        std::cout << "Max block dimensions: [" << prop.maxThreadsDim[0] << ", "
                                               << prop.maxThreadsDim[1] << ", "
                                               << prop.maxThreadsDim[2] << "]\n";
        std::cout << "-------------------------------------------\n";
    }
}

int main() {
    printCudaDeviceInfo();
    return 0;
}
