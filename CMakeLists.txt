cmake_minimum_required(VERSION 3.30)

project(gsql__ LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -arch=sm_52")
set(CMAKE_CUDA_ARCHITECTURES 52)

# Include and link directories
include_directories(libs)
include_directories(src)
link_directories(libs/hsql)
find_package(OpenMP REQUIRED)


set(SOURCES
        src/constants.hpp

        src/utils/clp.hpp
        src/tensor/tensor.hpp
        src/tensor/cpu_tensor.hpp
        src/tensor/gpu_tensor.hpp

        src/utils/memory.cu
        src/utils/memory.cuh

        src/db/column.cpp
        src/db/column.hpp

        src/utils/murmur_hash3_cuda.cu
        src/utils/murmur_hash3_cuda.cuh

        src/utils/murmur_hash3.cpp
        src/utils/murmur_hash3.hpp

        src/db/typing.hpp

        src/db/value_helper.cpp
        src/db/value_helper.hpp

        src/db/table.cpp
        src/db/table.hpp

        src/db/db_helper.cpp
        src/db/db_helper.hpp

        src/cli/cli.cpp
        src/cli/cli.hpp

        src/utils/konsol.cpp
        src/utils/konsol.hpp

        src/query/cpu_executor.cpp
        src/query/cpu_executor.hpp

        src/query/gpu_executor.cpp
        src/query/gpu_executor.hpp

        src/query/errors.cpp
        src/query/errors.hpp

        src/store.cpp
        src/store.hpp

        src/utils/string_utils.cpp
        src/utils/string_utils.hpp

        src/editor/NanoEditor.cpp
        src/editor/NanoEditor.h

        # CPU stuff
        src/query/cpu/from_resolver.cpp
        src/query/cpu/from_resolver.hpp
        src/query/cpu/select_executor.cpp
        src/query/cpu/select_executor.hpp
        src/query/cpu/filter_applier.cpp
        src/query/cpu/filter_applier.hpp

        src/query/cpu/ops/equality.cpp
        src/query/cpu/ops/equality.hpp

        src/query/cpu/ops/logical_and.cpp
        src/query/cpu/ops/logical_and.hpp
        src/query/cpu/ops/logical_or.cpp
        src/query/cpu/ops/logical_or.hpp
        src/query/cpu/ops/greater_than.cpp
        src/query/cpu/ops/greater_than.hpp
        src/query/cpu/agg/sum_avg_count.cpp
        src/query/cpu/agg/sum_avg_count.hpp
        src/query/cpu/agg/max_min.cpp
        src/query/cpu/agg/max_min.hpp

        # GPU stuff
        src/query/gpu/from_resolver.cpp
        src/query/gpu/from_resolver.hpp
        src/query/gpu/select_executor.cpp
        src/query/gpu/select_executor.hpp
        src/query/gpu/filter_applier.cpp
        src/query/gpu/filter_applier.hpp

        src/query/gpu/ops/equality.cpp
        src/query/gpu/ops/equality.hpp

        src/query/gpu/ops/logical_and.cpp
        src/query/gpu/ops/logical_and.hpp
        src/query/gpu/ops/logical_or.cpp
        src/query/gpu/ops/logical_or.hpp
        src/query/gpu/ops/greater_than.cpp
        src/query/gpu/ops/greater_than.hpp
        src/query/gpu/agg/sum_avg_count.cpp
        src/query/gpu/agg/sum_avg_count.hpp
        src/query/gpu/agg/max_min.cpp
        src/query/gpu/agg/max_min.hpp

        src/query/gpu/gpu_function_interface.cu
        src/query/gpu/gpu_function_interface.cuh

        src/query/gpu/gpu_buffer_pool.cu
        src/query/gpu/gpu_buffer_pool.cuh

        src/query/gpu/kernels/tensor_kernels.cu
        src/query/gpu/kernels/tensor_kernels.cuh

        src/query/gpu/kernels/equality_kernel.cu
        src/query/gpu/kernels/equality_kernel.cuh

        src/query/gpu/kernels/inequality_kernel.cu
        src/query/gpu/kernels/inequality_kernel.cuh

        src/query/query_optimizer.cpp
        src/query/query_optimizer.hpp
)

#
# ==============================
#              MAIN
# ==============================
#
add_executable(gsql__
        src/main.cpp
        ${SOURCES}
        src/query/resolve_result.hpp
        src/query/query_optimizer.cpp
        src/query/query_optimizer.hpp
)

set_target_properties(gsql__ PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)


# Link with external libraries
target_link_libraries(gsql__ PUBLIC
        sqlparser OpenMP::OpenMP_CXX
)

target_compile_options(gsql__ PRIVATE ${OpenMP_CXX_FLAGS})

# Make sure CUDA files are compiled with the CUDA compiler
set_source_files_properties(
        src/utils/memory.cu
        src/utils/memory.cu

        src/utils/murmur_hash3_cuda.cu
        src/utils/murmur_hash3_cuda.cuh

        src/query/gpu/gpu_function_interface.cu
        src/query/gpu/gpu_function_interface.cuh

        src/query/gpu/gpu_buffer_pool.cu
        src/query/gpu/gpu_buffer_pool.cuh

        src/query/gpu/kernels/tensor_kernels.cu
        src/query/gpu/kernels/tensor_kernels.cuh

        src/query/gpu/kernels/equality_kernel.cu
        src/query/gpu/kernels/equality_kernel.cuh

        src/query/gpu/kernels/inequality_kernel.cu
        src/query/gpu/kernels/inequality_kernel.cuh

        PROPERTIES LANGUAGE CUDA
)

#
# ==============================
#            SANDBOX
# ==============================
#
add_executable(
        Sandbox
        src/Sandbox.cpp
        ${SOURCES}
)

set_target_properties(Sandbox PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(Sandbox PUBLIC
        sqlparser OpenMP::OpenMP_CXX
)

target_compile_options(Sandbox PRIVATE ${OpenMP_CXX_FLAGS})


#
# ==============================
#          CUDA Helper
# ==============================
#
add_executable(details src/details.cu)

set_target_properties(details PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

set_source_files_properties(
        src/details.cu
        PROPERTIES LANGUAGE CUDA
)