cmake_minimum_required(VERSION 3.30)

project(gsql__ LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# Optional: Prevent host compiler warnings on CUDA code
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Include and link directories
include_directories(libs)
include_directories(src)
link_directories(libs/hsql)

# Source files
set(SOURCES
        src/main.cpp

        src/tensor/tensor.hpp
        src/tensor/cpu_tensor.hpp
        src/tensor/gpu_tensor.hpp

        src/utils/memory.cu
        src/utils/memory.cuh

        src/db/column.cpp
        src/db/column.hpp

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
        src/query/errors.cpp
        src/query/errors.hpp

        src/store.cpp
        src/store.hpp

        src/editor/NanoEditor.cpp
        src/editor/NanoEditor.h
        src/query/cpu/from_resolver.cpp
        src/query/cpu/from_resolver.hpp
        src/query/cpu/select_executor.cpp
        src/query/cpu/select_executor.hpp
        src/query/cpu/filter_applier.cpp
        src/query/cpu/filter_applier.hpp
        src/utils/string_utils.cpp
        src/utils/string_utils.hpp
)

# Create executable
add_executable(gsql__ ${SOURCES})

# Link with external libraries
target_link_libraries(gsql__ PUBLIC sqlparser)

# Make sure CUDA files are compiled with the CUDA compiler
set_source_files_properties(
        src/utils/memory.cu
        src/utils/memory.cu
        PROPERTIES LANGUAGE CUDA
)

add_executable(details src/details.cu)

# Make sure CUDA files are compiled with the CUDA compiler
set_source_files_properties(
        src/details.cu
        PROPERTIES LANGUAGE CUDA
)