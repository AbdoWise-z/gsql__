cmake_minimum_required(VERSION 3.30)
project(gsql__)

set(CMAKE_CXX_STANDARD 20)

include_directories(libs)
link_directories(libs/hsql)

add_executable(gsql__ src/main.cpp)

target_link_libraries(gsql__ PUBLIC sqlparser)