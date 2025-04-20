//
// Created by xabdomo on 4/15/25.
//

#ifndef STORE_HPP
#define STORE_HPP

#include <unordered_map>
#include <string>
#include "db/table.hpp"
#include <chrono>
#include <iostream>
#include <type_traits>

#define time_it(call) \
([&]() -> decltype(call) { \
    using namespace std::chrono; \
    const auto _time_it_start_ = high_resolution_clock::now(); \
    if constexpr (std::is_void_v<decltype(call)>) { \
        call; \
        const auto _time_it_end_ = high_resolution_clock::now(); \
        const auto _time_it_duration_ = _time_it_end_ - _time_it_start_; \
        const auto _sec_ = duration_cast<seconds>(_time_it_duration_); \
        const auto _ms_ = duration_cast<milliseconds>(_time_it_duration_); \
        const auto _ns_ = duration_cast<nanoseconds>(_time_it_duration_); \
        std::cout << #call << " Executed in: " \
                  << _sec_.count() << " seconds, " \
                  << _ms_.count() << " ms, " \
                  << _ns_.count() << " ns." << std::endl; \
    } else { \
        decltype(call) _time_it_result_ = call; \
        const auto _time_it_end_ = high_resolution_clock::now(); \
        const auto _time_it_duration_ = _time_it_end_ - _time_it_start_; \
        const auto _sec_ = duration_cast<seconds>(_time_it_duration_); \
        const auto _ms_ = duration_cast<milliseconds>(_time_it_duration_); \
        const auto _ns_ = duration_cast<nanoseconds>(_time_it_duration_); \
        std::cout << #call << " Executed in: " \
                  << _sec_.count() << " seconds, " \
                  << _ms_.count() << " ms, " \
                  << _ns_.count() << " ns." << std::endl; \
        return _time_it_result_; \
    } \
})()

extern std::unordered_map<std::string, table*> global_tables;

namespace Cfg {
    extern size_t HashTableExtendableSize;

    extern size_t maxTensorElements;

    extern size_t BlockDimX;
    extern size_t BlockDimY;
    extern size_t BlockDimZ;

    std::vector<size_t> getTileSizeFor(const std::vector<size_t>& inputSize);
}



#endif //STORE_HPP
