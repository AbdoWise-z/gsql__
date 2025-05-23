//
// Created by xabdomo on 4/20/25.
//

#ifndef GPU_FUNCTION_INTERFACE_CUH
#define GPU_FUNCTION_INTERFACE_CUH

#include "db/column.hpp"
#include "tensor/tensor.hpp"


// GFI for gpu function interface
namespace GFI {
    void fill(tensor<char, Device::GPU> *output_data, char value);
    void fill(tensor<char, Device::GPU> *output_data, char value, std::vector<size_t> position, std::vector<size_t> mask);

    void logical_and(const tensor<char, Device::GPU> *a, const tensor<char, Device::GPU> *b, tensor<char, Device::GPU> *out);
    void logical_or (const tensor<char, Device::GPU> *a, const tensor<char, Device::GPU> *b, tensor<char, Device::GPU> *out);
    void logical_not(const tensor<char, Device::GPU> *a, tensor<char, Device::GPU> *out);

    void equality(
        tensor<char, Device::GPU> *result,
        column *col_1,
        column *col_2,

        std::vector<size_t> tileOffset,
        std::vector<size_t> tileSize,

        size_t table_1_index,
        size_t table_2_index,

        std::vector<size_t> mask
    );

    void equality_date(
        tensor<char, Device::GPU> *result,
        column *col_1,
        tval value,

        std::vector<size_t> tileOffset,
        std::vector<size_t> tileSize,

        size_t table_1_index,

        std::vector<size_t> mask
    );

    void equality_time(
        tensor<char, Device::GPU> *result,
        column *col_1,
        tval value,

        std::vector<size_t> tileOffset,
        std::vector<size_t> tileSize,

        size_t table_1_index,

        std::vector<size_t> mask
    );


    void equality(
        tensor<char, Device::GPU> *result,
        column *col_1,
        tval value,
        bool isNull,

        std::vector<size_t> tileOffset,
        std::vector<size_t> tileSize,

        size_t table_1_index,

        std::vector<size_t> mask
    );

    void inequality(
        tensor<char, Device::GPU> *result,
        column *col_1,
        column *col_2,

        std::vector<size_t> tileOffset,
        std::vector<size_t> tileSize,

        size_t table_1_index,
        size_t table_2_index,

        std::vector<size_t> mask,
        column::SortedSearchType operation
    );

    void inequality(
        tensor<char, Device::GPU> *result,
        column *col_1,
        tval value,

        std::vector<size_t> tileOffset,
        std::vector<size_t> tileSize,

        size_t table_1_index,

        std::vector<size_t> mask,
        column::SortedSearchType operation
    );

    tval max(
        column *col_1
    );

    tval min(
        column *col_1
    );

    tval sum(
        column *col_1
    );

    tval avg(
        column *col_1
    );

    std::vector<size_t> iterator(tensor<char, Device::GPU> *a);

    std::vector<unsigned long long int> sort(column *col_1);

    void clearCache(void* col);

}



#endif //GPU_FUNCTION_INTERFACE_CUH
