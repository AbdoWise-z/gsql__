//
// Created by xabdomo on 5/2/25.
//

#ifndef ORDER_BY_CUH
#define ORDER_BY_CUH

#include <cstdint>


typedef unsigned long long int index_t;

namespace OrderBy {

    __global__ void histogram_kernel_indexed(
        const int64_t*  data,
        const index_t* indices,
        index_t* histogram,
        index_t* pins,
        size_t num_elements,
        index_t mask_bits,
        index_t shift_bits,
        index_t num_pins);

    __global__ void radix_scatter_pass(
        const  int64_t* data,
        const index_t* indices_in,
              index_t* indices_out,
        index_t* pin_offsets,        // size: num_bins, initialized from prefix_sums
        index_t* local_offsets,
        size_t num_elements,
        index_t mask_bits,
        index_t shift_bits
        );

    // kinda efficient :clown:
    __global__ void efficient_local_prefix_sum(const uint32_t* input, uint32_t* output, int n, int pins, uint32_t* aux);

    __global__ void add_local_aux(uint32_t* input, int n, int pins, const uint32_t* aux);

}



#endif //ORDER_BY_CUH
