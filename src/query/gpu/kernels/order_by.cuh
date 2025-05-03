//
// Created by xabdomo on 5/2/25.
//

#ifndef ORDER_BY_CUH
#define ORDER_BY_CUH

#include <cstdint>


namespace OrderBy {

    __global__ void histogram_kernel_indexed(
        const int64_t* data,
        const uint32_t* indices,
        uint32_t* histogram,
        uint32_t* pins,
        size_t num_elements,
        uint32_t mask_bits,
        uint32_t shift_bits,
        uint32_t num_pins);

    __global__ void radix_scatter_pass(
        const  int64_t* data,
        const uint32_t* indices_in,
              uint32_t* indices_out,
        uint32_t* pin_offsets,        // size: num_bins, initialized from prefix_sums
        uint32_t* local_offsets,
        size_t num_elements,
        uint32_t mask_bits,
        uint32_t shift_bits
        );

    // kinda efficient :clown:
    __global__ void efficient_local_prefix_sum(const uint32_t* input, uint32_t* output, int n, int pins, uint32_t* aux);

    __global__ void add_local_aux(uint32_t* input, int n, int pins, const uint32_t* aux);

}



#endif //ORDER_BY_CUH
