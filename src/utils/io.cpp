//
// Created by xabdomo on 5/8/25.
//

#include "io.hpp"

#include <fstream>
#include <iostream>
#include <streambuf>

static std::streambuf* original_cout_buf = nullptr;
static std::ofstream null_stream;

void io::disableOutput() {
    if (!original_cout_buf) {
        original_cout_buf = std::cout.rdbuf();
#if defined(_WIN32)
        null_stream.open("NUL");
#else
        null_stream.open("/dev/null");
#endif
    }

    std::cout.rdbuf(null_stream.rdbuf());
}

void io::enableOutput() {
    if (original_cout_buf) {
        std::cout.rdbuf(original_cout_buf);
    }
}
