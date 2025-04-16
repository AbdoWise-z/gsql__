//
// Created by xabdomo on 4/13/25.
//

#ifndef TYPING_HPP
#define TYPING_HPP
#include <string>

enum DataType {
    STRING = 0,
    INTEGER = 1,
    FLOAT = 2
};

std::string typeToString(DataType);

inline std::string typeToString(DataType d) {
    if (d == STRING) return "str";
    if (d == INTEGER) return "int";
    if (d == FLOAT) return "float";

    return "idk"; // idk
}

#endif //TYPING_HPP
