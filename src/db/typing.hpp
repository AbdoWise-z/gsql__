//
// Created by xabdomo on 4/13/25.
//

#ifndef TYPING_HPP
#define TYPING_HPP
#include <string>

enum DataType {
    INTEGER = 0,
    FLOAT = 1,
    DateTime = 2,
    STRING = 3,
};

std::string typeToString(DataType);

inline std::string typeToString(DataType d) {
    if (d == STRING) return "str";
    if (d == INTEGER) return "int";
    if (d == FLOAT) return "float";
    if (d == DateTime) return "dt";

    return "idk"; // idk
                    // he doesn't know
                    // they doesn't know
}

#endif //TYPING_HPP
