//
// Created by xabdomo on 5/8/25.
//

#include "file_utils.hpp"

#include <fstream>
#include <sstream>

std::string FileUtils::fileAsString(const std::string &path) {

    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + path);
    }

    std::string line;
    std::stringstream ss;
    while (std::getline(file, line)) {
        // Remove trailing '\r' if present (handles Windows \r\n) (fk windows btw)
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (line.empty()) continue;
        ss << line << std::endl;
    }

    return ss.str();
}
