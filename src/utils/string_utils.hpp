//
// Created by xabdomo on 4/16/25.
//

#ifndef STRING_UTILS_HPP
#define STRING_UTILS_HPP

#include <string>
#include <sstream>

namespace StringUtils {
    template <typename InputIt>
    std::string join(InputIt begin, InputIt end, const std::string& delimiter = ", ") {
        std::ostringstream oss;
        if (begin != end) {
            oss << *begin++;  // First element (no leading delimiter)
            while (begin != end) {
                oss << delimiter << *begin++;
            }
        }
        return oss.str();
    }

    inline std::string limit(std::string other, int size) {
        if (other.size() > size) {
            return other.substr(0, size);
        }

        return other;
    }
}



#endif //STRING_UTILS_HPP
