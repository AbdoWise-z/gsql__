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

    inline bool equalsIgnoreCase(const std::string& a, const std::string& b) {
        return a.size() == b.size() &&
            std::equal(a.begin(), a.end(), b.begin(),
            [](unsigned char x, unsigned char y) {
                return std::tolower(x) == std::tolower(y);
        });
    }

    inline std::string trim(const std::string& s) {
        auto start = s.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) return ""; // all whitespace

        auto end = s.find_last_not_of(" \t\r\n");
        return s.substr(start, end - start + 1);
    }
}



#endif //STRING_UTILS_HPP
