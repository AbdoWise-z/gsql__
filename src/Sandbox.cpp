//
// Created by xabdomo on 4/19/25.
//


#include <vector>

#include "store.hpp"
#include "query/cpu/select_executor.hpp"

int main() {
    std::vector<size_t> test_input = {1024, 1024, 1024 * 14};
    auto result = SelectExecutor::Schedule(test_input);
    for (auto i : result) {
        //
    }
    return 0;
}
