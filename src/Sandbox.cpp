//
// Created by xabdomo on 4/19/25.
//


#include <vector>
#include <iostream>
#include <map>
#include <unordered_map>
#include <hsql/SQLParser.h>
#include <hsql/SQLParserResult.h>
#include <hsql/sql/SelectStatement.h>

#include "db/table.hpp"
#include "query/query_optimizer.hpp"
#include "query/gpu/from_resolver.hpp"

int main() {

    const int N = 100;
    std::vector<int> data(N, 0);

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        // Critical section to prevent race condition
        #pragma omp critical
        {
            data[i] = i * i;  // Only one thread at a time can access this block
        }
    }

    // Output some values to confirm the result
    std::cout << "data[42] = " << data[42] << std::endl;

    return 0;
}
