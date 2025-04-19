//
// Created by xabdomo on 4/15/25.
//

#include "store.hpp"


std::unordered_map<std::string, table*> global_tables;

namespace Cfg {
    size_t HashTableExtendableSize = 4;

    size_t maxTensorElements = 512 * 512 * 512;

    std::vector<size_t> getTileSizeFor(const std::vector<size_t>& inputSize) {
        std::vector<long double> factors;
        uint64_t totalSize = 0;
        for (auto a: inputSize) {
            factors.push_back(a);
            totalSize += a;
        }

        for (long double & factor : factors) {
            factor /= static_cast<float>(totalSize);
        }

        // x * y * z <= maxTensorElements
        // x : y : z = a : b : c
        // x / z = a / c -> x = a / c * z
        //                  y = b / c * z
        // a / c * b / c * z^3 = maxTensorElements

        long double _m = 1;
        for (int i = 0;i < inputSize.size() - 1;i++) {
            _m = factors[i] * _m / factors[inputSize.size() - 1];
        }

        std::vector<size_t> result(inputSize.size(), 0);
        long double _f = maxTensorElements / _m;
        result[inputSize.size() - 1] = static_cast<int>(ceil(pow(_f, 1.0f / inputSize.size())));

        for (int i = 0;i < inputSize.size() - 1;i++) {
            result[i] = static_cast<int>(ceil(factors[i] / factors[inputSize.size() - 1] * result[inputSize.size() - 1]));
        }

        for (int i = 0;i < inputSize.size();i++) {
            result[i] = std::min(result[i], inputSize[i]);
        }

        return result;
    }
}
