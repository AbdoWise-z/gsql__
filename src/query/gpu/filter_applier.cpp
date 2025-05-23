//
// Created by xabdomo on 4/15/25.
//

#include "filter_applier.hpp"

#include "db/value_helper.hpp"
#include "ops/equality.hpp"
#include "ops/greater_than.hpp"
#include "ops/logical_and.hpp"
#include "ops/logical_or.hpp"
#include "ops/null_equality.hpp"
#include "query/errors.hpp"
#include "query/gpu/select_executor.hpp"

// #define FILTER_DEBUG

tensor<char, Device::GPU>* FilterApplier::GPU::apply(
        FromResolver::GPU::ResolveResult *input_data,
        hsql::Expr *eval,
        hsql::LimitDescription *limit,
        const std::vector<size_t>& tile_start,
        const std::vector<size_t>& tile_size
    ) {

    std::vector<size_t> result_size;

    for (int i = 0;i < input_data->table_names.size();i++) {
        auto table = input_data->tables[i];
        if (table->columns.empty()) {
            return new tensor<char, Device::GPU>({});
        } else {
            if (!tile_size.empty())
                result_size.push_back(tile_size[i]);
            else
                result_size.push_back(table->columns[0]->data.size());
        }
    }

    if (eval == nullptr) { // pass all
        auto* result = new tensor<char, Device::GPU>(result_size);
        result->setAll(1);
        return result;
    }

    auto expr_type = eval->type;
    if (expr_type == hsql::ExprType::kExprOperator) {
        // handle operators
        auto op = eval->opType;
#ifdef FILTER_DEBUG
        std::cout << "hsql::ExprType::kExprOperator" << op << std::endl;
#endif
        if (op == hsql::kOpAnd) {
            return Ops::GPU::logical_and(input_data, eval, limit, tile_start, tile_size);
        }

        if (op == hsql::kOpOr) {
            return Ops::GPU::logical_or(input_data, eval, limit, tile_start, tile_size);
        }

        if (op == hsql::kOpEquals) {
            return Ops::GPU::equality(input_data, eval, limit, tile_start, tile_size);
        }

        if (op == hsql::kOpGreater) {
            return Ops::GPU::greater_than(input_data, eval->expr, eval->expr2, limit, tile_start, tile_size);
        }

        if (op == hsql::kOpLess) {
            return Ops::GPU::greater_than(input_data, eval->expr2, eval->expr, limit, tile_start, tile_size);
        }

        if (op == hsql::kOpGreaterEq) {
            auto t1 = Ops::GPU::greater_than(input_data, eval->expr, eval->expr2, limit, tile_start, tile_size);
            auto t2 = Ops::GPU::equality(input_data, eval, limit, tile_start, tile_size);
            auto result = new tensor<char, Device::GPU>(*t1 || *t2);
            delete t1;
            delete t2;
            return result; // yes I am lazy sue me ig.
        }

        if (op == hsql::kOpLessEq) {
            auto t1 = Ops::GPU::greater_than(input_data, eval->expr2, eval->expr, limit, tile_start, tile_size);
            auto t2 = Ops::GPU::equality(input_data, eval, limit, tile_start, tile_size);
            auto result = new tensor<char, Device::GPU>(*t1 || *t2);
            delete t1;
            delete t2;
            return result;
        }

        if (op == hsql::kOpNotEquals) {
            auto t2 = Ops::GPU::equality(input_data, eval, limit, tile_start, tile_size);
            auto result = new tensor<char, Device::GPU>(!*t2);
            delete t2;
            return result;
        }

        if (op == hsql::kOpIsNull) {
            return Ops::GPU::null_equality(input_data, eval, limit, tile_start, tile_size);
        }

        if (op == hsql::kOpNot) {
            auto t2 = apply(input_data, eval->expr, limit, tile_start, tile_size);
            auto result = new tensor<char, Device::GPU>(!*t2);
            delete t2;
            return result;
        }

        auto* result = new tensor<char, Device::GPU>(result_size);
        auto literal = ValuesHelper::getLiteralFrom(eval, false, SelectExecutor::GPU::global_input, ValuesHelper::GPU);

        result->setAll(ValuesHelper::isZero(literal.first, literal.second) ? 0 : 1);
        ValuesHelper::deleteValue(literal.first, literal.second);
        return result;
    } else {
#ifdef FILTER_DEBUG
        std::cout << "hsql::ExprType::kLiteral" << std::endl;
#endif

        auto* result = new tensor<char, Device::GPU>(result_size);
        auto literal = ValuesHelper::getLiteralFrom(eval, false, SelectExecutor::GPU::global_input, ValuesHelper::GPU);

        result->setAll(ValuesHelper::isZero(literal.first, literal.second) ? 0 : 1);
        ValuesHelper::deleteValue(literal.first, literal.second);
        return result;
    }
}
