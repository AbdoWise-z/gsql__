//
// Created by xabdomo on 4/25/25.
//

#ifndef QUERY_OPTIMIZER_HPP
#define QUERY_OPTIMIZER_HPP

#include <vector>
#include <iostream>
#include <map>
#include <unordered_map>
#include <hsql/SQLParser.h>
#include <hsql/SQLParserResult.h>
#include <hsql/sql/SelectStatement.h>

#include "resolver.hpp"
#include "db/table.hpp"


namespace QueryOptimizer {
    struct ExecutionStep {
        hsql::Expr                       *query{} ;
        FromResolver::ResolveResult     input;
        std::set<std::string>         output_names;
    };

    std::string opTypeString(hsql::OperatorType operator_);

    std::string exprToString(const hsql::Expr* expr);

    // Helper: flatten an AND-tree into a vector of conjunctive predicates
    void collectConjuncts(hsql::Expr* e, std::vector<hsql::Expr*>& out);

    // Main recursive splitter: given any `where`, produce a list of steps
    // whose whereExpr is at most a binary-join predicate or conjunction thereof.
    void SegmentExpression_recv(hsql::Expr* where, std::vector<hsql::Expr*>& result);

    std::vector<hsql::Expr*> SegmentExpression(hsql::Expr* where);

    void getRequiredTables(FromResolver::ResolveResult& r, hsql::Expr* where,  std::set<std::string>& result);

    std::set<std::string> getRequiredTables(FromResolver::ResolveResult& r, hsql::Expr* where);

    FromResolver::ResolveResult constructSubInput(FromResolver::ResolveResult& r, hsql::Expr* where);

    std::vector<ExecutionStep> GeneratePlan(FromResolver::ResolveResult inputs, const hsql::SelectStatement* query);

    inline bool isSubSet(const std::set<std::string>&a, const std::set<std::string>&b) {
        if (a.size() > b.size()) return false;
        for (const auto& x : a) {
            if (!b.contains(x)) return false;
        }
        return true;
    }

    template <typename T>
    inline std::set<T> Union(const std::set<T>&a, const std::set<T>&b) {
        std::set<T> result;
        for (const auto& x : a) {
            result.insert(x);
        }
        for (const auto& x : b) {
            result.insert(x);
        }
        return result;
    }

    inline bool intersects(const std::set<std::string>&a, const std::set<std::string>&b) {
        for (const auto& x : a) {
            if (b.contains(x)) return true;
        }
        return false;
    }

    std::vector<ExecutionStep> ReducePlan(std::vector<ExecutionStep> before_steps);
};



#endif //QUERY_OPTIMIZER_HPP
