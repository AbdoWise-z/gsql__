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

struct ExecutionStep {
    hsql::Expr                       *query ;
    FromResolver::GPU::ResolveResult  result;
    std::set<std::string>       output_names;
};

static std::string opTypeString(hsql::OperatorType operator_) {
    switch (operator_) {
        case hsql::kOpOr:
            return "or";
        case hsql::kOpAnd:
            return "and";
        case hsql::kOpNot:
            return "!";
        case hsql::kOpGreater:
            return ">";
        case hsql::kOpLess:
            return "<";
        case hsql::kOpLessEq:
            return "<=";
        case hsql::kOpGreaterEq:
            return ">=";
        case hsql::kOpEquals:
            return "=";
    }

    return "idk";
}

static std::string exprToString(const hsql::Expr* expr) {
    if (!expr) return "";

    std::ostringstream os;

    switch (expr->type) {
        case hsql::kExprOperator: {
            // Some ops are unary
            if (!expr->expr2) {
                // Unary operator, e.g. NOT expr or -expr
                os << opTypeString(expr->opType)
                   << " "
                   << exprToString(expr->expr);
            } else {
                // Binary operator, e.g. expr op expr2
                os << "("
                   << exprToString(expr->expr)
                   << " "
                   << opTypeString(expr->opType)
                   << " "
                   << exprToString(expr->expr2)
                   << ")";
            }
            break;
        }
        case hsql::kExprColumnRef: {
            // ColumnRef: expr->name (and optional table)
            if (expr->table) {
                os << expr->table << ".";
            }
            os << expr->name;
            break;
        }
        case hsql::kExprLiteralInt: {
            os << expr->ival;
            break;
        }
        case hsql::kExprLiteralString: {
            os << "'" << expr->name << "'";
            break;
        }
        case hsql::kExprLiteralFloat: {
            os << expr->fval;
            break;
        }
        case hsql::kExprLiteralNull: {
            os << "NULL";
            break;
        }
        case hsql::kExprLiteralDate: {
            std::cout << "Date time";
            os << expr->name;
        }
        default: {
            // Fallback: if we have a raw name, print it
            if (strlen(expr->name)) {
                os << expr->name;
            }
            break;
        }
    }

    return os.str();
}

// Helper: flatten an AND-tree into a vector of conjunctive predicates
static void collectConjuncts(hsql::Expr* e, std::vector<hsql::Expr*>& out) {
    if (!e) return;
    if (e->type == hsql::kExprOperator && e->opType == hsql::kOpAnd) {
        collectConjuncts(e->expr,  out);
        collectConjuncts(e->expr2, out);
    } else {
        out.push_back(e);
    }
}

// Main recursive splitter: given any `where`, produce a list of steps
// whose whereExpr is at most a binary-join predicate or conjunction thereof.
static void GeneratePlan_recv(hsql::Expr*                         where,
                              std::vector<hsql::Expr*>&         result) {
    if (!where) return;

    // 1) If this is an OR, split into two branches and cartesian-combine
    if (where->type == hsql::kExprOperator && where->opType == hsql::kOpOr) {
        std::vector<hsql::Expr*> leftSteps, rightSteps;
        GeneratePlan_recv(where->expr ,  leftSteps);
        GeneratePlan_recv(where->expr2, rightSteps);

        // Combine each left predicate with each right via a new OR node
        for (auto& L : leftSteps) {
            for (auto& R : rightSteps) {
                auto* orExpr = new hsql::Expr(hsql::kExprOperator);
                orExpr->opType = hsql::kOpOr;
                orExpr->expr   = L;
                orExpr->expr2  = R;
                result.push_back({ orExpr });
            }
        }
        return;
    }

    // 2) If this is an AND, simply flatten it and emit ONE step
    //    whose predicate is the full conjunction of all its leaves.
    if (where->type == hsql::kExprOperator && where->opType == hsql::kOpAnd) {
        std::vector<hsql::Expr*> conjuncts;
        collectConjuncts(where, conjuncts);

        for (auto* pred : conjuncts) {
            result.push_back({ pred });
        }
        return;
    }

    // 3) Otherwise, this is a leaf predicate (e.g. col1 = col2, col > 5, etc.)
    //    It may mention one or two tables; either way, it's a valid step.
    result.push_back({ where });
}


std::vector<hsql::Expr*> GeneratePlan(hsql::Expr* where) {
    std::vector<hsql::Expr*> result;
    GeneratePlan_recv(where, result);
    return result;
}

void getRequiredTables(FromResolver::GPU::ResolveResult& r, hsql::Expr* where,  std::set<std::string>& result) {
    if (!where) return;

    if (where->type == hsql::kExprOperator) {
        getRequiredTables(r, where->expr, result);
        getRequiredTables(r, where->expr2, result);
    }

    if (where->type == hsql::kExprColumnRef) {
        std::string ref = where->table;
        std::string col = where->name;

        if (ref.empty()) {
            // we need to search for the actual table
            bool found = false;
            for (int i = 0;i < r.tables.size();i++) {
                auto table = r.tables[i];
                auto exists = std::find(table->headers.begin(), table->headers.end(), col);
                if (exists != table->headers.end()) {
                    for (const auto& _name : r.table_names[i]) {
                        result.insert(_name);
                    }

                    if (found) {
                        // ambiguous
                        throw std::runtime_error(col + ", is ambiguous");
                    }

                    found = true;
                }
            }
        } else {
            result.insert(ref);
        }
    }
}

std::set<std::string> getRequiredTables(FromResolver::GPU::ResolveResult& r, hsql::Expr* where) {
    std::set<std::string> result;
    getRequiredTables(r, where, result);
    return result;
}

FromResolver::GPU::ResolveResult constructSubInput(
    FromResolver::GPU::ResolveResult& r,
    hsql::Expr* where) {
    auto req = getRequiredTables(r, where);

    std::unordered_map<table*, std::set<std::string>> inverse_tables;
    for (const auto i: req) {
        const auto table_ptr = r.tables[FromResolver::GPU::find(&r, i)];
        if (!inverse_tables.contains(table_ptr)) {
            inverse_tables[table_ptr] = std::set<std::string>();
        }

        inverse_tables[table_ptr].insert(i);
    }

    FromResolver::GPU::ResolveResult result;
    for (const auto& [k, v] : inverse_tables) {
        result.table_names.push_back(v);
        result.tables.push_back(k);
    }

    return result;
}

std::vector<ExecutionStep> GeneratePlan(TableMap db, const hsql::SelectStatement* query) {
    auto sub_wheres = GeneratePlan(query->whereClause);
    auto inputs = FromResolver::GPU::resolve(query->fromTable, db);
    std::vector<ExecutionStep> steps;

    for (auto where: sub_wheres) {
        auto sub_input = constructSubInput(inputs, where);
        auto step = ExecutionStep();
        step.query = where;
        step.result = sub_input;
        step.output_names = getRequiredTables(sub_input, where);
        steps.push_back(step);
    }

    return steps;
}

inline bool isSubSet(const std::set<std::string>&a, const std::set<std::string>&b) {
    if (a.size() > b.size()) return false;
    for (const auto& x : a) {
        if (!b.contains(x)) return false;
    }
    return true;
}

std::vector<ExecutionStep> ReducePlan(std::vector<ExecutionStep>& before_steps) {
    std::sort(before_steps.begin(), before_steps.end(), [](const ExecutionStep& a, const ExecutionStep& b) {
        return a.output_names.size() > b.output_names.size();
    });

    std::vector<ExecutionStep> steps;

    std::vector<bool> skip;
    for (int i = 0; i < before_steps.size(); ++i) {
        skip.push_back(false);
        auto& a = before_steps[i];
        for (int j = i - 1; j > 0; --j) {
            auto& b = before_steps[j];
            if (skip[j]) continue;
            if (isSubSet(a.output_names, b.output_names)) {
                skip[i] = true;
                auto old_query = b.query;
                auto new_query = new hsql::Expr(hsql::kExprOperator);
                new_query->opType = hsql::kOpAnd;
                new_query->expr = old_query;
                new_query->expr2 = a.query;
                b.query = new_query;
                break;
            }
        }
    }

    for (int i = 0; i < before_steps.size(); ++i) {
        if (!skip[i]) steps.push_back(before_steps[i]);
    }

    return steps;
}


int main() {
    std::string query = "Select * from a as a, b as b, c as c where c.c3 = 15 and a.c1 = b.c1 and b.c2 = c.c2 and b.c3 > '2002-11-18 11:15:01'";
    hsql::SQLParserResult parser_result;
    hsql::SQLParser::parse(query, &parser_result);
    auto steps = GeneratePlan({
        {"a" , new table({"c1", "c2", "c3"}, {})},
        {"b" , new table({"c1", "c2", "c3"}, {})},
        {"c" , new table({"c1", "c2", "c3"}, {})}
        }, dynamic_cast<const hsql::SelectStatement*>(parser_result.getStatement(0)));

    for (int i = 0;i < steps.size();i++) {
        auto step = steps[i];
        std::cout << "step " << i << " -> { ";
        int j = 0;
        for (const auto& name: step.output_names) {
            std::cout << name;
            if (j++ != step.output_names.size() - 1) std::cout << ", ";
        }

        std::cout << " } where " << exprToString(step.query) << ";" << std::endl;
    }
    std::cout << std::endl << "After reduction: " << std::endl;

    steps = ::ReducePlan(steps);
    for (int i = 0;i < steps.size();i++) {
        auto step = steps[i];
        std::cout << "step " << i << " -> { ";
        int j = 0;
        for (const auto& name: step.output_names) {
            std::cout << name;
            if (j++ != step.output_names.size() - 1) std::cout << ", ";
        }

        std::cout << " } where " << exprToString(step.query) << ";" << std::endl;
    }
}
