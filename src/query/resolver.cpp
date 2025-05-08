//
// Created by xabdomo on 5/8/25.
//

#include "resolver.hpp"

#include <hsql/sql/SelectStatement.h>

#include "errors.hpp"


static void selectResolver(hsql::SelectStatement* stmnt, std::set<std::string>& needs);
static void exprResolver(hsql::Expr* expr, std::set<std::string>& needs);

static void fromResolver(hsql::TableRef* ref, std::set<std::string>& needs) {
    if (ref == nullptr) { // base case
        return;
    }

    if (ref->type == hsql::kTableName) {
        std::string name = ref->name;
        needs.insert(name);
    } else if (ref->type == hsql::kTableSelect) {
        selectResolver(ref->select, needs);
    } else if (ref->type == hsql::kTableCrossProduct) {
        for (auto tableName: *(ref->list)) {
            fromResolver(tableName, needs);
        }
    } else if (ref->type == hsql::kTableJoin) {
        auto join = ref->join;

        fromResolver(join->left, needs);
        fromResolver(join->right, needs);
    } else {
        throw std::runtime_error("Unknown \"from\" type.");
    }
}

static void exprResolver(hsql::Expr* expr, std::set<std::string>& needs) {
    if (expr == nullptr) return;

    exprResolver(expr->expr, needs);
    exprResolver(expr->expr2, needs);
    selectResolver(expr->select, needs);
}

static void selectResolver(hsql::SelectStatement* stmnt, std::set<std::string>& needs) {
    if (stmnt == nullptr) return;
    fromResolver(stmnt->fromTable, needs);

    auto proj = stmnt->selectList;
    for (hsql::Expr* expr: *proj) {
        exprResolver(expr, needs);
    }

    auto where = stmnt->whereClause;
    exprResolver(where, needs);
}

static void statementResolver(hsql::SQLStatement* statement, std::set<std::string>& needs) {
    switch (statement->type()) {
        case hsql::kStmtSelect:
            selectResolver(dynamic_cast<hsql::SelectStatement *>(statement), needs);
            break;
        default:
            throw UnsupportedOperationError(statement->type());
    }
}

std::set<std::string> FromResolver::resolveQueryNeeds(const hsql::SQLParserResult &query) {
    std::set<std::string> needs;

    for (const auto s: query.getStatements()) {
        statementResolver(s, needs);
    }

    return needs;
}