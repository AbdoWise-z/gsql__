//
// Created by xabdomo on 4/15/25.
//

#include "from_resolver.hpp"

#include <hsql/sql/SelectStatement.h>

#include "filter_applier.hpp"
#include "select_executor.hpp"
#include "store.hpp"
#include "query/errors.hpp"

FromResolver::ResolveResult FromResolver::CPU::resolve(hsql::TableRef * ref, TableMap& tables) {
    std::vector<std::set<std::string>> table_names;
    std::vector<table*> table_values;
    std::vector<bool> temporary_table;

    if (ref->type == hsql::kTableName) {
        std::string name = ref->name;
        std::string t_name = ref->name;

        auto table = FromResolver::find(tables, name, false);
        if (!table) {
            throw NoSuchTableError(name);
        }

        if (ref->alias != nullptr) {
            t_name = ref->alias->name;
        }

        table_names.push_back({t_name});
        table_values.push_back(table);
        temporary_table.push_back(false);

    } else if (ref->type == hsql::kTableSelect) {
        if (ref->select == nullptr || ref->alias == nullptr)
            throw std::runtime_error("Invalid table reference");

        table_names.push_back({ref->alias->name});
        table_values.push_back(SelectExecutor::CPU::Execute(ref->select, tables).second);
        temporary_table.push_back(true);

    } else if (ref->type == hsql::kTableCrossProduct) {
        for (auto tableName: *(ref->list)) { // execute each sub-import alone and return the final result
            auto result = FromResolver::CPU::resolve(tableName, tables);
            for (int i = 0;i < result.table_names.size();i++) {
                auto names = result.table_names[i];
                auto table = result.tables[i];
                auto temporary = result.isTemporary[i];

                if (tableName->alias != nullptr) {
                    names = { tableName->alias->name };
                }

                table_names.push_back(names);
                table_values.push_back(table);
                temporary_table.push_back(temporary);
            }
        }
    } else if (ref->type == hsql::kTableJoin) {
        auto join = ref->join;

        auto select = new hsql::SelectStatement();
        select->fromTable = new hsql::TableRef(hsql::kTableCrossProduct);
        select->fromTable->list = new std::vector<hsql::TableRef*>();
        select->fromTable->list->push_back(join->left);
        select->fromTable->list->push_back(join->right);
        select->selectList = new std::vector<hsql::Expr*>();
        select->selectList->push_back(new hsql::Expr(hsql::kExprStar));
        select->whereClause = join->condition;

        auto result = SelectExecutor::CPU::Execute(select, tables);

        table_names.push_back(result.first);
        table_values.push_back(result.second);
        temporary_table.push_back(true);
    } else {
        throw std::runtime_error("Unknown \"from\" type.");
    }

    return {
        .table_names = table_names,
        .tables = table_values,
        .isTemporary = temporary_table,
    };
}
