//
// Created by xabdomo on 4/15/25.
//

#include "from_resolver.hpp"

#include <hsql/sql/SelectStatement.h>

#include "filter_applier.hpp"
#include "select_executor.hpp"
#include "store.hpp"
#include "query/errors.hpp"

FromResolver::ResolveResult FromResolver::GPU::merge(ResolveResult *a, ResolveResult *b) {
    ResolveResult result;
    for (int i = 0;i < a->table_names.size();i++) { // add a normally
        auto names = a->table_names[i];
        auto table = a->tables[i];
        auto temporary = a->isTemporary[i];

        result.table_names.push_back(names);
        result.tables.push_back(table);
        result.isTemporary.push_back(temporary);
    }

    for (int i = 0;i < b->table_names.size();i++) {
        auto names = b->table_names[i];
        auto table = b->tables[i];
        auto temporary = b->isTemporary[i];

        // we need to check for dubs
        for (auto name: names) {
            auto k = FromResolver::GPU::find(&result, name);
            if (k != -1) {
                throw std::runtime_error("Duplicate table name: " + name);
            }
        }

        result.table_names.push_back(names);
        result.tables.push_back(table);
        result.isTemporary.push_back(temporary);
    }

    return result;
}

int FromResolver::GPU::find(ResolveResult *a, std::string tname) {
    for (int i = 0;i < a->table_names.size();i++) {
        if (a->table_names[i].contains(tname)) {
            return i;
        }
    }

    return -1;
}

FromResolver::ResolveResult FromResolver::GPU::resolve(hsql::TableRef * ref, TableMap& tables) {
    std::vector<std::set<std::string>> table_names;
    std::vector<table*> table_values;
    std::vector<bool> temporary_table;

    if (ref->type == hsql::kTableName) {
        std::string name = ref->name;
        std::string t_name = ref->name;

        if (!tables.contains(name)) {
            throw NoSuchTableError(name);
        }

        if (ref->alias != nullptr) {
            t_name = ref->alias->name;
        }

        table_names.push_back({t_name});
        table_values.push_back(tables[name]);
        temporary_table.push_back(false);

    } else if (ref->type == hsql::kTableSelect) {
        if (ref->select == nullptr || ref->alias == nullptr)
            throw std::runtime_error("Invalid table reference");

        table_names.push_back({ref->alias->name});
        table_values.push_back(SelectExecutor::GPU::Execute(ref->select, tables));
        temporary_table.push_back(true);

    } else if (ref->type == hsql::kTableCrossProduct) {
        for (auto tableName: *(ref->list)) { // execute each sub-import alone and return the final result
            auto result = FromResolver::GPU::resolve(tableName, tables);
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
        // auto join = ref->join;
        //
        // auto left   = FromResolver::GPU::resolve(join->left);
        // auto right  = FromResolver::GPU::resolve(join->right);
        // auto merged = FromResolver::GPU::merge(&left, &right);
        //
        // auto tensor = FilterApplier::GPU::apply(&merged, join->condition, nullptr);
        // auto result = SelectExecutor::GPU::ConstructTable(tensor, &merged);
        // delete tensor;
        //
        // std::unordered_set<std::string> final_result;
        // for (auto names: merged.table_names) {
        //     for (auto name: names) {
        //         final_result.insert(name);
        //     }
        // }
        //
        // for (int i = 0;i < merged.tables.size();i++) {
        //     // clean up
        //     if (merged.isTemporary[i]) {
        //         delete merged.tables[i];
        //     }
        // }
        //
        //
        // table_names.push_back(final_result);
        // table_values.push_back(result.result);
        // temporary_table.push_back(true);
        throw std::runtime_error("Joins with tiling needs a rework.");
    } else {
        throw std::runtime_error("Unknown \"from\" type.");
    }

    return {
        .table_names = table_names,
        .tables = table_values,
        .isTemporary = temporary_table,
    };
}
