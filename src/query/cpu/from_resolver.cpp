//
// Created by xabdomo on 4/15/25.
//

#include "from_resolver.hpp"

#include <hsql/sql/SelectStatement.h>

#include "select_executor.hpp"
#include "store.hpp"
#include "query/errors.hpp"

std::unordered_map<std::string, table*> FromResolver::resolve(hsql::TableRef * ref) {
    std::unordered_map<std::string, table*> res;

    if (ref->type == hsql::kTableName) {
        std::string name = ref->name;
        if (!tables.contains(name)) {
            throw NoSuchTableError(name);
        }
        res[name] = tables[name];
    } else if (ref->type == hsql::kTableSelect) {
        if (ref->select == nullptr || ref->alias == nullptr)
            throw std::runtime_error("Invalid table reference");
        res[ref->alias->name] = SelectExecutor::Execute(ref->select);
    } else if (ref->type == hsql::kTableCrossProduct) {
        for (auto tableName: *(ref->list)) { // execute each sub-import alone and return the final result
            auto result = FromResolver::resolve(tableName);
            for (auto& [name, table]: result) {
                if (res.contains(name)) {
                    throw std::runtime_error("Duplicate table name: " + name);
                }
                res[name] = table;
            }
        }
    } else if (ref->type == hsql::kTableJoin) {
        auto join = ref->join;
        //fixme: implement this after writing the code for WHERE / ON
        throw std::runtime_error("Joins will be implemented soon"); // need to implement Where / ON logic first ..
    } else {
        throw std::runtime_error("Unkown \"from\" type.");
    }

    return res;
}
