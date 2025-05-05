//
// Created by xabdomo on 4/25/25.
//

#ifndef RESOLVE_RESULT_HPP
#define RESOLVE_RESULT_HPP

#include "../db/table.hpp"

namespace FromResolver {
    struct ResolveResult {
        std::vector<std::set<std::string>> table_names;
        std::vector<table*> tables;
        std::vector<bool> isTemporary;
    };

    inline bool shouldDelete(FromResolver::ResolveResult r, table* t) {
        for (int i = 0;i < r.tables.size();i++) {
            if (r.tables[i] == t) {
                return r.isTemporary[i];
            }
        }

        return true; // not in the input so it was created by us
    }


    inline int find(ResolveResult *a, std::string tname) {
        for (int i = 0;i < a->table_names.size();i++) {
            if (a->table_names[i].contains(tname)) {
                return i;
            }
        }

        return -1;
    }

    inline ResolveResult merge(ResolveResult *a, ResolveResult *b) {
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
                auto k = find(&result, name);
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


}

#endif //RESOLVE_RESULT_HPP
