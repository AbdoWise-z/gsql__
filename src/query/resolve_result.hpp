//
// Created by xabdomo on 4/25/25.
//

#ifndef RESOLVE_RESULT_HPP
#define RESOLVE_RESULT_HPP

#include "store.hpp"
#include "../db/table.hpp"

namespace FromResolver {
    struct ResolveResult {
        std::vector<std::set<std::string>> table_names;
        std::vector<table*> tables;
        std::vector<bool> isTemporary;
    };

    inline bool shouldDeleteIntermediate(FromResolver::ResolveResult r, table* t) {
        for (int i = 0;i < r.tables.size();i++) {
            if (r.tables[i] == t) {
                return false;
            }
        }

        return true; // not in the input so it was created by us
    }

    inline bool shouldDelete(FromResolver::ResolveResult r, table* t) {
        for (int i = 0;i < r.tables.size();i++) {
            if (r.tables[i] == t) {
                return r.isTemporary[i];
            }
        }

        return true; // not in the input so it was created by us
    }


    inline int find(std::vector<std::set<std::string>> a, const std::string& tname) {
        int idx = -1;
        for (int i = 0;i < a.size();i++) {
            if (a[i].contains(tname)){
                if (idx != -1) return -2; // ambiguous
                idx = i;
            }
        }

        // fixme: handle exceptions here instead ?

        return idx;
    }

    inline table* find(TableMap t, const std::string& tname, bool autoError = true) {
        table* ret = nullptr;

        for (auto& [k, v] : t) {
            if (k.contains(tname)) {
                if (ret && autoError) {
                    throw std::invalid_argument("Table name [" + tname + "] is ambergris.");
                }

                ret = v;
            }
        }

        return ret;
    }

    inline TableMap::key_type find_it(TableMap t, const std::string& tname, bool autoError = true) {
        auto it = t.begin();
        auto result = t.end();
        while (it != t.end()) {
            if (it->first.contains(tname)) {
                if (result != t.end() && autoError) {
                    throw std::invalid_argument("Table name [" + tname + "] is ambergris.");
                }
                result = it;
            }
            ++it;
        }

        if (result == t.end()) {
            return {};
        }

        return result->first;
    }

    inline int find(ResolveResult *a, std::string tname) {
        return find(a->table_names, tname);
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
