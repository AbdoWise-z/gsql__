#include <iostream>

#include "hsql/SQLParser.h"
#include "csv.hpp"
#include "db/table.hpp"
#include "tensor/tensor.hpp"

#include <filesystem>

#include "cli/cli.hpp"
#include "db/db_helper.hpp"
#include "utils/konsol.hpp"

#include "store.hpp"
#include "editor/NanoEditor.h"
#include "query/cpu_executor.hpp"
#include "utils/string_utils.hpp"

namespace fs = std::filesystem;



void loadTable(std::vector<std::string> params) {
    std::string path_str = "";
    for (std::string param : params) {
        path_str += param;
    }

    fs::path p(path_str);

    if (!fs::exists(p)) {
        std::cout << "File does not exist" << std::endl;
        return;
    }

    std::string name = p.filename();
    name = name.substr(0, name.find_last_of('.'));
    if (global_tables.find(name) != global_tables.end()) {
        std::cout << "table with the same name already exists" << std::endl;
        return;
    }

    try {
        table* t = fromCSV(p);
        global_tables[name] = t;
        std::cout << "Loaded: " << color(p, GREEN_FG) << ", as: " << color(name, YELLOW_FG) << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Error loading: " << e.what() << std::endl;
    }
}

void removeTable(std::vector<std::string> params) {
    for (auto name: params) {
        if (global_tables.find(name) != global_tables.end()) {
            delete global_tables[name];
            global_tables.erase(name);
            std::cout << "Removed: " << color(name, GREEN_FG) << std::endl;
            return;
        } else {
            std::cout << "Table [" << color(name, RED_FG) << "] not found" << std::endl;
        }
    }
}

void show_details(std::vector<std::string> params) {
    for (const auto& name: params) {
        if (global_tables.contains(name)) {
            auto table = global_tables[name];

            std::cout << name << ": " << table::details(table) << std::endl;
        } else {
            std::cout << "Table [" << color(name, RED_FG) << "] not found" << std::endl;
        }
    }
}

void show_table(std::vector<std::string> params) {
    if (params.size() != 3) {
        std::cout << "Usage: show [table] [start-row] [count]" << std::endl;
        return;
    }

    auto name = params[0];
    auto start = stoi(params[1]);
    auto count = stoi(params[2]);

    // print the header
    std::vector<size_t> w;

    if (!global_tables.contains(name)) {
        std::cout << "Table " << color(name, RED_FG) << " not found." << std::endl;
        return;
    }

    auto table = global_tables[name];
    for (int i = 0;i < table->headers.size();i++) {
        switch (table->columns[i]->type) {
            case STRING:
                w.push_back(60);
                break;
            case INTEGER:
                w.push_back(25);
                break;
            case FLOAT:
                w.push_back(25);
                break;
        }

        count = std::min(count, static_cast<int>(table->columns[i]->data.size()) - start);

        std::cout << "";
        std::cout << std::setw(w[i]) << std::left << color(table->headers[i], GREEN_FG) << "|";
    }

    std::cout << std::endl;

    for (int i = start; i < start + count; i++) {
        for (int k = 0;k < table->headers.size();k++) {
            switch (table->columns[k]->type) {
                case STRING:
                    std::cout << std::setw(w[k]) << std::left << color(StringUtils::limit(*table->columns[k]->data[i].s, 30), RED_FG)                 << "|";
                    break;
                case INTEGER:
                    std::cout << std::setw(w[k]) << std::left << color(std::to_string(table->columns[k]->data[i].i) + " ", CYAN_FG) << "|";
                    break;
                case FLOAT:
                    std::cout << std::setw(w[k]) << std::left << color(std::to_string(table->columns[k]->data[i].d) + " ", CYAN_FG) << "|";
                    break;
            }


        }

        std::cout << std::endl;
    }
}

void hash_table(std::vector<std::string> params) {
    std::vector<std::string> cols;
    if (params.size() < 1) {
        std::cout << "Usage: hash [table] [col1] [col2] ..." << std::endl;
        return;
    }

    auto name = params[0];

    if (!global_tables.contains(name)) {
        std::cout << "Table " << color(name, RED_FG) << " not found." << std::endl;
        return;
    }

    auto t = global_tables[name];

    if (params.size() < 2) {
        for (auto head: t->headers) {
            cols.push_back(head);
        }
    } else {
        for (int i = 1; i < params.size();i++) {
            cols.push_back(params[i]);
        }
    }

    for (auto col: cols) {
        if (std::find(t->headers.begin(), t->headers.end(), col) == t->headers.end()) {
            std::cout << "Column " << color(col, RED_FG) << " not found." << std::endl;
            continue;
        }

        std::cout << "Building hashes for " << color(name, CYAN_FG) << "." << color(col, YELLOW_FG) << " ... ";
        auto index = std::distance(t->headers.begin(), std::find(t->headers.begin(), t->headers.end(), col));
        auto& c = t->columns[index];
        if (c->isHashIndexed()) {
            std::cout << color("Already indexed.", GREEN_FG) << std::endl;
            continue;
        }
        c->buildHashedIndexes(Cfg::HashTableExtendableSize);
        std::cout << color("Done.", GREEN_FG) << std::endl;
    }
}

void sort_table(std::vector<std::string> params) {
    std::vector<std::string> cols;
    if (params.size() < 1) {
        std::cout << "Usage: sort [table] [col1] [col2] ..." << std::endl;
        return;
    }

    auto name = params[0];

    if (!global_tables.contains(name)) {
        std::cout << "Table " << color(name, RED_FG) << " not found." << std::endl;
        return;
    }

    auto t = global_tables[name];

    if (params.size() < 2) {
        for (auto head: t->headers) {
            cols.push_back(head);
        }
    } else {
        for (int i = 1; i < params.size();i++) {
            cols.push_back(params[i]);
        }
    }

    for (auto col: cols) {
        if (std::find(t->headers.begin(), t->headers.end(), col) == t->headers.end()) {
            std::cout << "Column " << color(col, RED_FG) << " not found." << std::endl;
            continue;
        }

        std::cout << "Building sorted index for " << color(name, CYAN_FG) << "." << color(col, YELLOW_FG) << " ... ";
        auto index = std::distance(t->headers.begin(), std::find(t->headers.begin(), t->headers.end(), col));
        auto& c = t->columns[index];
        if (c->isHashIndexed()) {
            std::cout << color("Already indexed.", GREEN_FG) << std::endl;
            continue;
        }
        c->buildSortedIndexes();
        std::cout << color("Done.", GREEN_FG) << std::endl;
    }
}

void cfg(std::vector<std::string> params) {
    if (params.size() < 2) {
        std::cout << "Usage: cfg [option] [value]" << std::endl;
        return;
    }

    if (params[0] == "htEx") {
        try {
            Cfg::HashTableExtendableSize = std::stoi(params[1]);
            std::cout << "Updated: " << color("Cfg::HashTableExtendableSize", CYAN_FG) << " = " << Cfg::HashTableExtendableSize << std::endl;
            return;
        } catch (std::exception& e) {
            std::cout << "Error modifying configuration: " << e.what() << std::endl;
        }
    }


    if (params[0] == "mTE") {
        try {
            Cfg::maxTensorElements = std::stoi(params[1]);
            std::cout << "Updated: " << color("Cfg::maxTensorElements", CYAN_FG) << " = " << Cfg::maxTensorElements << std::endl;
            return;
        } catch (std::exception& e) {
            std::cout << "Error modifying configuration: " << e.what() << std::endl;
        }
    }

    std::cout << "Unknown param." << std::endl;
}

void sql(std::vector<std::string> params) {
    std::string query = "";
    for (int i = 0; i < params.size(); i++) {
        query += params[i];
        if (i != params.size() - 1) query += " ";
    }

    std::cout << "Running Query: " << color(query, CYAN_FG) << std::endl;

    hsql::SQLParserResult result;
    hsql::SQLParser::parse(query, &result);
    try {
        auto r_vec = CPUExecutor::executeQuery(result);
        for (auto t: r_vec) {
            std::cout << "Result: " << std::endl;
            std::cout << table::details(t) << std::endl;

            int i = 0;
            while (global_tables.contains("result-" + std::to_string(i))) {
                i++;
            }

            global_tables["result-" + std::to_string(i)] = t;
            std::cout << "Result saved on table: " << color("result-" + std::to_string(i), GREEN_FG) << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "Query failed: " << color(e.what(), RED_FG) << std::endl;
    }
}

void dummy(std::vector<std::string> params) {
    for (auto name: params) {
        if (global_tables.find(name) != global_tables.end()) {
            std::cout << "[" << color(name,  RED_FG) <<  "] table with the same name already exists" << std::endl;
            continue;
        }
        global_tables[name] = new table();
        std::cout << "Loaded: " << color("data/empty.csv", GREEN_FG) << ", as: " << color(name, YELLOW_FG) << std::endl;
    }
}

void editor(std::vector<std::string> params) {
    std::string result = NanoEditor::edit();
    sql({result});
}


int main() {
    global_tables.clear();

    std::cout << "gsql++ running." << std::endl;
    std::cout << "use load / add [path] to load a csv file as table," << std::endl;
    std::cout << "use remove [name] to remove a table," << std::endl;
    std::cout << "use details [name] / [names] to show table details" << std::endl;
    std::cout << "or enter a SQL query" << std::endl;

    CLI cli(sql);
    cli.addCommand("load", loadTable);
    cli.addCommand("add",    removeTable);
    cli.addCommand("remove", removeTable);
    cli.addCommand("details", show_details);
    cli.addCommand("editor", editor);
    cli.addCommand("dummy", dummy);
    cli.addCommand("show", show_table);
    cli.addCommand("hash", hash_table);
    cli.addCommand("sort", sort_table);
    cli.addCommand("cfg", cfg);

    cli.run();
    return 0;
}