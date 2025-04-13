#include <iostream>

#include "hsql/SQLParser.h"
#include "csv.hpp"
#include "db/table.hpp"
#include "tensor/tensor.hpp"

#include <filesystem>

#include "cli/cli.hpp"
#include "db/db_helper.hpp"

namespace fs = std::filesystem;

std::unordered_map<std::string, table*> tables;

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
    if (tables.find(name) != tables.end()) {
        std::cout << "table with the same name already exists" << std::endl;
        return;
    }

    try {
        table* t = fromCSV(p);
        tables[name] = t;
        std::cout << "Loaded: " << p << ", as: " << name << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Error loading: " << e.what() << std::endl;
    }
}

void removeTable(std::vector<std::string> params) {
    for (auto name: params) {
        if (tables.find(name) != tables.end()) {
            delete tables[name];
            tables.erase(name);
            std::cout << "Removed: " << name << std::endl;
            return;
        } else {
            std::cout << "Table: " << name << " not found" << std::endl;
        }
    }
}

void show_tables(std::vector<std::string> params) {
    for (auto name: params) {
        if (tables.find(name) != tables.end()) {
            auto table = tables[name];
            std::cout << name << ", " << table->headers.size() << " columns";
            if (table->columns.size() > 0) {
                std::cout << ", " << table->columns[0].data.size() << " rows";
            }
            std::cout << std::endl;

            for (int i = 0; i < table->headers.size(); i++) {
                std::cout << "| -- "
                    << "("
                    << typeToString(table->columns[i].type)
                    << ")  "
                    << table->headers[i] << std::endl;
            }

        } else {
            std::cout << "Table: " << name << " not found" << std::endl;
        }
    }
}


void sql(std::vector<std::string> params) {
    std::cout << "Coming soon in 2025" << std::endl;
}



int main() {
    std::cout << "gsql++ running." << std::endl;
    std::cout << "use load / add [path] to load a csv file as table," << std::endl;
    std::cout << "use remove [name] to remove a table" << std::endl;
    std::cout << "use table [name] / [names] to show table details" << std::endl;
    std::cout << "or enter an SQL query" << std::endl;

    CLI cli(sql);
    cli.addCommand("load", loadTable);
    cli.addCommand("add",    removeTable);
    cli.addCommand("remove", removeTable);
    cli.addCommand("table", show_tables);

    cli.run();
    return 0;
}