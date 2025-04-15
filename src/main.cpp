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
    if (tables.find(name) != tables.end()) {
        std::cout << "table with the same name already exists" << std::endl;
        return;
    }

    try {
        table* t = fromCSV(p);
        tables[name] = t;
        std::cout << "Loaded: " << color(p, GREEN_FG) << ", as: " << color(name, YELLOW_FG) << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Error loading: " << e.what() << std::endl;
    }
}

void removeTable(std::vector<std::string> params) {
    for (auto name: params) {
        if (tables.find(name) != tables.end()) {
            delete tables[name];
            tables.erase(name);
            std::cout << "Removed: " << color(name, GREEN_FG) << std::endl;
            return;
        } else {
            std::cout << "Table [" << color(name, RED_FG) << "] not found" << std::endl;
        }
    }
}

void show_details(std::vector<std::string> params) {
    for (const auto& name: params) {
        if (tables.contains(name)) {
            auto table = tables[name];

            std::cout << name << ": " << table::details(table) << std::endl;
        } else {
            std::cout << "Table [" << color(name, RED_FG) << "] not found" << std::endl;
        }
    }
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
        CPUExecutor::executeQuery(result);
    } catch (const std::exception& e) {
        std::cout << "Query failed: " << color(e.what(), RED_FG) << std::endl;
    }
}

void dummy(std::vector<std::string> params) {
    for (auto name: params) {
        if (tables.find(name) != tables.end()) {
            std::cout << "[" << color(name,  RED_FG) <<  "] table with the same name already exists" << std::endl;
            continue;
        }
        tables[name] = new table();
        std::cout << "Loaded: " << color("data/empty.csv", GREEN_FG) << ", as: " << color(name, YELLOW_FG) << std::endl;
    }
}

void editor(std::vector<std::string> params) {
    std::string result = NanoEditor::edit();
    sql({result});
}


int main() {
    tables.clear();

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

    cli.run();
    return 0;
}