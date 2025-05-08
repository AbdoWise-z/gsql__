#include <iostream>

#include "hsql/SQLParser.h"
#include "db/table.hpp"
#include "tensor/tensor.hpp"

#include <filesystem>

#include "cli/cli.hpp"
#include "db/db_helper.hpp"
#include "utils/konsol.hpp"

#include "store.hpp"
#include "editor/NanoEditor.h"
#include "query/cpu_executor.hpp"
#include "query/gpu_executor.hpp"
#include "query/cpu/from_resolver.hpp"
#include "utils/file_utils.hpp"
#include "utils/io.hpp"
#include "utils/string_utils.hpp"

namespace fs = std::filesystem;

void removeTable(std::vector<std::string> params) {
    for (auto name: params) {
        auto t = FromResolver::find_it(global_tables, name, false);
        if (!t.empty()) {
            delete global_tables[t];
            global_tables.erase(t);
            std::cout << "Removed: " << color(name, GREEN_FG) << std::endl;
            return;
        } else {
            std::cout << "Table [" << color(name, RED_FG) << "] not found" << std::endl;
        }
    }
}

void show_details(std::vector<std::string> params) {
    for (const auto& name: params) {
        auto t = FromResolver::find(global_tables, name, false);
        if (t) {
            std::cout << name << ": " << table::details(t) << std::endl;
        } else {
            std::cout << "Table [" << color(name, RED_FG) << "] not found" << std::endl;
        }
    }
}

void load_table(std::vector<std::string> params) {
    if (params.empty() || params.size() > 2) {
        std::cout << "Usage: load [path] [alias]" << std::endl;
        return;
    }

    const std::string& path_str = params[0];

    fs::path p(path_str);

    if (!fs::exists(p)) {
        std::cout << "File does not exist" << std::endl;
        return;
    }

    std::string name = p.filename();
    if (params.size() == 2) {
        name = params[1];
    }
    name = name.substr(0, name.find_last_of('.'));

    if (FromResolver::find(global_tables, name, false)) {
        std::cout << "Table " << color(name, RED_FG) << " already exists." << std::endl;
        return;
    }


    try {
        table* t = time_it(DBHelper::fromCSV_Unchecked(p));
        global_tables[{name}] = t;
        std::cout << "Loaded: " << color(p, GREEN_FG) << ", as: " << color(name, YELLOW_FG) << std::endl;
        show_details({name});
    } catch (const std::exception& e) {
        std::cout << "Error loading: " << e.what() << std::endl;
    }
}

void save_table(std::vector<std::string> params) {
    if (params.size() != 2) {
        std::cout << "Usage: save [name] [path]" << std::endl;
        return;
    }

    const std::string name = params[0];
    const std::string& path_str = params[1];

    fs::path p(path_str);

    if (fs::exists(p)) {
        std::cout << "File already exists" << std::endl;
        return;
    }

    if (FromResolver::find(global_tables, name, false) == nullptr) {
        std::cout << "Table " << color(name, RED_FG) << " not found." << std::endl;
        return;
    }


    auto t = FromResolver::find(global_tables, name, false);

    try {
        DBHelper::toCSV(t, p);
        std::cout << "Saved: " << color(name, YELLOW_FG) << ", To: " << color(p, GREEN_FG) << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Error saving: " << e.what() << std::endl;
    }
}

void show_table(std::vector<std::string> params) {
    if (params.empty() || params.size() > 3) {
        std::cout << "Usage: show [table] [start-row] [count]" << std::endl;
        return;
    }

    const auto& name = params[0];
    auto start = 0;

    // print the header
    std::vector<size_t> w;

    if (FromResolver::find(global_tables, name, false) == nullptr) {
        std::cout << "Table " << color(name, RED_FG) << " not found." << std::endl;
        return;
    }

    auto table = FromResolver::find(global_tables, name, false);

    auto count = table->size();

    if (params.size() == 3) {
        start = stoi(params[1]);
        count = stoi(params[2]);
    }

    if (table->headers.empty()) count = 0;

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
            case DateTime:
                w.push_back(30);
        }

        count = std::min(count, static_cast<size_t>(table->columns[i]->data.size()) - start);

        std::cout << "";
        std::cout << std::setw(w[i]) << std::left << color(table->headers[i], GREEN_FG) << "|";
    }

    std::cout << std::endl;

    for (int i = start; i < start + count; i++) {
        for (int k = 0;k < table->headers.size();k++) {
            if (table->columns[k]->nulls[i]) {
                std::cout << std::setw(w[k]) << std::left << color("null", RED_FG) << "|";
                continue;
            }

            switch (table->columns[k]->type) {
                case STRING:
                    std::cout << std::setw(w[k]) << std::left << color(StringUtils::limit(*table->columns[k]->data[i].s, 30), CYAN_FG) << "|";
                    break;
                case INTEGER:
                    std::cout << std::setw(w[k]) << std::left << color(std::to_string(table->columns[k]->data[i].i) + " ", CYAN_FG) << "|";
                    break;
                case FLOAT:
                    std::cout << std::setw(w[k]) << std::left << color(std::to_string(table->columns[k]->data[i].d) + " ", CYAN_FG) << "|";
                    break;
                case DateTime:
                    std::cout << std::setw(w[k]) << std::left << color(
                        std::to_string(table->columns[k]->data[i].t->year)   + "-" +
                            std::to_string(table->columns[k]->data[i].t->month)  + "-" +
                            std::to_string(table->columns[k]->data[i].t->day)    + " " +
                            std::to_string(table->columns[k]->data[i].t->hour)   + ":" +
                            std::to_string(table->columns[k]->data[i].t->minute) + ":" +
                            std::to_string(table->columns[k]->data[i].t->second)
                            + " ", CYAN_FG) << "|";
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

    if (FromResolver::find(global_tables, name, false) == nullptr) {
        std::cout << "Table " << color(name, RED_FG) << " not found." << std::endl;
        return;
    }

    auto t = FromResolver::find(global_tables, name, false);

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

    if (FromResolver::find(global_tables, name, false) == nullptr) {
        std::cout << "Table " << color(name, RED_FG) << " not found." << std::endl;
        return;
    }


    auto t = FromResolver::find(global_tables, name, false);

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

    if (params[0] == "Acc") {
        try {
            Cfg::useAccelerator = std::stoi(params[1]);
            std::cout << "Updated: " << color("Cfg::useAccelerator", CYAN_FG) << " = " << Cfg::useAccelerator << std::endl;
            return;
        } catch (std::exception& e) {
            std::cout << "Error modifying configuration: " << e.what() << std::endl;
        }
    }

    if (params[0] == "bd") {
        try {
            Cfg::BlockDim = std::stoi(params[1]);
            std::cout << "Updated: " << color("Cfg::BlockDim", CYAN_FG) << " = " << Cfg::BlockDim << std::endl;
            return;
        } catch (std::exception& e) {
            std::cout << "Error modifying configuration: " << e.what() << std::endl;
        }
    }

    if (params[0] == "bd2") {
        try {
            Cfg::BlockDim2D = std::stoi(params[1]);
            std::cout << "Updated: " << color("Cfg::BlockDim2D", CYAN_FG) << " = " << Cfg::BlockDim2D << std::endl;
            return;
        } catch (std::exception& e) {
            std::cout << "Error modifying configuration: " << e.what() << std::endl;
        }
    }

    if (params[0] == "mGM") {
        try {
            Cfg::maxGPUMemory = std::stoi(params[1]);
            std::cout << "Updated: " << color("Cfg::maxGPUMemory", CYAN_FG) << " = " << Cfg::maxGPUMemory << std::endl;
            return;
        } catch (std::exception& e) {
            std::cout << "Error modifying configuration: " << e.what() << std::endl;
        }
    }

    if (params[0] == "rIMS") {
        try {
            Cfg::radixIntegerMaskSize = std::stoi(params[1]);
            std::cout << "Updated: " << color("Cfg::radixIntegerMaskSize", CYAN_FG) << " = " << Cfg::radixIntegerMaskSize << std::endl;
            return;
        } catch (std::exception& e) {
            std::cout << "Error modifying configuration: " << e.what() << std::endl;
        }
    }

    if (params[0] == "nS") {
        try {
            Cfg::numStreams = std::stoi(params[1]);
            std::cout << "Updated: " << color("Cfg::numStreams", CYAN_FG) << " = " << Cfg::numStreams << std::endl;
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

    hsql::SQLParserResult parser_result;
    hsql::SQLParser::parse(query, &parser_result);
    try {
        std::vector<table*> r_vec;
        if (Cfg::useAccelerator) {
            r_vec = time_it(GPUExecutor::executeQuery(parser_result, global_tables));
        } else {
            r_vec = time_it(CPUExecutor::executeQuery(parser_result, global_tables));
        }
        for (auto t: r_vec) {
            std::cout << "Result: " << std::endl;
            std::cout << table::details(t) << std::endl;

            int i = 0;
            while (FromResolver::find(global_tables, "result_" + std::to_string(i), false)) {
                i++;
            }

            global_tables[{"result_" + std::to_string(i)}] = t;
            std::cout << "Result saved on table: " << color("result_" + std::to_string(i), GREEN_FG) << std::endl;
            std::cout << "First 10 rows:" << std::endl;
            show_table({"result_" + std::to_string(i), "0", "10"});
        }
    } catch (const std::exception& e) {
        std::cout << "Query failed: " << color(e.what(), RED_FG) << std::endl;
    }
}

void dummy(std::vector<std::string> params) {
    for (auto name: params) {

        if (FromResolver::find(global_tables, name, false)) {
            std::cout << "[" << color(name,  RED_FG) <<  "] table with the same name already exists" << std::endl;
            continue;
        }
        global_tables[{name}] = new table();
        std::cout << "Loaded: " << color("data/empty.csv", GREEN_FG) << ", as: " << color(name, YELLOW_FG) << std::endl;
    }
}

void editor(std::vector<std::string> params) {
    std::string result = NanoEditor::edit();
    sql({result});
}


int main(int argc, char** argv) {
    if (argc == 1) {
        global_tables.clear();

        std::cout << "gsql++ running." << std::endl;
        std::cout << "use load / add [path] to load a csv file as table," << std::endl;
        std::cout << "use remove [name] to remove a table," << std::endl;
        std::cout << "use details [name] / [names] to show table details" << std::endl;
        std::cout << "or enter a SQL query" << std::endl;

        CLI cli(sql);
        cli.addCommand("load", load_table);
        cli.addCommand("add", load_table);
        cli.addCommand("save", save_table);
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

    if (argc != 3) {
        std::cout << "Usage: ./main [data-folder] [query-file]";
        return 0;
    }

    std::string data_folder = argv[1];
    std::string query_file  = argv[2];

    std::string query = FileUtils::fileAsString(query_file);

    hsql::SQLParserResult parser_result;
    hsql::SQLParser::parse(query, &parser_result);

    if (!parser_result.isValid()) {
        std::cerr << "Invalid SQL query" << std::endl;
        return 0;
    }

    auto inputs = FromResolver::resolveQueryNeeds(parser_result);
    global_tables.clear();

    io::disableOutput();

    try {
        fs::path data_path(data_folder);
        for (const auto& table: inputs) {
            load_table({(data_path / fs::path(table + ".csv")).string()});
        }

        std::vector<table*> r_vec;

        r_vec = time_it(GPUExecutor::executeQuery(parser_result, global_tables));

        if (r_vec.size() > 1) {
            for (size_t i = 0; i < r_vec.size(); i++) {

            }
        } else {
            auto output_path = data_folder / fs::path("team6.csv");
            global_tables[{"__team_6__query_result__"}] = r_vec[0];
            if (fs::exists(output_path)) {
                fs::remove_all(output_path);
            }
            save_table({"__team_6__query_result__", output_path.string()});
        }
    } catch (std::exception& e) {
        std::cerr << "Query failed: " + std::string(e.what()) << std::endl;
    } catch (...) {
        std::cerr << "Query failed: Unknown error" << std::endl;
    }
    return 0;
}