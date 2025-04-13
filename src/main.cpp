#include <iostream>
#include "hsql/SQLParser.h"
#include "csv.hpp"

int main() {
    csv::CSVReader reader("data/Authors.csv");
    std::vector<std::string> headers = reader.get_col_names();
    std::cout << "Headers:\n";
    for (const auto& h : headers)
        std::cout << " - " << h << "\n";

    // Iterate through rows
    for (csv::CSVRow& row : reader) {
        for (csv::CSVField& field : row) {
            std::cout << field.get<>() << " | ";
        }
        std::cout << "\n";
    }

    return 0;
}