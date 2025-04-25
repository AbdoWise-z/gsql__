//
// Created by xabdomo on 4/13/25.
//

#include "db_helper.hpp"

#include <csv.hpp>
#include <hsql/sql/ColumnType.h>
#include <regex>
#include <optional>

std::optional<dateTime> parseDateTime(const std::string& input) {
    // Regex capturing groups: year, month, day, hour, minute, second
    static const std::regex re(
        R"(^([0-9]{4})-([0-1][0-9])\-([0-3][0-9])\s+([0-2][0-9]):([0-5][0-9]):([0-5][0-9])$)"
    );

    std::smatch m;
    if (!std::regex_match(input, m, re)) {
        return std::nullopt;
    }

    // Convert captured strings to integers
    int y = std::stoi(m[1].str());
    int M = std::stoi(m[2].str());
    int d = std::stoi(m[3].str());
    int h = std::stoi(m[4].str());
    int mnt = std::stoi(m[5].str());
    int s = std::stoi(m[6].str());

    // Validate ranges more strictly if desired
    if (M < 1 || M > 12 || d < 1 || d > 31 ||
        h > 23 || mnt > 59 || s > 59)
    {
        return std::nullopt;
    }

    dateTime dt{
        static_cast<uint16_t>(y),
        static_cast<uint16_t>(M),
        static_cast<uint16_t>(d),
        static_cast<uint8_t>(h),
        static_cast<uint8_t>(mnt),
        static_cast<uint8_t>(s)
    };

    return dt;
}


static std::vector<DataType> inferTypes(const csv::CSVRow& row) {
    std::vector<DataType> types;
    std::optional<dateTime> dt;

    for (auto field: row) {
        switch (field.type()) {
            case csv::DataType::CSV_STRING:
                dt = parseDateTime(field.get<std::string>());
                if (dt != std::nullopt)
                    types.push_back(DateTime);
                else
                    types.push_back(STRING);
                break;
            case csv::DataType::CSV_INT8:
            case csv::DataType::CSV_INT16:
            case csv::DataType::CSV_INT32:
            case csv::DataType::CSV_INT64:
                types.push_back(INTEGER);
                break;
            case csv::DataType::CSV_DOUBLE:
                types.push_back(FLOAT);

            default:
                throw std::runtime_error("Not implemented");
        }
    }

    return types;
}

table * fromCSV(std::string path) {
    csv::CSVReader csv(path);
    auto headers = csv.get_col_names();
    auto headers_fixed = std::vector<std::string>(headers.size());
    for (int i = 0;i < headers.size();i++) {
        headers_fixed[i] = headers[i];
        if (headers_fixed[i].ends_with("(P)")) {
            headers_fixed[i] = headers[i].substr(0, headers[i].find_last_of(" (P)"));
        }
    }

    auto table   = new ::table();

    bool first_row = true;
    for (const auto& row : csv) {
        std::vector<tval> values;
        if (first_row) {
            std::vector<DataType> types = inferTypes(row);
            first_row = false;
            table->setHeaders(headers, types);
        }

        std::optional<dateTime> dt;

        for (auto field: row) {
            tval val{};
            switch (field.type()) {
                case csv::DataType::CSV_STRING:
                    dt = parseDateTime(field.get<std::string>());
                    if (dt == std::nullopt)
                        val.s = new std::string(field.get<std::string>());
                    else
                        val.t = new dateTime(dt.value());
                    break;
                case csv::DataType::CSV_INT8:
                case csv::DataType::CSV_INT16:
                case csv::DataType::CSV_INT32:
                case csv::DataType::CSV_INT64:
                    val.i = field.get<int64_t>();
                    break;
                case csv::DataType::CSV_DOUBLE:
                    val.d = field.get<double>();
                    break;
                default:
                    throw std::runtime_error("Not implemented");
            }
            values.push_back(val);
        }

        table->addRecord(values);
    }

    return table;
}
