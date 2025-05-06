//
// Created by xabdomo on 4/13/25.
//

#include "db_helper.hpp"

#include <csv.hpp>
#include <iomanip>
#include <hsql/sql/ColumnType.h>


static std::vector<DataType> inferTypes(const csv::CSVRow& row) {
    std::vector<DataType> types;
    std::optional<dateTime> dt;

    for (auto field: row) {
        switch (field.type()) {
            case csv::DataType::CSV_STRING:
                dt = ValuesHelper::parseDateTime(field.get<std::string>());
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
            case csv::DataType::UNKNOWN:
            case csv::DataType::CSV_DOUBLE:
                types.push_back(FLOAT);
                break;
            default:
                throw std::runtime_error("Not implemented");
        }
    }

    return types;
}

table * DBHelper::fromCSV(std::string path) {
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
                    dt = ValuesHelper::parseDateTime(field.get<std::string>());
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
                case csv::DataType::UNKNOWN:
                case csv::DataType::CSV_DOUBLE: // fixme: I just assume it's double ..
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

static std::string asString(const tval &value, DataType type)
{
    std::string result;

    switch (type)
    {
    case STRING:
        if (value.s != nullptr)
        {
            result = *(value.s);
        }
        break;

    case INTEGER:
        result = std::to_string(value.i);
        break;

    case FLOAT:
        result = std::to_string(value.d);
        break;

    case DateTime:
        if (value.t != nullptr)
        {
            dateTime &dt = *(value.t);
            // Format as YYYY-MM-DD HH:MM:SS
            std::ostringstream oss;
            oss << dt.year << "-"
                << std::setw(2) << std::setfill('0') << dt.month << "-"
                << std::setw(2) << std::setfill('0') << dt.day << " "
                << std::setw(2) << std::setfill('0') << static_cast<int>(dt.hour) << ":"
                << std::setw(2) << std::setfill('0') << static_cast<int>(dt.minute) << ":"
                << std::setw(2) << std::setfill('0') << static_cast<int>(dt.second);
            result = oss.str();
        }
        break;
    }

    return result;
}


bool DBHelper::toCSV(table *t, const std::string& path) {
    auto headers = t->headers;
    auto cols = t->columns;
    std::ofstream out(path);
    if (!out.is_open())
    {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    // Write headers
    for (size_t i = 0; i < headers.size(); ++i)
    {
        out << headers[i];
        if (i < headers.size() - 1)
            out << ",";
    }
    out << "\n";

    // Determine number of rows from the first column
    size_t rowCount = 0;
    if (!headers.empty() && !cols.empty())
    {
        rowCount = cols[0]->data.size();
    }

    for (size_t rowIdx = 0; rowIdx < rowCount; ++rowIdx){
        for (size_t colIdx = 0; colIdx < headers.size(); ++colIdx){
            auto type = cols[colIdx]->type;
            std::string value = asString(cols[colIdx]->data[rowIdx], type);
            out << value;

            if (colIdx < headers.size() - 1)
                out << ",";
        }
        out << "\n";
    }

    out.close();

    return true;
}
