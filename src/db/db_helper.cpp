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


inline std::vector<std::string> parseCSVLine(const std::string& line) {
    std::vector<std::string> result;
    std::string field;
    bool inQuotes = false;

    for (size_t i = 0; i < line.length(); ++i) {
        char c = line[i];

        if (inQuotes) {
            if (c == '"') {
                if (i + 1 < line.length() && line[i + 1] == '"') {
                    field += '"';  // Escaped quote
                    ++i;
                } else {
                    inQuotes = false;  // End of quoted field
                }
            } else {
                field += c;
            }
        } else {
            if (c == '"') {
                inQuotes = true;
            } else if (c == ',') {
                result.push_back(field);
                field.clear();
            } else {
                field += c;
            }
        }
    }
    result.push_back(field);  // Add last field
    return result;
}


static std::vector<DataType> inferTypes(std::string row) {
    std::vector<DataType> types;
    std::optional<dateTime> dt;


    auto fields = parseCSVLine(row);

    for (const auto& field: fields) {
        try {
            if (field.find(".") == field.npos) {
                auto int_val = std::stoll(field);
                types.push_back(INTEGER);
                continue;
            }
        } catch (...) {}

        try {
            auto double_val = std::stod(field);
            types.push_back(FLOAT);
            continue;
        } catch (...) {}

        try {
            auto dt_val = ValuesHelper::parseDateTime(field);
            if (dt_val != std::nullopt) {
                types.push_back(DateTime);
                continue;
            }
        } catch (...) {}

        types.push_back(STRING);
    }

    return types;
}


table * DBHelper::fromCSV_Unchecked(std::string path) {
    auto table   = new ::table();
    std::vector<std::string> lines;
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + path);
    }

    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    if (lines.empty()) {
        return table;
    }

    table->headers = parseCSVLine(lines[0]);
    std::vector<DataType> types;
    for (int i = 0;i < table->headers.size();i++) {
        types.push_back(STRING);

    }

    if (lines.size() > 1) {
        types = inferTypes(lines[1]);
    }

    for (auto type: types) {
        table->columns.push_back(new column());
        // pre - reserve anything that we need
        table->columns[table->columns.size() - 1]->type = type;
        table->columns[table->columns.size() - 1]->data = std::vector<tval>(lines.size() - 1, {nullptr});
    }

    #pragma omp parallel for default(none) shared(lines, types, table, path, ValuesHelper::DefaultIntegerValue, ValuesHelper::DefaultFloatValue, ValuesHelper::DefaultDateTimeValue) schedule(static)
    for (size_t i = 1; i < lines.size(); i++) {
        auto vals = parseCSVLine(lines[i]);

        if (types.size() != vals.size()) {
            #pragma omp critical
            {
                throw std::runtime_error("Wrong number of columns in file: " + path + ", line " + std::to_string(i));
            }
        }

        for (size_t j = 0; j < types.size(); j++) {
            tval value{};
            switch (types[j]) {
                case INTEGER:
                    try {
                        value = ValuesHelper::create_from(static_cast<int64_t>(std::stoll(vals[j])));
                    } catch (...) {
                        value = ValuesHelper::create_from(ValuesHelper::DefaultIntegerValue);
                    }
                    break;
                case FLOAT:
                    try {
                        value = ValuesHelper::create_from(std::stod(vals[j]));
                    } catch (...) {
                        value = ValuesHelper::create_from(ValuesHelper::DefaultFloatValue);
                    }
                    break;
                case DateTime:
                    try {
                        value = ValuesHelper::create_from(ValuesHelper::parseDateTime(vals[j]).value());
                    } catch (...) {
                        value = ValuesHelper::create_from(ValuesHelper::DefaultDateTimeValue);
                    }
                    break;
                case STRING:
                    value = ValuesHelper::create_from(vals[j]);
                    break;
            }

            table->columns[j]->data[i - 1] = value;
        }
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
