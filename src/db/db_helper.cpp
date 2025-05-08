//
// Created by xabdomo on 4/13/25.
//

#include "db_helper.hpp"

#include <csv.hpp>
#include <iomanip>
#include <hsql/sql/ColumnType.h>
#include <omp.h>
#include <regex>

#include "utils/string_utils.hpp"


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


inline std::pair<std::string, std::vector<char>> strip_tags(std::string input) {
    std::vector<char> tags;

    // Regular expression to match trailing (x) groups
    std::regex tag_regex(R"(\s*\(([a-zA-Z0-9])\)\s*$)");
    std::smatch match;

    // Keep removing tags from the end
    while (std::regex_search(input, match, tag_regex)) {
        tags.push_back(match[1].str()[0]);  // Extract the character inside ()
        input = StringUtils::trim(input.substr(0, match.position()));  // Remove the match from the end
    }

    // Reverse tags to preserve original order
    std::reverse(tags.begin(), tags.end());

    return {input, tags};
}


// table * DBHelper::fromCSV(std::string path) {
//     csv::CSVReader csv(path);
//     auto headers = csv.get_col_names();
//     for (auto & header : headers) {
//         header = fixHeaderName(header);
//     }
//
//     auto table   = new ::table();
//
//     bool first_row = true;
//     for (const auto& row : csv) {
//         std::vector<tval> values;
//         if (first_row) {
//             std::vector<DataType> types = inferTypes(row);
//             first_row = false;
//             table->setHeaders(headers, types);
//         }
//
//         std::optional<dateTime> dt;
//
//         for (auto field: row) {
//             tval val{};
//             switch (field.type()) {
//                 case csv::DataType::CSV_STRING:
//                     dt = ValuesHelper::parseDateTime(field.get<std::string>());
//                     if (dt == std::nullopt)
//                         val.s = new std::string(field.get<std::string>());
//                     else
//                         val.t = new dateTime(dt.value());
//                     break;
//                 case csv::DataType::CSV_INT8:
//                 case csv::DataType::CSV_INT16:
//                 case csv::DataType::CSV_INT32:
//                 case csv::DataType::CSV_INT64:
//                     val.i = field.get<int64_t>();
//                     break;
//                 case csv::DataType::UNKNOWN:
//                 case csv::DataType::CSV_DOUBLE: // fixme: I just assume it's double ..
//                     val.d = field.get<double>();
//                     break;
//                 default:
//                     throw std::runtime_error("Not implemented");
//             }
//             values.push_back(val);
//         }
//
//         table->addRecord(values);
//     }
//
//     return table;
// }


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


    auto fields = parseCSVLine(row);

    for (const auto& field: fields) {
        try {
            auto dt_val = ValuesHelper::parseDateTime(field);
            if (dt_val != std::nullopt) {
                types.push_back(DateTime);
                continue;
            }
        } catch (...) {}

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

        types.push_back(STRING);
    }

    return types;
}

table * DBHelper::fromCSV_Unchecked(const std::string& path) {
    auto table   = new ::table();
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + path);
    }

    std::vector<std::string> lines;


    std::string line;
    while (std::getline(file, line)) {
        // Remove trailing '\r' if present (handles Windows \r\n) (fk windows btw)
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (line.empty()) continue;
        lines.push_back(line);
    }

    if (lines.empty()) {
        return table;
    }

    std::vector<DataType> types;
    auto headers = parseCSVLine(lines[0]);
    for (auto & header : headers) {
        auto fixed_header = strip_tags(header);
        header = fixed_header.first;

        DataType dt;
        for (char c : fixed_header.second) {
            if (c == 'N') dt = FLOAT;
            if (c == 'd') dt = DateTime;
            if (c == 'T') dt = STRING;
        }

        types.push_back(dt);
    }


    // we love democracy (we are lazy to actually do it properly)
    if (lines.size() > 1) {
        int sampleIdx = 0;
        std::vector<std::vector<int>> votes = std::vector<std::vector<int>>(types.size(), std::vector<int>(4, 0));
        while (sampleIdx < 20) {
            auto l = lines[random() % lines.size() + 1];
            auto _t = inferTypes(l);
            if (_t.size() == types.size()) {
                for (size_t i = 0; i < types.size(); ++i) {
                    votes[i][_t[i]]++;
                }
            }
            sampleIdx++;
        }

        for (size_t i = 0; i < types.size(); ++i) {
            auto _idx = 0;
            for (size_t j = 0;j < votes[i].size(); ++j) {
                if (votes[i][j] >= votes[i][_idx]) _idx = j;
            }

            types[i] = static_cast<DataType>(_idx);
        }
    }

    table->setHeaders(headers, types);

    for (int i = 0;i < types.size();i++) {
        // pre-reserve anything that we need
        table->columns[i]->data  = std::vector<tval>(lines.size() - 1, {nullptr});
        table->columns[i]->nulls = std::vector<char>(lines.size() - 1, 0);
    }

    int nThreads = omp_get_max_threads();
    std::vector localNulls(nThreads, std::vector<std::int64_t>(table->columns.size(), 0));
    // shared(lines, types, table, path, ValuesHelper::DefaultIntegerValue, ValuesHelper::DefaultFloatValue, ValuesHelper::DefaultDateTimeValue)
    #pragma omp parallel for schedule(static)
    for (size_t i = 1; i < lines.size(); i++) {
        int tid = omp_get_thread_num();
        auto vals = parseCSVLine(lines[i]);

        if (types.size() != vals.size()) {
            #pragma omp critical
            {
                throw std::runtime_error("Wrong number of columns in file: " + path + ", line " + std::to_string(i));
            }
        }

        for (size_t j = 0; j < types.size(); j++) {
            tval value{};
            char nil = 0;
            switch (types[j]) {
                case INTEGER:
                    try {
                        value = ValuesHelper::create_from(static_cast<int64_t>(std::stoll(vals[j])));
                    } catch (...) {
                        value = ValuesHelper::create_from(ValuesHelper::DefaultIntegerValue);
                        nil = 1;
                    }
                    break;
                case FLOAT:
                    try {
                        value = ValuesHelper::create_from(std::stod(vals[j]));
                    } catch (...) {
                        value = ValuesHelper::create_from(ValuesHelper::DefaultFloatValue);
                        nil = 1;
                    }
                    break;
                case DateTime:
                    try {
                        value = ValuesHelper::create_from(ValuesHelper::parseDateTime(vals[j], false).value());
                    } catch (...) {
                        value = ValuesHelper::create_from(ValuesHelper::DefaultDateTimeValue);
                        nil = 1;
                    }
                    break;
                case STRING:
                    value = ValuesHelper::create_from(vals[j]);
                    break;
            }

            table->columns[j]->data[i - 1]  = value;
            table->columns[j]->nulls[i - 1] = nil;

            if (nil) {
                localNulls[tid][j]++;
            }
        }
    }

    for (int t = 0; t < nThreads; ++t) {
        for (size_t j = 0; j < table->columns.size(); ++j) {
            table->columns[j]->nullsCount += localNulls[t][j];
        }
    }

    return table;
}

static std::string asString(const tval &value, DataType type, bool isNull)
{
    std::string result;

    if (isNull) return "";

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
            std::string value = asString(cols[colIdx]->data[rowIdx], type, cols[colIdx]->nulls.size() > rowIdx ? cols[colIdx]->nulls[rowIdx] != 0 : false);
            out << value;

            if (colIdx < headers.size() - 1)
                out << ",";
        }
        out << "\n";
    }

    out.close();

    return true;
}

