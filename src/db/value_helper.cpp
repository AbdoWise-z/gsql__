//
// Created by xabdomo on 4/13/25.
//

#include "value_helper.hpp"

#include <string>
#include <regex>
#include <optional>

namespace ValuesHelper {
    dateTime DefaultDateTimeValue = {
        .year = 0,
        .month = 0,
        .day = 0,
        .hour = 0,
        .minute = 0,
        .second = 0
    };
    int64_t  DefaultIntegerValue = 0;
    double   DefaultFloatValue   = 0;

    std::map<std::pair<DataType, DataType>, DataType> conversionMap = {
        {{INTEGER, INTEGER}, INTEGER},
        {{INTEGER, FLOAT}, FLOAT},
        {{INTEGER, DateTime}, DateTime},
        {{INTEGER, STRING}, STRING},

        {{FLOAT, INTEGER}, FLOAT},
        {{FLOAT, FLOAT}, FLOAT},
        {{FLOAT, DateTime}, INTEGER},
        {{FLOAT, STRING}, STRING},

        {{DateTime, INTEGER}, INTEGER},
        {{DateTime, FLOAT}, INTEGER},
        {{DateTime, DateTime}, DateTime},
        {{DateTime, STRING}, STRING},

        {{STRING, INTEGER}, STRING},
        {{STRING, FLOAT}, STRING},
        {{STRING, DateTime}, STRING},
        {{STRING, STRING}, STRING},
    };
}

int ValuesHelper::cmp(const int64_t& a, const int64_t& b) {
    return a > b ? 1 : (a < b ? -1 : 0);
}

int ValuesHelper::cmp(const double& a, const double& b) {
    return a > b ? 1 : (a < b ? -1 : 0);
}

int ValuesHelper::cmp(const char* a, const char* b) {
    return strcmp(a, b);
}

int ValuesHelper::cmp(const dateTime &a, const dateTime &b) {
    int64_t a_v = dateTimeToInt(a);
    int64_t b_v = dateTimeToInt(b);
    return 1 * (a_v > b_v) + -1 * (a_v < b_v);
}

int ValuesHelper::cmp(const tval& a, const tval& b, const DataType& t) {
    switch (t) {
        case STRING:
            return strcmp(a.s->c_str(), b.s->c_str());
        case INTEGER:
            return a.i > b.i ? 1 : (a.i < b.i ? -1 : 0);
        case FLOAT:
            return a.d > b.d ? 1 : (a.d < b.d ? -1 : 0);
        case DateTime:
            if (a.t->year - b.t->year) return a.t->year - b.t->year;
            if (a.t->month - b.t->month) return a.t->month - b.t->month;
            if (a.t->day - b.t->day) return a.t->day - b.t->day;
            if (a.t->hour - b.t->hour) return a.t->hour - b.t->hour;
            if (a.t->minute - b.t->minute) return a.t->minute - b.t->minute;
            return a.t->second - b.t->second;
    }

    return 0;
}

size_t ValuesHelper::sizeOf(const tval &v, const DataType &t) {
    switch (t) {
        case STRING:
            return v.s->length();
        case INTEGER:
            return sizeof(int64_t);
        case FLOAT:
            return sizeof(double);
        case DateTime:
            return sizeof(dateTime);
    }

    return 0;
}

tval ValuesHelper::copy(const tval v, const DataType& t) {
    tval result{};

    if (t == STRING) {
        result.s = new std::string(*v.s);
    } else if (t == INTEGER) {
        result.i = static_cast<int64_t>(v.i);
    } else if (t == FLOAT) {
        result.d = static_cast<double>(v.d);
    } else {
        result.t = new dateTime(*v.t);
    }

    return result;
}

void ValuesHelper::deleteValue(tval v, DataType t) {
    if (t == STRING) delete v.s;
    if (t == DateTime) delete v.t;

    v.s = nullptr; // v.t == nullptr;
}

std::string ValuesHelper::to_string(tval v, DataType t) {
    switch (t) {
        case STRING:
            return *v.s;
        case INTEGER:
            return std::to_string(v.i);
        case FLOAT:
            return std::to_string(v.d);
        case DateTime:
            return  std::to_string(v.t->year)   + "-" +
                    std::to_string(v.t->month)  + "-" +
                    std::to_string(v.t->day)    + " " +
                    std::to_string(v.t->hour)   + ":" +
                    std::to_string(v.t->minute) + ":" +
                    std::to_string(v.t->second);
    }

    return "__non_type__";
}

std::optional<dateTime> ValuesHelper::parseDateTimeDateOnly(const std::string &input) {
    static const std::regex re(
            R"(^\s*([0-9]{4})-([0-1]?[0-9])-([0-3]?[0-9])\s*$)"
        );

    std::smatch m;
    if (!std::regex_match(input, m, re)) {
        return std::nullopt;
    }

    // Convert captured strings to integers
    int y = std::stoi(m[1].str());
    int M = std::stoi(m[2].str());
    int d = std::stoi(m[3].str());
    int h   = 0;
    int mnt = 0;
    int s   = 0;

    // Validate ranges more strictly if desired
    if (M < 1 || M > 12 || d < 1 || d > 31)
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

std::optional<dateTime> ValuesHelper::parseDateTimeTimeOnly(const std::string &input) {
    // Regex capturing groups: year, month, day, hour, minute, second
    static const std::regex re(
        R"(^\s*([0-2]?[0-9]):([0-5]?[0-9]):([0-5]?[0-9])\s*$)"
    );

    std::smatch m;
    if (!std::regex_match(input, m, re)) {
        return std::nullopt;
    }

    // Convert captured strings to integers
    int y = 1970;
    int M = 1;
    int d = 1;
    int h = std::stoi(m[1].str());
    int mnt = std::stoi(m[2].str());
    int s = std::stoi(m[3].str());

    // Validate ranges more strictly if desired
    if (h > 23 || mnt > 59 || s > 59)
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

std::optional<dateTime> ValuesHelper::parseDateTime(const std::string &input, bool strict) {
    // Regex capturing groups: year, month, day, hour, minute, second
    static const std::regex re(
        R"(^\s*([0-9]{4})-([0-1]?[0-9])-([0-3]?[0-9])\s+([0-2]?[0-9]):([0-5]?[0-9]):([0-5]?[0-9])\s*$)"
    );

    std::smatch m;
    if (!std::regex_match(input, m, re)) {
        if (!strict) {
            auto result = parseDateTimeDateOnly(input);
            if (result != std::nullopt)
                return *result;

            result = parseDateTimeTimeOnly(input);
            if (result != std::nullopt)
                return *result;

        }
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

static std::pair<std::pair<tval, tval>, DataType> resolveBinaryOp(hsql::Expr * literal) {
    auto inner1 = ValuesHelper::getLiteralFrom(literal->expr);
    auto inner2 = ValuesHelper::getLiteralFrom(literal->expr2);
    auto common = ValuesHelper::conversionMap[{inner1.second, inner2.second}];
    if (common != inner1.second) {
        const auto temp = inner1;
        inner1.first = ValuesHelper::castTo(temp.first, inner1.second, common);
        ValuesHelper::deleteValue(temp.first, temp.second);
        inner1.second = common;
    }

    if (common != inner2.second) {
        const auto temp = inner2;
        inner2.first = ValuesHelper::castTo(temp.first, inner2.second, common);
        ValuesHelper::deleteValue(temp.first, temp.second);
        inner2.second = common;
    }

    return {{inner1.first, inner2.first}, common};
}

std::pair<tval, DataType> ValuesHelper::getLiteralFrom(hsql::Expr * literal) {
    tval literal_v {};
    DataType literal_t = STRING;
    if (literal->type == hsql::ExprType::kExprLiteralString) {
        auto lt = ValuesHelper::parseDateTime(literal->name, false);
        if (lt != std::nullopt) {
            literal_v = ValuesHelper::create_from(*lt);
            literal_t = DateTime;
        } else {
            literal_v = ValuesHelper::create_from(literal->name);
            literal_t = STRING;
        }
    } else if (literal->type == hsql::ExprType::kExprLiteralInt) {
        literal_t = INTEGER;
        literal_v = ValuesHelper::create_from(literal->ival);
    } else if (literal->type == hsql::ExprType::kExprLiteralFloat) {
        literal_t = FLOAT;
        literal_v = ValuesHelper::create_from(literal->fval);
    } else if (literal->type == hsql::kExprOperator) {
        if (literal->opType == hsql::kOpUnaryMinus) {
            auto inner = getLiteralFrom(literal->expr);
            auto result = neg(inner.first, inner.second);
            ValuesHelper::deleteValue(inner.first, inner.second);
            literal_v = result;
            literal_t = inner.second;
        } else if (literal->opType == hsql::kOpPlus) {
            auto inner = resolveBinaryOp(literal);
            auto res = add(inner.first.first, inner.first.second, inner.second);
            ValuesHelper::deleteValue(inner.first.first, inner.second);
            ValuesHelper::deleteValue(inner.first.second, inner.second);
            return {res, inner.second};
        } else if (literal->opType == hsql::kOpMinus) {
            auto inner = resolveBinaryOp(literal);
            auto res = sub(inner.first.first, inner.first.second, inner.second);
            ValuesHelper::deleteValue(inner.first.first, inner.second);
            ValuesHelper::deleteValue(inner.first.second, inner.second);
            return {res, inner.second};
        } else if (literal->opType == hsql::kOpAsterisk) {
            auto inner = resolveBinaryOp(literal);
            auto res = mul(inner.first.first, inner.first.second, inner.second);
            ValuesHelper::deleteValue(inner.first.first, inner.second);
            ValuesHelper::deleteValue(inner.first.second, inner.second);
            return {res, inner.second};
        } else if (literal->opType == hsql::kOpSlash) {
            auto inner = resolveBinaryOp(literal);
            auto res = div(inner.first.first, inner.first.second, inner.second);
            ValuesHelper::deleteValue(inner.first.first, inner.second);
            ValuesHelper::deleteValue(inner.first.second, inner.second);
            return {res, inner.second};
        } else {
            throw UnsupportedOperatorError(literal->getName());
        }
    } else {
        throw UnsupportedLiteralError();
    }

    return {literal_v, literal_t};
}

tval ValuesHelper::add(tval a, tval b, DataType t) {
    tval result = {};
    switch (t) {
        case INTEGER:
            result.i = a.i + b.i;
            break;
        case FLOAT:
            result.d = a.d + b.d;
            break;
        case DateTime:
            result.t = new dateTime{};
            result.t->year = a.t->year + b.t->year;
            result.t->month = a.t->month + b.t->month;
            result.t->day = a.t->day + b.t->day;
            result.t->hour = a.t->hour + b.t->hour;
            result.t->minute = a.t->minute + b.t->minute;
            result.t->second = a.t->second + b.t->second;
            break;
        case STRING:
            result.s = new std::string(*a.s + *b.s);
            break;
    }

    return result;
}

tval ValuesHelper::sub(tval a, tval b, DataType t) {
    tval result = {};
    switch (t) {
        case INTEGER:
            result.i = a.i - b.i;
            break;
        case FLOAT:
            result.d = a.d - b.d;
            break;
        case DateTime:
            result.t = new dateTime{};
            result.t->year = a.t->year - b.t->year;
            result.t->month = a.t->month - b.t->month;
            result.t->day = a.t->day - b.t->day;
            result.t->hour = a.t->hour - b.t->hour;
            result.t->minute = a.t->minute - b.t->minute;
            result.t->second = a.t->second - b.t->second;
            break;
        case STRING:
            throw UnsupportedOperatorError(" string - string is not a thing. ");
            break;
    }

    return result;
}


tval ValuesHelper::mul(tval a, tval b, DataType t) {
    tval result = {};
    switch (t) {
        case INTEGER:
            result.i = a.i * b.i;
            break;
        case FLOAT:
            result.d = a.d * b.d;
            break;
        case DateTime:
            result.t = new dateTime{};
            result.t->year = a.t->year * b.t->year;
            result.t->month = a.t->month * b.t->month;
            result.t->day = a.t->day * b.t->day;
            result.t->hour = a.t->hour * b.t->hour;
            result.t->minute = a.t->minute * b.t->minute;
            result.t->second = a.t->second * b.t->second;
            break;
        case STRING:
            throw UnsupportedOperatorError(" string * string is not a thing. ");
            break;
    }

    return result;
}

tval ValuesHelper::div(tval a, tval b, DataType t) {
    tval result = {};
    switch (t) {
        case INTEGER:
            result.i = a.i / b.i;
            break;
        case FLOAT:
            result.d = a.d / b.d;
            break;
        case DateTime:
            result.t = new dateTime{};
            result.t->year = a.t->year / b.t->year;
            result.t->month = a.t->month / b.t->month;
            result.t->day = a.t->day / b.t->day;
            result.t->hour = a.t->hour / b.t->hour;
            result.t->minute = a.t->minute / b.t->minute;
            result.t->second = a.t->second / b.t->second;
            break;
        case STRING:
            throw UnsupportedOperatorError(" string / string is not a thing. ");
            break;
    }

    return result;
}

tval ValuesHelper::neg(tval a, DataType t) {
    tval result = {};
    switch (t) {
        case INTEGER:
            result.i = - a.i;
        break;
        case FLOAT:
            result.d = - a.d;
        break;
        case DateTime:
            result.t = new dateTime{};
        result.t->year = -a.t->year;
        result.t->month = -a.t->month;
        result.t->day = -a.t->day;
        result.t->hour = -a.t->hour;
        result.t->minute = -a.t->minute;
        result.t->second = -a.t->second;
        break;
        case STRING:
            throw UnsupportedOperatorError(" - string is not a thing. ");
        break;
    }

    return result;
}

bool ValuesHelper::isZero(tval a, DataType t) {
    switch (t) {
        case INTEGER:
            return a.i == 0;
        case FLOAT:
            return a.d == 0.0;
        case DateTime:
            return dateTimeToInt(*a.t) == 0;
        case STRING:
            return a.s->size() == 0;
    }

    return true;
}


