//
// Created by xabdomo on 4/13/25.
//

#ifndef VALUE_HELPER_HPP
#define VALUE_HELPER_HPP

#include <cstdint>
#include <map>
#include <optional>
#include <string>

#include "typing.hpp"
#include "utils/murmur_hash3.hpp"

#include "constants.hpp"
#include "store.hpp"
#include "query/errors.hpp"




namespace ValuesHelper {
    int cmp(const int64_t& a, const int64_t& b);
    int cmp(const double& a, const double& b);
    int cmp(const char* a, const char* b);
    int cmp(const dateTime& a, const dateTime& b);

    int cmp(const tval& a, const tval& b, const DataType& t);

    size_t sizeOf(const tval& v, const DataType& t);

    tval copy(tval v, const DataType& t);

    extern dateTime DefaultDateTimeValue;
    extern int64_t  DefaultIntegerValue;
    extern double   DefaultFloatValue;

    extern std::map<std::pair<DataType, DataType>, DataType> conversionMap;

    inline uint64_t hash(const tval& v, const DataType& t) {
        return MurmurHash3_x64_64((t == STRING || t == DateTime) ? static_cast<const void*>(v.s) : static_cast<const void*>(&v), sizeOf(v, t), SEED);
    }

    inline tval create_from(const std::string& str) {
        tval res{};
        res.s = new std::string(str);
        return res;
    }

    inline tval create_from(int64_t i) {
        tval res{};
        res.i = i;
        return res;
    }


    inline tval create_from(double d) {
        tval res{};
        res.d = d;
        return res;
    }

    inline tval create_from(const dateTime dt) {
        tval res{};
        res.t = new dateTime(dt);
        return res;
    }

    inline int64_t dateTimeToInt(dateTime a) {
        return a.year * 33177600LL + a.month * 2764800LL + a.day * 86000LL + a.hour * 3600LL + a.minute * 60LL + a.second;
    }

    inline dateTime dateTimeFromInt(int64_t i) {
        dateTime dt{};
        dt.second = i % 60; i /= 60;
        dt.minute = i % 60; i /= 60;
        dt.hour   = i % 24; i /= 24;
        dt.day    = i % 32; i /= 32;
        dt.month  = i % 12; i /= 12;
        dt.year   = i;
        return dt;
    }

    void deleteValue(tval, DataType);

    std::string to_string(tval, DataType);

    std::optional<dateTime> parseDateTimeTimeOnly(const std::string& input);
    std::optional<dateTime> parseDateTimeDateOnly(const std::string& input);
    std::optional<dateTime> parseDateTime(const std::string& input, bool strict = true);

    inline tval castTo(tval in, DataType from, DataType to, bool strict = true) {
        switch (from) {
            case INTEGER:
                switch (to) {
                    case INTEGER:
                        return create_from(static_cast<int64_t>(in.i));
                    case FLOAT:
                        return create_from(static_cast<double>(in.i));
                    case DateTime:
                        return create_from(dateTimeFromInt(in.i));
                    case STRING:
                        return create_from(to_string(in, INTEGER));
                }
            case FLOAT:
                switch (to) {
                    case INTEGER:
                        return create_from(static_cast<int64_t>(in.d));
                    case FLOAT:
                        return create_from(static_cast<double>(in.d));
                    case DateTime:
                        return create_from(dateTimeFromInt(static_cast<int64_t>(in.d)));
                    case STRING:
                        return create_from(to_string(in, FLOAT));
                }
            case DateTime:
                switch (to) {
                    case INTEGER:
                        return create_from(static_cast<int64_t>(dateTimeToInt(*in.t)));
                    case FLOAT:
                        return create_from(static_cast<double>(dateTimeToInt(*in.t)));
                    case DateTime:
                        return create_from(*in.t);
                    case STRING:
                        return create_from(to_string(in, DateTime));
                }
            case STRING:
                switch (to) {
                    case INTEGER:
                        return create_from(static_cast<int64_t>(std::stoll(*in.s)));
                    case FLOAT:
                        return create_from(static_cast<double>(std::stod(*in.s)));
                    case DateTime:
                        return create_from(parseDateTime(*in.s, strict).value());
                    case STRING:
                        return create_from(*in.s);
                }
        }

        throw UnsupportedLiteralError();
    }

    inline int cmp(tval a, tval b, DataType a_t, DataType b_t) {
        auto common_type = conversionMap[{a_t, b_t}];
        tval a_v{ nullptr };
        tval b_v{ nullptr };
        if (common_type != a_t) {
            a_v = castTo(a, a_t, common_type, false);
        } else {
            a_v = a;
        }

        if (common_type != b_t) {
            b_v = castTo(b, b_t, common_type, false);
        } else {
            b_v = b;
        }

        const auto result = cmp(a_v, b_v, common_type);
        if (a_v.s != a.s) deleteValue(a_v, common_type); // if they have different pointer (we created a new value
        if (b_v.s != b.s) deleteValue(b_v, common_type);

        return result;
    }

    enum Resolver {
        CPU,
        GPU
    };

    std::pair<tval, DataType> getLiteralFrom(hsql::Expr*, bool strict = false, TableMap global_input = {}, Resolver resolver = CPU);

    tval add(tval, tval, DataType);
    tval sub(tval, tval, DataType);
    tval div(tval, tval, DataType);
    tval mul(tval, tval, DataType);
    tval neg(tval, DataType);

    bool isZero(tval, DataType);

}

#endif //VALUE_HELPER_HPP
