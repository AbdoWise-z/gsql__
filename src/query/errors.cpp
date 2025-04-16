//
// Created by xabdomo on 4/15/25.
//

#include "errors.hpp"

#include <utility>

UnsupportedOperationError::UnsupportedOperationError(hsql::StatementType type) : msg(std::to_string(type)) {
}

UnsupportedOperationError::UnsupportedOperationError(std::string msg) : msg(std::move(msg)) {

}

UnsupportedOperationError::~UnsupportedOperationError() noexcept = default;

const char * UnsupportedOperationError::what() const noexcept {
    static std::string s;
    s =  "This type of SQL operation [";
    s += msg;
    s += "] is not supported";
    return s.c_str();
}

NoSuchTableError::NoSuchTableError(std::string name): name(std::move(name)) {

}

NoSuchTableError::~NoSuchTableError() noexcept = default;

const char * NoSuchTableError::what() const noexcept {
    static std::string s;
    s =  "Table with name [";
    s += name;
    s += "] doesn't exist";
    return s.c_str();
}

NoSuchColumnError::NoSuchColumnError(std::string name): name(std::move(name)) {
}

NoSuchColumnError::~NoSuchColumnError() noexcept {
}

const char * NoSuchColumnError::what() const noexcept {
    static std::string s;
    s =  "Column with signature [";
    s += name;
    s += "] doesn't exist";
    return s.c_str();
}


DuplicateTableError::DuplicateTableError(std::string name): name(std::move(name)) {

}

DuplicateTableError::~DuplicateTableError() noexcept = default;

const char * DuplicateTableError::what() const noexcept {
    static std::string s;
    s =  "Table with name [";
    s += name;
    s += "] already exist.";
    return s.c_str();
}

UnsupportedOperatorError::UnsupportedOperatorError(std::string name): name(std::move(name)) {}

UnsupportedOperatorError::~UnsupportedOperatorError() noexcept = default;

const char * UnsupportedOperatorError::what() const noexcept {
    static std::string s;
    s =  "Operator [";
    s += name;
    s += "] is not yet supported.";
    return s.c_str();
}

UnsupportedLiteralError::UnsupportedLiteralError() {}

UnsupportedLiteralError::~UnsupportedLiteralError() noexcept {
}

const char * UnsupportedLiteralError::what() const noexcept {
    return "Only literals of type [string, int, float] are supported atm.";
}

TableSizeMismatch::TableSizeMismatch() {
}

TableSizeMismatch::~TableSizeMismatch() noexcept {
}

const char * TableSizeMismatch::what() const noexcept {
    return "Table size mismatched (this probably will happened due to using Aggregators with normal columns in the same result)";
}
