//
// Created by xabdomo on 4/15/25.
//

#include "errors.hpp"

#include <utility>

UnsupportedOperationError::UnsupportedOperationError(hsql::StatementType type) : type(type) {
}

UnsupportedOperationError::~UnsupportedOperationError() noexcept = default;

const char * UnsupportedOperationError::what() const noexcept {
    static std::string s;
    s =  "This type of SQL operation [";
    s += type;
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
