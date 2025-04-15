//
// Created by xabdomo on 4/15/25.
//

#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include <string>
#include <hsql/sql/SQLStatement.h>

class UnsupportedOperationError final : public std::exception {
private:
    hsql::StatementType type;

public:
    explicit UnsupportedOperationError(hsql::StatementType type);
    ~UnsupportedOperationError() noexcept override;

    const char * what() const noexcept override;
};


class NoSuchTableError final : public std::exception {
private:
    std::string name;

public:
    explicit NoSuchTableError(std::string  name);
    ~NoSuchTableError() noexcept override;

    const char * what() const noexcept override;
};


class DuplicateTableError final : public std::exception {
private:
    std::string name;

public:
    explicit DuplicateTableError(std::string  name);
    ~DuplicateTableError() noexcept override;

    const char * what() const noexcept override;
};


class UnsupportedOperatorError final : public std::exception {
private:
    std::string name;

public:
    explicit UnsupportedOperatorError(std::string  name);
    ~UnsupportedOperatorError() noexcept override;

    const char * what() const noexcept override;
};

#endif //EXECUTOR_HPP
