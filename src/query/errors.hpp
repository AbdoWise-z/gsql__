//
// Created by xabdomo on 4/15/25.
//

#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include <string>
#include <hsql/sql/SQLStatement.h>

class UnsupportedOperationError final : public std::exception {
private:
    std::string msg;
public:
    explicit UnsupportedOperationError(hsql::StatementType type);
    explicit UnsupportedOperationError(std::string msg);
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

class NoSuchColumnError final : public std::exception {
private:
    std::string name;

public:
    explicit NoSuchColumnError(std::string  name);
    ~NoSuchColumnError() noexcept override;

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

class UnsupportedLiteralError final : public std::exception {
public:
    explicit UnsupportedLiteralError();
    ~UnsupportedLiteralError() noexcept override;

    const char * what() const noexcept override;
};

class TableSizeMismatch final : public std::exception {
public:
    explicit TableSizeMismatch();
    ~TableSizeMismatch() noexcept override;

    const char * what() const noexcept override;
};

#endif //EXECUTOR_HPP
