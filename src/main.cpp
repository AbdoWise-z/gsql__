#include <iostream>
#include "hsql/SQLParser.h"

int main() {
    auto sql = "SELECT * FROM some_table WHERE id in (SELECT id FROM some_table2);";
    hsql::SQLParserResult result;
    hsql::SQLParser::parse(sql, &result);
    std::cout << result.isValid() << std::endl;
    std::cout << result.getStatements()[0]->type() << std::endl;
    return 0;
}