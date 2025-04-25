//
// Created by xabdomo on 4/15/25.
//

#ifndef FROM_RESOLVER_HPP
#define FROM_RESOLVER_HPP

#include <hsql/sql/Table.h>

#include "db/table.hpp"
#include "query/resolve_result.hpp"

namespace FromResolver::CPU {
    typedef ResolveResult ResolveResult; // because I don't wanna remove the GPU:: / or CPU:: prefix

    ResolveResult merge(ResolveResult* a, ResolveResult* b);
    int find(ResolveResult* a, std::string tname);

    ResolveResult resolve(hsql::TableRef*);
};



#endif //FROM_RESOLVER_HPP
