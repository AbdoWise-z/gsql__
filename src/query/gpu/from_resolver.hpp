//
// Created by xabdomo on 4/15/25.
//

#ifndef FROM_RESOLVER_HPP
#define FROM_RESOLVER_HPP

#include <hsql/sql/Table.h>

#include "store.hpp"
#include "db/table.hpp"
#include "query/resolve_result.hpp"


namespace FromResolver::GPU {
    typedef ResolveResult ResolveResult; // because I don't wanna remove the GPU:: / or CPU:: prefix

    ResolveResult resolve(hsql::TableRef*, TableMap& tables);
};



#endif //FROM_RESOLVER_HPP
