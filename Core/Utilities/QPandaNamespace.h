#ifndef QPANDA_NAMESPACE
#define QPANDA_NAMESPACE

#include <iostream>

#define QCERR(x) std::cerr<<__FILE__<<" " <<__LINE__<<" "<<__FUNCTION__<<" "\
                          <<(x)<<std::endl

#define QPANDA_BEGIN namespace QPanda {
#define QPANDA_END }
#define USING_QPANDA using namespace QPanda;
#endif // !QPANDA_NAMESPACE

