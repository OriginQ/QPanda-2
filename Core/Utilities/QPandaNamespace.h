#ifndef QPANDA_NAMESPACE
#define QPANDA_NAMESPACE

#include <iostream>
#include <vector>
#include <complex>

#define QCERR(x) std::cerr<<__FILE__<<" " <<__LINE__<<" "<<__FUNCTION__<<" "\
                          <<(x)<<std::endl

typedef double qstate_type;
typedef std::complex <qstate_type> qcomplex_t;
typedef std::vector <qcomplex_t> QStat;
typedef std::vector<size_t> Qnum;


#define QPANDA_BEGIN namespace QPanda {
#define QPANDA_END }
#define USING_QPANDA using namespace QPanda;
#endif // !QPANDA_NAMESPACE

