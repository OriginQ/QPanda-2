#ifndef QPANDA_NAMESPACE
#define QPANDA_NAMESPACE

#include <iostream>
#include <vector>
#include <complex>

#define QCERR(x) std::cerr<<__FILE__<<" " <<__LINE__<<" "<<__FUNCTION__<<" "\
                          <<(x)<<std::endl

typedef std::complex <double> qcomplex_t;
typedef std::vector<size_t> Qnum;
typedef std::vector <std::complex<double>> QStat;

#define QPANDA_BEGIN namespace QPanda {
#define QPANDA_END }
#define USING_QPANDA using namespace QPanda;
#endif // !QPANDA_NAMESPACE

