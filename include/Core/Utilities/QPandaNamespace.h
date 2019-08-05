/*! \file QPandaNamespace.h */
#ifndef QPANDA_NAMESPACE
#define QPANDA_NAMESPACE

#include <iostream>
#include <vector>
#include <map>
#include <complex>
#include <unordered_map>

/**
* @def QCERR
* @brief QPanda2 cout error message
* @ingroup Core
*/
#define QCERR(x) std::cerr<<__FILE__<<" " <<__LINE__<<" "<<__FUNCTION__<<" "\
                          <<(x)<<std::endl

/**
* @def qstate_type
* @brief QPanda2 quantum state data type
* @ingroup Core
*/
typedef double qstate_type;

/**
* @def qcomplex_t
* @brief QPanda2 quantum state
* @ingroup Core
*/
typedef std::complex <qstate_type> qcomplex_t;
typedef std::vector <qcomplex_t> QStat;
typedef std::vector<size_t> Qnum;

using prob_map = std::unordered_map<std::string, qstate_type>;
using stat_map = std::unordered_map<std::string, qcomplex_t>;

/**
* @namespace QPanda
* @brief QPanda2 base namespace
* @ingroup Core
*/
#define QPANDA_BEGIN namespace QPanda {
#define QPANDA_END }
#define USING_QPANDA using namespace QPanda;
#endif // !QPANDA_NAMESPACE

