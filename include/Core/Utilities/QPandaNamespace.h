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
  output the error string to standard error and throw a standard exception.
  A standard exception can be of the following types:
  runtime_error, invalid_argument, range_error, etc
*/
#define QCERR_AND_THROW_ERRSTR(std_exception, x) {\
    QCERR(x);\
    throw std_exception(#x);}

/**
* @def qstate_type
* @brief QPanda2 quantum state data type
* @ingroup Core
*/
typedef float qstate_type;

/**
* @def qcomplex_t
* @brief QPanda2 quantum state
* @ingroup Core
*/
typedef std::complex <qstate_type> qcomplex_t;
typedef std::vector <qcomplex_t> QStat;
typedef std::vector<size_t> Qnum;

using prob_vec = std::vector<double>;
using prob_map = std::unordered_map<std::string, double>;
using stat_map = std::unordered_map<std::string, qcomplex_t>;
using prob_dict = std::map<std::string, double>;
using prob_tuple = std::vector<std::pair<size_t, double>>;


/**
* @namespace QPanda
* @brief QPanda2 base namespace
* @ingroup Core
*/
#define QPANDA_BEGIN namespace QPanda {
#define QPANDA_END }
#define USING_QPANDA using namespace QPanda;
#endif // !QPANDA_NAMESPACE

