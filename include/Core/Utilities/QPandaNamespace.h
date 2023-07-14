/*! \file QPandaNamespace.h */
#ifndef QPANDA_NAMESPACE
#define QPANDA_NAMESPACE

#include <iostream>
#include <vector>
#include <map>
#include <complex>
#include <unordered_map>
#include <cstring>
#include <bitset>
/**
* @namespace QPanda
* @brief QPanda2 base namespace
*/
#define QPANDA_BEGIN namespace QPanda {
#define QPANDA_END }
#define USING_QPANDA using namespace QPanda;

QPANDA_BEGIN

inline std::string _file_name(const char* file = "") {
	const auto _p_linux = strrchr(file, '/');
	const auto _p_win = strrchr(file, '\\');
	if ((nullptr != _p_linux) || (nullptr != _p_win)) { return nullptr != _p_linux ? (_p_linux + 1):(_p_win + 1); }
	return file;
}

#define __FILENAME__ QPanda::_file_name(__FILE__)

/**
* @def QCERR
* @brief QPanda2 cout error message
*/
#define QCERR(x) std::cerr<<__FILENAME__<<" " <<__LINE__<<" "<<__FUNCTION__<<" "\
                          <<x<<std::endl

/**
  output the error string to standard error and throw a standard exception.
  A standard exception can be of the following types:
  runtime_error, invalid_argument, range_error, etc
*/
#define QCERR_AND_THROW_ERRSTR(std_exception, x) {\
    QCERR(x);\
    throw std_exception(#x);}

#define QCERR_AND_THROW(std_exception, _Message_){ \
        std::ostringstream ss;\
        ss << _Message_;\
        QCERR(ss.str()); \
        throw std_exception(ss.str());}

#define QPANDA_ASSERT(con, argv)    do{                                     \
                                        if (con)                            \
                                        {                                   \
                                            throw std::runtime_error(argv); \
                                        }                                   \
                                    }while(0)


#define QPANDA_RETURN(con, value)    do{                                    \
                                        if (con)                            \
                                        {                                   \
                                            return value;                   \
                                        }                                   \
                                    }while(0)

#define QPANDA_OP(con, op)          do{                                     \
                                        if (con)                            \
                                        {                                   \
                                            op;                             \
                                        }                                   \
                                    }while(0)


/**
* @def qstate_type
* @brief QPanda2 quantum state data type
*/
typedef double qstate_type;

/**
* @def qcomplex_t
* @brief QPanda2 quantum state
*/
typedef std::complex <qstate_type> qcomplex_t;
typedef std::vector <qcomplex_t> QStat;
typedef std::vector<size_t> Qnum;

template <size_t num_qubits>
using qubit_label_type = std::bitset<num_qubits>;

// Wavefunctions are hash maps of some key (std::bitset or a string)
template <typename key>
using abstract_wavefunction = std::unordered_map<key, qcomplex_t>;

// Wavefunctions with strings as keys are "universal" in that they do not depend
// on the total number of qubits
using universal_wavefunction = abstract_wavefunction<std::string>;


using prob_vec = std::vector<double>;
using prob_map = std::unordered_map<std::string, qstate_type>;
using stat_map = std::unordered_map<std::string, qcomplex_t>;
using prob_dict = std::map<std::string, double>;
using prob_tuple = std::vector<std::pair<size_t, double>>;


/*struct for QSolver Oracle*/
struct OracleUserData
{
    std::vector<double> vd;
    std::vector<std::vector<double>> mat;
    std::vector<std::vector<size_t>> mNonzero;
    //std::vector<double> vd;
    //std::vector<std::vector<size_t>> vvi;
    //std::vector<std::vector<size_t>> vvj;
};

QPANDA_END

#endif // !QPANDA_NAMESPACE