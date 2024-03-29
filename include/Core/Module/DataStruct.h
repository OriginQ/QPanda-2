/*
Copyright (c) 2017-2023 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

DataStruct.h

Author: LiYe
Created in 2018-09-06


*/
#ifndef DATASTRUCT_H
#define DATASTRUCT_H

#include <map>
#include <vector>
#include <functional>
#include <complex>
#include "Core/Utilities/QPandaNamespace.h"

#ifdef _MSC_VER
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

QPANDA_BEGIN

#define DEF_WARING                    ("Warning: ")
#define DEF_ITERATIONS                ("Iterations")
#define DEF_EVALUATIONS               ("Function evaluations")
#define DEF_VALUE                     ("Value")
#define DEF_KEY                       ("Key")
#define DEF_MESSAGE                   ("Message")
#define DEF_OPTI_STATUS_PARA_ERROR    ("Optimizer parameter setting error.")
#define DEF_OPTI_STATUS_CALCULATING   ("Calculating")
#define DEF_OPTI_STATUS_SUCCESS       ("Optimization terminated successfully.")
#define DEF_OPTI_STATUS_MAX_FEV       ("Maximum number of function evaluations" \
                                       " has been exceeded.")
#define DEF_OPTI_STATUS_MAX_ITER      ("Maximum number of iterations has been" \
                                       " exceeded.")
#ifndef DEF_UNINIT_INT
#define DEF_UNINIT_INT                (-1234567)
#endif

#ifndef DEF_UNINIT_FLOAT
#define DEF_UNINIT_FLOAT              (-1234567.0)
#endif

#define Q_PI       3.14159265358979323846   /**< pi */
#define Q_PI_2     1.57079632679489661923   /**< pi/2 */

    /**
    * @brief Graph element structure
    */
    struct QGraphItem
    {
        size_t first;
        size_t second;
        double weight;
        QGraphItem() :
            first(0),
            second(0),
            weight(0.0)
        {}

        QGraphItem(size_t first, size_t second, double weight) :
            first(first),
            second(second),
            weight(weight)
        {}
    };

    using complex_d = std::complex<double>;
    using vector_i = std::vector<int>;
    using vector_d = std::vector<double>;
    using vector_s = std::vector<std::string>;
    using QGraph = std::vector<QGraphItem>;

    //typedef QResultPair(*QOptFunc_c)(const vector_d &x, vector_d& grad, int, int);

    using QProbMap = std::map<std::string, double>;

    /**
    * @note
        The QTerm value char only will be 'X','Y','Z'.
        If QTerm is empty it reperents 'I'.
    */
    using QTerm = std::map<size_t, char>;
    using QTermPair = std::pair<size_t, char>;
    using QHamiltonianItem = std::pair<QTerm, double>;
    using QHamiltonian = std::vector<QHamiltonianItem>;
    using QPauliPair = std::pair<QTerm, std::string>;
    using QPauliMap = std::map<std::string, complex_d>;
    using QIndexMap = std::map<size_t, size_t>;

	/**
	* @brief Position structure
	*/
    struct QPosition
    {
        double x;
        double y;
        double z;

        QPosition(double x = 0.0, double y = 0.0, double z = 0.0) :
            x(x),
            y(y),
            z(z)
        {}
    };

    using QAtomPos = std::pair<std::string, QPosition>;
    using QMoleculeGeometry = std::vector<QAtomPos>;
    using QAtomsPosGroup = std::vector<std::vector<QPosition>>;


    const std::map<std::string, size_t> g_kAtomElectrons
    {
        {"H", 1},{"He", 2},{"Li", 3},{"Be",4},{"B", 5},{"C", 6},{"N", 7},      
        {"O", 8},{"F", 9},{"Ne", 10},{"Na", 11},{"Mg", 12},{"Al", 13},
        {"Si", 14},{"P", 15},{"S", 16},{"Cl", 17},{"Ar", 18}

    };

	/**
	* @brief TransForm Type
	*/
    enum class TransFormType
    {
        Jordan_Wigner,
        Parity,
        Bravyi_Ktaev
    };

	/**
	* @brief Ucc Type
	*/
    enum class UccType
    {
        UCCS,
        UCCSD
    };

    /**
    * @brief Ansatz gate type
    */
    enum class AnsatzGateType
    {
        AGT_X,
        AGT_H,
        AGT_RX,
        AGT_RY,
        AGT_RZ,
        AGT_X1,
        AGT_Y1,
        AGT_Z1,
        AGT_Z,
        AGT_Y
    };

    /**
    * @brief Ansatz gate structure
    */
    struct AnsatzGate
    {
        AnsatzGateType type;
        int target;
        double theta;
        int control;
        bool constant;

        AnsatzGate(
            AnsatzGateType type_,
            int target_,
            double theta_ = -1,
            int control_ = -1,
            bool constant_ = false) :
            type(type_),
            target(target_),  
            theta(theta_),
            control(control_),
            constant(constant_)
        {}
    };
    
QPANDA_END

#endif // DATASTRUCT_H
