#ifndef MATRIX_DECOMPOSITION_H
#define MATRIX_DECOMPOSITION_H
#include <math.h>
#include "QPanda.h"
#include "ThirdParty/Eigen/Dense"
#include "ThirdParty/Eigen/Sparse"
QPANDA_BEGIN


enum class MatrixUnit
{
    SINGLE_P0,
    SINGLE_P1,
    SINGLE_I2,
    SINGLE_V2
};


using MatrixSequence = std::vector<MatrixUnit>;
using DecomposeEntry = std::pair<int, MatrixSequence>;

using ColumnOperator = std::vector<DecomposeEntry>;
using MatrixOperator = std::vector<ColumnOperator>;

class QMatrix : public QStat
{
    using BaseClass = QStat;

public:
    QMatrix(BaseClass::iterator iter_begin, BaseClass::iterator iter_end)
    {
        for (auto aiter = iter_begin; aiter != iter_end; aiter++)
        {
            push_back(*aiter);
        }
    }

    QMatrix(const std::initializer_list<qcomplex_t> &args)
    {
        std::for_each(args.begin(), args.end(),[&](qcomplex_t c) 
        {
            push_back(c); 
        });
    }

    QMatrix(const QMatrix &old)
    {
        std::for_each(old.begin(), old.end(), [&](qcomplex_t c) 
        {
            push_back(c); 
        });
    }

    QMatrix(const BaseClass &base)
    {
        std::for_each(base.begin(), base.end(), [&](qcomplex_t c) 
        {
            push_back(c); 
        });
    }


    inline qcomplex_t operator[](size_t  pos)
    {
        if (pos > (cbit_size_t)size())
        {
            QCERR("pos overflow");
            throw std::invalid_argument("pos overflow");
        }
        return BaseClass::operator[](pos);
    }

    void decompose();
};



QPANDA_END
#endif // MATRIX_DECOMPOSITION_H