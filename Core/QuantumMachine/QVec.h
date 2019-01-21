#ifndef _QVEC_H
#define _QVEC_H
#include <vector>
#include "QubitReference.h"
QPANDA_BEGIN

class QVec : public std::vector<Qubit *>
{
    typedef std::vector<Qubit *> BaseClass;
public:
    QVec(BaseClass::iterator  iter_begen, BaseClass::iterator  iter_end )
    {
        for (auto aiter = iter_begen; aiter != iter_end; aiter++)
        {
            push_back(*aiter);
        }
    }

    QVec(const std::initializer_list<Qubit *> & args)
    {
        for (auto aiter: args)
        {
            push_back(aiter);   
        }
    }
    QVec() {}

    QVec(const QVec & old) 
    {
        for (auto aiter : old)
        {
            push_back(aiter);
        }
    }
    QVec(BaseClass & vector)
    {
        for (auto aiter = vector.begin(); aiter != vector.end(); aiter++)
        {
            push_back(*aiter);
        }
    }
    inline Qubit * operator[](ClassicalCondition & classical_cond)
    {
        std::vector<Qubit *> qvec;
        for(auto aiter : *this)
        {
            qvec.push_back(aiter);
        }
        QubitReference *temp = new QubitReference(classical_cond,qvec);
        return temp;
    }

    inline Qubit * operator[](cbit_size_t  pos)
    {
        if(pos > (cbit_size_t)size())
        {
            QCERR("pos overflow");
            throw std::invalid_argument("pos overflow");
        }
        return BaseClass::operator[](pos);
    }

    inline QVec &operator<<(int)
    {
        return *this;
    }
};
QPANDA_END

#endif