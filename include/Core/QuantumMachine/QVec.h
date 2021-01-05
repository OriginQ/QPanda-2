/*! \file QVec.h */
#ifndef _QVEC_H
#define _QVEC_H
#include <vector>
#include "Core/QuantumMachine/QubitReference.h"
QPANDA_BEGIN

/**
* @class QVec
* @brief Qubit vector  basic class
* @ingroup QuantumMachine
*/
class QVec : public std::vector<Qubit *>
{
    typedef std::vector<Qubit *> BaseClass;
public:
    QVec(BaseClass::iterator iter_begin, BaseClass::iterator iter_end)
    {
        for (auto aiter = iter_begin; aiter != iter_end; aiter++)
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

    QVec(Qubit * q)
    {
        push_back(q);
    }

    QVec(BaseClass &vector)
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

    inline Qubit * operator[](size_t  pos) const
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

	QVec operator +(QVec vec)
	{
		QVec new_vec(*this);
		new_vec += vec;
		return new_vec;
	}

	QVec& operator +=(QVec vec)
	{
		for (auto aiter = vec.begin(); aiter != vec.end(); aiter++)
		{
			auto biter = begin();
			for (; biter != end(); biter++)
			{
				if (*aiter == *biter)
				{
					break;
				}
			}

			if (biter == end())
			{
				(*this).push_back(*aiter);
			}
		}

		return *this;
	}

	QVec operator -(QVec vec)
	{
		QVec new_vec(*this);
		new_vec -= vec;
		
		return new_vec;
	}

	QVec& operator -=(QVec vec)
	{
		for (auto aiter = begin(); aiter != end(); )
		{
			auto biter = vec.begin();
			for (; biter != vec.end(); biter++)
			{
				if (*aiter == *biter)
				{
					break;
				}
			}

			if (biter != vec.end())
			{
				aiter = (*this).erase(aiter);
				continue;
			}

			++aiter;
		}
		return *this;
	}
};
QPANDA_END

#endif