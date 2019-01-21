/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

ComplexVector.h
Author: Wangjing
Created in 2018-8-31

Classes for vector caculate

*/


#ifndef COMPLEXVECTOR_H
#define COMPLEXVECTOR_H

#include <iostream>
#include <complex>
#include <exception>
#include <vector>
#include <initializer_list>
#include "ComplexMatrix.h"
#include <assert.h>
#include "QPandaNamespace.h"

QPANDA_BEGIN
template<size_t qubit_number, typename precision_t=double>

class ComplexVector
{
    typedef  std::complex<precision_t> data_t;
    template<size_t, typename> class ComplexMatrix;
public:
    ComplexVector()
    {
        static_assert(qubit_number > 0, "qubitnum must > 0");
        size_t size = getSize();
        m_data.resize(size, 0);
    }

    ComplexVector(const ComplexVector<qubit_number, precision_t> & other)
    {
        size_t size = other.getSize();
        m_data.resize(size, 0);

        for (size_t i = 0; i < size; i++)
        {
            m_data[i] = other[i];
        }
    }

    ComplexVector<qubit_number,precision_t>& operator =(const ComplexVector<qubit_number, precision_t>& other)
    {
        if (this == &other)
        {
            return *this;
        }

        size_t size = other.getSize();
        m_data.resize(size, 0);

        for (size_t i = 0; i < size; i++)
        {
            m_data[i] = other[i];
        }

        return *this;
    }

    const data_t& operator[](size_t x) const
    {
        assert(x < getSize());
        return m_data[x];
    }

    data_t& operator[](size_t x)
    {
        assert(x < getSize());
        return m_data[x];
    }

    ComplexVector<qubit_number,precision_t>(const std::initializer_list<data_t> &args)
    {
        static_assert(qubit_number > 0, "qubitnum must > 0");
        size_t size = getSize();
        m_data.resize(size, 0);
        auto p = args.begin();

        for (size_t i = 0; i < size; i++)
        {
            if (i < args.size())
            {
                m_data[i] = *(p+i);
            }
        }
    }

    static ComplexVector<qubit_number,precision_t>GetZero()
    {
        ComplexVector<qubit_number, precision_t> ret_vec;
        ret_vec[0] = 1;
        return ret_vec;
    }

    void setEle(const size_t &x, const data_t &value)
    {
        assert(x < getSize());
        m_data[x] = value;
    }

    friend ComplexVector<qubit_number, precision_t>
    operator+(const ComplexVector<qubit_number, precision_t> &a,
              const ComplexVector<qubit_number, precision_t> &b)
    {
        ComplexVector<qubit_number> result;
        size_t size = a.getSize();
        for (size_t i = 0; i < size; i++)
        {
            result[i] = a.m_data[i] + b.m_data[i];
        }

        return result;
    }

    friend ComplexVector<qubit_number, precision_t>
    operator+(const data_t &a, const ComplexVector<qubit_number, precision_t> &b)
    {
        ComplexVector<qubit_number> result;
        size_t size = b.getSize();
        for (size_t i = 0; i < size; i++)
        {
            result[i] = a + b.m_data[i];
        }

        return result;
    }

    friend ComplexVector<qubit_number, precision_t>
    operator+(const ComplexVector<qubit_number, precision_t> &a, const data_t &b)
    {
        ComplexVector<qubit_number> result;
        size_t size = a.getSize();
        for (size_t i = 0; i < size; i++)
        {
            result[i] = a.m_data[i] + b;
        }

        return result;
    }

    friend ComplexVector<qubit_number, precision_t>
    operator-(const ComplexVector<qubit_number, precision_t> &a,
              const ComplexVector<qubit_number, precision_t> &b)
    {
        ComplexVector<qubit_number> result;
        size_t size = a.getSize();
        for (size_t i = 0; i < size; i++)
        {
            result[i] = a.m_data[i] - b.m_data[i];
        }

        return result;
    }

    friend ComplexVector<qubit_number, precision_t>
    operator-(const data_t &a, const ComplexVector<qubit_number, precision_t> &b)
    {
        ComplexVector<qubit_number> result;
        size_t size = b.getSize();
        for (size_t i = 0; i < size; i++)
        {
            result[i] = a - b.m_data[i];
        }

        return result;
    }

    friend ComplexVector<qubit_number, precision_t>
    operator-(const ComplexVector<qubit_number, precision_t> &a, const data_t &b)
    {
        ComplexVector<qubit_number> result;
        size_t size = a.getSize();
        for (size_t i = 0; i < size; i++)
        {
            result[i] = a.m_data[i] - b;
        }

        return result;
    }

    inline size_t getSize() const
    {
        return 1u << qubit_number;
    }

    friend std::ostream& operator<<(std::ostream& out, const ComplexVector<qubit_number, precision_t>& vec)
    {
        size_t size = vec.getSize();
        for (size_t i = 0; i < size; i++)
        {
            out << vec.m_data[i] << "\t";
        }

        return out;
    }

    template<size_t _qubitnum,typename _precisiton_t> friend
    ComplexVector<_qubitnum, _precisiton_t> operator *
    (const ComplexMatrix<_qubitnum, _precisiton_t> &mat,
     const ComplexVector<_qubitnum, _precisiton_t> &vec
     );

    ~ComplexVector()
    {
    }

private:
    std::vector<data_t> m_data;
};
QPANDA_END
#endif // COMPLEXVECTOR_H
