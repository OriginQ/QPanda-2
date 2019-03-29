/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

ComplexMatrix.h
Author: Wangjing
Created in 2018-8-31

Classes for matrix caculate

*/

#ifndef COMPLEXMATRIX_H
#define COMPLEXMATRIX_H

#include <iostream>
#include <complex>
#include <exception>
#include <vector>
#include <stdexcept>
#include <initializer_list>
#include "Core/Utilities/ComplexVector.h"

USING_QPANDA
template<size_t qubit_number, typename precision_t=double>
class ComplexMatrix
{
    typedef std::complex<precision_t> data_t;
public:
    ComplexMatrix()
    {
        static_assert(qubit_number > 0, "qubitnum must > 0");
        initialize();
    }

    ComplexMatrix<qubit_number,precision_t>(const std::initializer_list<data_t> &args)
    {
        static_assert(qubit_number > 0, "qubitnum must > 0");
        initialize();

        size_t dimension = getDimension();
        auto p = args.begin();
        for (size_t i = 0; i < dimension; i++)
        {
            for (size_t j = 0; j < dimension; j++)  
            {
                if ((i* dimension + j) < args.size())
                {
                    m_data[i][j] = p[i * dimension + j];
                }
            }
        }

    }

    ComplexMatrix(const ComplexMatrix<qubit_number, precision_t>& other_matrix)
    {
        initialize();
        size_t dimension = other_matrix.getDimension();

        for (size_t i = 0; i < dimension; i++)
        {
            for (size_t j = 0; j < dimension; j++)
            {
                m_data[i][j] = other_matrix.getElem(i,j);
            }
        }
    }

    ComplexMatrix<qubit_number, precision_t>& operator=(const ComplexMatrix<qubit_number, precision_t>& other)
    {
        if (this == &other)
        {
            return *this;
        }

        size_t dimension = other.getDimension();
        for (size_t i = 0; i < dimension; i++)
        {
            for (size_t j = 0; j < dimension; j++)
            {
                m_data[i][j] = other.getElem(i, j);
            }
        }
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& out, const ComplexMatrix<qubit_number, precision_t>& mat)
    {
        size_t size = mat.getDimension();
        for (size_t i = 0; i < size; i++)
        {
            for (size_t j = 0; j < size; j++)
            {
                out << mat.getElem(i, j) << "\t";
            }
            out << "\n";
        }

        return out;
    }

    const data_t& getElem(const size_t x, const size_t y) const
    {
        assert(x < getDimension() && y < getDimension());
        return this->m_data[x][y];
    }

    static ComplexMatrix<qubit_number, precision_t> GetIdentity()
    {
        ComplexMatrix<qubit_number, precision_t> matrix;
        size_t dimension = matrix.getDimension();

        for (size_t i = 0; i < dimension; i++)
        {
            for (size_t j = 0; j < dimension; j++)
            {
                if (i == j)
                {
                    matrix.setElem(i, j, 1);
                }
            }
        }

        return matrix;
    }

    ComplexMatrix<qubit_number, precision_t> dagger()
    {
        ComplexMatrix<qubit_number, precision_t> matrix;
        size_t dimension = getDimension();

        for (size_t i = 0; i < dimension; i++)
        {
            for (size_t j = 0; j < dimension; j++)
            {
                matrix.setElem(i, j, std::conj(m_data[j][i]));
            }
        }

        return matrix;
    }

    friend ComplexMatrix<qubit_number, precision_t>
    operator+(const ComplexMatrix<qubit_number, precision_t> &a,
              const ComplexMatrix<qubit_number, precision_t> &b)
    {
        ComplexMatrix<qubit_number, precision_t> result;
        size_t dimension = a.getDimension();

        for (size_t i = 0; i < dimension; i++)
        {
            for (size_t j = 0; j < dimension; j++)
            {
                data_t tmp = a.m_data[i][j] + b.m_data[i][j];
                result.setElem(i, j, tmp);
            }
        }
        return result;
    }

    friend ComplexMatrix<qubit_number, precision_t>
    operator+(const data_t &a, const ComplexMatrix<qubit_number, precision_t> &b)
    {
        ComplexMatrix<qubit_number, precision_t> result;
        size_t dimension = b.getDimension();

        for (size_t i = 0; i < dimension; i++)
        {
            for (size_t j = 0; j < dimension; j++)
            {
                data_t tmp = a + b.m_data[i][j];
                result.setElem(i, j, tmp);
            }
        }
        return result;
    }

    friend ComplexMatrix<qubit_number, precision_t>
    operator+(const ComplexMatrix<qubit_number, precision_t> &a, const data_t &b)
    {
        ComplexMatrix<qubit_number, precision_t> result;
        size_t dimension = a.getDimension();

        for (size_t i = 0; i < dimension; i++)
        {
            for (size_t j = 0; j < dimension; j++)
            {
                data_t tmp = a.m_data[i][j] + b;
                result.setElem(i, j, tmp);
            }
        }
        return result;
    }

    friend ComplexMatrix<qubit_number, precision_t>
    operator-(const ComplexMatrix<qubit_number, precision_t> &a,
              const ComplexMatrix<qubit_number, precision_t> &b)
    {
        ComplexMatrix<qubit_number, precision_t> result;
        size_t dimension = a.getDimension();

        for (size_t i = 0; i < dimension; i++)
        {
            for (size_t j = 0; j < dimension; j++)
            {
                data_t tmp = a.m_data[i][j] - b.m_data[i][j];
                result.setElem(i, j, tmp);
            }
        }

        return result;
    }

    friend ComplexMatrix<qubit_number, precision_t>
    operator-(const data_t &a, const ComplexMatrix<qubit_number, precision_t> &b)
    {
        ComplexMatrix<qubit_number, precision_t> result;
        size_t dimension = b.getDimension();

        for (size_t i = 0; i < dimension; i++)
        {
            for (size_t j = 0; j < dimension; j++)
            {
                data_t tmp = a - b.m_data[i][j];
                result.setElem(i, j, tmp);
            }
        }

        return result;
    }

    friend ComplexMatrix<qubit_number, precision_t>
    operator-(const ComplexMatrix<qubit_number, precision_t> &a, const data_t &b)
    {
        ComplexMatrix<qubit_number, precision_t> result;
        size_t dimension = a.getDimension();

        for (size_t i = 0; i < dimension; i++)
        {
            for (size_t j = 0; j < dimension; j++)
            {
                data_t tmp = a.m_data[i][j] - b;
                result.setElem(i, j, tmp);
            }
        }

        return result;
    }

    friend ComplexMatrix<qubit_number, precision_t>
    operator*(const ComplexMatrix<qubit_number, precision_t> &a,
              const ComplexMatrix<qubit_number, precision_t> &b)
    {
        ComplexMatrix<qubit_number, precision_t> result;
        size_t dimension = a.getDimension();

        for(size_t i = 0; i < dimension; i++)
        {
            for(size_t j = 0; j < dimension; j++)
            {
                data_t tmp = 0;
                for(size_t k = 0; k < dimension; k++)
                {
                    tmp += a.m_data[i][k] * b.m_data[k][j];
                }

                result.setElem(i, j, tmp);
            }
        }

        return result;
    }

    friend ComplexMatrix<qubit_number, precision_t>
    operator*(const data_t &a, const ComplexMatrix<qubit_number, precision_t> &b)
    {
        ComplexMatrix<qubit_number, precision_t> result;
        size_t dimension = b.getDimension();

        for(size_t i = 0; i < dimension; i++)
        {
            for(size_t j = 0; j < dimension; j++)
            {
                data_t tmp = 0;
                tmp = a * b.m_data[i][j];
                result.setElem(i, j, tmp);
            }
        }

        return result;
    }

    friend ComplexMatrix<qubit_number, precision_t>
    operator*(const ComplexMatrix<qubit_number, precision_t> &a, const data_t &b)
    {
        ComplexMatrix<qubit_number, precision_t> result;
        size_t dimension = a.getDimension();

        for(size_t i = 0; i < dimension; i++)
        {
            for(size_t j = 0; j < dimension; j++)
            {
                data_t tmp = 0;
                tmp = a.m_data[i][j] * b;
                result.setElem(i, j, tmp);
            }
        }

        return result;
    }

    friend ComplexVector<qubit_number, precision_t>
    operator *(const ComplexMatrix<qubit_number, precision_t> &mat,
               const ComplexVector<qubit_number, precision_t> &vec)
    {
        ComplexVector<qubit_number, precision_t> result;
        size_t size = mat.getDimension();

        for (size_t i = 0; i < size; i++)
        {
            for (size_t j = 0; j < size; j++)
            {
                result[i] = result[i] + mat.m_data[i][j] * vec[j];
            }
        }

        return result;
    }

    inline size_t getDimension() const
    {
        return 1u << qubit_number;
    }

    void setElem(const size_t x, const size_t y, const data_t &value)
    {
        assert(x < getDimension() && y < getDimension());
        m_data[x][y] = value;
        return ;
    }

    void initialize()
    {
        size_t dimension = getDimension();
        m_data.resize(dimension);

        for (size_t i = 0; i < dimension; i++)
        {
            m_data[i].resize(dimension, 0);
        }
    }

    ~ComplexMatrix()
    {
    }

private:
    std::vector< std::vector<data_t> > m_data;
};

#endif // COMPLEXMATRIX_H
