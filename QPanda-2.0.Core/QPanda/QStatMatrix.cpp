#include "QStatMatrix.h"
#include "QPandaException.h"
#include <cmath>


bool isPerfectSquare(int number)
{
    for(int i = 1; number > 0; i += 2)
    {
        number -= i;
    }
    return  0 == number;
}


QStat operator+(const QStat &matrix_left, const QStat &matrix_right)
{
    if ((matrix_left.size() != matrix_right.size())    // insure dimension of the two matrixes is same
		|| (!isPerfectSquare((int)matrix_left.size())))   
    {
        throw param_error_exception("QStat is illegal", false);
    }

    int size = (int)matrix_left.size();
    QStat matrix_result(size, 0);

    for (int i = 0; i <size; i++)
    {
        matrix_result[i] = matrix_left[i] + matrix_right[i];
    }

    return matrix_result;
}


QStat operator+(const QStat &matrix_left, const complex<double> value)
{
    if (!isPerfectSquare((int)matrix_left.size()))
    {
        throw param_error_exception("QStat is illegal", false);
    }

    int size = (int)matrix_left.size();
    QStat matrix_result(size, 0);

    for (int i = 0; i < size; i++)
    {
        matrix_result[i] = matrix_left[i] + value;
    }

    return matrix_result;
}



QStat operator+(const complex<double> value, const QStat &matrix_right)
{
    if (!isPerfectSquare((int)matrix_right.size()))
    {
        throw param_error_exception("QStat is illegal", false);
    }

    int size = (int)matrix_right.size();
    QStat matrix_result(size, 0);

    for (int i = 0; i < size; i++)
    {
        matrix_result[i] = value + matrix_right[i];
    }

    return matrix_result;
}


QStat operator-(const QStat &matrix_left, const QStat &matrix_right)
{
    if ((matrix_left.size() != matrix_right.size())  // insure dimension of the two matrixes is same
		|| (!isPerfectSquare((int)matrix_left.size()))) 
    {
        throw param_error_exception("QStat is illegal", false);
    }

    int size = (int)matrix_left.size();
    QStat matrix_result(size, 0);

    for (int i = 0; i < size; i++)
    {
        matrix_result[i] = matrix_left[i] - matrix_right[i];
    }

    return matrix_result;
}



QStat operator-(const QStat &matrix_left, const complex<double> &value)
{
    if (!isPerfectSquare((int)matrix_left.size()))
    {
        throw param_error_exception("QStat is illegal", false);
    }

    int size = (int)matrix_left.size();
    QStat matrix_result(size, 0);

    for (int i = 0; i < size; i++)
    {
        matrix_result[i] = matrix_left[i] - value;
    }

    return matrix_result;
}



QStat operator-(const complex<double> &value, const QStat &matrix_right)
{
    if (!isPerfectSquare((int)matrix_right.size()))
    {
        throw param_error_exception("QStat is illegal", false);
    }

    int size = (int)matrix_right.size();
    QStat matrix_result(size, 0);

    for (int i = 0; i < size; i++)
    {
        matrix_result[i] = value - matrix_right[i];
    }

    return matrix_result;
}



QStat operator*(const QStat &matrix_left, const QStat &matrix_right)
{
    if ((matrix_left.size() != matrix_right.size())  // insure dimension of the two matrixes is same
		|| (!isPerfectSquare((int)matrix_left.size())))
    {
        throw param_error_exception("QStat is illegal", false);
    }

    int size = (int)matrix_left.size();
    QStat matrix_result(size, 0);
    int dimension = (int)sqrt(size);

    for (int i = 0; i < dimension; i++)
    {
        for (int j = 0; j < dimension; j++)
        {
            complex<double> temp = 0;
            for (int k = 0; k < dimension; k++)
            {
                temp += matrix_left[i*dimension + k] * matrix_right[k*dimension + j];
            }
            matrix_result[i*dimension + j] = temp;
        }
    }

    return matrix_result;
}


QStat operator*(const QStat &matrix_left, const complex<double> &value)
{
    if (!isPerfectSquare((int)matrix_left.size()))
    {
        throw param_error_exception("QStat is illegal", false);
    }

    int size = (int)matrix_left.size();
    QStat matrix_reslut(size, 0);

    for (int i = 0; i < size; i++)
    {
        matrix_reslut[i] = matrix_left[i] * value;
    }

    return matrix_reslut;
}


QStat operator*(const complex<double> &value, const QStat &matrix_right)
{
    if (!isPerfectSquare((int)matrix_right.size()))
    {
        throw param_error_exception("QStat is illegal", false);
    }

    int size =(int)matrix_right.size();
    QStat matrix_result(size, 0);

    for (int i = 0; i < size; i++)
    {
        matrix_result[i] = value * matrix_right[i];
    }

    return matrix_result;
}
