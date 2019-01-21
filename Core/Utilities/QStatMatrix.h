/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QstatMatrix.h
Author: Wangjing
Created in 2018-8-31

Classes for get the shortes path of graph

*/



#ifndef QSTATMATRIX_H
#define QSTATMATRIX_H

#include "QPandaNamespace.h"
#include <iostream>
#include <complex>
#include <vector>
QPANDA_BEGIN

typedef std::vector<std::complex<double>> QStat;


bool isPerfectSquare(int number);
QStat operator+(const QStat &matrix_left, const QStat &matrix_right);
QStat operator+(const QStat &matrix_left, const std::complex<double> value);
QStat operator+(const std::complex<double> value, const QStat &matrix_right);

QStat operator-(const QStat &matrix_left, const QStat &matrix_right);
QStat operator-(const QStat &matrix_left, const std::complex<double> &value);
QStat operator-(const std::complex<double> &value, const QStat &matrix_right);

QStat operator*(const QStat &matrix_left, const QStat &matrix_right);
QStat operator*(const QStat &matrix_left, const std::complex<double> &value);
QStat operator*(const std::complex<double> &value, const QStat &matrix_right);

QPANDA_END
#endif // QSTATMATRIX_H
