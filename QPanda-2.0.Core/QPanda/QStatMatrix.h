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

#pragma once

#include <iostream>
#include <complex>
#include <vector>

using namespace std;

typedef vector<complex<double>> QStat;


bool isPerfectSquare(int number);
QStat operator+(const QStat &matrix_left, const QStat &matrix_right);
QStat operator+(const QStat &matrix_left, const complex<double> value);
QStat operator+(const complex<double> value, const QStat &matrix_right);

QStat operator-(const QStat &matrix_left, const QStat &matrix_right);
QStat operator-(const QStat &matrix_left, const complex<double> &value);
QStat operator-(const complex<double> &value, const QStat &matrix_right);

QStat operator*(const QStat &matrix_left, const QStat &matrix_right);
QStat operator*(const QStat &matrix_left, const complex<double> &value);
QStat operator*(const complex<double> &value, const QStat &matrix_right);


#endif // QSTATMATRIX_H
