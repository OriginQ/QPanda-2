/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

Quanatum Fourier transform

*/

#ifndef  QFT_H
#define  QFT_H

#include "Core/Core.h"

QPANDA_BEGIN

inline QCircuit QFT(QVec qvec)
{
	QCircuit  qft = CreateEmptyCircuit();
	for (auto i = 0; i < qvec.size(); i++)
	{
		qft << H(qvec[qvec.size() - 1 - i]);
		for (auto j = i + 1; j < qvec.size(); j++)
		{
			qft << CR(qvec[qvec.size() - 1 - j],
				qvec[qvec.size() - 1 - i], 2 * PI / (1 << (j - i + 1)));
		}
	}
	return qft;
}

QPANDA_END

#endif