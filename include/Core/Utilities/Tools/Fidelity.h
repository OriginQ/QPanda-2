/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

Fidelity.h
Author: Wangjing
Created in 2020-9-24

funtions for get fidelity

*/

#ifndef _FIDELITY_H_
#define _FIDELITY_H_

#include "Core/Utilities/QPandaNamespace.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/Tools/QStatMatrix.h"



QPANDA_BEGIN

double state_fidelity(const QStat &state1, const QStat &state2, bool validate = true);
double state_fidelity(const std::vector<QStat> &matrix1, const std::vector<QStat> &matrix2, bool validate = true);
double state_fidelity(const QStat &state, const std::vector<QStat> &matrix, bool validate = true);
double state_fidelity(const std::vector<QStat> &matrix, const QStat &state, bool validate = true);

QPANDA_END


#endif // _FIDELITY_H_
