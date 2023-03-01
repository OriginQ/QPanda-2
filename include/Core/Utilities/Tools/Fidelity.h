/*
Copyright (c) 2017-2023 Origin Quantum Computing. All Right Reserved.
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
#include <map>


QPANDA_BEGIN

double state_fidelity(const QStat &state1, const QStat &state2, bool validate = true);
double state_fidelity(const std::vector<QStat> &matrix1, const std::vector<QStat> &matrix2, bool validate = true);
double state_fidelity(const QStat &state, const std::vector<QStat> &matrix, bool validate = true);
double state_fidelity(const std::vector<QStat> &matrix, const QStat &state, bool validate = true);
double process_fidelity(const QStat& state1, const QStat& state2, bool validate = true);
double average_gate_fidelity(const QMatrixXcd& matrix, const QStat& state, bool validate = true);
double average_gate_fidelity(const QMatrixXcd& matrix, const QMatrixXcd& state, bool validate = true);
double hellinger_fidelity(const std::map<std::string, size_t>, const std::map<std::string, size_t>, int);
QPANDA_END


#endif // _FIDELITY_H_
