/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef _QGlobalVariable_H
#define _QGlobalVariable_H

#include <complex>
#include <vector>

enum NodeType
{
    NODE_UNDEFINED = -1,
    GATE_NODE,
    CIRCUIT_NODE,
    PROG_NODE,
    MEASURE_GATE,
    WHILE_START_NODE,
    QIF_START_NODE,
    CLASS_COND_NODE
};

enum GateType {
    PAULI_X_GATE,
    PAULI_Y_GATE,
    PAULI_Z_GATE,
    X_HALF_PI,
    Y_HALF_PI,
    Z_HALF_PI,
    HADAMARD_GATE,
    T_GATE,
    S_GATE,
    RX_GATE,
    RY_GATE,
    RZ_GATE,
    U1_GATE,
    U2_GATE,
    U3_GATE,
    U4_GATE,
    CU_GATE,
    CNOT_GATE,
    CZ_GATE,
    CPHASE_GATE,
    ISWAP_THETA_GATE,
    ISWAP_GATE,
    SQISWAP_GATE,
    TWO_QUBIT_GATE
};

enum Operatortype
{
    OP_AND = 1,
    OP_OR,
    OP_NOT,
    OP_ADD,
    OP_MINUS,
    OP_MULTIPLY,
    OP_DIVIDE,
    OP_MODE,
    OP_EQUAL,
    OP_LESS,
    OP_MORE,
    OP_NO_MORE,
    OP_NO_LESS,
};
enum OperatorType
{
    TYPE_OPERATOR_NODE,
    TYPE_CBIT_NODE,
    TYPE_CINT_NODE
};

typedef std::complex <double> qcomplex_t;
typedef std::vector <std::complex<double>> QStat;
const double PI = 3.14159265358979;

constexpr double SQRT2 = 1.4142135623730950488016887242097;
#endif
