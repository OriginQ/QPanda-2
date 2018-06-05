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

#ifndef _QGlobalVariable_H_
#define _QGlobalVariable_H_
enum NodeType
{
    NODE_UNDEFINED = -1,
    GATE_NODE,
    CIRCUIT_NODE,
    PROG_NODE,
    MEASURE_GATE,
    WHILE_START_NODE,
    QIF_START_NODE
};

enum GateType
{
    GATE_UNDEFINED = -1,
    H_GATE,
    RX_GATE,
    RY_GATE,
    RZ_GATE,
    S_GATE,
    CNOT_GATE,
    CZ_GATE,


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
    TYPE_CINT_NODE,
    /*TYPE_CINT8_NODE,
    TYPE_CINT16_NODE,
    TYPE_CVEC_NODE,*/
};
//const map<int, string> Operator_String_Map
//{
//    { OP_AND,"&&"},
//    { OP_OR,"||" },
//    { OP_NOT,"!" },
//    { OP_ADD,"+" },
//    { OP_MINUS,"-" },
//    { OP_MULTIPLY,"*" },
//    { OP_DIVIDE,"/" },
//    { OP_MODE,"%" },
//    { OP_EQUAL,"==" },
//    { OP_LESS,"<" },
//    { OP_MORE,">" },
//    { OP_NO_MORE,"<=" },
//    { OP_NO_LESS,">=" }
//};

constexpr double SQRT2 = 1.4142135623730950488016887242097;
#endif
