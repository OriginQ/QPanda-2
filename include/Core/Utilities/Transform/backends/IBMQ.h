/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

IBMQ.h
Author: Zhaodongyi
Updated in 2019/08/08 15:05

*/
/*! \file IBMQ.h */
#ifndef  IBMQ_H_
#define  IBMQ_H_

#include "Core/Utilities/XMLConfigParam.h"
#include <iostream>
#include <bitset>

QPANDA_BEGIN

#ifndef IBMQ_BACKENDS_CONFIG
#define IBMQ_BACKENDS_CONFIG R"(<?xml version="1.0" encoding="utf-8" ?><IBMQ>    <!-- IBMQ backends  -->    <backends>        <!-- 32 qubits simulator -->        <ibmq_qasm_simulator>            <QubitCount>32</QubitCount>            <QubitMatrix>                <!-- no QubitMatrix on ibmq_qasm_simulator-->            </QubitMatrix>        </ibmq_qasm_simulator>        <!-- 14 qubits real quantum chip-->        <ibmq_16_melbourne>            <QubitCount>14</QubitCount>            <QubitMatrix>                <Qubit QubitNum="1">                    <AdjacentQubit QubitNum="2">1</AdjacentQubit>                </Qubit>                <Qubit QubitNum="2">                    <AdjacentQubit QubitNum="1">1</AdjacentQubit>                    <AdjacentQubit QubitNum="3">1</AdjacentQubit>                    <AdjacentQubit QubitNum="14">1</AdjacentQubit>                </Qubit>                <Qubit QubitNum="3">                    <AdjacentQubit QubitNum="2">1</AdjacentQubit>                    <AdjacentQubit QubitNum="4">1</AdjacentQubit>                    <AdjacentQubit QubitNum="13">1</AdjacentQubit>                </Qubit>                <Qubit QubitNum="4">                    <AdjacentQubit QubitNum="3">1</AdjacentQubit>                    <AdjacentQubit QubitNum="5">1</AdjacentQubit>                    <AdjacentQubit QubitNum="12">1</AdjacentQubit>                </Qubit>                <Qubit QubitNum="5">                    <AdjacentQubit QubitNum="4">1</AdjacentQubit>                    <AdjacentQubit QubitNum="6">1</AdjacentQubit>                    <AdjacentQubit QubitNum="11">1</AdjacentQubit>                </Qubit>                <Qubit QubitNum="6">                    <AdjacentQubit QubitNum="5">1</AdjacentQubit>                    <AdjacentQubit QubitNum="7">1</AdjacentQubit>                    <AdjacentQubit QubitNum="10">1</AdjacentQubit>                </Qubit>                <Qubit QubitNum="7">                    <AdjacentQubit QubitNum="6">1</AdjacentQubit>                    <AdjacentQubit QubitNum="9">1</AdjacentQubit>                </Qubit>                <Qubit QubitNum="8">                    <AdjacentQubit QubitNum="9">1</AdjacentQubit>                </Qubit>                <Qubit QubitNum="9">                    <AdjacentQubit QubitNum="7">1</AdjacentQubit>                    <AdjacentQubit QubitNum="8">1</AdjacentQubit>                    <AdjacentQubit QubitNum="10">1</AdjacentQubit>                </Qubit>                <Qubit QubitNum="10">                    <AdjacentQubit QubitNum="6">1</AdjacentQubit>                    <AdjacentQubit QubitNum="9">1</AdjacentQubit>                    <AdjacentQubit QubitNum="11">1</AdjacentQubit>                </Qubit>                <Qubit QubitNum="11">                    <AdjacentQubit QubitNum="5">1</AdjacentQubit>                    <AdjacentQubit QubitNum="10">1</AdjacentQubit>                    <AdjacentQubit QubitNum="12">1</AdjacentQubit>                </Qubit>                <Qubit QubitNum="12">                    <AdjacentQubit QubitNum="4">1</AdjacentQubit>                    <AdjacentQubit QubitNum="11">1</AdjacentQubit>                    <AdjacentQubit QubitNum="13">1</AdjacentQubit>                </Qubit>                <Qubit QubitNum="13">                    <AdjacentQubit QubitNum="3">1</AdjacentQubit>                    <AdjacentQubit QubitNum="12">1</AdjacentQubit>                    <AdjacentQubit QubitNum="14">1</AdjacentQubit>                </Qubit>                <Qubit QubitNum="14">                    <AdjacentQubit QubitNum="2">1</AdjacentQubit>                    <AdjacentQubit QubitNum="13">1</AdjacentQubit>                </Qubit>            </QubitMatrix>        </ibmq_16_melbourne>        <!-- 5 qubits real quantum chip Qx2 -->        <ibmqx2>            <QubitCount>5</QubitCount>            <QubitMatrix>                <Qubit QubitNum="1">                    <AdjacentQubit QubitNum="2">1</AdjacentQubit>                    <AdjacentQubit QubitNum="3">1</AdjacentQubit>                </Qubit>                <Qubit QubitNum="2">                    <AdjacentQubit QubitNum="1">1</AdjacentQubit>                    <AdjacentQubit QubitNum="3">1</AdjacentQubit>                </Qubit>                <Qubit QubitNum="3">                    <AdjacentQubit QubitNum="1">1</AdjacentQubit>                    <AdjacentQubit QubitNum="2">1</AdjacentQubit>                    <AdjacentQubit QubitNum="4">1</AdjacentQubit>                    <AdjacentQubit QubitNum="5">1</AdjacentQubit>                </Qubit>                <Qubit QubitNum="4">                    <AdjacentQubit QubitNum="3">1</AdjacentQubit>                    <AdjacentQubit QubitNum="5">1</AdjacentQubit>                </Qubit>                <Qubit QubitNum="5">                    <AdjacentQubit QubitNum="3">1</AdjacentQubit>                    <AdjacentQubit QubitNum="4">1</AdjacentQubit>                </Qubit>            </QubitMatrix>        </ibmqx2>        <!-- 5 qubits real quantum chip Qx4 -->        <ibmqx4>            <QubitCount>5</QubitCount>            <QubitMatrix>                <Qubit QubitNum="1">                    <AdjacentQubit QubitNum="2">1</AdjacentQubit>                    <AdjacentQubit QubitNum="3">1</AdjacentQubit>                </Qubit>                <Qubit QubitNum="2">                    <AdjacentQubit QubitNum="1">1</AdjacentQubit>                    <AdjacentQubit QubitNum="3">1</AdjacentQubit>                </Qubit>                <Qubit QubitNum="3">                    <AdjacentQubit QubitNum="1">1</AdjacentQubit>                    <AdjacentQubit QubitNum="2">1</AdjacentQubit>                    <AdjacentQubit QubitNum="4">1</AdjacentQubit>                    <AdjacentQubit QubitNum="5">1</AdjacentQubit>                </Qubit>                <Qubit QubitNum="4">                    <AdjacentQubit QubitNum="3">1</AdjacentQubit>                    <AdjacentQubit QubitNum="5">1</AdjacentQubit>                </Qubit>                <Qubit QubitNum="5">                    <AdjacentQubit QubitNum="3">1</AdjacentQubit>                    <AdjacentQubit QubitNum="4">1</AdjacentQubit>                </Qubit>            </QubitMatrix>        </ibmqx4>    </backends></IBMQ>)"
#endif // !IBMQ_BACKENDS_CONFIG

#define IBMQ_BACKENDS_CONFIG_FILE ("IBMQ_backends_config.xml")

static bool loadIBMQuantumTopology(const std::string &xmlStr, const std::string& backendName, int &qubitsCnt, std::vector<std::vector<int>> &vec)
{
    std::ifstream f(IBMQ_BACKENDS_CONFIG_FILE);
    if (!f.good())
    {
	    TiXmlDocument doc;
	    FILE *xmlBakFd = fopen(IBMQ_BACKENDS_CONFIG_FILE, "w+");
		doc.Parse(xmlStr.c_str());
		doc.Print(xmlBakFd);
	    fclose(xmlBakFd);
    }

    return XmlConfigParam::loadQuantumTopoStructure(xmlStr, backendName, qubitsCnt, vec, IBMQ_BACKENDS_CONFIG_FILE);
}

QPANDA_END
#endif