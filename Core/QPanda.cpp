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

#include "Core/QPanda.h"
#include "Core/Utilities/ConfigMap.h"
#include "Core/Utilities/Transform/QProgToQRunes.h"
#include "Core/Utilities/Transform/QProgToQuil.h"
#include "Core/Utilities/Transform/QRunesToQProg.h"
#include "Core/Utilities/TranformQGateTypeStringAndEnum.h"
#include "Core/Utilities/Transform/QProgClockCycle.h"
#include "Core/Utilities/OriginCollection.h"
#include "Core/QuantumMachine/Factory.h"
#include "Core/Utilities/QPandaException.h"
#include "Core/Utilities/Utilities.h"
USING_QPANDA
using namespace std;
static QuantumMachine* qvm = nullptr;

QuantumMachine *QPanda::initQuantumMachine(const QMachineType class_type)
{
    auto class_name = QMachineTypeTarnfrom::getInstance()[class_type];
    switch (class_type)
    {
    case QMachineType::CPU:
        qvm = new CPUQVM();
        break;
    case QMachineType::CPU_SINGLE_THREAD:
        qvm = new CPUSingleThreadQVM();
        break;
    case QMachineType::GPU:
        qvm = new GPUQVM();
        break;
    case QMachineType::NOISE:
        qvm = new NoiseQVM();
        break;
    default:
        qvm = nullptr;
        break;
    }
    if (nullptr == qvm)
    {
        QCERR("quantum machine alloc fail");
        throw bad_alloc();
    }
    try
    {
        qvm->init();
        return qvm;
    }
    catch (const init_fail&e)
    {
        delete qvm;
        qvm = nullptr;
        return nullptr;
    }

}

void QPanda::destroyQuantumMachine(QuantumMachine * qvm)
{
    if (nullptr == qvm)
    {
        return;
    }
    else
    {
        qvm->finalize();
    }
}

bool QPanda::init(const QMachineType class_type )
{
    qvm = initQuantumMachine(class_type);
    return true;
}

void QPanda::finalize()
{
    if (nullptr == qvm)
    {
        return;
    }
    qvm->finalize();
    delete qvm;
    qvm = nullptr;
}

Qubit* QPanda::qAlloc()
{
    if (nullptr == qvm)
    {
        QCERR("qvm init fail");
        throw init_fail("qvm init fail");
    }
    return qvm->allocateQubit();
}

Qubit* QPanda::qAlloc(size_t stQubitAddr)
{
    if (nullptr == qvm)
    {
        QCERR("qvm init fail");
        throw init_fail("qvm init fail");
    }
    return qvm->allocateQubitThroughPhyAddress(stQubitAddr);
}

QVec QPanda::qAllocMany(size_t stQubitNumber)
{
    if (nullptr == qvm)
    {
        QCERR("qvm init fail");
        throw init_fail("qvm init fail");
    }
    return qvm->allocateQubits(stQubitNumber);
}

size_t QPanda::getAllocateQubitNum()
{
    if (nullptr == qvm)
    {
        QCERR("qvm init fail");
        throw init_fail("qvm init fail");
    }
    return qvm->getAllocateQubit();
}

size_t QPanda::getAllocateCMem()
{
    if (nullptr == qvm)
    {
        QCERR("qvm init fail");
        throw init_fail("qvm init fail");
    }
    return qvm->getAllocateCMem();
}


ClassicalCondition QPanda::cAlloc()
{
    if (nullptr == qvm)
    {
        QCERR("qvm init fail");
        throw init_fail("qvm init fail");
    }
    return qvm->allocateCBit();
}

ClassicalCondition QPanda::cAlloc(size_t stCBitaddr)
{
    if (nullptr == qvm)
    {
        QCERR("qvm init fail");
        throw init_fail("qvm init fail");
    }
    return qvm->allocateCBit(stCBitaddr);
}

vector<ClassicalCondition> QPanda::cAllocMany(size_t stCBitNumber)
{
    if (nullptr == qvm)
    {
        QCERR("qvm init fail");
        throw init_fail("qvm init fail");
    }
    return qvm->allocateCBits(stCBitNumber);
}

void QPanda::cFree(ClassicalCondition& classical_cond)
{
    if (nullptr == qvm)
    {
        QCERR("qvm init fail");
        throw init_fail("qvm init fail");
    }
    qvm->Free_CBit(classical_cond);
}

void cFreeAll(vector<ClassicalCondition> vCBit)
{
    if (nullptr == qvm)
    {
        QCERR("qvm init fail");
        throw init_fail("qvm init fail");
    }
    qvm->Free_CBits(vCBit);
}

QMachineStatus* QPanda::getstat()
{
    if (nullptr == qvm)
    {
        QCERR("qvm init fail");
        throw init_fail("qvm init fail");
    }
    return qvm->getStatus();
}

map<string, bool> QPanda::directlyRun(QProg & qProg)
{
    if (nullptr == qvm)
    {
        QCERR("qvm init fail");
        throw init_fail("qvm init fail");
    }
    return qvm->directlyRun(qProg);
}

vector<pair<size_t, double>> QPanda::getProbTupleList(QVec & vQubit, int selectMax)
{
    if (nullptr == qvm)
    {
        QCERR("qvm init fail");
        throw init_fail("qvm init fail");
    }
    auto temp = dynamic_cast<IdealMachineInterface *>(qvm);
    if (nullptr == temp)
    {
        QCERR("qvm is not ideal machine");
        throw runtime_error("qvm is not ideal machine");
    }
    return temp->getProbTupleList(vQubit, selectMax);
}

vector<double> QPanda::getProbList(QVec & vQubit, int selectMax)
{
    if (nullptr == qvm)
    {
        QCERR("qvm init fail");
        throw init_fail("qvm init fail");
    }
    auto temp = dynamic_cast<IdealMachineInterface *>(qvm);
    if (nullptr == temp)
    {
        QCERR("qvm is not ideal machine");
        throw runtime_error("qvm is not ideal machine");
    }
    return temp->getProbList(vQubit, selectMax);
}
map<string, double> QPanda::getProbDict(QVec & vQubit, int selectMax)
{
    if (nullptr == qvm)
    {
        QCERR("qvm init fail");
        throw init_fail("qvm init fail");
    }
    auto temp = dynamic_cast<IdealMachineInterface *>(qvm);
    if (nullptr == temp)
    {
        QCERR("qvm is not ideal machine");
        throw runtime_error("qvm is not ideal machine");
    }
    return temp->getProbDict(vQubit, selectMax);
}

vector<pair<size_t, double>> QPanda::probRunTupleList(QProg & qProg,QVec & vQubit, int selectMax)
{
    if (nullptr == qvm)
    {
        QCERR("qvm init fail");
        throw init_fail("qvm init fail");
    }
    auto temp = dynamic_cast<IdealMachineInterface *>(qvm);
    if (nullptr == temp)
    {
        QCERR("qvm is not ideal machine");
        throw runtime_error("qvm is not ideal machine");
    }
    return temp->probRunTupleList(qProg, vQubit, selectMax);
}



vector<double> QPanda::probRunList(QProg & qProg, QVec & vQubit, int selectMax)
{
    if (nullptr == qvm)
    {
        QCERR("qvm init fail");
        throw init_fail("qvm init fail");
    }
    auto temp = dynamic_cast<IdealMachineInterface *>(qvm);
    if (nullptr == temp)
    {
        QCERR("qvm is not ideal machine");
        throw runtime_error("qvm is not ideal machine");
    }
    return temp->probRunList(qProg, vQubit, selectMax);
}
map<string, double> QPanda::probRunDict(QProg & qProg, QVec & vQubit, int selectMax)
{
    if (nullptr == qvm)
    {
        QCERR("qvm init fail");
        throw init_fail("qvm init fail");
    }
    auto temp = dynamic_cast<IdealMachineInterface *>(qvm);
    if (nullptr == temp)
    {
        QCERR("qvm is not ideal machine");
        throw runtime_error("qvm is not ideal machine");
    }
    return temp->probRunDict(qProg, vQubit, selectMax);
}

string QPanda::qProgToQRunes(QProg &qProg)
{

    QProgToQRunes qRunesTraverse;
    qRunesTraverse.qProgToQRunes(dynamic_cast<AbstractQuantumProgram*>(qProg.getImplementationPtr().get()));
    return qRunesTraverse.insturctionsQRunes();
}

string QPanda::qProgToQASM(QProg &pQProg)
{
    QProgToQASM pQASMTraverse;
    pQASMTraverse.progToQASM(dynamic_cast<AbstractQuantumProgram*>(pQProg.getImplementationPtr().get()));
    return pQASMTraverse.insturctionsQASM();
}

void QPanda::qRunesToQProg(std::string sFilePath, QProg& newQProg)
{
    QRunesToQprog qRunesTraverse(sFilePath);
    qRunesTraverse.qRunesParser(newQProg);
}



vector<pair<size_t, double>> QPanda::PMeasure(QVec& qubit_vector,
    int select_max)
{
    if (nullptr == qvm)
    {
        QCERR("qvm init fail");
        throw init_fail("qvm init fail");
    }
    auto temp = dynamic_cast<IdealMachineInterface *>(qvm);
    if (nullptr == temp)
    {
        QCERR("qvm is not ideal machine");
        throw runtime_error("qvm is not ideal machine");
    }
    return temp->PMeasure(qubit_vector, select_max);
}

vector<double> QPanda::PMeasure_no_index(QVec& qubit_vector)
{
    if (nullptr == qvm)
    {
        QCERR("qvm init fail");
        throw init_fail("qvm init fail");
    }
    auto temp = dynamic_cast<IdealMachineInterface *>(qvm);
    if (nullptr == temp)
    {
        QCERR("qvm is not ideal machine");
        throw runtime_error("qvm is not ideal machine");
    }
    return temp->PMeasure_no_index(qubit_vector);
}



vector<double> QPanda::accumulateProbability(vector<double>& prob_list)
{
    vector<double> accumulate_prob(prob_list);
    for (int i = 1; i<prob_list.size(); ++i)
    {
        accumulate_prob[i] = accumulate_prob[i - 1] + prob_list[i];
    }
    return accumulate_prob;
}

static void add_up_a_map(map<string, size_t> &meas_result, string key)
{
    if (meas_result.find(key) != meas_result.end())
    {
        meas_result[key]++;
    }
    else
    {
        meas_result[key] = 1;
    }
}

map<string, size_t> QPanda::quick_measure(QVec& qubit_vector, int shots,
    vector<double>& accumulate_probabilites)
{
    if (nullptr == qvm)
    {
        QCERR("qvm init fail");
        throw init_fail("qvm init fail");
    }
    map<string, size_t> meas_result;
    for (int i = 0; i < shots; ++i)
    {
        double rng = RandomNumberGenerator();
        if (rng < accumulate_probabilites[0])
            add_up_a_map(meas_result, dec2bin(0, qubit_vector.size()));
        for (int i = 1; i < accumulate_probabilites.size(); ++i)
        {
            if (rng < accumulate_probabilites[i] &&
                rng >= accumulate_probabilites[i - 1]
                )
            {
                add_up_a_map(meas_result,
                    dec2bin(i, qubit_vector.size())
                );
                break;
            }
        }
    }
    return meas_result;
}

size_t QPanda::getQProgClockCycle(QProg &prog)
{
    QProgClockCycle counter(qvm->getGateTimeMap());
    return counter.countQProgClockCycle(&prog);
}

map<string, size_t> QPanda::runWithConfiguration(QProg & qProg, vector<ClassicalCondition>& vCBit, int shots)
{
    if (nullptr == qvm)
    {
        QCERR("qvm init fail");
        throw init_fail("qvm init fail");
    }
    rapidjson::Document doc;
    doc.Parse("{}");
    doc.AddMember("shots",
        shots,
        doc.GetAllocator());
    return qvm->runWithConfiguration(qProg, vCBit, doc);
}

map<string, size_t> QPanda::quickMeasure(QVec& vQubit, int shots)
{
    auto temp = dynamic_cast<IdealMachineInterface *>(qvm);
    if (nullptr == temp)
    {
        QCERR("qvm is not ideal machine");
        throw runtime_error("qvm is not ideal machine");
    }
    return temp->quickMeasure(vQubit, shots);
}

QProg QPanda::MeasureAll(QVec& vQubit, vector<ClassicalCondition> &vCBit)
{
    QProg qprog = CreateEmptyQProg();
    if (vQubit.size() != vCBit.size())
    {
        QCERR("vQubit != vCBit");
        throw invalid_argument("vQubit != vCBit");
    }
    for (size_t i = 0; i < vQubit.size(); i++)
    {
        qprog << Measure(vQubit[i], vCBit[i]);
    }
    return qprog;
}

QStat QPanda::getQState()
{
    if (nullptr == qvm)
    {
        QCERR("qvm is nullptr");
        throw invalid_argument("qvm is nullptr");
    }
    return qvm->getQState();
}
