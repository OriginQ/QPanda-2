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
QuantumMachine* global_quantum_machine = nullptr;

QuantumMachine *QPanda::initQuantumMachine(const QMachineType class_type)
{
    auto class_name = QMachineTypeTarnfrom::getInstance()[class_type];
	QuantumMachine* qm;
    switch (class_type)
    {
    case QMachineType::CPU:
        qm = new CPUQVM();
        break;
    case QMachineType::CPU_SINGLE_THREAD:
		qm = new CPUSingleThreadQVM();
        break;
    case QMachineType::GPU:
		qm = new GPUQVM();
        break;
    case QMachineType::NOISE:
		qm = new NoiseQVM();
        break;
    default:
		qm = nullptr;
        break;
    }
    if (nullptr == qm)
    {
        QCERR("quantum machine alloc fail");
        throw bad_alloc();
    }
    try
    {
        qm->init();
        return qm;
    }
    catch (const init_fail&e)
    {
        delete qm;
        qm = nullptr;
        return nullptr;
    }
}

void QPanda::destroyQuantumMachine(QuantumMachine *qvm)
{
    if (!qvm)
        qvm->finalize();
}

bool QPanda::init(const QMachineType class_type)
{
    global_quantum_machine = initQuantumMachine(class_type);
    return global_quantum_machine;
}

void QPanda::finalize()
{
    if (nullptr == global_quantum_machine)
    {
        return;
    }
    global_quantum_machine->finalize();
    delete global_quantum_machine;
    global_quantum_machine = nullptr;
}

Qubit* QPanda::qAlloc()
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }
    return global_quantum_machine->allocateQubit();
}

Qubit* QPanda::qAlloc(size_t stQubitAddr)
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }
    return global_quantum_machine->allocateQubitThroughPhyAddress(stQubitAddr);
}

QVec QPanda::qAllocMany(size_t stQubitNumber)
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }
    return global_quantum_machine->allocateQubits(stQubitNumber);
}

size_t QPanda::getAllocateQubitNum()
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }
    return global_quantum_machine->getAllocateQubit();
}

size_t QPanda::getAllocateCMem()
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }
    return global_quantum_machine->getAllocateCMem();
}


ClassicalCondition QPanda::cAlloc()
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }
    return global_quantum_machine->allocateCBit();
}

ClassicalCondition QPanda::cAlloc(size_t stCBitaddr)
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }
    return global_quantum_machine->allocateCBit(stCBitaddr);
}

vector<ClassicalCondition> QPanda::cAllocMany(size_t stCBitNumber)
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }
    return global_quantum_machine->allocateCBits(stCBitNumber);
}

void QPanda::cFree(ClassicalCondition& classical_cond)
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }
    global_quantum_machine->Free_CBit(classical_cond);
}

void cFreeAll(vector<ClassicalCondition> vCBit)
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }
    global_quantum_machine->Free_CBits(vCBit);
}

QMachineStatus* QPanda::getstat()
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }
    return global_quantum_machine->getStatus();
}

map<string, bool> QPanda::directlyRun(QProg & qProg)
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }
    return global_quantum_machine->directlyRun(qProg);
}

vector<pair<size_t, double>> QPanda::getProbTupleList(QVec & vQubit, int selectMax)
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }
    auto temp = dynamic_cast<IdealMachineInterface *>(global_quantum_machine);
    if (nullptr == temp)
    {
        QCERR("global_quantum_machine is not ideal machine");
        throw runtime_error("global_quantum_machine is not ideal machine");
    }
    return temp->getProbTupleList(vQubit, selectMax);
}

vector<double> QPanda::getProbList(QVec & vQubit, int selectMax)
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }
    auto temp = dynamic_cast<IdealMachineInterface *>(global_quantum_machine);
    if (nullptr == temp)
    {
        QCERR("global_quantum_machine is not ideal machine");
        throw runtime_error("global_quantum_machine is not ideal machine");
    }
    return temp->getProbList(vQubit, selectMax);
}
map<string, double> QPanda::getProbDict(QVec & vQubit, int selectMax)
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }
    auto temp = dynamic_cast<IdealMachineInterface *>(global_quantum_machine);
    if (nullptr == temp)
    {
        QCERR("global_quantum_machine is not ideal machine");
        throw runtime_error("global_quantum_machine is not ideal machine");
    }
    return temp->getProbDict(vQubit, selectMax);
}

vector<pair<size_t, double>> QPanda::probRunTupleList(QProg & qProg,QVec & vQubit, int selectMax)
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }
    auto temp = dynamic_cast<IdealMachineInterface *>(global_quantum_machine);
    if (nullptr == temp)
    {
        QCERR("global_quantum_machine is not ideal machine");
        throw runtime_error("global_quantum_machine is not ideal machine");
    }
    return temp->probRunTupleList(qProg, vQubit, selectMax);
}

vector<double> QPanda::probRunList(QProg & qProg, QVec & vQubit, int selectMax)
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }
    auto temp = dynamic_cast<IdealMachineInterface *>(global_quantum_machine);
    if (nullptr == temp)
    {
        QCERR("global_quantum_machine is not ideal machine");
        throw runtime_error("global_quantum_machine is not ideal machine");
    }
    return temp->probRunList(qProg, vQubit, selectMax);
}
map<string, double> QPanda::probRunDict(QProg & qProg, QVec & vQubit, int selectMax)
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }
    auto temp = dynamic_cast<IdealMachineInterface *>(global_quantum_machine);
    if (nullptr == temp)
    {
        QCERR("global_quantum_machine is not ideal machine");
        throw runtime_error("global_quantum_machine is not ideal machine");
    }
    return temp->probRunDict(qProg, vQubit, selectMax);
}

vector<pair<size_t, double>> QPanda::PMeasure(QVec& qubit_vector,
    int select_max)
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }
    auto temp = dynamic_cast<IdealMachineInterface *>(global_quantum_machine);
    if (nullptr == temp)
    {
        QCERR("global_quantum_machine is not ideal machine");
        throw runtime_error("global_quantum_machine is not ideal machine");
    }
    return temp->PMeasure(qubit_vector, select_max);
}

vector<double> QPanda::PMeasure_no_index(QVec& qubit_vector)
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }
    auto temp = dynamic_cast<IdealMachineInterface *>(global_quantum_machine);
    if (nullptr == temp)
    {
        QCERR("global_quantum_machine is not ideal machine");
        throw runtime_error("global_quantum_machine is not ideal machine");
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
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
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

map<string, size_t> QPanda::runWithConfiguration(QProg & qProg, vector<ClassicalCondition>& vCBit, int shots)
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }
    rapidjson::Document doc;
    doc.Parse("{}");
    doc.AddMember("shots",
        shots,
        doc.GetAllocator());
    return global_quantum_machine->runWithConfiguration(qProg, vCBit, doc);
}

map<string, size_t> QPanda::quickMeasure(QVec& vQubit, int shots)
{
    auto temp = dynamic_cast<IdealMachineInterface *>(global_quantum_machine);
    if (nullptr == temp)
    {
        QCERR("global_quantum_machine is not ideal machine");
        throw runtime_error("global_quantum_machine is not ideal machine");
    }
    return temp->quickMeasure(vQubit, shots);
}

QProg QPanda::MeasureAll(QVec vQubit, vector<ClassicalCondition> vCBit)
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
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine is nullptr");
        throw invalid_argument("global_quantum_machine is nullptr");
    }
    return global_quantum_machine->getQState();
}

