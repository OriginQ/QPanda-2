/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.

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

#include "Core/Core.h"
#include "Core/Utilities/QProgInfo/ConfigMap.h"
#include "Core/Utilities/QProgInfo/QProgClockCycle.h"
#include "Core/Utilities/Compiler/QProgToOriginIR.h"
#include "Core/Utilities/Compiler/QProgToQuil.h"
#include "Core/Utilities/Compiler/QRunesToQProg.h"
#include "Core/Utilities/Tools/TranformQGateTypeStringAndEnum.h"
#include "Core/Utilities/Tools/QPandaException.h"
#include "Core/Utilities/Tools/Utils.h"
#include "Core/Utilities/Tools/OriginCollection.h"
#include "Core/QuantumMachine/Factory.h"
#include "Core/QuantumMachine/QuantumMachineFactory.h"
#include "Core/QuantumMachine/QCloudMachine.h"
#include "Core/QuantumNoise/NoiseModelV2.h"


USING_QPANDA
using namespace std;
QuantumMachine* global_quantum_machine = nullptr;

QuantumMachine *QPanda::initQuantumMachine(const QMachineType class_type)
{
    auto qm = QuantumMachineFactory::GetFactoryInstance()
        .CreateByType(class_type);

    if (nullptr == qm)
    {
        QCERR("quantum machine alloc fail");
        throw bad_alloc();
    }
    try
    {
        global_quantum_machine = qm;
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
    if (qvm)
    {
        global_quantum_machine = nullptr;
        qvm->finalize();
        delete qvm;
    }
}

bool QPanda::init(const QMachineType class_type)
{
    global_quantum_machine = initQuantumMachine(class_type);
    if(nullptr == global_quantum_machine)
    {
        return false;
    }

    return true;
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
size_t QPanda::getAllocateCMemNum()
{
	return getAllocateCMem();
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
    
QMachineStatus* QPanda::getstat()
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }
    return global_quantum_machine->getStatus();
}

map<string, bool> QPanda::directlyRun(QProg & qProg, const NoiseModel& noise_model)
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }
    return global_quantum_machine->directlyRun(qProg, noise_model);
}

prob_tuple QPanda::getProbTupleList(QVec vQubit, int selectMax)
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

prob_vec QPanda::getProbList(QVec vQubit, int selectMax)
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
prob_dict QPanda::getProbDict(QVec vQubit, int selectMax)
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

prob_tuple QPanda::probRunTupleList(QProg & qProg,QVec vQubit, int selectMax)
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

prob_vec QPanda::probRunList(QProg & qProg, QVec vQubit, int selectMax)
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
prob_dict QPanda::probRunDict(QProg & qProg, QVec vQubit, int selectMax)
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

prob_tuple QPanda::PMeasure(QVec qubit_vector,
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

prob_tuple QPanda::pMeasure(QVec qubit_vector,
	int select_max)
{
	return PMeasure(qubit_vector, select_max);
}

prob_vec QPanda::PMeasure_no_index(QVec qubit_vector)
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

prob_vec QPanda::pMeasureNoIndex(QVec qubit_vector)
{
	return PMeasure_no_index(qubit_vector);
}

prob_vec QPanda::accumulateProbability(prob_vec & prob_list)
{
    prob_vec accumulate_prob(prob_list);
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


map<string, size_t> QPanda::quick_measure(QVec qubit_vector, int shots,
    prob_vec& accumulate_probabilites)
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

map<string, size_t> QPanda::runWithConfiguration(QProg & qProg, vector<ClassicalCondition>& vCBit, int shots, const NoiseModel& noise_model)
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
    return global_quantum_machine->runWithConfiguration(qProg, vCBit, doc, noise_model);
}

map<string, size_t> QPanda::quickMeasure(QVec vQubit, int shots)
{
    auto temp = dynamic_cast<IdealMachineInterface *>(global_quantum_machine);
    if (nullptr == temp)
    {
        QCERR("global_quantum_machine is not ideal machine");
        throw runtime_error("global_quantum_machine is not ideal machine");
    }
    return temp->quickMeasure(vQubit, shots);
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

QGate QPanda::QOracle(const QVec& qubits, const EigenMatrixXc& matrix)
{
    return(QOracle(qubits, Eigen_to_QStat(matrix)));
}

void QPanda::cFreeAll()
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }

    global_quantum_machine->cFreeAll();
    return ;
}

void QPanda::qFree(Qubit *q)
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }

    global_quantum_machine->qFree(q);
    return ;
}

void QPanda::qFreeAll(QVec &qv)
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }

    global_quantum_machine->qFreeAll(qv);
    return ;
}

void QPanda::qFreeAll()
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }

    global_quantum_machine->qFreeAll();
    return ;
}

void QPanda::cFreeAll(std::vector<ClassicalCondition> &vCBit)
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }

    global_quantum_machine->cFreeAll(vCBit);
}


size_t QPanda::get_allocate_qubits(QVec &qubits)
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }

    return global_quantum_machine->get_allocate_qubits(qubits);
}

size_t QPanda:: get_allocate_cbits(std::vector<ClassicalCondition> &cc_vec)
{
    if (nullptr == global_quantum_machine)
    {
        QCERR("global_quantum_machine init fail");
        throw init_fail("global_quantum_machine init fail");
    }

    return global_quantum_machine->get_allocate_cbits(cc_vec);
}
