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
#include "Factory.h"

USING_QPANDA
using namespace std;
static QuantumMachine* qvm;

QuantumMachine *QPanda::initQuantumMachine(QuantumMachine_type type)
{
    qvm = QuantumMachineFactory
        ::GetFactoryInstance().CreateByName(ConfigMap::getInstance()["QuantumMachine"]);// global
    if (!qvm->init(type))
    {
        return nullptr;
    }
    else
    {
        return qvm;
    }
}

void QPanda::destroyQuantumMachine(QuantumMachine * qvm)
{
    try
    {
        qvm->finalize();
    }
    catch (const std::exception&)
    {
        throw runtime_error("free quantum machine error");
    }

}

bool QPanda::init(QuantumMachine_type type)
{
    qvm = initQuantumMachine(type);
    return true;
}

void QPanda::finalize()
{
    qvm->finalize();
    delete qvm;
    qvm = nullptr;
}

Qubit* QPanda::qAlloc()
{
    return qvm->Allocate_Qubit();
}

Qubit* QPanda::qAlloc(size_t stQubitAddr)
{
    return qvm->Allocate_Qubit(stQubitAddr);
}

QVec QPanda::qAllocMany(size_t stQubitNumber)
{
    return qvm->Allocate_Qubits(stQubitNumber);
}

size_t QPanda::getAllocateQubitNum()
{
    return qvm->getAllocateQubit();
}

size_t QPanda::getAllocateCMem()
{
    return qvm->getAllocateCMem();
}


ClassicalCondition QPanda::cAlloc()
{
    return qvm->Allocate_CBit();
}

ClassicalCondition QPanda::cAlloc(size_t stCBitaddr)
{
    return qvm->Allocate_CBit(stCBitaddr);
}

vector<ClassicalCondition> QPanda::cAllocMany(size_t stCBitNumber)
{
    return qvm->Allocate_CBits(stCBitNumber);
}

void QPanda::cFree(ClassicalCondition& classical_cond)
{
    qvm->Free_CBit(classical_cond);
}

void cFreeAll(vector<ClassicalCondition> vCBit)
{
    qvm->Free_CBits(vCBit);
}

QMachineStatus* QPanda::getstat()
{
    return qvm->getStatus();
}

map<string, bool> QPanda::directlyRun(QProg & qProg)
{
    if (nullptr == qvm)
    {
        QCERR("qvm is not ideal machine");
        throw runtime_error("qvm is not ideal machine");
    }
    return qvm->directlyRun(qProg);
}

vector<pair<size_t, double>> QPanda::getProbTupleList(QVec & vQubit, int selectMax)
{
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
    qRunesTraverse.qProgToQRunes(&qProg);
    return qRunesTraverse.insturctionsQRunes();
}

string QPanda::qProgToQASM(QProg &pQPro)
{
    QProgToQASM pQASMTraverse;
    pQASMTraverse.qProgToQasm(&pQPro);
    return pQASMTraverse.insturctionsQASM();
}

void QPanda::qRunesToQProg(std::string sFilePath, QProg& newQProg)
{
    QRunesToQprog qRunesTraverse(sFilePath);
    qRunesTraverse.qRunesParser(newQProg);
}

static string dec2bin(unsigned n, size_t size)
{
    string binstr = "";
    for (int i = 0; i < size; ++i)
    {
        binstr = (char)((n & 1) + '0') + binstr;
        n >>= 1;
    }
    return binstr;
}

vector<pair<size_t, double>> QPanda::PMeasure(QVec& qubit_vector,
    int select_max)
{
    if (0 == qubit_vector.size())
        throw invalid_argument("qubit is zero");

    auto pmeasure_vector = qvm->PMeasure(qubit_vector,select_max);

    return pmeasure_vector;
}

vector<double> QPanda::PMeasure_no_index(QVec& qubit_vector)
{
    if (0 == qubit_vector.size())
        throw exception();

    auto pmeasure_vector = qvm->PMeasure(qubit_vector,-1);

    vector<double> temp;

    for(auto aiter : pmeasure_vector)
    {
        temp.push_back(aiter.second);
    }

    return temp;
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

static double RandomNumberGenerator()
{
    /*
    *  difine constant number in 16807 generator.
    */
    int  ia = 16807, im = 2147483647, iq = 127773, ir = 2836;
#ifdef _WIN32
    time_t rawtime;
    struct tm  timeinfo;
    time(&rawtime);
    localtime_s(&timeinfo, &rawtime);
    static int irandseed = timeinfo.tm_year + 70 *
        (timeinfo.tm_mon + 1 + 12 *
        (timeinfo.tm_mday + 31 *
            (timeinfo.tm_hour + 23 *
            (timeinfo.tm_min + 59 * timeinfo.tm_sec))));
#else
    time_t rawtime;
    struct tm * timeinfo;
    time(&rawtime);
    timeinfo = localtime(&rawtime);

    static int irandseed = timeinfo->tm_year + 70 *
        (timeinfo->tm_mon + 1 + 12 *
        (timeinfo->tm_mday + 31 *
            (timeinfo->tm_hour + 23 *
            (timeinfo->tm_min + 59 * timeinfo->tm_sec))));
#endif
    static int irandnewseed = 0;
    if (ia * (irandseed % iq) - ir * (irandseed / iq) >= 0)
    {
        irandnewseed = ia * (irandseed % iq) - ir * (irandseed / iq);
    }
    else
    {
        irandnewseed = ia * (irandseed % iq) - ir * (irandseed / iq) + im;
    }
    irandseed = irandnewseed;
    return (double)irandnewseed / im;
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
