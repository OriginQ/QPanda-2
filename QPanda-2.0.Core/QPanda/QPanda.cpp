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

#include "QPanda.h"
#include "QPanda/ConfigMap.h"
#include "QuantumMachine/QuantumMachineFactory.h"
#include "Transform/QProgToQRunes.h"
#include "Transform/QProgToQuil.h"
#include "Transform/QRunesToQProg.h"
#include "QPanda/TranformQGateTypeStringAndEnum.h"
#include "Transform/QProgClockCycle.h"


static QuantumMachine* qvm;


void init()
{
    qvm = QuantumMachineFactory
        ::GetFactoryInstance().CreateByName(ConfigMap::getInstance()["QuantumMachine"]);// global
	qvm->init();
}

void finalize()
{
	qvm->finalize();
	delete qvm;
	qvm = nullptr;
}

Qubit * qAlloc()
{
	return qvm->Allocate_Qubit();
}

Qubit * qAlloc(size_t stQubitAddr)
{
    return qvm->Allocate_Qubit(stQubitAddr);
}

size_t getAllocateQubitNum()
{
	return qvm->getAllocateQubit();
}

size_t getAllocateCMem()
{
	return qvm->getAllocateCMem();
}


void qFree(Qubit * q)
{
	qvm->Free_Qubit(q);
}

CBit * cAlloc()
{
	return qvm->Allocate_CBit();
}

CBit * cAlloc(size_t stCBitaddr)
{
    return qvm->Allocate_CBit(stCBitaddr);
}

void cFree(CBit * c)
{
	qvm->Free_CBit(c);
}

void load(QProg & q)
{
	qvm->load(q);
}

void append(QProg & q)
{
	qvm->append(q);
}

QMachineStatus* getstat()
{
	return qvm->getStatus();
}

QResult* getResult()
{
	return qvm->getResult();
}

ClassicalCondition bind_a_cbit(CBit * c)
{
	return ClassicalCondition(c);
}

void run()
{
	qvm->run();
}




map<string, bool> getResultMap()
{
	return qvm->getResult()->getResultMap();
}

bool getCBitValue(CBit * cbit)
{
	auto resmap = getResultMap();
	return resmap[cbit->getName()];
}



string qProgToQRunes(QProg &qProg)
{
    QProgToQRunes qRunesTraverse;
    qRunesTraverse.qProgToQRunes(&qProg);
    return qRunesTraverse.insturctionsQRunes();
}

string qProgToQasm(QProg &pQPro)
{
    QProgToQASM pQASMTraverse;
    pQASMTraverse.qProgToQasm(&pQPro);
    return pQASMTraverse.insturctionsQASM();
}

QProg qRunesToProg()
{
    QRunesToQprog qRunesTraverse;
    return qRunesTraverse.qRunesParser();
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

vector<pair<size_t, double>> PMeasure(vector<Qubit*>& qubit_vector,
    int select_max)
{
    if (0 == qubit_vector.size())
        throw exception();
    Qnum vqubit;
    for (auto aiter = qubit_vector.begin(); aiter != qubit_vector.end(); ++aiter)
    {
        vqubit.push_back((*aiter)->getPhysicalQubitPtr()->getQubitAddr());
    }

    vector<pair<size_t, double>> pmeasure_vector;
    auto gates = qvm->getQuantumGates();
    gates->pMeasure(vqubit, pmeasure_vector, select_max);

    return pmeasure_vector;
}

vector<double> PMeasure_no_index(vector<Qubit*>& qubit_vector)
{
    if (0 == qubit_vector.size())
        throw exception();
    Qnum vqubit;
    for (auto aiter = qubit_vector.begin(); aiter != qubit_vector.end(); ++aiter)
    {
        vqubit.push_back((*aiter)->getPhysicalQubitPtr()->getQubitAddr());
    }
    vector<double> pmeasure_vector;
    auto gates = qvm->getQuantumGates();
    gates->pMeasure(vqubit, pmeasure_vector);

    return pmeasure_vector;
}

vector<double> accumulateProbability(vector<double>& prob_list)
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

map<string, size_t> quick_measure(vector<Qubit*>& qubit_vector, int shots,
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

size_t getQProgClockCycle(QProg &prog)
{
    QProgClockCycle counter(qvm->getGateTimeMap());
    return counter.countQProgClockCycle(&prog);
}
