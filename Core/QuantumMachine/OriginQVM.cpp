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

#include "OriginQuantumMachine.h"
#include "Factory.h"
#include "QuantumMachineFactory.h"
#include "Utilities/ConfigMap.h"
#include "config.h"
#include "VirtualQuantumProcessor/GPUImplQPU.h"
#include "VirtualQuantumProcessor/CPUImplQPU.h"
#include "VirtualQuantumProcessor/CPUImplQPUSingleThread.h"

USING_QPANDA
using namespace std;
bool OriginQVM::init(QuantumMachine_type type)
{
    _Qubit_Pool = 
        QubitPoolFactory::GetFactoryInstance().
        GetPoolWithoutTopology(_Config.maxQubit);
    _CMem = 
        CMemFactory::GetFactoryInstance().
        GetInstanceFromSize(_Config.maxCMem);
    _QResult =
        QResultFactory::GetFactoryInstance().
        GetEmptyQResult();
    _QMachineStatus =
        QMachineStatusFactory::
        GetQMachineStatus();

    bool is_success = false;
    if (CPU == type)
    {
        _pGates = new CPUImplQPU();
        is_success = true;
    }
    else if (GPU == type)
    {
    #ifdef USE_CUDA
        _pGates = new GPUImplQPU();
        is_success = true;
    #else
        _pGates = nullptr;
        is_success = false;;
    #endif // USE_CUDA
    }
    else if (CPU_SINGLE_THREAD == type)
    {
        _pGates = new CPUImplQPUSingleThread();
        is_success = true;
    }
    else
    {
        is_success = false;
    }

    return is_success;
}


Qubit * OriginQVM::Allocate_Qubit()
{
    if (_Qubit_Pool == nullptr)
    {
        // check if the pointer is nullptr
        // Before init
        // After finalize
        QCERR("Must initialize the system first");
        throw(runtime_error("Must initialize the system first"));
    }
    else
    {
        return _Qubit_Pool->Allocate_Qubit();
    }
        
}

QVec OriginQVM::Allocate_Qubits(size_t qubitNumber)
{
    if (_Qubit_Pool == nullptr)
    {
        // check if the pointer is nullptr
        // Before init
        // After finalize
        QCERR("Must initialize the system first");
        throw(runtime_error("Must initialize the system first"));
    }
    QVec vQubit;
    for (size_t i = 0; i < qubitNumber; i++)
    {
        vQubit.push_back(_Qubit_Pool->Allocate_Qubit());
    }
    return vQubit;
}


ClassicalCondition OriginQVM::Allocate_CBit()
{
    if (_CMem == nullptr)
    {
        // check if the pointer is nullptr
        // Before init
        // After finalize
        QCERR("Must initialize the system first");
        throw(runtime_error("Must initialize the system first"));
    }
    else
    {
        auto cbit = _CMem->Allocate_CBit();
        ClassicalCondition temp(cbit);
        return temp;
    }
}


vector<ClassicalCondition> OriginQVM::Allocate_CBits(size_t cbitNumber)
{
    if (_CMem == nullptr)
    {
        // check if the pointer is nullptr
        // Before init
        // After finalize
        QCERR("Must initialize the system first");
        throw(runtime_error("Must initialize the system first"));
    }
    else
    {
        vector<ClassicalCondition> cbit_vector;
        for (size_t i = 0; i < cbitNumber; i++)
        {
            auto cbit = _CMem->Allocate_CBit();
            cbit_vector.push_back(cbit);
        }
        return cbit_vector;
    }
}


ClassicalCondition OriginQVM::Allocate_CBit(size_t stCBitaddr)
{
    if (_CMem == nullptr)
    {
        // check if the pointer is nullptr
        // Before init
        // After finalize
        QCERR("Must initialize the system first");
        throw(runtime_error("Must initialize the system first"));
    }
    else
    {
        auto cbit = _CMem->Allocate_CBit(stCBitaddr);
        ClassicalCondition temp(cbit);
        return temp;
    }
}

Qubit * OriginQVM::Allocate_Qubit(size_t stQubitNum)
{
    if (_Qubit_Pool == nullptr)
    {
        // check if the pointer is nullptr
        // Before init
        // After finalize
        QCERR("Must initialize the system first");
        throw(runtime_error("Must initialize the system first"));
    }
    else
    {
        return _Qubit_Pool->Allocate_Qubit(stQubitNum);
    }
}

void OriginQVM::Free_Qubit(Qubit *qubit)
{
    this->_Qubit_Pool->Free_Qubit(qubit);
}

void OriginQVM::Free_Qubits(QVec &vQubit)
{
    for (auto iter : vQubit)
    {
        this->_Qubit_Pool->Free_Qubit(iter);
    }
}

void OriginQVM::Free_CBit(ClassicalCondition & class_cond)
{
    auto cbit = class_cond.getExprPtr()->getCBit();
    if (nullptr == cbit)
    {
        QCERR("cbit is null");
        throw invalid_argument("cbit is null");
    }
    _CMem->Free_CBit(cbit);
}

void OriginQVM::Free_CBits(vector<ClassicalCondition> & vCBit)
{
    for (auto iter : vCBit)
    {
        auto cbit = iter.getExprPtr()->getCBit();
        if (nullptr == cbit)
        {
            QCERR("cbit is null");
            throw invalid_argument("cbit is null");
        }
        this->_CMem->Free_CBit(cbit);
    }
}

void OriginQVM::run(QProg & node)
{
    
    _pParam = new QuantumGateParam();

    _pParam->m_qbit_number = _Qubit_Pool->getMaxQubit(); 

    _pGates->initState(_pParam);

    node.getImplementationPtr()->execute(_pGates, _pParam);

    /* aiter has been used in line 120 */
    for (auto aiter : _pParam->m_return_value)
    {
        _QResult->append(aiter);
    }

    _pGates->endGate(_pParam,nullptr);
    delete _pParam;
    _pParam = nullptr;
    return;
}

QMachineStatus * OriginQVM::getStatus() const
{
    return _QMachineStatus;
}

QResult * OriginQVM::getResult()
{
    return _QResult;
}

void OriginQVM::finalize()
{

    _QProgram.reset();
    delete _Qubit_Pool;
    delete _CMem;
    delete _QResult;
    delete _QMachineStatus;
    delete _pGates;

    _Qubit_Pool = nullptr;
    _CMem = nullptr;
    _QResult = nullptr;
    _QMachineStatus = nullptr;
    _pGates = nullptr;
}

size_t OriginQVM::getAllocateQubit()
{
    return _Qubit_Pool->getMaxQubit() - _Qubit_Pool->getIdleQubit();
}

size_t OriginQVM::getAllocateCMem()
{
    return _CMem->getMaxMem()- _CMem->getIdleMem();
}

map<string, bool> OriginQVM::getResultMap()
{
    if (nullptr == _QResult)
    {
        QCERR("QResult is null");
        throw runtime_error("QResult is null");
    }
    return _QResult->getResultMap();
}

vector<pair<size_t, double>> OriginQVM::PMeasure(QVec qubit_vector, int select_max)
{
    if (0 == qubit_vector.size())
    {
        QCERR("the size of qubit_vector is zero");
        throw invalid_argument("the size of qubit_vector is zero");
    }

    Qnum vqubit;
    for (auto aiter = qubit_vector.begin(); aiter != qubit_vector.end(); ++aiter)
    {
        vqubit.push_back((*aiter)->getPhysicalQubitPtr()->getQubitAddr());
    }

    vector<pair<size_t, double>> pmeasure_vector;
    _pGates->pMeasure(vqubit, pmeasure_vector, select_max);

    return pmeasure_vector;
}

vector<double> OriginQVM::PMeasure_no_index(QVec qubit_vector)
{
    if (0 == qubit_vector.size())
    {
        QCERR("the size of qubit_vector is zero");
        throw invalid_argument("the size of qubit_vector is zero");
    }

    Qnum vqubit;
    for (auto aiter = qubit_vector.begin(); aiter != qubit_vector.end(); ++aiter)
    {
        vqubit.push_back((*aiter)->getPhysicalQubitPtr()->getQubitAddr());
    }
    vector<double> pmeasure_vector;
    _pGates->pMeasure(vqubit, pmeasure_vector);

    return pmeasure_vector;
}

map<string, bool> OriginQVM::directlyRun(QProg & qProg)
{
    run(qProg);
    return _QResult->getResultMap();
}

vector<pair<size_t, double>> OriginQVM::getProbTupleList(QVec vQubit,  int selectMax)
{
    if (0 == vQubit.size())
    {
        QCERR("the size of qubit_vector is zero");
        throw invalid_argument("the size of qubit_vector is zero");
    }

    vector<pair<size_t, double>> vResult;
    Qnum vqubitAddr;
    for (auto aiter = vQubit.begin(); aiter != vQubit.end(); ++aiter)
    {
        vqubitAddr.push_back((*aiter)->getPhysicalQubitPtr()->getQubitAddr());
    }
    _pGates->pMeasure(vqubitAddr, vResult, selectMax);
    return vResult;
}

vector<double> OriginQVM::getProbList(QVec vQubit, int selectMax)
{
    if (0 == vQubit.size())
    {
        QCERR("the size of qubit_vector is zero");
        throw invalid_argument("the size of qubit_vector is zero");
    }
    vector<double> vResult;
    Qnum vqubitAddr;
    for (auto aiter = vQubit.begin(); aiter != vQubit.end(); ++aiter)
    {
        vqubitAddr.push_back((*aiter)->getPhysicalQubitPtr()->getQubitAddr());
    }
    _pGates->pMeasure(vqubitAddr, vResult);
    return vResult;
}

static string dec2bin(size_t n, size_t size)
{
    string binstr = "";
    for (int i = 0; i < size; ++i)
    {
        binstr = (char)((n & 1) + '0') + binstr;
        n >>= 1;
    }
    return binstr;
}
string OriginQVM::ResultToBinaryString(vector<ClassicalCondition> & vCBit)
{
    string sTemp;
    if (nullptr == _QResult)
    {
        QCERR("_QResult is null");
        throw runtime_error("_QResult is null");
    }
    auto resmap = _QResult->getResultMap();
    for (auto c : vCBit)
    {
        auto cbit = c.getExprPtr()->getCBit();
        if (nullptr == cbit)
        {
            QCERR("vcbit is error");
            throw invalid_argument("vcbit is error");
        }
        if (resmap[cbit->getName()])
        {
            sTemp.push_back('1');
        }
        else
        {
            sTemp.push_back('0');
        }
    }
    return sTemp;
}

map<string, double> OriginQVM::getProbDict(QVec vQubit, int selectMax)
{
    if (0 == vQubit.size())
    {
        QCERR("the size of qubit_vector is zero");
        throw invalid_argument("the size of qubit_vector is zero");
    }

    map<string, double> mResult;
    
    size_t stLength = vQubit.size();
    auto vTemp = PMeasure(vQubit, selectMax);
    for (auto iter : vTemp)
    {
        mResult.insert(make_pair(dec2bin(iter.first, stLength), iter.second));
    }
    return mResult;
}

vector<pair<size_t, double>> OriginQVM::
probRunTupleList(QProg & qProg, QVec vQubit, int selectMax)
{
    run(qProg);
    return getProbTupleList(vQubit, selectMax);
}

vector<double> OriginQVM::
probRunList(QProg & qProg, QVec vQubit,int selectMax)
{
    run(qProg);
    return getProbList(vQubit, selectMax);
}
map<string, double> OriginQVM::
probRunDict(QProg & qProg, QVec vQubit, int selectMax)
{
    run(qProg);
    return getProbDict(vQubit,  selectMax);
}

map<string, size_t> OriginQVM::
runWithConfiguration(QProg & qProg, vector<ClassicalCondition>& vCBit, rapidjson::Document & param)
{
    map<string, size_t> mResult;
    if (!param.HasMember("shots"))
    {
        QCERR("OriginCollection don't  have shots");
        throw invalid_argument("OriginCollection don't  have shots");
    }
    size_t shots = 0;
    if (param["shots"].IsString())
    {
        shots = (size_t)atoll(param["shots"].GetString());
    }
    else
    {
        shots = param["shots"].GetUint64();
    }

    for (size_t i = 0; i < shots; i++)
    {
        run(qProg);
        string sResult = ResultToBinaryString(vCBit);
        if (mResult.find(sResult) == mResult.end())
        {
            mResult[sResult] = 1;
        }
        else
        {
            mResult[sResult] += 1;
        }
    }
    return mResult;
}

static void accumulateProbability(vector<double>& probList, vector<double> & accumulateProb)
{
    accumulateProb.clear();
    accumulateProb.push_back(probList[0]);
    for (int i = 1; i < probList.size(); ++i)
    {
        accumulateProb.push_back(accumulateProb[i - 1] + probList[i]);
    }
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

map<string, size_t> OriginQVM::quickMeasure(QVec vQubit, size_t shots)
{
    map<string, size_t>  meas_result;
    vector<double> probList=getProbList(vQubit,-1);
    vector<double> accumulate_probabilites;
    accumulateProbability(probList, accumulate_probabilites);
    for (int i = 0; i < shots; ++i)
    {
        double rng = RandomNumberGenerator();
        if (rng < accumulate_probabilites[0])
            add_up_a_map(meas_result, dec2bin(0, vQubit.size()));
        for (int i = 1; i < accumulate_probabilites.size(); ++i)
        {
            if (rng < accumulate_probabilites[i] &&
                rng >= accumulate_probabilites[i - 1]
                )
            {
                add_up_a_map(meas_result,
                    dec2bin(i, vQubit.size())
                );
                break;
            }
        }
    }
    return meas_result;
}


map<int, size_t> OriginQVM::getGateTimeMap() const
{
    return map<int, size_t>();
}

QStat OriginQVM::getQStat()
{
    if (nullptr == _pGates)
    {
        QCERR("_pGates is null");
        throw runtime_error("_pGates is null");
    }
}


