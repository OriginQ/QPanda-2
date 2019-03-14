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
#include "Utilities/ConfigMap.h"
#include "config.h"
#include "VirtualQuantumProcessor/GPUImplQPU.h"
#include "VirtualQuantumProcessor/CPUImplQPU.h"
#include "VirtualQuantumProcessor/CPUImplQPUSingleThread.h"

USING_QPANDA
using namespace std;

QuantumMachine* CPUQVM_Constructor()
{
    return new CPUQVM();
}
volatile QuantumMachineFactoryHelper _Quantum_Machine_Factory_Helper_CPUQVM(
    "CPUQVM",
    CPUQVM_Constructor
);

//REGISTER_QUANTUM_MACHINE(CPUQVM);
REGISTER_QUANTUM_MACHINE(CPUSingleThreadQVM);
REGISTER_QUANTUM_MACHINE(GPUQVM);


Qubit * QVM::Allocate_Qubit()
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

QVec QVM::Allocate_Qubits(size_t qubitNumber)
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

ClassicalCondition QVM::Allocate_CBit()
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


vector<ClassicalCondition> QVM::Allocate_CBits(size_t cbitNumber)
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


ClassicalCondition QVM::Allocate_CBit(size_t stCBitaddr)
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

Qubit * QVM::Allocate_Qubit(size_t stQubitNum)
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

void QVM::Free_Qubit(Qubit *qubit)
{
    this->_Qubit_Pool->Free_Qubit(qubit);
}

void QVM::Free_Qubits(QVec &vQubit)
{
    for (auto iter : vQubit)
    {
        this->_Qubit_Pool->Free_Qubit(iter);
    }
}

void QVM::Free_CBit(ClassicalCondition & class_cond)
{
    auto cbit = class_cond.getExprPtr()->getCBit();
    if (nullptr == cbit)
    {
        QCERR("cbit is null");
        throw invalid_argument("cbit is null");
    }
    _CMem->Free_CBit(cbit);
}

void QVM::Free_CBits(vector<ClassicalCondition> & vCBit)
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

void QVM::run(QProg & node)
{
    
    _pParam = new QuantumGateParam();

    _pParam->m_qbit_number = _Qubit_Pool->getMaxQubit()- _Qubit_Pool->getIdleQubit();

    _pGates->initState(_pParam);

    node.getImplementationPtr()->execute(_pGates, _pParam);

    /* aiter has been used in line 120 */
    for (auto aiter : _pParam->m_return_value)
    {
        _QResult->append(aiter);
    }
    delete _pParam;
    _pParam = nullptr;
    return;
}

QMachineStatus * QVM::getStatus() const
{
    return _QMachineStatus;
}

QStat QVM::getQState() const
{
    if (nullptr == _pGates)
    {
        QCERR("pgates is nullptr");
        throw std::runtime_error("pgates is nullptr");
    }
    return _pGates->getQState();
}

QResult * QVM::getResult()
{
    return _QResult;
}

void QVM::finalize()
{
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

size_t QVM::getAllocateQubit()
{
    return _Qubit_Pool->getMaxQubit() - _Qubit_Pool->getIdleQubit();
}

size_t QVM::getAllocateCMem()
{
    return _CMem->getMaxMem()- _CMem->getIdleMem();
}

map<string, bool> QVM::getResultMap()
{
    if (nullptr == _QResult)
    {
        QCERR("QResult is null");
        throw runtime_error("QResult is null");
    }
    return _QResult->getResultMap();
}

vector<pair<size_t, double>> CPUQVM::PMeasure(QVec qubit_vector, int select_max)
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

vector<double> CPUQVM::PMeasure_no_index(QVec qubit_vector)
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

map<string, bool> QVM::directlyRun(QProg & qProg)
{
    run(qProg);
    return _QResult->getResultMap();
}

vector<pair<size_t, double>> CPUQVM::getProbTupleList(QVec vQubit,  int selectMax)
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

vector<double> CPUQVM::getProbList(QVec vQubit, int selectMax)
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
string QVM::_ResultToBinaryString(vector<ClassicalCondition> & vCBit)
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

void QVM::_start()
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

    if ((nullptr == _Qubit_Pool) ||
        (nullptr == _CMem) ||
        (nullptr == _QResult) ||
        (nullptr == _QMachineStatus))
    {
        QCERR("new fail");
        throw std::runtime_error("new fail");
    }
}

map<string, double> CPUQVM::getProbDict(QVec vQubit, int selectMax)
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

vector<pair<size_t, double>> CPUQVM::
probRunTupleList(QProg & qProg, QVec vQubit, int selectMax)
{
    run(qProg);
    return getProbTupleList(vQubit, selectMax);
}

vector<double> CPUQVM::
probRunList(QProg & qProg, QVec vQubit,int selectMax)
{
    run(qProg);
    return getProbList(vQubit, selectMax);
}
map<string, double> CPUQVM::
probRunDict(QProg & qProg, QVec vQubit, int selectMax)
{
    run(qProg);
    return getProbDict(vQubit,  selectMax);
}

map<string, size_t> QVM::
runWithConfiguration(QProg & qProg, vector<ClassicalCondition>& vCBit, rapidjson::Document & param)
{
    map<string, size_t> mResult;
    if (!param.HasMember("shots"))
    {
        QCERR("OriginCollection don't  have shots");
        throw invalid_argument("OriginCollection don't  have shots");
    }
    size_t shots = 0;
    if (param["shots"].IsUint64())
    {
        shots = param["shots"].GetUint64();
    }
    else
    {
        QCERR("shots data type error");
        throw invalid_argument("shots data type error");
    }

    for (size_t i = 0; i < shots; i++)
    {
        run(qProg);
        string sResult = _ResultToBinaryString(vCBit);
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

map<string, size_t> CPUQVM::quickMeasure(QVec vQubit, size_t shots)
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


map<int, size_t> QVM::getGateTimeMap() const
{
    return map<int, size_t>();
}

QStat CPUQVM::getQStat()
{
    if (nullptr == _pGates)
    {
        QCERR("_pGates is null");
        throw runtime_error("_pGates is null");
    }
    return _pGates->getQState();
}

void CPUQVM::init()
{
    _start();

    _pGates = new CPUImplQPU();
    if (nullptr == _pGates)
    {
        QCERR("new _pGates fail");
        throw std::runtime_error("new _pGates fail");
    }
}

void GPUQVM::init()
{
    _start();

#ifdef USE_CUDA
    _pGates = new GPUImplQPU();
#else
    _pGates = nullptr;
#endif // USE_CUDA
    if (nullptr == _pGates)
    {
        QCERR("new _pGates fail");
        throw std::runtime_error("new _pGates fail");
    }
}

void CPUSingleThreadQVM::init()
{
    _start();

    _pGates = new CPUImplQPUSingleThread();
    if (nullptr == _pGates)
    {
        QCERR("new _pGates fail");
        throw std::runtime_error("new _pGates fail");
    }
}
