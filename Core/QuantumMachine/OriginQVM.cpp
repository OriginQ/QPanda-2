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
#include "QPandaConfig.h"
#include "VirtualQuantumProcessor/GPUImplQPU.h"
#include "VirtualQuantumProcessor/CPUImplQPU.h"
#include "VirtualQuantumProcessor/CPUImplQPUSingleThread.h"
#include "Core/Utilities/QPandaException.h"
#include "Core/Utilities/Utilities.h"
#include "Core/Utilities/QuantumMetadata.h"

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




void QVM::setConfig(const Configuration & config)
{
    _Config.maxQubit = config.maxQubit;
    _Config.maxCMem = config.maxCMem;
}

Qubit * QVM::allocateQubit()
{
    if (_Qubit_Pool == nullptr)
    {
        // check if the pointer is nullptr
        // Before init
        // After finalize
        QCERR("Must initialize the system first");
        throw(qvm_attributes_error("Must initialize the system first"));
    }
    else
    {
        try
        {
            return _Qubit_Pool->allocateQubit();
        }
        catch (const std::exception&e)
        {
            QCERR(e.what());
            throw(qalloc_fail(e.what()));
        }

    }
        
}

QVec QVM::allocateQubits(size_t qubitNumber)
{
    if (_Qubit_Pool == nullptr)
    {
        // check if the pointer is nullptr
        // Before init
        // After finalize
        QCERR("Must initialize the system first");
        throw(qvm_attributes_error("Must initialize the system first"));
    }

    if (qubitNumber > _Config.maxQubit)
    {
        QCERR("qubitNumber > maxQubit");
        throw(qalloc_fail("qubitNumber > maxQubit"));
    }

    try
    {
        QVec vQubit;

        for (size_t i = 0; i < qubitNumber; i++)
        {
            vQubit.push_back(_Qubit_Pool->allocateQubit());
        }
        return vQubit;
    }
    catch (const std::exception &e)
    {
        QCERR(e.what());
        throw(qalloc_fail(e.what()));
    }
}

ClassicalCondition QVM::allocateCBit()
{
    if (_CMem == nullptr)
    {
        // check if the pointer is nullptr
        // Before init
        // After finalize
        QCERR("Must initialize the system first");
        throw(qvm_attributes_error("Must initialize the system first"));
    }
    else
    {
        try
        {
            auto cbit = _CMem->Allocate_CBit();
            ClassicalCondition temp(cbit);
            return temp;
        }
        catch (const std::exception&e)
        {
            QCERR(e.what());
            throw(calloc_fail(e.what()));
        }
    }
}


vector<ClassicalCondition> QVM::allocateCBits(size_t cbitNumber)
{
    if (_CMem == nullptr)
    {
        // check if the pointer is nullptr
        // Before init
        // After finalize
        QCERR("Must initialize the system first");
        throw(qvm_attributes_error("Must initialize the system first"));
    }
    else
    {
        if (cbitNumber > _Config.maxCMem)
        {
            QCERR("cbitNumber > maxCMem");
            throw(calloc_fail("cbitNumber > maxCMem"));
        }
        try
        {
            vector<ClassicalCondition> cbit_vector;
            for (size_t i = 0; i < cbitNumber; i++)
            {
                auto cbit = _CMem->Allocate_CBit();
                cbit_vector.push_back(cbit);
            }
            return cbit_vector;
        }
        catch (const std::exception&e)
        {
            QCERR(e.what());
            throw(calloc_fail(e.what()));
        }
    }
}


ClassicalCondition QVM::allocateCBit(size_t stCBitaddr)
{
    if (_CMem == nullptr)
    {
        // check if the pointer is nullptr
        // Before init
        // After finalize
        QCERR("Must initialize the system first");
        throw(qvm_attributes_error("Must initialize the system first"));
    }
    else
    {
        try
        {
            auto cbit = _CMem->Allocate_CBit(stCBitaddr);
            if (nullptr == cbit)
            {
                QCERR("stCBitaddr > maxCMem");
                throw calloc_fail("stCBitaddr > maxCMem");
            }
            ClassicalCondition temp(cbit);
            return temp;
        }
        catch (const std::exception&e)
        {
            QCERR(e.what());
            throw(calloc_fail(e.what()));
        }
    }
}

Qubit * QVM::allocateQubitThroughPhyAddress(size_t stQubitNum)
{
    if (_Qubit_Pool == nullptr)
    {
        // check if the pointer is nullptr
        // Before init
        // After finalize
        QCERR("Must initialize the system first");
        throw(qvm_attributes_error("Must initialize the system first"));
    }
    else
    {
        try
        {
            return _Qubit_Pool->allocateQubitThroughPhyAddress(stQubitNum);
        }
        catch (const std::exception&e)
        {
            QCERR(e.what());
            throw(qalloc_fail(e.what()));
        }

    }
}

Qubit * QVM::allocateQubitThroughVirAddress(size_t qubit_num)
{
    if (nullptr == _Qubit_Pool)
    {
        QCERR("_Qubit_Pool is nullptr ,you must init qvm at first");
        throw qvm_attributes_error("_Qubit_Pool is nullptr ,you must init qvm at first");
    }
    return _Qubit_Pool->allocateQubitThroughVirAddress(qubit_num);
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
    
    try
    {
        auto _pParam = new QuantumGateParam();
        _ptrIsNull(_pParam, "_pParam");

        _pParam->m_qubit_number = _Qubit_Pool->getMaxQubit() - _Qubit_Pool->getIdleQubit();

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
    catch (const std::exception&e)
    {
        QCERR(e.what());
        throw run_fail(e.what());
    }
}

QMachineStatus * QVM::getStatus() const
{
    if (nullptr == _QMachineStatus)
    {
        QCERR("_QMachineStatus is null");
        throw qvm_attributes_error("_QMachineStatus is null");
    }
    return _QMachineStatus;
}

QStat QVM::getQState() const
{
    if (nullptr == _pGates)
    {
        QCERR("pgates is nullptr");
        throw qvm_attributes_error("pgates is nullptr");
    }
    return _pGates->getQState();
}

size_t QVM::getVirtualQubitAddress(Qubit *qubit)const
{
    if (nullptr == qubit)
    {
        QCERR("qubit is nullptr");
        throw invalid_argument("qubit is nullptr");
    }

    if (nullptr == _Qubit_Pool)
    {
        QCERR("_Qubit_Pool is nullptr,you must init qvm");
        throw qvm_attributes_error("_Qubit_Pool is nullptr,you must init qvm");
    }

    return _Qubit_Pool->getVirtualQubitAddress(qubit);
}

bool QVM::swapQubitPhysicalAddress(Qubit * first_qubit, Qubit* second_qubit)
{
    if ((nullptr == first_qubit) || (nullptr == second_qubit))
    {
        return false;
    }

    auto first_addr = first_qubit->getPhysicalQubitPtr()->getQubitAddr();
    auto second_addr = second_qubit->getPhysicalQubitPtr()->getQubitAddr();

    first_qubit->getPhysicalQubitPtr()->setQubitAddr(second_addr);
    second_qubit->getPhysicalQubitPtr()->setQubitAddr(first_addr);
}

QResult * QVM::getResult()
{
    if (nullptr == _QResult)
    {
        QCERR("_QResult is nullptr");
        throw qvm_attributes_error("_QResult is nullptr");
    }
    return _QResult;
}

void QVM::finalize()
{
    if (nullptr != _Qubit_Pool)
    {
        delete _Qubit_Pool;
    }

    if (nullptr != _CMem)
    {
        delete _CMem;
    }

    if (nullptr != _QResult)
    {
        delete _QResult;
    }

    if (nullptr != _QMachineStatus)
    {
        delete _QMachineStatus;
    }

    if (nullptr != _pGates)
    {
        delete _pGates;
    }

    _Qubit_Pool = nullptr;
    _CMem = nullptr;
    _QResult = nullptr;
    _QMachineStatus = nullptr;
    _pGates = nullptr;
}

size_t QVM::getAllocateQubit()
{
    if (nullptr == _Qubit_Pool)
    {
        QCERR("_QResult is nullptr");
        throw qvm_attributes_error("_QResult is nullptr");
    }
    return _Qubit_Pool->getMaxQubit() - _Qubit_Pool->getIdleQubit();
}

size_t QVM::getAllocateCMem()
{
    if (nullptr == _CMem)
    {
        QCERR("_CMem is nullptr");
        throw qvm_attributes_error("_CMem is nullptr");
    }
    return _CMem->getMaxMem()- _CMem->getIdleMem();
}

map<string, bool> QVM::getResultMap()
{
    if (nullptr == _QResult)
    {
        QCERR("QResult is null");
        throw qvm_attributes_error("QResult is null");
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

    if (nullptr == _pGates)
    {
        QCERR("_pGates is null");
        throw qvm_attributes_error("_pGates is null");
    }
    try
    {
        Qnum vqubit;
        for (auto aiter = qubit_vector.begin(); aiter != qubit_vector.end(); ++aiter)
        {
            vqubit.push_back((*aiter)->getPhysicalQubitPtr()->getQubitAddr());
        }

        vector<pair<size_t, double>> pmeasure_vector;
        _pGates->pMeasure(vqubit, pmeasure_vector, select_max);

        return pmeasure_vector;
    }
    catch (const std::exception&e)
    {
        QCERR(e.what());
        throw result_get_fail(e.what());
    }

}

vector<double> CPUQVM::PMeasure_no_index(QVec qubit_vector)
{
    if (0 == qubit_vector.size())
    {
        QCERR("the size of qubit_vector is zero");
        throw invalid_argument("the size of qubit_vector is zero");
    }

    if (nullptr == _pGates)
    {
        QCERR("_pGates is null");
        throw qvm_attributes_error("_pGates is null");
    }

    try
    {

        Qnum vqubit;
        for (auto aiter = qubit_vector.begin(); aiter != qubit_vector.end(); ++aiter)
        {
            vqubit.push_back((*aiter)->getPhysicalQubitPtr()->getQubitAddr());
        }
        vector<double> pmeasure_vector;
        _pGates->pMeasure(vqubit, pmeasure_vector);

        return pmeasure_vector;
    }
    catch (const std::exception&e)
    {
        QCERR(e.what());
        throw result_get_fail(e.what());
    }


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

    if (nullptr == _pGates)
    {
        QCERR("_pGates is null");
        throw qvm_attributes_error("_pGates is null");
    }

    try
    {
        vector<pair<size_t, double>> vResult;
        Qnum vqubitAddr;
        for (auto aiter = vQubit.begin(); aiter != vQubit.end(); ++aiter)
        {
            vqubitAddr.push_back((*aiter)->getPhysicalQubitPtr()->getQubitAddr());
        }
        _pGates->pMeasure(vqubitAddr, vResult, selectMax);
        return vResult;
    }
    catch (const std::exception&e)
    {
        QCERR(e.what());
        throw result_get_fail(e.what());
    }
}

vector<double> CPUQVM::getProbList(QVec vQubit, int selectMax)
{
    if (0 == vQubit.size())
    {
        QCERR("the size of qubit_vector is zero");
        throw invalid_argument("the size of qubit_vector is zero");
    }

    if (nullptr == _pGates)
    {
        QCERR("_pGates is null");
        throw qvm_attributes_error("_pGates is null");
    }

    try
    {
        vector<double> vResult;
        Qnum vqubitAddr;
        for (auto aiter = vQubit.begin(); aiter != vQubit.end(); ++aiter)
        {
            vqubitAddr.push_back((*aiter)->getPhysicalQubitPtr()->getQubitAddr());
        }
        _pGates->pMeasure(vqubitAddr, vResult);
        return vResult;
    }
    catch (const std::exception&e)
    {
        QCERR(e.what());
        throw result_get_fail(e.what());
    }


}

string QVM::_ResultToBinaryString(vector<ClassicalCondition> & vCBit)
{
    string sTemp;
    if (nullptr == _QResult)
    {
        QCERR("_QResult is null");
        throw qvm_attributes_error("_QResult is null");
    }
    auto resmap = _QResult->getResultMap();
    for (auto c : vCBit)
    {
        auto cbit = c.getExprPtr()->getCBit();
        if (nullptr == cbit)
        {
            QCERR("vcbit is error");
            throw runtime_error("vcbit is error");
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

void QVM::_ptrIsNull(void * ptr, std::string name)
{
    if (nullptr == ptr)
    {
        stringstream error;
        error << "alloc " << name << " fail";
        QCERR(error.str());
        throw bad_alloc();
    }
}

void QVM::_start()
{
    _Qubit_Pool =
        QubitPoolFactory::GetFactoryInstance().
        GetPoolWithoutTopology(_Config.maxQubit);
    _ptrIsNull(_Qubit_Pool, "_Qubit_Pool");

    _CMem =
        CMemFactory::GetFactoryInstance().
        GetInstanceFromSize(_Config.maxCMem);

    _ptrIsNull(_CMem, "_CMem");

    _QResult =
        QResultFactory::GetFactoryInstance().
        GetEmptyQResult();

    _ptrIsNull(_QResult, "_QResult");

    _QMachineStatus =
        QMachineStatusFactory::
        GetQMachineStatus();

    _ptrIsNull(_QMachineStatus, "_QMachineStatus");
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
        throw run_fail("runWithConfiguration param don't  have shots");
    }
    size_t shots = 0;
    if (param["shots"].IsUint64())
    {
        shots = param["shots"].GetUint64();
    }
    else
    {
        QCERR("shots data type error");
        throw run_fail("shots data type error");
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


map<GateType, size_t> QVM::getGateTimeMap() const
{
    QuantumMetadata metadata;
    map<GateType, size_t> gate_time;
    metadata.getGateTime(gate_time);

    return gate_time;
}

QStat CPUQVM::getQStat()
{
    if (nullptr == _pGates)
    {
        QCERR("_pGates is null");
        throw qvm_attributes_error("_pGates is null");
    }
    return _pGates->getQState();
}

void CPUQVM::init()
{
    try
    {
        _start();
        _pGates = new CPUImplQPU();
        _ptrIsNull(_pGates, "CPUImplQPU");
    }
    catch (const std::exception &e)
    {
        QCERR(e.what());
        throw init_fail(e.what());
    }

}

void GPUQVM::init()
{
    try
    {
        _start();
#ifdef USE_CUDA
        _pGates = new GPUImplQPU();
#else
        _pGates = nullptr;
#endif // USE_CUDA
        _ptrIsNull(_pGates, "GPUImplQPU");
    }
    catch (const std::exception&e)
    {
        QCERR(e.what());
        throw init_fail(e.what());
    }

}

void CPUSingleThreadQVM::init()
{
    try
    {
        _start();
        _pGates = new CPUImplQPUSingleThread();
        _ptrIsNull(_pGates, "CPUImplQPUSingleThread");
    }
    catch (const std::exception &e)
    {
        QCERR(e.what());
        throw init_fail(e.what());
    }

}
