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
/*! \file OriginQuantumMachine.h */
#ifndef ORIGIN_QUANTUM_MACHINE_H
#define ORIGIN_QUANTUM_MACHINE_H
#include "Core/QuantumMachine/Factory.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Core/VirtualQuantumProcessor/QPUImpl.h"
#include "Core/VirtualQuantumProcessor/QuantumGateParameter.h"
#include "Core/Utilities/QPandaException.h"
#include "Core/VirtualQuantumProcessor/RandomEngine/RandomEngine.h"
USING_QPANDA
/**
* @namespace QPanda
*/

/**
* @defgroup QuantumMachine
* @brief    QPanda2 quantum virtual machine
*/

class OriginPhysicalQubit : public PhysicalQubit
{
private:
    size_t addr;
    bool bIsOccupied;
public:
    OriginPhysicalQubit(); 
    inline size_t getQubitAddr() { return addr; }
    inline void setQubitAddr(size_t iaddr) { this->addr = iaddr; }
    bool getOccupancy() const;
    void setOccupancy(bool);
};

class OriginQubit : public Qubit
{
private:
    PhysicalQubit * ptPhysicalQubit;
public:

    OriginQubit(PhysicalQubit*);

    inline PhysicalQubit* getPhysicalQubitPtr()
    {
        if (nullptr == ptPhysicalQubit)
        {
            QCERR("ptPhysicalQubit is nullptr");
            throw std::runtime_error("ptPhysicalQubit is nullptr");
        }
        return ptPhysicalQubit;
    }

    inline bool getOccupancy()
    {
        return ptPhysicalQubit->getOccupancy();
    }
};

class OriginQubitPool : public QubitPool
{
    // implementation of the QubitPool
private:
    std::vector<PhysicalQubit*> vecQubit;

public:
    OriginQubitPool(size_t maxQubit);

    void clearAll();
    size_t getMaxQubit() const;
    size_t getIdleQubit() const;

    Qubit* allocateQubit();
    Qubit* allocateQubitThroughPhyAddress(size_t);
    Qubit* allocateQubitThroughVirAddress(size_t qubit_num); // allocate and return a qubit
    void Free_Qubit(Qubit*);
    size_t getPhysicalQubitAddr(Qubit*);
    size_t getVirtualQubitAddress(Qubit*) const;
    ~OriginQubitPool();
};

class OriginCBit : public CBit
{
    std::string name;
    bool bOccupancy;
    cbit_size_t m_value;
public:
    OriginCBit(std::string name);
    inline bool getOccupancy() const
    {
        return bOccupancy;
    }
    inline void setOccupancy(bool _bOc)
    {
        bOccupancy = _bOc;
    }
    inline std::string getName() const {return name;}
    cbit_size_t getValue() const noexcept { return m_value; };
    void setValue(const cbit_size_t value) noexcept { m_value = value; };
};

class OriginCMem : public CMem
{
    std::vector<CBit*> vecBit;

public:

    OriginCMem(size_t maxMem);

    CBit * Allocate_CBit();
    CBit * Allocate_CBit(size_t stCBitNum);
    size_t getMaxMem() const;
    size_t getIdleMem() const;
    void Free_CBit(CBit*);
    void clearAll();
    ~OriginCMem();
};

class OriginQResult : public QResult
{
private:
    std::map<std::string, bool> _Result_Map;
public:

    OriginQResult();
    inline std::map<std::string, bool> getResultMap() const    
    {
        return _Result_Map;
    }
    void append(std::pair<std::string, bool>);

    ~OriginQResult() {}
};

class OriginQMachineStatus : public QMachineStatus
{
private:
    int iStatus = -1;
public:
    OriginQMachineStatus();
    friend class QMachineStatusFactory;

    inline int getStatusCode() const
    {
        return iStatus;
    }
    inline void setStatusCode(int miStatus)
    {
        iStatus = miStatus;
    }
};

class QVM : public QuantumMachine
{
protected:
	RandomEngine* random_engine;
    QubitPool * _Qubit_Pool = nullptr;
    CMem * _CMem = nullptr;
    QResult* _QResult = nullptr;
    QMachineStatus* _QMachineStatus = nullptr;
    QPUImpl     * _pGates = nullptr;
    Configuration _Config;
    virtual void run(QProg&);
    std::string _ResultToBinaryString(std::vector<ClassicalCondition>& vCBit);
    virtual void _start();
    QVM() {
        _Config.maxQubit = 29;
        _Config.maxCMem = 256;
    }
    void _ptrIsNull(void * ptr, std::string name);
    virtual ~QVM() {}
    virtual void init() {}
public:
    virtual void setConfig(const Configuration &config);
    virtual Qubit* allocateQubit();
    virtual Qubit* allocateQubitThroughPhyAddress(size_t qubit_num);
    virtual Qubit* allocateQubitThroughVirAddress(size_t qubit_num); // allocate and return a qubit
    virtual QVec allocateQubits(size_t qubit_count);
    virtual QMachineStatus* getStatus() const;
    virtual QResult* getResult();
    virtual std::map<std::string, bool> getResultMap();
    virtual void finalize();
    virtual size_t getAllocateQubit();
    virtual size_t getAllocateCMem();
    virtual void Free_Qubit(Qubit*);
    virtual void Free_Qubits(QVec&); //free a list of qubits
    virtual void Free_CBit(ClassicalCondition &);
    virtual void Free_CBits(std::vector<ClassicalCondition>&);
    virtual ClassicalCondition allocateCBit();
    virtual std::vector<ClassicalCondition> allocateCBits(size_t cbit_count);
    virtual ClassicalCondition allocateCBit(size_t stCbitNum);
    virtual std::map<std::string, bool> directlyRun(QProg & qProg);
    virtual std::map<std::string, size_t> runWithConfiguration(QProg &, std::vector<ClassicalCondition> &, rapidjson::Document &);
    virtual std::map<GateType, size_t> getGateTimeMap() const;
    virtual QStat getQState() const;
    virtual size_t getVirtualQubitAddress(Qubit *) const;
    virtual bool swapQubitPhysicalAddress(Qubit *, Qubit*);
	virtual void set_random_engine(RandomEngine* rng);
};


class IdealQVM : public QVM, public IdealMachineInterface
{
public:
    std::vector<std::pair<size_t, double>> PMeasure(QVec qubit_vector, int select_max);
    std::vector<double> PMeasure_no_index(QVec qubit_vector);
    std::vector<std::pair<size_t, double>> getProbTupleList(QVec , int);
    std::vector<double> getProbList(QVec , int);
    std::map<std::string, double> getProbDict(QVec , int);
    std::vector<std::pair<size_t, double>> probRunTupleList(QProg &, QVec , int);
    std::vector<double> probRunList(QProg &, QVec , int);
    std::map<std::string, double> probRunDict(QProg &, QVec , int);
    std::map<std::string, size_t> quickMeasure(QVec , size_t);
    QStat getQStat();
};

class CPUQVM : public IdealQVM {
public:
	CPUQVM() {}
	void init();
};

class GPUQVM : public IdealQVM
{
public:
    GPUQVM() {}
    void init();
};

class CPUSingleThreadQVM : public IdealQVM
{
public:
    CPUSingleThreadQVM() { }
    void init();
};

class NoiseQVM : public QVM
{
private:
    std::vector<std::vector<std::string>> m_gates_matrix;
    std::vector<std::vector<std::string>> m_valid_gates_matrix;
    void _getValidGatesMatrix();
    void run(QProg&);
    void initGates(rapidjson::Document &);
public:
    NoiseQVM();
    void init();
    void init(rapidjson::Document &);
};

#endif
