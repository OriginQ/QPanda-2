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

#ifndef ORIGIN_QUANTUM_MACHINE_H
#define ORIGIN_QUANTUM_MACHINE_H
#include "Factory.h"
#include "QuantumMachineInterface.h"
#include "VirtualQuantumProcessor/QuantumGates.h"
#include "VirtualQuantumProcessor/QuantumGateParameter.h"
USING_QPANDA

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

    Qubit* Allocate_Qubit();
    Qubit* Allocate_Qubit(size_t);
    void Free_Qubit(Qubit*);
    size_t getPhysicalQubitAddr(Qubit*);
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

class OriginQVM : public QuantumMachine
{
private:
    QubitPool * _Qubit_Pool = nullptr;
    CMem * _CMem = nullptr;
    qmap_size_t _QProgram = -1;
    QResult* _QResult = nullptr;
    QMachineStatus* _QMachineStatus = nullptr;
    QuantumGateParam * _pParam;
    QuantumGates     * _pGates;
    struct Configuration
    {
        size_t maxQubit=25;
        size_t maxCMem=256;
    };
    Configuration _Config;

public:
    OriginQVM() {}
    bool init(QuantumMachine_type type = CPU);
    Qubit* Allocate_Qubit();
    Qubit* Allocate_Qubit(size_t qubit_num);
    QVec Allocate_Qubits(size_t qubit_count);
    ClassicalCondition Allocate_CBit();
    std::vector<ClassicalCondition> Allocate_CBits(size_t cbit_count);
    ClassicalCondition Allocate_CBit(size_t stCbitNum);
    void Free_Qubit(Qubit*);
    void Free_Qubits(QVec&); //free a list of qubits
    void Free_CBit(ClassicalCondition &);
    void Free_CBits(std::vector<ClassicalCondition>&);
    void load(QProg &);
    void append(QProg&);
    void run();
    QMachineStatus* getStatus() const;
    QResult* getResult();
    void finalize();
    size_t getAllocateQubit();
    size_t getAllocateCMem();
    std::map<std::string, bool> getResultMap();
    std::vector<std::pair<size_t, double>> PMeasure(QVec qubit_vector, int select_max);
    std::vector<double> PMeasure_no_index(QVec qubit_vector);
    std::map<std::string, bool> directlyRun(QProg & qProg);
    std::vector<std::pair<size_t, double>> getProbTupleList(QVec , int);
    std::vector<double> getProbList(QVec , int);
    std::map<std::string, double> getProbDict(QVec , int);
    std::vector<std::pair<size_t, double>> probRunTupleList(QProg &, QVec , int);
    std::vector<double> probRunList(QProg &, QVec , int);
    std::map<std::string, double> probRunDict(QProg &, QVec , int);
    std::string ResultToBinaryString(std::vector<ClassicalCondition>& vCBit);
    std::map<std::string, size_t> runWithConfiguration(QProg &, std::vector<ClassicalCondition> &, int);
    std::map<std::string, size_t> quickMeasure(QVec , size_t);
    virtual std::map<int, size_t> getGateTimeMap() const;
};

#endif
