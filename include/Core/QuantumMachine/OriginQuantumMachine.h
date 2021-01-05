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
/*! \file OriginQuantumMachine.h */
#ifndef ORIGIN_QUANTUM_MACHINE_H
#define ORIGIN_QUANTUM_MACHINE_H
#include "Core/QuantumMachine/Factory.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Core/VirtualQuantumProcessor/QPUImpl.h"
#include "Core/VirtualQuantumProcessor/QuantumGateParameter.h"
#include "Core/Utilities/Tools/QPandaException.h"
#include "Core/VirtualQuantumProcessor/RandomEngine/RandomEngine.h"
#include "Core/VirtualQuantumProcessor/NoiseQPU/NoiseModel.h"  
#include "Core/QuantumMachine/QProgCheck.h"
#include <map>
QPANDA_BEGIN

/**
* @brief Implementation  class of  PhysicalQubit
* @ingroup QuantumMachine
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

/**
* @brief Implementation  class of Qubit
* @ingroup QuantumMachine
*/
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

/**
* @brief Implementation  class of QubitPool
* @ingroup QuantumMachine
*/
class OriginQubitPool : public QubitPool
{
private:
    std::vector<PhysicalQubit*> vecQubit;

public:
    OriginQubitPool(size_t maxQubit);

    void clearAll();
    size_t getMaxQubit() const;
    size_t getIdleQubit() const;
	size_t get_max_usedqubit_addr() const;

    Qubit* allocateQubit();
    Qubit* allocateQubitThroughPhyAddress(size_t);
    Qubit* allocateQubitThroughVirAddress(size_t qubit_num); // allocate and return a qubit
    void Free_Qubit(Qubit*);
    size_t getPhysicalQubitAddr(Qubit*);
    size_t getVirtualQubitAddress(Qubit*) const;
    ~OriginQubitPool();
};

/**
* @brief Implementation  class of QubitPool
* @ingroup QuantumMachine
*/
class OriginQubitPoolv2 : public QubitPool
{
private:
	std::vector<PhysicalQubit*> vecQubit;
	std::map<Qubit*,size_t> allocated_qubit;

public:
	OriginQubitPoolv2(size_t maxQubit);

	void clearAll();
	size_t getMaxQubit() const;
	size_t getIdleQubit() const;
	size_t get_max_usedqubit_addr() const;
	Qubit* allocateQubit();
	Qubit* allocateQubitThroughPhyAddress(size_t);
	Qubit* allocateQubitThroughVirAddress(size_t qubit_num); // allocate and return a qubit
	void Free_Qubit(Qubit*);
	size_t getPhysicalQubitAddr(Qubit*);
	size_t getVirtualQubitAddress(Qubit*) const;
	~OriginQubitPoolv2();
};

/**
* @brief Implementation  class of CBit
* @ingroup QuantumMachine
*/
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
    void set_val(const cbit_size_t value) noexcept { m_value = value; };
};

/**
* @brief Implementation  class of CMem
* @ingroup QuantumMachine
*/
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

/**
* @brief Implementation  class of QResult
* @ingroup QuantumMachine
*/
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

/**
* @brief Implementation  class of QMachineStatus
* @ingroup QuantumMachine
*/
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
    virtual std::map<std::string, size_t> run_with_optimizing(QProg &prog, std::vector<ClassicalCondition> &cbits,
        int shots, TraversalConfig &traver_param);
    virtual std::map<std::string, size_t> run_with_normal(QProg &prog, std::vector<ClassicalCondition> &cbits, int shots);
public:
    virtual Qubit* allocateQubitThroughPhyAddress(size_t qubit_num);
    virtual Qubit* allocateQubitThroughVirAddress(size_t qubit_num); // allocate and return a qubit
    virtual QMachineStatus* getStatus() const;
    virtual QResult* getResult();
    virtual std::map<std::string, bool> getResultMap();
    virtual void finalize();
    virtual std::map<std::string, bool> directlyRun(QProg & qProg);
    virtual std::map<std::string, size_t> runWithConfiguration(QProg &, std::vector<ClassicalCondition> &, rapidjson::Document &);
	virtual std::map<std::string, size_t> runWithConfiguration(QProg &, std::vector<ClassicalCondition> &, int );

    virtual std::map<GateType, size_t> getGateTimeMap() const;
    virtual QStat getQState() const;
    virtual size_t getVirtualQubitAddress(Qubit *) const;
    virtual bool swapQubitPhysicalAddress(Qubit *, Qubit*);
	virtual void set_random_engine(RandomEngine* rng);

    /*will delete*/
    virtual void setConfig(const Configuration &config);
    virtual Qubit* allocateQubit();
    virtual QVec allocateQubits(size_t qubit_count);
    virtual ClassicalCondition allocateCBit();
    virtual std::vector<ClassicalCondition> allocateCBits(size_t cbit_count);
    virtual ClassicalCondition allocateCBit(size_t stCbitNum);
    virtual size_t getAllocateQubit();
    virtual size_t getAllocateCMem();
    virtual void Free_Qubit(Qubit*);
    virtual void Free_Qubits(QVec&); //free a list of qubits
    virtual void Free_CBit(ClassicalCondition &);
    virtual void Free_CBits(std::vector<ClassicalCondition>&);

    /* new interface */
    virtual void setConfigure(const Configuration &); //! To initialize the quantum machine
    virtual Qubit* qAlloc(); //! Allocate and return a qubit
    virtual QVec qAllocMany(size_t qubit_count);//! allocateQubits
    virtual ClassicalCondition cAlloc(); //! Allocate and run a cbit
    virtual ClassicalCondition cAlloc(size_t); //! Allocate and run a cbit
    virtual std::vector<ClassicalCondition> cAllocMany(size_t); //! Allocate and return a list of cbits
    virtual void qFree(Qubit*); //! Free a qubit
    virtual void qFreeAll(QVec &); //!Gree a list of qubits
    virtual void cFree(ClassicalCondition &); //! Gree a cbit
    virtual void cFreeAll(std::vector<ClassicalCondition > &); //!Gree a list of CBits
    virtual size_t getAllocateQubitNum();//! getAllocateQubit
    virtual size_t getAllocateCMemNum();//! getAllocateCMem
};


class IdealQVM : public QVM, public IdealMachineInterface
{
public:
    prob_tuple getProbTupleList(QVec , int);
    prob_vec getProbList(QVec , int);
    prob_dict getProbDict(QVec , int);
    prob_tuple probRunTupleList(QProg &, QVec , int);
    prob_vec probRunList(QProg &, QVec , int);
    prob_dict probRunDict(QProg &, QVec , int);
    std::map<std::string, size_t> quickMeasure(QVec , size_t);

    /*will delete*/
    QStat getQStat();
    prob_tuple PMeasure(QVec qubit_vector, int select_max);
    prob_vec PMeasure_no_index(QVec qubit_vector);

    /* new interface */
    QStat getQState();
    prob_tuple pMeasure(QVec qubit_vector, int select_max);
    prob_vec pMeasureNoIndex(QVec qubit_vector);
};

class CPUQVM : public IdealQVM {
public:
	CPUQVM() {}
    void init();
    void initState(const QStat &state = {}) { _pGates->initState(getAllocateQubitNum(), state); return; }
};

class GPUQVM : public IdealQVM
{
public:
    GPUQVM() {}
    void init();
    void initState(const QStat &state = {}) { _pGates->initState(getAllocateQubitNum(), state);  return; }
};

class CPUSingleThreadQVM : public IdealQVM
{
public:
    CPUSingleThreadQVM() { }
    void init();
    void initState(const QStat &state = {}) { _pGates->initState(getAllocateQubitNum(), state);  return; }
};

class NoiseQVM : public QVM
{
private:
    std::vector<std::vector<std::string>> m_gates_matrix;
    std::vector<std::vector<std::string>> m_valid_gates_matrix;

    std::vector<int > m_models_vec;
    std::vector<std::string > m_gates_vec;

    std::vector <std::vector <double>> m_params_vecs;
    double m_rotation_angle_error{0};

    std::vector<std::vector <QStat> >  m_kraus_mats_vec;
    std::vector<std::string > m_kraus_gates_vec;
    NoisyQuantum m_quantum_noise;
public:
    NoiseQVM();
    void init();
    void initState(const QStat &state = {});
    void init(rapidjson::Document &);
    virtual std::map<std::string, size_t> runWithConfiguration(QProg &, std::vector<ClassicalCondition> &, rapidjson::Document &);
    virtual std::map<std::string, size_t> runWithConfiguration(QProg &, std::vector<ClassicalCondition> &, int);
    std::map<std::string, bool> directlyRun(QProg &prog);
    void run(QProg &prog);


    //void set_single_gate_noise_model(const NOISE_MODEL &model, const GateType &type, double prob, const std::vector<size_t> &qubits);
    //void set_noise_model(const NOISE_MODEL &model, const GateType &type, double prob, const std::vector<std::vector<size_t>> &qubits);
    void set_noise_model(const NOISE_MODEL &model, const GateType &type, double prob);
    void set_noise_model(const NOISE_MODEL &model, const GateType &type, double prob, const QVec &qubits);
    void set_noise_model(const NOISE_MODEL &model, const GateType &type, double prob, const std::vector<QVec> &qubits);

    void set_noise_model(const NOISE_MODEL &model, const GateType &type, double T1, double T2, double t_gate);
    void set_noise_model(const NOISE_MODEL &model, const GateType &type, double T1, double T2, double t_gate,
                         const QVec &qubits);
    void set_noise_model(const NOISE_MODEL &model, const GateType &type, double T1, double T2, double t_gate,
                         const std::vector<QVec> &qubits);

    void set_measure_error(const NOISE_MODEL &model, double prob, const QVec &qubits = {});
    void set_measure_error(const NOISE_MODEL &model, double T1, double T2, double t_gate,
                           const QVec &qubits = {});

    void set_mixed_unitary_error(const GateType &type, const std::vector<QStat> &unitary_matrices,
                                 const std::vector<double> &probs);
    void set_mixed_unitary_error(const GateType &type, const std::vector<QStat> &unitary_matrices,
                                 const std::vector<double> &probs, const QVec &qubits);
    void set_mixed_unitary_error(const GateType &type, const std::vector<QStat> &unitary_matrices,
                                 const std::vector<double> &probs, const std::vector<QVec> &qubits);
    void set_reset_error(double p0, double p1, const QVec &qubits = {});
    void set_readout_error(const std::vector<std::vector<double>> &probs_list, const QVec &qubits = {});


    /**
    * @brief  set QGate rotation angle errors
    * @param[in]  double rotation angle errors
    * @return     void
    * @see  QNode
    */
    void set_rotation_error(double error);
};
QPANDA_END
#endif
