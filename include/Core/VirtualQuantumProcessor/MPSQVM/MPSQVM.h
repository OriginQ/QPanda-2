#ifndef MPSQVM_H 
#define MPSQVM_H

#include "Core/VirtualQuantumProcessor/MPSQVM/MPSImplQPU.h"
#include "Core/VirtualQuantumProcessor/MPSQVM/NoiseSimulator.h"

QPANDA_BEGIN

/**
* @brief MPS quantum virtual machine
* @ingroup VirtualQuantumProcessor
*/
class MPSQVM : public QVM, TraversalInterface<QCircuitConfig&>
{
public:
    virtual void init();
    virtual std::map<std::string, bool> directlyRun(QProg &prog);
    virtual std::map<std::string, size_t> runWithConfiguration(QProg &prog,
        std::vector<ClassicalCondition> &cbits, int shots);
    virtual std::map<std::string, size_t> runWithConfiguration(QProg &prog,
        std::vector<ClassicalCondition> &cbits, rapidjson::Document &doc);
    virtual QStat getQState();
    virtual prob_tuple pMeasure(QVec qubits, int select_max = -1);

    prob_tuple getProbTupleList(QVec, int);
    prob_vec getProbList(QVec, int  selectMax = -1);
    prob_dict getProbDict(QVec, int);
    prob_tuple probRunTupleList(QProg &, QVec, int);
    prob_vec probRunList(QProg &, QVec, int);
    prob_dict probRunDict(QProg &, QVec, int);

    void execute(std::shared_ptr<AbstractQGateNode>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractClassicalProg>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractQuantumMeasure>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractQuantumReset>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractQuantumCircuit>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractQuantumProgram>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractControlFlowNode>, std::shared_ptr<QNode>, QCircuitConfig &config);

    void set_noise_model(NOISE_MODEL model, std::string gate, Qnum qubits_vec, std::vector<double> params_vec);
    void set_noise_model(NOISE_MODEL model, std::vector<double> params_vec);

protected:
    void handle_one_target(std::shared_ptr<AbstractQGateNode> gate, const QCircuitConfig &config);
    void handle_two_targets(std::shared_ptr<AbstractQGateNode> gate, const QCircuitConfig &config);
    virtual void run(QProg &prog);
    void run_cannot_optimize_measure(QProg &prog);

    std::map<std::string, size_t> run_configuration_with_noise(QProg &prog, std::vector<ClassicalCondition> &cbits, int shots);
    std::map<std::string, size_t> run_configuration_without_noise(QProg &prog, std::vector<ClassicalCondition> &cbits, int shots);

    void run_cannot_optimize_measure_with_noise(QProg &prog);

private:
    TensorNoiseGenerator m_noise_manager;
    std::shared_ptr<MPSImplQPU> m_simulator = nullptr;
    std::vector<std::pair<size_t, CBit * > > m_measure_obj;
    unsigned short m_qubit_num = 0;
};



QPANDA_END

#endif//!MPSQVM_H


