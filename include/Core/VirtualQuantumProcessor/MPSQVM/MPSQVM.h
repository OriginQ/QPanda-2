#ifndef MPSQVM_H 
#define MPSQVM_H

#include <map>
#include <string>
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
    void initState(const QStat &state = {});
    virtual std::map<std::string, bool> directlyRun(QProg &prog);
    std::map<std::string, size_t> quickMeasure(QVec vQubit, size_t shots);

    virtual std::map<std::string, size_t> runWithConfiguration(QProg &prog,
        std::vector<ClassicalCondition> &cbits, int shots);
    virtual std::map<std::string, size_t> runWithConfiguration(QProg &prog,
        std::vector<ClassicalCondition> &cbits, rapidjson::Document &doc);
    virtual QStat getQState();
    virtual prob_tuple pMeasure(QVec qubits, int select_max = -1);
    virtual prob_tuple PMeasure(QVec qubits, int select_max = -1);

    prob_vec PMeasure_no_index(QVec qubits);
    qcomplex_t pmeasure_bin_index(QProg prog, std::string str);
    qcomplex_t pmeasure_dec_index(QProg prog, std::string str);

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

    //The all next functions are only for noise simulation

    /*combine error*/
    /*use KarusError.tensor(KarusError),KarusError.expand(KarusError),KarusError.compose(KarusError)*/
    //void set_error(GateType gate_type, const KarusError& karus_error);
    //void set_error(GateType gate_type, const KarusError& karus_error, const std::vector<QVec>& qubits_vecs);

    /*mixed unitary error*/
    void set_mixed_unitary_error(GateType gate_type, const std::vector<QStat>&);
    void set_mixed_unitary_error(GateType gate_type, const std::vector<QStat>&, const std::vector<double>&);

    void set_mixed_unitary_error(GateType gate_type, const std::vector<QStat>&, const std::vector<QVec>&);
    void set_mixed_unitary_error(GateType gate_type, const std::vector<QStat>&, const std::vector<double>& , const std::vector<QVec>& );

    /* bit-flip,phase-flip,bit-phase-flip,phase-damping,amplitude-damping,depolarizing*/
    void set_noise_model(NOISE_MODEL model, GateType gate_type, double param);
    void set_noise_model(NOISE_MODEL model, GateType gate_type, double param, const QVec& qubits_vec);
    void set_noise_model(NOISE_MODEL model, GateType gate_type, double param, const std::vector<QVec>& qubits_vecs);

    /*decoherence error*/
    void set_noise_model(NOISE_MODEL model, GateType gate_type, double T1, double T2, double time_param);
    void set_noise_model(NOISE_MODEL model, GateType gate_type, double T1, double T2, double time_param, const QVec& qubits_vec);
    void set_noise_model(NOISE_MODEL model, GateType gate_type, double T1, double T2, double time_param, const std::vector<QVec>& qubits_vecs);

    /*readout error*/
    void set_readout_error(const std::vector<std::vector<double>>& readout_params, const QVec& qubits);

    /*measurement error*/
    void set_measure_error(NOISE_MODEL model, double param);
    void set_measure_error(NOISE_MODEL model, double param, const QVec& qubits_vec);
    void set_measure_error(NOISE_MODEL model, double T1, double T2, double time_param);
    void set_measure_error(NOISE_MODEL model, double T1, double T2, double time_param, const QVec& qubits_vec);

    /*rotation error*/
    void set_rotation_error(double param);

    /*reset error*/
    void set_reset_error(double reset_0_param, double reset_1_param);

protected:
    void handle_one_target(std::shared_ptr<AbstractQGateNode> gate, const QCircuitConfig &config);
    void handle_two_targets(std::shared_ptr<AbstractQGateNode> gate, const QCircuitConfig &config);
    virtual void run(QProg &prog);
    void run_cannot_optimize_measure(QProg &prog);

    std::map<std::string, size_t> run_configuration_with_noise(QProg &prog, std::vector<ClassicalCondition> &cbits, int shots);
    
    //The all next functions are only for noise simulator
    std::map<std::string, size_t> run_configuration_without_noise(QProg &prog, std::vector<ClassicalCondition> &cbits, int shots);
    void run_cannot_optimize_measure_with_noise(QProg &prog);

private:
    NoiseSimulator m_noise_simulator;
    std::shared_ptr<MPSImplQPU> m_simulator = nullptr;
    std::vector<std::pair<size_t, CBit * > > m_measure_obj;
    unsigned short m_qubit_num = 0;
};

QPANDA_END

#endif//!MPSQVM_H


