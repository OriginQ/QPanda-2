#ifndef NOISE_SIMULATOR_H
#define NOISE_SIMULATOR_H

#include "Core/VirtualQuantumProcessor/MPSQVM/MPSImplQPU.h"
#include "Core/VirtualQuantumProcessor/RandomEngine/RandomEngine.h"

QPANDA_BEGIN

using DoubleQubits = std::pair<size_t, size_t>;

enum class KarusErrorType
{
    KARUS_MATRIICES,
    UNITARY_MATRIICES,
};

class NonKarusError
{
public:

    bool has_non_karus_error();

    //rotation error
    void set_rotation_error(double param) { m_rotation_param = param; }
    double get_rotation_error() { return m_rotation_param; }

    //reset error
    void set_reset_error(double p0_param, double p1_param) { m_reset_p0 = p0_param; m_reset_p1 = p1_param; }
    double get_reset_p0_error() { return m_reset_p0; }
    double get_reset_p1_error() { return m_reset_p1; }

    //measure error
    bool has_measure_qubit(size_t qubit);
    void set_measure_qubit(const Qnum& qubits);
    bool has_measure_error() { return !m_measure_error_karus_matrices.empty(); }
    void set_measure_error(int qubit, std::vector<QStat>& karus_matrices);
    void get_measure_error(int qubit, std::vector<QStat>& karus_matrices);

    //readout error
    bool has_readout_error();
    bool get_readout_result(bool result, size_t qubit);
    void set_readout_error(const std::vector<std::vector<double>>& readout_probabilities, const Qnum& qvec);

private:

    //rotation error param
    double m_rotation_param = .0;

    //reset error param
    double m_reset_p0 = .0;  /*probabilities for reset to |0>*/
    double m_reset_p1 = .0;  /*probabilities for reset to |1>*/ 

    //measure error param
    Qnum m_measure_qubits;
    std::map<int, std::vector<QStat>> m_measure_error_karus_matrices = {};

    //readout error param
    Qnum m_readout_qubits = {};  /*measure qubits for readout*/
    std::vector<std::vector<double>> m_readout_probabilities = {};  /*probabilities for readout*/
};

class KarusError
{
public:
    KarusError() {}
    KarusError(const std::vector<QStat>&);
    KarusError(const std::vector<QStat>&, const std::vector<double>&);

    bool has_karus_error();

    //bit-flip, phase-flip, bit-phase-flip, phase-damping, depolarizing
    void set_unitary_probs(std::vector<double>& probs_vec);
    void get_unitary_probs(std::vector<double>& probs_vec) const;

    void set_unitary_matrices(std::vector<QStat>& unitary_matrices);
    void get_unitary_matrices(std::vector<QStat>& unitary_matrices) const;

    //decoherence error, amplitude-damping
    void set_karus_matrices(std::vector<QStat>& karus_matrices);
    void get_karus_matrices(std::vector<QStat>& karus_matrices) const;

    KarusError tensor(const KarusError& karus_error);
    KarusError expand(const KarusError& karus_error);
    KarusError compose(const KarusError& karus_error);
    
    size_t get_qubit_num() const { return m_qubit_num; }
    KarusErrorType get_karus_error_type() { return m_karus_error_type; }

private:

    //1 qubit error or 2 qubit error
    size_t m_qubit_num = 1;
     
    //KARUS_MATRIICES or UNITARY_MATRIICES
    KarusErrorType  m_karus_error_type;

    //karus matrices for all karus noise model 
    std::vector<QStat>  m_karus_matrices = {};

    //unitary matrices only for bit-flip,phase-flip,bit-phase-flip,phase-damping,depolarizing
    std::vector<double> m_unitary_probs = {};   //*all probabilities must sum to 1.0*/
    std::vector<QStat>  m_unitary_matrices = {};
};

class NoiseSimulator : public TraversalInterface<bool>
{
public:
    NoiseSimulator() :m_mps_qpu(nullptr), m_result(nullptr) {}

    bool has_quantum_error();

    /*The next error are non-karus error*/

    //rotation error
    void set_rotation_error(double);
     
    //measure error
    void set_measure_error(NOISE_MODEL model, double param);
    void set_measure_error(NOISE_MODEL model, double param, const Qnum& qubits_vec);

    void set_measure_error(NOISE_MODEL model, double T1, double T2, double time_param);
    void set_measure_error(NOISE_MODEL model, double T1, double T2, double time_param, const Qnum& qubits_vec);

    //reset error
    void set_reset_error(double, double);

    //readout error
    void set_readout_error(const std::vector<std::vector<double>>& readout_params, const Qnum& qubits);

    /*The next error are karus error*/
    void set_combining_error(GateType gate_type, const KarusError& karus_error, const std::vector<Qnum>& qubits_vecs);

    /*mixed unitary error*/
    void set_mixed_unitary_error(GateType gate_type, const std::vector<QStat>& karus_matrices);
    void set_mixed_unitary_error(GateType gate_type, const std::vector<QStat>& unitary_matrices, const std::vector<double>& probs_vec);

    void set_mixed_unitary_error(GateType gate_type, const std::vector<QStat>& karus_matrices, const std::vector<Qnum>& qubits_vecs);
    void set_mixed_unitary_error(GateType gate_type, const std::vector<QStat>& unitary_matrices, const std::vector<double>& probs_vec, const std::vector<Qnum>& qubits_vecs);
    
    /*combining error*/

    /* bit-flip,phase-flip,bit-phase-flip,phase-damping,amplitude-damping,depolarizing*/
    void set_noise_model(NOISE_MODEL model, GateType gate_type, double param);
    void set_noise_model(NOISE_MODEL model, GateType gate_type, double param, const Qnum& qubits_vec);
    void set_noise_model(NOISE_MODEL model, GateType gate_type, double param, const std::vector<Qnum>& qubits_vecs);

    //The next 2 set_noise_model functions is only appear in DECOHERENCE_KRAUS_OPERATOR
    void set_noise_model(NOISE_MODEL model, GateType gate_type, double T1, double T2, double time_param);
    void set_noise_model(NOISE_MODEL model, GateType gate_type, double T1, double T2, double time_param, const Qnum& qubits_vecs);
    void set_noise_model(NOISE_MODEL model, GateType gate_type, double T1, double T2, double time_param, const std::vector<Qnum>& qubits_vecs);

    void set_mps_qpu_and_result(std::shared_ptr<MPSImplQPU> mps_qpu, QResult* result);

    //traversal component
    void execute(std::shared_ptr<AbstractQGateNode>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractClassicalProg>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractQuantumMeasure>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractQuantumReset>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractQuantumCircuit>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractQuantumProgram>, std::shared_ptr<QNode>, QCircuitConfig &config);
    void execute(std::shared_ptr<AbstractControlFlowNode>, std::shared_ptr<QNode>, QCircuitConfig &config);

private:
     
    QResult* m_result;
    std::shared_ptr<MPSImplQPU> m_mps_qpu; 

    //qubits config for GateType
    std::map<GateType, Qnum> m_single_qubits;
    std::map<GateType, std::vector<DoubleQubits>> m_double_qubits;

private:

    NonKarusError m_non_karus_error;

    std::vector<std::tuple<GateType, int, KarusError>> m_one_qubit_karus_error_tuple;
    std::vector<std::tuple<GateType, int, int, KarusError>> m_two_qubit_karus_error_tuple;


    void handle_karus_matrices(std::vector<QStat>& karus_matrices, const QVec& qubits);
    void handle_unitary_matrices(const std::vector<QStat>& unitary_matrices, const std::vector<double> m_unitary_probs, const QVec& qubits);

    void handle_noise_gate(GateType gate_type, QVec& qubits);

    bool has_error_for_current_gate(GateType gate_type, QVec qubits);

    void set_gate_and_qnum(GateType gate_type, const Qnum& qubits);
    void set_gate_and_qnums(GateType gate_type, const std::vector<Qnum>& qubits);

    void set_single_karus_error_tuple(GateType gate_type, const KarusError &karus_error, const Qnum& qubits);
    void set_double_karus_error_tuple(GateType gate_type, const KarusError &karus_error, const std::vector<Qnum>& qubits);

    void handle_quantum_gate(std::shared_ptr<AbstractQGateNode> gate_type, bool is_dagger);
    
    void update_karus_error_tuple(GateType gate_type, int tar_qubit, const KarusError& karus_error);
    void update_karus_error_tuple(GateType gate_type, int ctr_qubit, int tar_qubit, const KarusError& karus_error);

    std::shared_ptr<AbstractQGateNode> handle_rotation_error(std::shared_ptr<AbstractQGateNode>);

    KarusError get_karus_error(GateType gate_type,const QVec& qubits);
};

QPANDA_END

#endif  //!NOISE_SIMULATOR_H 