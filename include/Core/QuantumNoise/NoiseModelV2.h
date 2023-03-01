#pragma once

#include "Core/Utilities/QPandaNamespace.h"
#include "Core/VirtualQuantumProcessor/NoiseQPU/NoiseModel.h"
#include "Core/QuantumCircuit/QuantumGate.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QuantumMeasure.h"
#include "Core/QuantumCircuit/QReset.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/ClassicalProgram.h"
#include "Core/QuantumCircuit/QNodeDeepCopy.h"
#include "Core/QuantumNoise/QNoise.h"
#include "Core/QuantumNoise/OriginNoise.h"
#include "Core/QuantumNoise/DynamicOriginNoise.h"
#include "Core/Utilities/Tools/Traversal.h"

QPANDA_BEGIN

class NoiseModel;

/**
 * @brief virtual noise simulate gate node generator
 *
 */
class NoiseGateGenerator
{
public:
    static void append_noise_gate(GateType gate_type, QVec target, NoisyQuantum &noise, AbstractNodeManager &noise_qc);

private:
    /**
     * @brief used for dynamicly generate noise matrix
     */
    class KrausOpGenerator : AbstractOpsGenerator
    {
    public:
        KrausOpGenerator(const Qnum &qns, const NoiseOp &ops)
            : m_qns(qns),
              m_noise_ops(ops)
        {
        }

        virtual ~KrausOpGenerator() = default;

        virtual QStat generate_op();

    private:
        double kraus_expectation(const Qnum &qns, const QStat &op);
        Qnum m_qns;
        NoiseOp m_noise_ops;
    };

    static RandomEngine19937 m_rng;
};

/**
 * @brief virtual reset noise node generator
 *
 */
class NoiseResetGenerator
{
public:
    static void append_noise_reset(GateType gate_type, QVec target, NoisyQuantum &noise, AbstractNodeManager &noise_qc);

private:
    static RandomEngine19937 m_rng;
};

/**
 * @brief mixture readout noise with result
 *
 */
class NoiseReadOutGenerator
{
public:
    static void append_noise_readout(const NoiseModel &noise_model, std::map<std::string, bool> &result);

private:
    static RandomEngine19937 m_rng;
};

/**
 * @brief noise model for user define noise. most code copied from NoiseCPUImplQPU
 *
 */
class NoiseModel
{
public:
    NoiseModel() = default;
    ~NoiseModel() = default;
    /* same code for NoiseCPUImplQPU to make QuantamError */
    void add_noise_model(const NOISE_MODEL &model, const GateType &type, double prob);
    void add_noise_model(const NOISE_MODEL &model, const std::vector<GateType> &types, double prob);
    void add_noise_model(const NOISE_MODEL &model, const GateType &type, double prob, const QVec &qubits);
    void add_noise_model(const NOISE_MODEL &model, const std::vector<GateType> &types, double prob, const QVec &qubits);
    void add_noise_model(const NOISE_MODEL &model, const GateType &type, double prob, const std::vector<QVec> &qubits);

    void add_noise_model(const NOISE_MODEL &model, const GateType &type, double T1, double T2, double t_gate);
    void add_noise_model(const NOISE_MODEL &model, const std::vector<GateType> &types, double T1, double T2, double t_gate);
    void add_noise_model(const NOISE_MODEL &model, const GateType &type, double T1, double T2, double t_gate, const QVec &qubits);
    void add_noise_model(const NOISE_MODEL &model, const std::vector<GateType> &types, double T1, double T2, double t_gate, const QVec &qubits);
    void add_noise_model(const NOISE_MODEL &model, const GateType &type, double T1, double T2, double t_gate, const std::vector<QVec> &qubits);

    void add_mixed_unitary_error(const GateType &type, const std::vector<QStat> &unitary_matrices,
                                 const std::vector<double> &probs);
    void add_mixed_unitary_error(const GateType &type, const std::vector<QStat> &unitary_matrices,
                                 const std::vector<double> &probs, const QVec &qubits);
    void add_mixed_unitary_error(const GateType &type, const std::vector<QStat> &unitary_matrices,
                                 const std::vector<double> &probs, const std::vector<QVec> &qubits);

    void set_measure_error(const NOISE_MODEL &model, double prob, const QVec &qubits = {});
    void set_measure_error(const NOISE_MODEL &model, double T1, double T2, double t_gate, const QVec &qubits = {});

    void set_reset_error(double p0, double p1, const QVec &qubits = {});
    void set_readout_error(const std::vector<std::vector<double>> &probs_list, const QVec &qubits = {});

    void set_rotation_error(double error);

    /* get noise */
    double rotation_error() const;

    const NoisyQuantum &quantum_noise() const;

    /* noise status */
    bool enabled() const;
    bool readout_error_enabled() const;

    uint32_t get_noise_model_type() const;
    std::vector<double> get_single_params() const;
    std::vector<double> get_double_params() const;

private:
    uint32_t noise_model_type;
    std::vector<double> single_params;
    std::vector<double> double_params;

    double m_rotation_angle_error{0};
    bool m_enable{false};
    bool m_readout_error_enable{false};
    NoisyQuantum m_quantum_noise;
};

//--------------------------------------------------------------------------------------------------------------
/**
 * @brief Utilities class for deepcopy quantum program and inserte virtual noise node to new qprog implicitly
 *
 */
class NoiseProgGenerator : public QNodeDeepCopy
{
public:
    virtual void execute(std::shared_ptr<AbstractQGateNode> cur_node,
                         std::shared_ptr<QNode> parent_node) override;

    virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node,
                         std::shared_ptr<QNode> parent_node) override;

    virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node,
                         std::shared_ptr<QNode> parent_node) override;

    /**
     * @brief generate qprog mixed with simulate noise node
     * deep copy from input qc, insert noise node into copied qprog
     *
     * @param qnode qnode without noise
     */
    // QProg generate_noise_prog(const NoiseModel &noise, const QProg &qprog);
    template <typename qnode_t>
    auto generate_noise_prog(const NoiseModel &noise, std::shared_ptr<qnode_t> qnode) -> decltype(copy_node(qnode))
    {
        // static_assert(std::is_base_of<QNode, qnode_t>::value, "Only support type derived from QNode");
        m_qnoise = noise.quantum_noise();
        return copy_node(qnode);
    }

private:
    NoisyQuantum m_qnoise;
};

QPANDA_END