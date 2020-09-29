#pragma once
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Components/Operator/PauliOperator.h"
#include "ThirdParty/Eigen/Dense"

QPANDA_BEGIN
class QuantumMachine;

/**
* @brief Variational Quantum Imagine Time Evolution Algorithem Class
* @ingroup	QAlg
*/
class QITE
{   
public:
    enum class UpdateMode
    {
        GD_VALUE,
        GD_DIRECTION
    };

public:
    QITE();

    /**
    * @brief  Set problem hamitonian
    * @param[in]  const PauliOperator& problem hamiltoinan
    * @see PauliOperator
    */
    void setHamiltonian(const PauliOperator& h)
    {
        m_pauli = h;
    }

    /**
    * @brief  Set ansatz gate
    * @param[in]  const std::vector<AnsatzGate>& ansatz gate vector
    * @see AnsatzGate
    */
    void setAnsatzGate(const std::vector<AnsatzGate>& ansatz_gate)
    {
        m_ansatz = ansatz_gate;
    }

    /**
    * @brief  Set delta tau value
    * @param[in]  double delta tau value
    */
    void setDeltaTau(double delta_tau)
    {
        m_delta_tau = delta_tau;
    }

    /**
    * @brief  Set iteration number
    * @param[in]  size_t iteration number
    */
    void setIterNum(size_t num)
    {
        m_iter_num = num;
    }

    /**
    * @brief  Set parameters update mode
    * @param[in]  UpdateMode parameters update mode
    * @see UpdateMode
    */
    void setParaUpdateMode(UpdateMode mode)
    {
        m_update_mode = mode;
    }

    /**
    * @brief  Set upthrow number
    * @param[in]  size_t upthrow number
    */
    void setUpthrowNum(size_t num)
    {
        m_upthrow_num = num;
    }

    /**
    * @brief  Set convergence factor Q
    * @param[in]  size_t convergence factor Q
    */
    void setConvergenceFactorQ(double value)
    {
        m_Q = value;
    }

    /**
    * @brief  Set the quantum machine type
    * @param[in]  QMachineType quantum machine type
    * @see QMachineType
    */
    void setQuantumMachineType(QMachineType type)
    {
        m_quantum_machine_type = type;
    }

    /**
    * @brief  Set log file
    * @param[in]  const std::string& log file name
    */
    void setLogFile(const std::string& filename)
    {
        m_log_file = filename;
    }

    /**
    * @brief  Set arbitary cofficient
    * @param[in]  double arbitary cofficient
    */
    void setArbitaryCofficient(double arbitary_cofficient)
    {
        m_arbitary_cofficient = arbitary_cofficient;
    }

    /**
    * @brief  Execute algorithem
    * @return  int  success flag, 0: success, -1: fail
    */
    int exec();

    /**
    * @brief  Get calculation result of the algorithem
    * @return  prob_tuple  calculation result
    */
    prob_tuple getResult();
private:
    void initEnvironment();
    double getExpectation(const std::vector<AnsatzGate>& ansatz);
    double getExpectationOneTerm(QCircuit c, const QHamiltonianItem& component);
    bool ParityCheck(size_t state, const QTerm& paulis) const;

    void calcParaA();
    void calcParaC();
    QCircuit constructCircuit(const std::vector<AnsatzGate>& ansatz);
    std::complex<double> complexDagger(std::complex<double> &value);
    
    int getAnsatzDerivativeParaNum(int i);
    int getHamiltonianItemNum();
    std::complex<double> getAnsatzDerivativePara(int i, int cnt);
    double getHamiltonianItemPara(int i);
    QCircuit getAnsatzDerivativeCircuit(int i, int cnt);
    QCircuit getHamiltonianItemCircuit(int cnt);
    double calcSubCircuit(
        int index1,
        int index2,
        double theta,
        QCircuit cir1,
        QCircuit cir2);
    QCircuit convertAnsatzToCircuit(const AnsatzGate &u);
    Eigen::MatrixXd pseudoinverse(Eigen::MatrixXd matrix);
private:
    std::vector<AnsatzGate> m_ansatz;
    PauliOperator m_pauli;
    QVec m_qlist; // the last qubit is auxiliary qubit 
    double m_delta_tau{ 0.1 };
    size_t m_iter_num{ 100 };
    double m_arbitary_cofficient{ 1e-6 };
    double m_Q{ 0.95 };
    size_t m_upthrow_num{ 1 };
    UpdateMode m_update_mode{ UpdateMode::GD_DIRECTION };
    QMachineType m_quantum_machine_type{ CPU };
    
    std::unique_ptr<QuantumMachine> m_machine;
    QHamiltonian m_hamiltonian;
    std::vector<size_t> m_theta_index_vec;
    int qubits_num{ 0 };
    double m_expectation{ 0.0 };
    std::vector<AnsatzGate> m_best_ansatz;
    Eigen::MatrixXd m_A;
    Eigen::VectorXd m_C;
    std::string m_log_file;
    std::fstream m_log_writer;
};

/*
* @brief Quantum imagine time evolution algorithem interface
* @param[in]  const PauliOperator& problem hamiltoinan
* @param[in]  const std::vector<AnsatzGate>& ansatz gate vector
* @param[in]  size_t iteration number
* @param[in]  const std::string& log file name
* @param[in]  QITE::UpdateMode parameters update mode
* @param[in]  size_t upthrow number
* @param[in]  double delta tau value
* @param[in]  size_t convergence factor Q
* @param[in]  double arbitary cofficient
* @param[in]  QMachineType quantum machine type
* @return  prob_tuple  calculation result
*/
prob_tuple qite(
    const PauliOperator& h,
    const std::vector<AnsatzGate>& ansatz_gate,
    size_t iter_num = 100,
    std::string log_file = "",
    QITE::UpdateMode mode = QITE::UpdateMode::GD_DIRECTION,
    size_t up_throw_num = 3,
    double delta_tau = 0.1,
    double convergence_factor_Q = 0.95,
    double arbitary_cofficient = 1e-6,
    QMachineType type = QMachineType::CPU_SINGLE_THREAD
    );

QPANDA_END