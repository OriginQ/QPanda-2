#pragma once

#include "Components/DataStruct.h"
#include "Components/Operator/PauliOperator.h"
#include "ThirdParty/Eigen/Dense"

QPANDA_BEGIN

/**
* @brief Node Sort Problem Generator
* @ingroup	Components
*/
class NodeSortProblemGenerator
{
public:
    NodeSortProblemGenerator() {}

    /**
    * @brief  Set problem graph
    * @param[in]  const std::vector<std::vector<double>>& problem graph
    */
    void setProblemGraph(const std::vector<std::vector<double>>& graph)
    {
        m_graph = graph;
    }

    /**
    * @brief  Set model parameter lamda1
    * @param[in]  double lambda
    */
    void setLambda1(double lambda)
    {
        m_lambda1 = lambda;
    }

    /**
    * @brief  Set model parameter lamda2
    * @param[in]  double lambda
    */
    void setLambda2(double lambda)
    {
        m_lambda2 = lambda;
    }

    /**
    * @brief  Set model parameter lamda3
    * @param[in]  double lambda
    */
    void setLambda3(double lambda)
    {
        m_lambda3 = lambda;
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
    * @brief  Execute
    */
    void exec();

    /**
    * @brief  Get hamiltonian from the problem model
    * @return  PauliOperator hamiltonian
    * @see PauliOperator
    */
    PauliOperator getHamiltonian() const
    {
        return m_pauli;
    }

    /**
    * @brief  Get ansatz from the problem model
    * @return  std::vector<QITE::AnsatzGate> ansatz
    * @see AnsatzGate
    */
    std::vector<AnsatzGate> getAnsatz() const
    {
        return m_ansatz;
    }

    /**
    * @brief  Get linear solver result of the problem model
    * @return  Eigen::VectorXd linear solver result
    * @see AnsatzGate
    */
    Eigen::VectorXd getLinearSolverResult() const
    {
        return m_linear_solver_result;
    }

    /**
    * @brief  Get the coefficient matrix
    * @return  Eigen::MatrixXd parameters of Matirx
    * @see AnsatzGate
    */
    Eigen::MatrixXd getMatrixA() const
    {
        return m_A;
    }

    /**
    * @brief  Get the constent term
    * @return  Eigen::VectorXd constent term
    * @see AnsatzGate
    */
    Eigen::VectorXd getVectorB() const
    {
        return m_b;
    }
private:
    void calcGraphPara(
        const std::vector<std::vector<double>>& graph,
        double lambda,
        std::vector<double>& U_hat_vec,
        std::vector<double>& D_hat_vec) const;
    PauliOperator genHamiltonian(
        const std::vector<std::vector<double>>& graph,
        double lambda_u,
        double lambda_d,
        const std::vector<double>& U_hat_vec,
        const std::vector<double>& D_hat_vec) const;
    std::vector<AnsatzGate> genAnsatz(
        const std::vector<std::vector<double>>& graph,
        const std::vector<double>& U_hat_vec,
        const std::vector<double>& D_hat_vec) const;
    Eigen::VectorXd genLinearSolverResult(
        const std::vector<std::vector<double>>& graph,
        double lambda_u,
        double lambda_d,
        const std::vector<double>& U_hat_vec,
        const std::vector<double>& D_hat_vec,
        Eigen::MatrixXd & A,
        Eigen::VectorXd & b) const;
    Eigen::MatrixXd pseudoinverse(Eigen::MatrixXd matrix) const;
private:
    std::vector<std::vector<double>> m_graph;

    double m_lambda1{ 0.2 };
    double m_lambda2{ 0.5 };
    double m_lambda3{ 0.5 };
    double m_arbitary_cofficient{ 1e-6 };

    PauliOperator m_pauli;
    std::vector<AnsatzGate> m_ansatz;
    Eigen::MatrixXd m_A;
    Eigen::VectorXd m_b;
    Eigen::VectorXd m_linear_solver_result;
};

QPANDA_END