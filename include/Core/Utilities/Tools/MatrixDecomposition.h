#ifndef MATRIX_DECOMPOSITION_H
#define MATRIX_DECOMPOSITION_H

#include "ThirdParty/Eigen/Dense"
#include "ThirdParty/Eigen/Sparse"
#include "Core/Utilities/Tools/QStatMatrix.h"
#include <complex>
#include <ctime>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <EigenUnsupported/Eigen/KroneckerProduct>
#include <Eigen/QR>
#include <Eigen/SVD>
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/Utilities/Tools/QStatMatrix.h"
#include "Core/Utilities/Tools/Utils.h"
#include "Core/QuantumCircuit/QCircuit.h"


QPANDA_BEGIN

using EigenMatrixX = Eigen::Matrix<double, -1, -1>;
using PualiOperatorLinearCombination = std::vector<std::pair<double, QCircuit>>;

enum class MatrixUnit
{
    SINGLE_P0,
    SINGLE_P1,
    SINGLE_I2,
    SINGLE_V2
};

enum class DecompositionMode
{
    QR = 0,
    HOUSEHOLDER_QR = 1,
    QSD = 2,
    CSD = 3
};


/**
* @brief  matrix decomposition
* @ingroup Utilities
* @param[in]  QVec& the used qubits
* @param[in]  QStat& The target matrix
* @param[in]  DecompositionMode decomposition mode, default is HOUSEHOLDER_QR
* @param[in]  const bool true for positive sequence(q0q1q2), false for inverted order(q2q1q0),
                default is true
* @return    QCircuit The quantum circuit for target matrix
* @see Decomposition of quantum gates by Chi Kwong Li and Diane Christine Pelejo
        Un，Un-1，，，U1，U = I
*/
QCircuit matrix_decompose_qr(QVec qubits, const QStat& src_mat, const bool b_positive_seq = true);
QCircuit matrix_decompose_qr(QVec qubits, QMatrixXcd& src_mat, const bool b_positive_seq = true);
QCircuit diagonal_matrix_decompose(const QVec& qubits, const QStat& src_mat);



/*************************************************************************************
*      decomposition partition: (matrix  to  pauli operator's linear combination)
**************************************************************************************/


/**
* @brief  : convert matrix to vector in row major order
* @ingroup Utilities
* @param[in]  QMatrixXd& mat: the original matrix, element type:double.
* @return : array
*/
std::vector<double> mat2array_d(const QMatrixXd& mat);

class QCircuitToPauliOperator : public TraversalInterface<>
{
public:

    QCircuitToPauliOperator() = delete;
    QCircuitToPauliOperator(double val)
    {
        m_pauli_value = complex_d(val, 0);
    };

    template <typename _Ty>
    std::pair<std::string, complex_d> traversal(_Ty& node)
    {
        TraversalInterface::execute(node.getImplementationPtr(), nullptr);
        return std::make_pair(get_final_result(), m_pauli_value);
    }

    std::string get_final_result();
    complex_d get_final_pauli_value() { return m_pauli_value; }
    virtual void execute(std::shared_ptr<AbstractQGateNode>,
        std::shared_ptr<QNode>);

private:
    complex_d m_pauli_value;
    std::vector<std::string> m_pauli_strings;
};

class QCircuitSpiltToPauliOperator : public TraversalInterface<>
{
public:
    QCircuitSpiltToPauliOperator() {}

    std::vector<QCircuit> traversal(QCircuit& node)
    {
        TraversalInterface::execute(node.getImplementationPtr(), nullptr);
        return spilt_to_pauli_subcircuits();
    }

    virtual void execute(std::shared_ptr<AbstractQGateNode>, std::shared_ptr<QNode>);

private:

    std::vector<QCircuit> spilt_to_pauli_subcircuits();

    //key : qubit addr : 0,1,2,3
    //val : pauli gate : X,Y,Z,I
    std::map<size_t, std::vector<GateType>> m_pauli_circuits;
};


class MatrixToPauli
{
public:
    MatrixToPauli(QuantumMachine* qvm);
    MatrixToPauli(QVec qubits);
    virtual ~MatrixToPauli();

    /* Decompose the diagnal element into paulis */
    void add2CirAndCoeII(std::vector<double>& mat, const QVec& a);

    void add2CirAndCoeIJ(std::vector<double>& mat, int i, int j, const QVec& a);

    void add2CirAndCoeIorJ(std::vector<double>& mat, int i, int j, const QVec& a);

    void matrixDecompositionNew(EigenMatrixX& qmat);
    void matrixDecompositionNew_v2(EigenMatrixX& qmat);

    void matrixDecompositionSub(std::vector<double>& mat, int i, int j, unsigned short index, int numbits, const QVec& a);

    std::vector<int> ASCII2BIN(int a);
    std::pair<std::vector<int>, std::vector<int>> convert2FullBinaryIndex(int numbits, unsigned long i, unsigned long j);
    std::vector<int> convert2Coefficient(const std::vector<int>& i_s, const std::vector<int>& j_s);
    std::pair<std::vector<QCircuit>, std::vector<int>> convert2PauliOperator(const std::vector<int>& i_s, const std::vector<int>& j_s, const QVec& a);

    template <typename V>
    void addCoeAndCirAtMij(double ma, const std::vector<QCircuit>& cir, V& sign);

    std::vector<double> getQMcoe()
    {
        return m_QMcoeMerged;
    }

    std::vector<QCircuit> getQMcir()
    {
        return m_QMcirMerged;
    }

    void combine_same_circuit();
    bool matchIndex(int i, const std::vector<int>& repeatedIndex);
    bool matchTwoCircuit(const QCircuit& a, const QCircuit& b, bool criteria_matrix_circuit = false);
    void addtoSimplyCircuit(int i, const std::vector<int>& index, int num);

private:
    std::unique_ptr<QuantumMachine> m_machine;
    QMachineType m_quantum_machine_type{ CPU };
    QuantumMachine* m_qvm;
    QVec m_qubits;
    std::vector<double> m_QMcoe;    /*brief Decompostion coeffient of QMatrix M  */
    std::vector<QCircuit> m_QMcir;   /*brief Decompostion unitary matrix of QMatrix M  */
    std::vector<double> m_QMcoeMerged;    /*brief Decompostion coeffient(merged) of QMatrix M  */
    std::vector<QCircuit> m_QMcirMerged;   /*brief Decompostion unitary matrix(merged) of QMatrix M  */
    QCircuit m_bcirts;
    std::vector<std::vector<double>> m_theta;
};

/**
* @brief  matrix_decompose_paulis
* @ingroup Utilities
* @param[in]  QuantumMachine* qvm: QuantumMachine
* @param[in]  QMatrixXd& mat: the matrix
* @param[out]  PualiOperatorLinearCombination& linearcom
*/
void matrix_decompose_paulis(QuantumMachine* qvm, EigenMatrixX& mat, PualiOperatorLinearCombination& linearcom);
void matrix_decompose_paulis(QVec qubits, EigenMatrixX& mat, PualiOperatorLinearCombination& linearcom);
void matrix_decompose_paulis_duplicates(const PualiOperatorLinearCombination& in_linear_com, PualiOperatorLinearCombination& out_linear_com);
PualiOperatorLinearCombination pauli_combination_replace(const PualiOperatorLinearCombination& in_linear_com, QuantumMachine* machine, std::string src, std::string dst);

QPANDA_END
#endif // MATRIX_DECOMPOSITION_H