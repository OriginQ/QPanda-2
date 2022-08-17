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
#include <Eigen/QR>
#include <Eigen/SVD>
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/Utilities/Tools/QStatMatrix.h"
#include "Core/QuantumCircuit/QCircuit.h"


QPANDA_BEGIN

using matrix = Eigen::Matrix<double, -1, -1>;
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
	HOUSEHOLDER_QR =1,
	QSD = 2,
	CSD =3
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



/*******************************************************************
*              decomposition  (matrix  to  pualis)
********************************************************************/



template<class Ty>
class QMatrix {
public:
	unsigned long size;
	std::vector<Ty> data;

	unsigned long* n0col_ptr;
	unsigned long* n0col_ind;
	unsigned long* n0row_ind;
	unsigned long n0Line;

	QMatrix(int size_);

	QMatrix(std::vector<Ty>& val);
	QMatrix(int size_, std::vector<Ty>& val);

	void initialQMatrix();

	Ty& operator()(int x, int y) {
		return data[x * size + y];
	}
	void display();
	~QMatrix();
};


template<class Ty>
QMatrix<Ty>::QMatrix(int size_, std::vector<Ty>& val)
{
	size = size_;
	n0row_ind = nullptr;
	n0col_ind = nullptr;
	n0col_ptr = nullptr;
	n0Line = 0;
	int wholesize = size * size;
	for (int i = 0; i < wholesize; i++)
		data.push_back(val[i]);
}

template<class Ty>
QMatrix<Ty>::QMatrix(std::vector<Ty>& val)
{
	size = _msize(val) / sizeof(Ty);
	n0row_ind = nullptr;
	n0col_ind = nullptr;
	n0col_ptr = nullptr;
	n0Line = 0;
	int wholesize = size * size;
	for (int i = 0; i < wholesize; i++)
	{
		data[i] = val[i];
	}
}


template<class Ty>
QMatrix<Ty>::QMatrix(int size_)
{
	size = size_;
	n0row_ind = nullptr;
	n0col_ind = nullptr;
	n0col_ptr = nullptr;
	n0Line = 0;
	unsigned long wholesize = size * size;
	for (unsigned long i = 0; i < wholesize; i++)
		data[i] = 0;
}


template<class Ty>
void QMatrix<Ty>::initialQMatrix()
{

	std::vector<unsigned long> row;
	std::vector<unsigned long> col;
	std::vector<unsigned long> n0rowInd;
	int n0count = 0;
	row.push_back(n0count);
	for (int i = 0; i < size; i++)
	{
		int count = 0;
		for (int j = 0; j < size; j++) {
			if (data[i * size + j] != 0)
			{
				col.push_back(j);
				count++;
			}
		}
		if (count != 0) {
			n0count += count;
			row.push_back(n0count);
			n0rowInd.push_back(i);
		}
	}
	n0col_ptr = new unsigned long[row.size()];
	n0row_ind = new unsigned long[n0rowInd.size()];
	n0col_ind = new unsigned long[col.size()];
	n0Line = row.size() - 1;
	for (int i = 0; i < row.size(); i++)
		n0col_ptr[i] = row[i];
	for (int i = 0; i < col.size(); i++)
		n0col_ind[i] = col[i];
	for (int i = 0; i < n0rowInd.size(); i++)
		n0row_ind[i] = n0rowInd[i];
}


template<class Ty>
void QMatrix<Ty>::display()
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			if (j != size - 1)
				std::cout << data[i * size + j] << "\t";
			else
				std::cout << data[i * size + j];
		}
		std::cout << std::endl;
	}
}


template<class Ty>
QMatrix<Ty>::~QMatrix()
{
	delete n0col_ptr;
	delete n0col_ind;
	delete n0row_ind;
}


class MatrixToPauli
{
public:
	MatrixToPauli(QuantumMachine* qvm);
	virtual ~MatrixToPauli();

	/* Decompose the diagnal element into paulis */
	void add2CirAndCoeII(std::vector<double>& mat, const QVec& a);

	void add2CirAndCoeIJ(std::vector<double>& mat, int i, int j, const QVec& a);

	void add2CirAndCoeIorJ(std::vector<double>& mat, int i, int j, const QVec& a);

	void matrixDecompositionNew(QMatrix<double>& qmat);

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
* @param[in]  PualiOperatorLinearCombination& linearcom
*/
void matrix_decompose_paulis(QuantumMachine* qvm, const QMatrixXd& mat, PualiOperatorLinearCombination& linearcom);

QPANDA_END
#endif // MATRIX_DECOMPOSITION_H