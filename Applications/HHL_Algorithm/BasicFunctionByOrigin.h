/*
 * This preconditioner is coded by ZhuY!
 */

#pragma once

#include "QPanda.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/QR>
#include <Eigen/Core>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <regex>
#include <algorithm>
#include <stdlib.h>
#include "Core/Utilities/QPandaNamespace.h"
#include "SparseQMatrix.h"
//#include "GMRES_VQLS.h"

using namespace Eigen;
using namespace std;

extern vector<vector<int> > sparI;
extern vector<vector<int> > No0Index;
extern vector<vector<int> > No0IndexCol;
extern vector<double> dright;

double MaxVectorElementNew(double* B, int size);

template<typename Ty>
Ty MinVectorElementNew(vector<Ty> B) {
	int size = B.size();
	Ty minElem = 0;
	for (int i = 0; i < size; i++)
	{
		if (minElem > std::abs(B[i])) {
			minElem = std::abs(B[i]);
		}
	}
	return minElem;
}

/*  \brief write matrix to csv file
 *  \param[in] Matrix A, name of file.
 */
void writeMatrixtoCSV(MatrixXd m, string s);

template<typename Ty>
void writeVectortoCSV(vector<Ty> m, const string& s) {
	ofstream outcsvFile;
	stringstream file;
	file << "Vector-" << s << ".csv";
	outcsvFile.open(file.str(), ios::out);
	int size = m.size();

	for (int i = 0; i < size; i++)
	{
		outcsvFile << m[i] << endl;
	}
	outcsvFile.close();
}

/*  \brief get the row elments of matrix m
 *  \param[in] Matrix m
 *  return row vectors
 */
vector<vector<double>> getRowVector(MatrixXd m);

/*  \brief get the column elments of matrix m
 *  \param[in] Matrix m
 *  return column vectors
 */
vector<vector<double>> getColumnVector(MatrixXd m);

/*  \brief get non-zero sub-matrix subA
 *  \param[in] Sparse structure
 *  return subA and non-zero index 
 */
pair<MatrixXd, vector<int>> getSubMatrix(const vector<int>& Sp, SparseQMatrix<double>& Ae);

/*  \brief get  small size b
 *  \param[in] iteration
 *  \param[in] non-zero index
 *  return vector b
 */
VectorXd getb(int index, const vector<int>& No0Ind);

/*  \brief get the position to add element
 *  \param[in] vector r=Ax-b
 *  \param[in] norm of r
 *  \param[in] non-zero index
 *  \param[in] initial sparse structure
 *  return position jadd
 */
int getRhoJ(VectorXd& r, double dd, vector<int>& No0Ind, vector<int>& SpIni, SparseQMatrix<double>& Ae);

/*  \brief calculate matrix matrix product
 *  \param[in] MatrixXd M 
 *  \param[in] MatrixXd A
 *  \param[in] initial sparsity
 *  return matrix matrix product
 */
SparseQMatrix<double> MprodcutA(SparseQMatrix<double>& M, SparseQMatrix<double>& A, int SparseIndex);

/*  \brief calculate dense matrix vector product
 *  \param[in] MatrixXd M
 *  \param[in] VectorXd b
 *  return matrix vector product
 */ 
VectorXd MproductbDen(const SparseQMatrix<double>& M, const VectorXd& b);

/*  \brief calculate dense matrix vector product
 *  \param[in] MatrixXd M
 *  \param[in] vector<double> b
 *  return matrix vector product
 */
vector<double> MproductbDen(const SparseQMatrix<double>& M, const vector<double>& b);

/*  \brief initialize the sparsity
 *  \param[in] matrix dimension
 */
void initialSparsity(int N);

/*  \brief check the vector elements are or not all zero
 *  \param[in] vector clos
 *  return true-> all zero; false->there is at least one element is not zero.
 */
bool JudgeNoneZero(VectorXd& clos);

/*  \brief get the non-zero index of known rows. 
 *  \param[in] row index
 *  return non-zero index
 */
vector<int> getNZeroIndex(int* Sp, int size);

/*  \brief get the non-zero index of known cols.
 *  \param[in] col index
 *  return non-zero index
 */
vector<int> getNZeroIndexCol(vector<int>& Sp);

/*  \brief get small size matrix subA.
 *  \param[in] non-zero row index of big matrix
 *  \param[in] sparse structure index
 *  return subA
 */
MatrixXd FastGetSubMatrix(const int* TotalIndex, const int* Sp, int cols, int rows, SparseQMatrix<double>& Ae);

/*  \brief add new non-index to sparse structure index.
 *  \param[in] initial small size matrix subA
 *  \param[in] non-zero row/column index of big matrix
 *  \param[in] new sparse structure index
 *  return subA
 */
void AddNZeroIndex(vector<int>& TotalIndex, int jadd);

/*  \brief get the index of non-zero element.
 *  \param[in] iteration
 *  \param[in] non-zero index.
 *  return index of non-zero element
 */
int No0_in_b(int index, vector<int>& No0Ind);

/*  \brief add new non-index to sparse structure index.
 *  \param[in] initial small size matrix subA
 *  \param[in] non-zero row/column index of big matrix
 *  \param[in] new vector of sparse structure index
 *  return subA
 */
vector<int> AddNNZIndex(VectorXd& a, vector<int>& sp);

/*  \brief get norm-1 of matrix A.
 *  \param[in] matrix A
 *  return norm-1
 */
double Get1_Norm(MatrixXd Ae);

/*  \brief get norm-1 of matrix A transpose.
 *  \param[in] matrix A
 *  return norm-1
 */
double Get1_Norm_Row(MatrixXd Ae);

/*  \brief get number of non-zero elements.
 *  \param[in] vector x
 *  return nnz
 */
int Getnnz(VectorXd x);

/*  \brief calculate matrix vector product.
 *  \param[in] matrix M and Qstat y
 *  return matrix vector product
 */
VectorXd MproductY(SparseQMatrix<double> &M, QStat &y);

/*  \brief calculate matrix vector product.
 *  \param[in] matrix M and vector<double> y
 *  return matrix vector product
 */
VectorXd MproductY(SparseQMatrix<double> &M, vector<double> &y);

/*  \brief get small size matrix subA.
 *  \param[in] initial small size matrix subA
 *  \param[in] non-zero col index of big matrix
 *  \param[in] sparse structure index
 *  return subA
 */
MatrixXd FastGetSubMatrix(const int* TotalIndex, const int* Sp, int cols, int rows, const MatrixXd& Ae);

void FastGetSubMatrixRight(MatrixXd& subA, const vector<int>& TotalIndex, const vector<int>& Sp, SparseQMatrix<double>& Ae);

/*  \brief get new index of sparse structure.
 *  \param[in] initial sparse structure
 *  \param[in] zero index in vector x
 *  return new index of sparse structure
 */
vector<int> GetFreshIndex(vector<int> &tempSp, const vector<int> &zIndex);

/*  \brief calculate new A*mk in right precondition.
 *  \param[in] sparse structure
 *  \param[in] zero index in vector x
 *  \param[in] small size matrix subA 
 *  \param[in] vector of new x 
 *  return new A*mk
 */
VectorXd FastcalculateAMk(vector<int> &tempSp, vector<int> &nzindex, MatrixXd &subA, vector<double> &mk, SparseQMatrix<double>& Ae);

/*  \brief calculate new A*mk in left precondition.
 *  \param[in] sparse structure
 *  \param[in] zero index in vector x
 *  \param[in] small size matrix subA
 *  \param[in] vector of new x
 *  return new A*mk
 */
VectorXd FastcalculateAMkLeft(vector<int>& tempSp, vector<int>& nzindex, MatrixXd& subA, const vector<double>& mk, SparseQMatrix<double>& Ae);


/*  \brief calculate Vector dot product.
 *  \param[in] sparse vector
 *  \param[in] eigen vector b
 *  return dot product
 */
double VecDotVec(SparseQVector<double>& a, VectorXd& b);

/*  \brief calculate diagnoal matrix.
 *  \param[in] sparse matrix.
 *  return diagnoal matrix.
 */
vector<double> GetJacobiPrecondition(SparseQMatrix<double>& A);

/*  \brief calculate diagnoal matrix and sparse matrix product.
 *  \param[in] diagnoal matrix.
 *  \param[in] sparse matrix.
 *  return matrix matrix product.
 */
SparseQMatrix<double> JacobiMatrixMatrixProduct(const double* diagonal, SparseQMatrix<double>& A);

/*  \brief read the binary vqls variable theta.
 *  return vector<double> theta.
 */
vector<double> read_vqls_theta();

/*  \brief write the binary vqls variable theta.
 *  return void.
 */
//void write_vqls_theta();

/*  \brief read the binary sparsity for dynamic SPAI.
 *  return sparsity structure.
 */
vector<vector<int>> read_sparsity();

/*  \brief write the binary sparsity.
 *  return void.
 */
void write_sparsity();


enum preconditioner {
	NO_PRE = 0,
	DIAGNOAL_SCALING = 1,
	STATIC_SPAI = 2,
	DYNAMIC_SPAI = 3,
	POWER_SAI_LEFT = 4,
	JACOBI_P = 5
};

void get_dR_dC(vector<double>& dR, vector<double>& dC, SparseQMatrix<double>& m);

void update_d1_d2(const vector<double>& dR, const vector<double>& dC,
	vector<double>& d1, vector<double>& d2,
	SparseQMatrix<double>& m);

void get_Bi_Bj(vector<double>& Bi, vector<double>& Bj, SparseQMatrix<double>& m);

//MatrixXd FastGetSubMatrix(int* TotalIndex, int* Sp, int cols, int rows, const MatrixXd& Ae);

int getRhoJ(VectorXd& r, double dd, vector<int>& No0Ind, vector<int>& SpIni, MatrixXd& Ae);

pair<MatrixXd, vector<double>> read_jacobi_res();

void write_N0index();

vector<vector<int>> read_N0index();

void initialNo0Index(MatrixXd& Ae);