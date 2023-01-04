/*
 * This preconditioner is coded by ZhuY!
 */

#include "BasicFunctionByOrigin.h"
#include "QPandaConfig.h"
#include <cstdio> 
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

#ifdef WIN32
#include <io.h>
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif // USE_OPENMP

using namespace Eigen;
using namespace std;

vector<vector<int>> sparI;
vector<vector<int>> No0Index;
vector<vector<int>> No0IndexCol;
vector<double> dright;
//SparseQMatrix<double> JacobMatrix;

void writeMatrixtoCSV(MatrixXd m, const string& s) {
	ofstream outcsvFile;
	stringstream file;
	file << "matrix-" << s << ".csv";
	outcsvFile.open(file.str(), ios::out);
	int size = m.rows();
	for (int i = 0; i < size; i++)
	{
		stringstream ss1;
		for (int j = 0; j < size; j++)
		{
			if (j < size - 1) {
				ss1 << m(i, j) << ',';
			}
			else
			{
				ss1 << m(i, j);
			}
		}
		outcsvFile << ss1.str() << endl;
	}
	outcsvFile.close();
}

double MatrixMaxElementNew(MatrixXd m)
{
	int size = m.rows();
	double maxElem = 0;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			if (maxElem < abs(m(i, j))) {
				maxElem = abs(m(i, j));
			}
		}
	}
	return maxElem;
}

vector<vector<double>> getRowVector(MatrixXd m)
{
	int size = m.rows();
	vector<vector<double>> RowVector(size);

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			if (m(i, j) != 0) {
				RowVector[i].push_back(m(i, j));
			}

		}
	}
	return RowVector;
}


vector<vector<double>> getColumnVector(MatrixXd m)
{
	int size = m.rows();
	vector<vector<double>> ColumnVector(size);

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			if (m(j, i) != 0) {
				ColumnVector[i].push_back(m(j, i));
			}
		}
	}
	return ColumnVector;
}

pair<MatrixXd, vector<int>> getSubMatrix(const vector<int>& Sp, SparseQMatrix<double>& Ae) {
	int size = Sp.size();
	//cout << "size=" << size << endl;
	//sort(Sp.begin(), Sp.end()); 
	int cols = Ae.size;
	MatrixXd subA(size, cols);
	for (int i = 0; i < size; i++)
	{
		subA.row(i) = Ae.getSparseEigenRowVector(Sp[i]);
	}

	vector<int> NoZeroIndex;
	for (int j = 0; j < cols; j++) {
		VectorXd col_i = subA.col(j);
		bool NoneZero = JudgeNoneZero(col_i);
		if (NoneZero) {
			NoZeroIndex.push_back(j);
		}
	}
	int NoneZeroSize = NoZeroIndex.size();
	MatrixXd No0subA(size, NoneZeroSize);
	for (int i = 0; i < NoneZeroSize; i++) {
		No0subA.col(i) = subA.col(NoZeroIndex[i]);
	}
	return make_pair(No0subA, NoZeroIndex);
}


VectorXd getb(int index, const vector<int>& No0Ind) {
	VectorXd b(No0Ind.size());
	for (int i = 0; i < No0Ind.size(); i++) {
		if (No0Ind[i] == index)
		{
			b(i) = 1;
		}
		else {
			b(i) = 0;
		}
	}
	return b;
}

int No0_in_b(int index, const vector<int>& No0Ind) {
	VectorXd b(No0Ind.size());
	int index_1=0;
	for (int i = 0; i < No0Ind.size(); i++) {
		if (No0Ind[i] == index)
		{
			b(i) = 1;
			index_1 = i;
			break;
		}
	}
	return index_1;
}

int getRhoJ(VectorXd& r, double dd, vector<int>& No0Ind, vector<int>& SpIni, MatrixXd& Ae) {

	int size = r.size();
	int rows = Ae.rows();

	VectorXd r_all(rows);
	vector<int> Jset;
	//vector<int> JsetNew;

	for (int i = 0; i < size; i++) {
		if (r(i) != 0) {
			Jset.push_back(No0Ind[i]);
		}
	}

	for (auto it = SpIni.begin(); it != SpIni.end(); ++it) {
		Jset.erase(std::remove(Jset.begin(), Jset.end(), *it), Jset.end());
	}

	MatrixXd sA(rows, No0Ind.size());
	for (int i = 0; i < No0Ind.size(); i++) {
		sA.col(i) = Ae.col(No0Ind[i]);
	}
	//vector<double> rho_j;

	double min_rho = 10;
	int J_add=0;
	for (int j = 0; j < Jset.size(); j++) {
		
		auto up = pow(r.dot(sA.row(Jset[j])), 2);
		auto down = pow(sA.row(Jset[j]).norm(), 2);
		auto rho = dd * dd - up / down;
		if (rho < min_rho) {
			min_rho = rho;
			J_add = j;
		}
	}
	return Jset[J_add];
}

int getRhoJ(VectorXd& r, double dd, vector<int>& No0Ind, vector<int>& SpIni, SparseQMatrix<double>& Ae) {

	int size = r.size();
	int rows = Ae.size;

	VectorXd r_all(rows);
	vector<int> Jset;
	//vector<int> JsetNew;

	for (int i = 0; i < size; i++) {
		if (r(i) != 0) {
			Jset.push_back(No0Ind[i]);
		}
	}

	for (auto it = SpIni.begin(); it != SpIni.end(); ++it) {
		Jset.erase(std::remove(Jset.begin(), Jset.end(), *it), Jset.end());
	}

	MatrixXd sA(rows, No0Ind.size());
	for (int i = 0; i < No0Ind.size(); i++) {
		sA.col(i) = Ae.getSparseEigenColVector(No0Ind[i]);
	}

	//vector<double> rho_j;

	double min_rho = 10;
	int J_add=0;
	for (int j = 0; j < Jset.size(); j++) {
		auto up = pow(r.dot(sA.row(Jset[j])), 2);
		auto down = pow(sA.row(Jset[j]).norm(), 2);
		auto rho = dd * dd - up / down;
		if (rho < min_rho) {
			min_rho = rho;
			J_add = j;
		}
		//rho_j.push_back(rho);
	}
	/*vector<double>::iterator myMin = min_element(rho_j.begin(), rho_j.end());
	int jadd = distance(rho_j.begin(), myMin);*/
	return Jset[J_add];
}

SparseQMatrix<double> MprodcutA(SparseQMatrix<double>& M, SparseQMatrix<double>& A, int SparseIndex) {
	int size = A.size;
 
	vector<vector<pair<int, double>>> sparse_dataM(size);

#pragma omp for
	for (int i = 0; i < size; i++) {
		auto rowV = M.getSparseEigenRowVector(i);
		for (int j = 0; j < size; j++) {
			auto colV = A.getSparseEigenColVector(j);
			auto val = rowV.dot(colV);
			if (val != 0) {
				sparse_dataM[i].emplace_back(j, val);
			}
		}
	}
	SparseQMatrix<double> MA;
	FastTransfer2SparseQMat(sparse_dataM.data(), size, MA);
	return MA;
}

VectorXd MproductbDen(const SparseQMatrix<double>& M, const VectorXd &b) {
	int size = M.size;
	VectorXd Mb(size);
	auto data = M.data;
	auto col_ptr = M.nnzcol_ptr;
	auto col_ind = M.nnzcol_ind;
	if (size == 0) { 
		cout << "wrong is here" << endl; 
	}
#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		double sum = 0;
		for (int j = col_ptr[i]; j < col_ptr[i + 1]; j++) {
			auto temp = data[j] * b(col_ind[j]);
			sum += temp;
		}	
		Mb(i) = sum;
	}

	return Mb;
}

vector<double> MproductbDen(const SparseQMatrix<double>& M, const vector<double>& b) {
	int size = M.size;
	vector<double> Mb(size);
	auto data = M.data;
	auto col_ptr = M.nnzcol_ptr;
	auto col_ind = M.nnzcol_ind;
#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		double sum = 0;
		for (int j = col_ptr[i]; j < col_ptr[i + 1]; j++) {
			auto temp = data[j] * b[col_ind[j]];
			sum += temp;
		}
		Mb[i] = sum;
	}
	return Mb;
}

double VecDotVec(const SparseQVector<double>& a, const VectorXd& b) {
	int nnz = a.nnz_num;
	auto nnz_index = a.nnz_ind;
	double sum = 0;
	for (int i = 0; i < nnz; i++) {
		auto temp = a.data[i] * b(nnz_index[i]);
		sum += temp;
	}
	return sum;
}

void initialSparsity(int N) {
	sparI.resize(N);
	for (int i = 0; i < N; i++) {
		sparI[i].push_back(i);
	}
}


bool JudgeNoneZero(VectorXd& clos) {
	int size = clos.size();
	bool NoZero = false;
	for (int i = 0; i < size; i++) {
		if (clos(i) != 0) {
			NoZero = true;
			break;
		}
	}
	return NoZero;
}

vector<int> getNZeroIndex(int* Sp, int size) {
	//int size = Sp.size();
	vector<vector<int>> N0index;
	for (int i = 0; i < size; i++) {
		N0index.push_back(No0Index[Sp[i]]);
	}
	vector<int> TotalIndex;
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < N0index[i].size(); j++) {
			TotalIndex.push_back(N0index[i][j]);
		}
	}
	sort(TotalIndex.begin(), TotalIndex.end());
	TotalIndex.erase(unique(TotalIndex.begin(), TotalIndex.end()), TotalIndex.end());
	return TotalIndex;
}

vector<int> getNZeroIndexCol(vector<int>& Sp) {
	int size = Sp.size();
	
	vector<int> TotalIndex;
	for (int i = 0; i < size; i++) {
		int num = No0IndexCol[Sp[i]].size();
		for (int j = 0; j < num; j++) {
			TotalIndex.push_back(No0IndexCol[Sp[i]][j]);
		}
	}
	sort(TotalIndex.begin(), TotalIndex.end());
	TotalIndex.erase(unique(TotalIndex.begin(), TotalIndex.end()), TotalIndex.end());
	return TotalIndex;
}

MatrixXd FastGetSubMatrix(const int* TotalIndex, const int* Sp, int cols, int rows, SparseQMatrix<double>& Ae) {

	//int rows = Sp.size();
	//int cols = TotalIndex.size();
	MatrixXd subA(cols, rows);
	for (int i = 0; i < rows; i++) {
		auto l = Sp[i];
		for (int j = 0; j < cols; j++) {
			auto m = TotalIndex[j];
			subA(j, i) = Ae(l, m);
		}
	}
	return subA;
}

MatrixXd FastGetSubMatrix(const int* TotalIndex, const int* Sp, int cols, int rows, const MatrixXd& Ae) {
	MatrixXd subA(cols, rows);
	for (int i = 0; i < rows; i++) {
		auto l = Sp[i];
		for (int j = 0; j < cols; j++) {
			auto m = TotalIndex[j];
			subA(j, i) = Ae(l, m);
		}
	}
	return subA;
}

void AddNZeroIndex(vector<int>& TotalIndex, int jadd) {
	auto index_add = No0Index[jadd];
	TotalIndex.insert(TotalIndex.end(), index_add.begin(), index_add.end());
	sort(TotalIndex.begin(), TotalIndex.end());
	TotalIndex.erase(unique(TotalIndex.begin(), TotalIndex.end()), TotalIndex.end());
}

void FastGetSubMatrixRight(MatrixXd &subA, const vector<int> &TotalIndex, const vector<int> &Sp, SparseQMatrix<double>& Ae) {

	int rows = Sp.size();
	int cols = TotalIndex.size();
	subA.resize(cols, rows);
	for (int i = 0; i < rows; i++) {
		auto m = Sp[i];
		for (int j = 0; j < cols; j++) {
			auto l = TotalIndex[j];
			subA(j, i) = Ae(l, m);
		}
	}
}

vector<int> AddNNZIndex(VectorXd& a, vector<int>& sp) {
	int size = a.size();
	vector<int> Jnnz;
	for (int i = 0; i < size; i++) {
		if (a(i) != 0) {
			Jnnz.push_back(i);
		}
	}
	for (auto it = sp.begin(); it != sp.end(); ++it) {
		Jnnz.erase(std::remove(Jnnz.begin(), Jnnz.end(), *it), Jnnz.end());
	}
	return Jnnz;
}

double Get1_Norm(MatrixXd Ae) {
	Eigen::ArrayXXd Ae_arr = Ae.array().abs();
	MatrixXd Abs = Ae_arr.matrix();
	int Cols = Abs.cols();
	VectorXd col_sum(Cols);
#pragma omp parallel for num_threads(6) 
	for (int i = 0; i < Cols; i++) {
		col_sum(i) = Abs.col(i).sum();
	}
	return col_sum.maxCoeff();
}

double Get1_Norm_Row(MatrixXd Ae) {
	Eigen::ArrayXXd Ae_arr = Ae.array().abs();
	MatrixXd Abs = Ae_arr.matrix();
	int Cols = Abs.rows();
	VectorXd col_sum(Cols);
#pragma omp parallel for num_threads(6) 
	for (int i = 0; i < Cols; i++) {
		col_sum(i) = Abs.row(i).sum();
	}
	return col_sum.maxCoeff();
}

int Getnnz(VectorXd x) {
	int nnz=0;
	for (int j = 0; j < x.size(); j++) {
		if (x(j) != 0) {
			nnz++;
		}
	}
	return nnz;
}

VectorXd MproductY(SparseQMatrix<double> &M, QStat &y) {
	int size = M.size;
	VectorXd My(size);

	for (int i = 0; i < size; i++)
	{
		double sum = 0;
		for (int j = 0; j < size; j++)
		{
			sum = M(i, j) * y[j].real() + sum;
		}
		My(i) = sum;
	}

	return My;
}

VectorXd MproductY(SparseQMatrix<double> &M, const vector<double> &y) {
	int size = M.size;
	VectorXd My(size);
#pragma omp parallel for num_threads(6) 
	for (int i = 0; i < size; i++)
	{
		double sum = 0;
		for (int j = 0; j < size; j++)
		{
			if(M(i, j)!=0){
				sum = M(i, j) * y[j] + sum;
			}	
		}
		My(i) = sum;
	}
	return My;
}

vector<int> GetFreshIndex(vector<int> &tempSp, const vector<int> &zIndex) {
	vector<int> newSp;
	for (int i = 0; i < tempSp.size(); i++) {
		bool samecheck = false;
		for (int j = 0; j < zIndex.size(); j++) {
			if (i == zIndex[j]) {
				samecheck = true;
				break;
			}
		}
		if (!samecheck) newSp.push_back(tempSp[i]);
	}
	return newSp;
}

VectorXd FastcalculateAMk(vector<int> &tempSp, vector<int> &nzindex, MatrixXd &subA, const vector<double> &mk, SparseQMatrix<double>& Ae)
{
	FastGetSubMatrixRight(subA, nzindex, tempSp, Ae);
	VectorXd x(mk.size());
	for (int i = 0; i < mk.size(); i++) x(i) = mk[i];
	if (subA.cols() != x.size()) cout << "Please check the input matrix and vector!! " << endl;
	auto amk = subA * x;
	return amk;
}

VectorXd FastcalculateAMkLeft(vector<int> &tempSp, vector<int> &nzindex, MatrixXd &subA, const vector<double> &mk, SparseQMatrix<double>& Ae)
{
	subA = FastGetSubMatrix(nzindex.data(), tempSp.data(), nzindex.size(), tempSp.size(), Ae);
	VectorXd x(mk.size());
	for (int i = 0; i < mk.size(); i++) x(i) = mk[i];
	if (subA.cols() != x.size()) cout << "Please check the input matrix and vector!! " << endl;
	auto amk = subA * x;
	return amk;
}

double MaxVectorElementNew(const double* B, int size)
{
	//int size = B.size();
	double maxElem = 0;
#pragma omp for
	for (int i = 0; i < size; i++)
	{
		if (maxElem < B[i]) {
			maxElem = B[i];
		}
	}
	return maxElem;
}

vector<double> GetJacobiPrecondition(SparseQMatrix<double>& A) {
	int size = A.size;
	vector<double> Dia(size);
#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		Dia[i] = 1.0 / A(i, i);
	}
	return Dia;
}

SparseQMatrix<double>JacobiMatrixMatrixProduct(const double* diagonal, SparseQMatrix<double>& Ae) {
	int size = Ae.size;
	auto col_ptr = Ae.nnzcol_ptr;

#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		for (int j = col_ptr[i]; j < col_ptr[i + 1]; j++)
		{
			Ae.data[j] = diagonal[i] * Ae.data[j];
		}
	}
	return Ae;
}

vector<double> read_vqls_theta() {
	FILE* f_r = fopen("vqls_theta.txt", "rb");
	vector<double> theta;
	if (f_r == NULL) {
		cout << "Error occurred when QCFD read vqls_theta\n";
		exit(0);
	}
	else {
		fseek(f_r, 0, SEEK_END);
		int filesize = ftell(f_r);
		int size = filesize / sizeof(double);
		theta.resize(size);
		fseek(f_r, 0, SEEK_SET);
		for (int i = 0; i < size; i++) {
			double* theta_i = new double;
			fread(theta_i, sizeof(double), 1, f_r);
			theta[i] = *theta_i;
			delete theta_i;
		}
	}
	fclose(f_r);
	return theta;
}

vector<vector<int>> read_sparsity() {
	FILE* f_r = fopen("sparsity.txt", "rb");
	FILE* f_n = fopen("sparsity_num.txt", "rb");
	vector<vector<int>> sparI_s;
	if (f_r == NULL || f_n == NULL) {
		cout << "Error occurred when QCFD read sparsity data\n";
		exit(0);
	}
	else {
		fseek(f_n, 0, SEEK_END);
		int filesize = ftell(f_n);
		int size = filesize / sizeof(int);
		sparI_s.resize(size);
		fseek(f_n, 0, SEEK_SET);
		for (int i = 0; i < size; i++) {
			int* num = new int;
			fread(num, sizeof(int), 1, f_n);
			for (int j = 0; j < *num; j++) {
				int* rr = new int;
				fread(rr, sizeof(int), 1, f_r);
				sparI_s[i].push_back(*rr);
				delete rr;
			}
			delete num;
		}
	}
	fclose(f_r);
	fclose(f_n);
	return sparI_s;
}

void write_sparsity() {
	/*string sparsity = "sparsity.txt";
	string num_str = "sparsity_num.txt";*/
	int size = sparI.size();

	FILE* f_j = fopen("sparsity.txt", "wb");
	FILE* f_n = fopen("sparsity_num.txt", "wb");
	if (f_j == NULL || f_n == NULL)
	{
		cout << "ERROR: fopen out error\n";
		exit(0);
	}
	else {
		for (int i = 0; i < size; i++) {
			int temp_num = sparI[i].size();
			int* num = &temp_num;
			fwrite(num, sizeof(int), 1, f_n);
			for (int j = 0; j < sparI[i].size(); j++) {
				auto temp = sparI[i][j];
				int* py = &temp;
				fwrite(py, sizeof(int), 1, f_j);
			}
		}
	}
	fclose(f_j);
	fclose(f_n);
}

//void write_vqls_theta() {
//	string theta = "vqls_theta.txt";
//	FILE* f_j = fopen("vqls_theta.txt", "wb");
//	int size = VQLS_theta.size();
//	if (f_j == NULL)
//	{
//		cout << "ERROR: fopen out error\n";
//		exit(0);
//	}
//	else {
//		for (int i = 0; i < size; i++) {
//			double temp_theta = VQLS_theta[i];
//			double* theta_i = &temp_theta;
//			fwrite(theta_i, sizeof(double), 1, f_j);
//		}
//	}
//	fclose(f_j);
//}

double norm(VectorXd& b) {
	double norm = 0.0;
#pragma omp parallel for reduction(+:norm)
	for (int i = 0; i < b.size(); i++) {
		norm += b(i) * (i);
	}
	return norm;
}

void get_dR_dC(vector<double>& dR, vector<double>& dC, SparseQMatrix<double>& m) {
	int size = m.size;
#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		dR[i] = 1 / sqrt(m.getRowMaxCoeff(i));
		dC[i] = 1 / sqrt(m.getColMaxCoeff(i));
	}
}

void update_d1_d2(const vector<double>& dR, const vector<double>& dC,
	vector<double>& d1, vector<double>& d2,
	SparseQMatrix<double>& m) {
	int size = m.size;
	auto col_ptr = m.nnzcol_ptr;
	auto col_index = m.nnzcol_ind;
#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		for (int j = col_ptr[i]; j < col_ptr[i + 1]; j++) {
			m.data[j] = m.data[j] * dR[i] * dC[col_index[j]];
		}
		d1[i] = dR[i] * d1[i];
		d2[i] = dC[i] * d2[i];
	}
}

void get_Bi_Bj(vector<double>& Bi, vector<double>& Bj, SparseQMatrix<double>& m) {
	int size = m.size;
#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		Bi[i] = m.getRowMaxCoeff(i);
		Bj[i] = m.getColMaxCoeff(i);
	}
}

pair<MatrixXd, vector<double>> read_jacobi_res() {
	FILE* f_r = fopen("D:/residual_out.txt", "rb");
	size_t dim;
	double* p1 = new double[1];
	if (f_r != NULL) {
		fread(p1, sizeof(double), 1, f_r);
		dim = p1[0];
		cout << "dimension = " << dim << endl;
	}
	else {
		cout << "Error occurred when QCFD read residual data\n";
		exit(0);
	}
	delete[] p1;
	std::vector<double> residual(dim, 0);
	double* rr = new double[dim];
	fread(rr, sizeof(double), dim, f_r);
	for (size_t i = 0; i < dim; i++) residual[i] = rr[i];
	delete[] rr;
	fclose(f_r);

	MatrixXd s_jacobi = MatrixXd::Zero(dim, dim);
	string infilename = "D:/jacobian_out_num.txt";
	ifstream infile(infilename, std::ios::in);
	size_t num;
	if (infile.fail()) {
		cout << "Error occurred when QCFD read jacobian numbers data\n";
		exit(0);
	}
	else {
		string line;
		getline(infile, line);
		num = atoi(line.c_str());
	}
	infile.close();

	FILE* f_j = fopen("D:/jacobian_out.txt", "rb");
	No0Index.resize(dim);
	if (f_j != NULL) {
		double* rj = new double[num];
		fread(rj, sizeof(double), num, f_j);
		for (int i = 1; i < num; i++) {
			size_t a = rj[i];
			i++;
			size_t b = rj[i];
			i++;
			double c = rj[i];
			s_jacobi(a, b) = c;
			No0Index[a].emplace_back(b);
		}
		delete[] rj;
	}
	else {
		cout << "Error occurred when QCFD read jacobian data\n";
		exit(0);
	}
	fclose(f_j);
	write_N0index();
	return make_pair(s_jacobi, residual);
}

void write_N0index() {
	int size = No0Index.size();
	FILE* f_j = fopen("N0index.txt", "wb");
	FILE* f_n = fopen("N0index_num.txt", "wb");
	if (f_j == NULL || f_n == NULL)
	{
		cout << "ERROR: fopen out error\n";
		exit(0);
	}
	else {
		for (int i = 0; i < size; i++) {
			int temp_num = No0Index[i].size();
			int* num = &temp_num;
			fwrite(num, sizeof(int), 1, f_n);
			for (int j = 0; j < No0Index[i].size(); j++) {
				auto temp = No0Index[i][j];
				int* py = &temp;
				fwrite(py, sizeof(int), 1, f_j);
			}
		}
	}
	fclose(f_j);
	fclose(f_n);
}

vector<vector<int>> read_N0index() {
	FILE* f_r = fopen("N0index.txt", "rb");
	FILE* f_n = fopen("N0index_num.txt", "rb");
	vector<vector<int>> sparI_s;
	if (f_r == NULL || f_n == NULL) {
		cout << "Error occurred when QCFD read sparsity data\n";
		exit(0);
	}
	else {
		fseek(f_n, 0, SEEK_END);
		int filesize = ftell(f_n);
		int size = filesize / sizeof(int);
		sparI_s.resize(size);
		fseek(f_n, 0, SEEK_SET);
		for (int i = 0; i < size; i++) {
			int* num = new int;
			fread(num, sizeof(int), 1, f_n);
			for (int j = 0; j < *num; j++) {
				int* rr = new int;
				fread(rr, sizeof(int), 1, f_r);
				sparI_s[i].push_back(*rr);
				delete rr;
			}
			delete num;
		}
	}
	fclose(f_r);
	fclose(f_n);
	return sparI_s;
}

void initialNo0Index(MatrixXd& Ae) {
	No0Index.resize(Ae.rows());
	for (size_t i = 0; i < Ae.rows(); i++) {
		for (size_t j = 0; j < Ae.cols(); j++) {
			if (Ae(i, j) != 0) {
				No0Index[i].emplace_back(j);
			}
		}
	}
}
