/*
 * This sparse matrix class is coded by ZhuY!
 */

#pragma once

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
#include "SparseQVector.h"

using namespace Eigen;
using namespace std;

template<class Ty>
class QMat {
public:
	unsigned long size;
	std::vector<Ty> data;
	unsigned long n0Line;

	QMat(int size_);

	QMat(std::vector<Ty>& val);
	QMat(int size_, std::vector<Ty>& val);

	QMat<Ty> Transpose();
	Ty& operator()(int x, int y) {
		return data[x * size + y];
	}
	QMat<Ty> operator*(QMat<Ty>& m) {
		unsigned long size_ = m.size;
		QMat<Ty> prod(size_);
		for (unsigned long i = 0; i < size_; i++) {
			for (unsigned long j = 0; j < size_; j++) {
				Ty sum = 0;
				for (unsigned long k = 0; k < size_; k++) {
					sum += data[i * size_ + k] * m(k, j);
				}
				prod(i, j) = sum;
			}
		}
		return prod;
	};

	vector<Ty> operator*(const vector<Ty>& b) {
		unsigned long size_ = b.size();
		vector<Ty> prod(size_);
		for (unsigned long i = 0; i < size_; i++) {
			double sum = 0;
			for (unsigned long j = 0; j < size_; j++) {
				sum += data[i * size_ + j] * b[j];
			}
			prod[i] = sum;
		}
		return prod;
	};

	friend vector<Ty> operator*(const vector<Ty>& b, const QMat<Ty>& M)
	{
		unsigned long size_ = b.size();
		vector<Ty> prod(size_);
		for (unsigned long i = 0; i < size_; i++) {
			Ty sum = 0;
			for (unsigned long j = 0; j < size_; j++) {
				sum += M.data[j * size_ + i] * b[j];
			}
			prod[i] = sum;
		}
		return prod;

	};

	QMat<Ty> operator-(const QMat<Ty>& M) {
		unsigned long size_ = M.size;
		QMat<Ty> Minus(size_);
		for (unsigned long i = 0; i < size_ * size_; i++) {
			Minus.data[i] = data[i] - M.data[i];
		}
		return Minus;
	};
	void display();
	~QMat();
};


template<class Ty>
QMat<Ty>::QMat(int size_, std::vector<Ty>& val)
{
	size = size_;
	n0Line = 0;
	int wholesize = size * size;
	for (int i = 0; i < wholesize; i++)
		data.emplace_back(val[i]);
}

template<class Ty>
QMat<Ty>::QMat(std::vector<Ty>& val)
{
	size = _msize(val) / sizeof(Ty);
	n0Line = 0;
	int wholesize = size * size;
	for (int i = 0; i < wholesize; i++)
	{
		data[i] = val[i];
	}
}


template<class Ty>
QMat<Ty>::QMat(int size_)
{
	size = size_;
	n0Line = 0;
	unsigned long wholesize = size * size;
	for (unsigned long i = 0; i < wholesize; i++)
		data.emplace_back(0);
}

template<class Ty>
QMat<Ty> QMat<Ty>::Transpose() {
	QMat<Ty> transpose(size);
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			transpose(j, i) = data[i * size + j];
		}
	}
	return transpose;
}

template<class Ty>
void QMat<Ty>::display()
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
QMat<Ty>::~QMat()
{
}


template<class Ty>
class SparseQMatrix {
public:
	unsigned long size;
	unsigned long nnz_num;

	Ty* data;
	unsigned long* nnzcol_ptr;
	unsigned long* nnzcol_ind;
	/*unsigned long* n0row_ind;
	unsigned long n0Line;*/

	void InitialSparseQMatrix(const int* NNZ_col_ptr, const int* nnz_col_index, const Ty* val, int wholesize, int rownum);
	void Initialiaztion(int nnz, int size_);

	Ty operator()(unsigned long x, unsigned long y) {
		return getValue(x, y);
	}

	void operator=(const SparseQMatrix<Ty>& m) {
		Initialiaztion(m.nnz_num, m.size);
		for (int i = 0; i < nnz_num; ++i) {
			data[i] = m.data[i];
			nnzcol_ind[i] = m.nnzcol_ind[i];
		}
		for (int i = 0; i < size + 1; ++i)nnzcol_ptr[i] = m.nnzcol_ptr[i];
	}

	SparseQMatrix(const SparseQMatrix<Ty>& m) {
		size = m.size;
		nnz_num = m.nnz_num;
		Initialiaztion(nnz_num, size);
		for (int i = 0; i < nnz_num; ++i) {
			data[i] = m.data[i];
			nnzcol_ind[i] = m.nnzcol_ind[i];
		}
		for (int i = 0; i < size + 1; ++i)nnzcol_ptr[i] = m.nnzcol_ptr[i];
	}

	void SetRowNum(int size_) {
		size = size_;
		nnzcol_ptr = new unsigned long[size_ + 1];
	}

	void SetNNZ(int nnz) {
		nnzcol_ind = new unsigned long[nnz];
		data = new Ty[nnz];
		nnz_num = nnz;
	}

	Ty getValue(unsigned long x, unsigned long y);

	Ty getMaxValue();

	vector<Ty> getCompressedRowVector(unsigned long x);

	vector<Ty> getSparseRowVector(unsigned long x);

	vector<Ty> getCompressedColVector(unsigned long x);

	vector<Ty> getSparseColVector(unsigned long x);

	Ty getColMaxCoeff(unsigned long x);

	Ty getRowMaxCoeff(unsigned long x);

	Eigen::Matrix<Ty, Dynamic, 1> getSparseEigenRowVector(unsigned long x);

	Eigen::Matrix<Ty, Dynamic, 1> getCompressedEigenRowVector(unsigned long x);

	Eigen::Matrix<Ty, Dynamic, 1> getSparseEigenColVector(unsigned long x);

	Eigen::Matrix<Ty, Dynamic, 1> getCompressedEigenColVector(unsigned long x);

	SparseQVector<Ty> getRowSparseQVector(unsigned long x);
	//void Display();
	/*SparseQMatrix<Ty> Transpose();*/
	SparseQMatrix();

	~SparseQMatrix(void) {
		delete[] data;
		delete[] nnzcol_ptr;
		delete[] nnzcol_ind;	
	};
};

template<class Ty>
SparseQMatrix<Ty>::SparseQMatrix() {
	size = 0;
	nnz_num = 0;
	nnzcol_ptr = nullptr;
	nnzcol_ind = nullptr;
	data = nullptr;
}

template<class Ty>
void SparseQMatrix<Ty>::Initialiaztion(int nnz, int size_) {
	size = size_;
	nnz_num = nnz;
	nnzcol_ptr = new unsigned long[size_+1];
	nnzcol_ind = new unsigned long[nnz];
	data = new Ty[nnz];
}


template<class Ty>
Ty SparseQMatrix<Ty>::getValue(unsigned long x, unsigned long y) {
	Ty value = 0;
	for (int i = nnzcol_ptr[x]; i < nnzcol_ptr[x + 1]; i++) {
		if (y == nnzcol_ind[i]) {
			value = data[i];
			break;
		}
	}
	return value;
}

template<class Ty>
Ty SparseQMatrix<Ty>::getMaxValue() {
	Ty maxVal = -10e9;
	for (int i = 0; i < nnz_num; i++) {
		if (maxVal < data[i]) {
			maxVal = data[i];
		}
	}
	return maxVal;
}

template<class Ty>
void SparseQMatrix<Ty>::InitialSparseQMatrix(const int* NNZ_col_ptr, const int* nnz_col_index, const Ty* val, int wholesize, int rownum) {
	size = rownum;
	nnz_num = wholesize;
	nnzcol_ind = new unsigned long[wholesize];
	nnzcol_ptr = new unsigned long[rownum + 1];
	data = new Ty[wholesize];
#pragma omp parallel for
	for (int i = 0; i < rownum + 1; i++) nnzcol_ptr[i] = NNZ_col_ptr[i];
#pragma omp parallel for
	for (int i = 0; i < wholesize; i++) nnzcol_ind[i] = nnz_col_index[i];
#pragma omp parallel for
	for (int i = 0; i < wholesize; i++) data[i] = val[i];
}

template<class Ty>
vector<Ty> SparseQMatrix<Ty>::getSparseRowVector(unsigned long x) {
	vector<Ty> row_vector(size, 0);
	for (int i = nnzcol_ptr[x]; i < nnzcol_ptr[x + 1]; i++) {
		row_vector[nnzcol_ind[i]] = data[i];
	}
	return row_vector;
}

template<class Ty>
vector<Ty> SparseQMatrix<Ty>::getCompressedRowVector(unsigned long x) {
	vector<Ty> row_vector;
	for (int i = nnzcol_ptr[x]; i < nnzcol_ptr[x + 1]; i++) {
		row_vector.emplace_back(data[i]);
	}
	return row_vector;
}

template<class Ty>
vector<Ty> SparseQMatrix<Ty>::getSparseColVector(unsigned long x) {
	vector<Ty> col_vector(size, 0);
	for (int i = 0; i < size; i++) {
		for (int j = nnzcol_ptr[i]; j < nnzcol_ptr[i + 1]; j++) {
			if (x == nnzcol_ind[j]) {
				col_vector[i] = data[j];
			}
		}
	}
	return col_vector;
}

template<class Ty>
vector<Ty> SparseQMatrix<Ty>::getCompressedColVector(unsigned long x) {
	vector<Ty> col_vector;
	for (int i = 0; i < size; i++) {
		for (int j = nnzcol_ptr[i]; j < nnzcol_ptr[i + 1]; j++) {
			if (x == nnzcol_ind[j]) {
				col_vector.emplace_back[data[j]];
			}
		}
	}
	return col_vector;
}

template<class Ty>
Ty SparseQMatrix<Ty>::getColMaxCoeff(unsigned long x) {
	Ty max = -10e9;
	for (int i = 0; i < size; i++) {
		for (int j = nnzcol_ptr[i]; j < nnzcol_ptr[i + 1]; j++) {
			if (x == nnzcol_ind[j]) {
				if (max < data[j]) {
					max = data[j];
				}
			}
		}
	}
	return max;
}

template<class Ty>
Ty SparseQMatrix<Ty>::getRowMaxCoeff(unsigned long x) {
	Ty row_max = -10e9;
	for (int i = nnzcol_ptr[x]; i < nnzcol_ptr[x + 1]; i++) {
		if (row_max < data[i]) {
			row_max = data[i];
		}
	}
	return row_max;
}

template<class Ty>
Eigen::Matrix<Ty, Dynamic, 1> SparseQMatrix<Ty>::getSparseEigenRowVector(unsigned long x) {
	Eigen::Matrix<Ty, Dynamic, 1> row_vector = Eigen::Matrix<Ty, Dynamic, 1>::Zero(size, 1);
	for (int i = nnzcol_ptr[x]; i < nnzcol_ptr[x + 1]; i++) {
		row_vector(nnzcol_ind[i]) = data[i];
	}
	return row_vector;
}

template<class Ty>
Eigen::Matrix<Ty, Dynamic, 1> SparseQMatrix<Ty>::getCompressedEigenRowVector(unsigned long x) {
	int length = nnzcol_ptr[x + 1] - nnzcol_ptr[x];
	Eigen::Matrix<Ty, Dynamic, 1> row_vector(length);
	for (int i = nnzcol_ptr[x]; i < nnzcol_ptr[x + 1]; i++) {
		row_vector(i) = data[i];
	}
	return row_vector;
}

template<class Ty>
Eigen::Matrix<Ty, Dynamic, 1> SparseQMatrix<Ty>::getSparseEigenColVector(unsigned long x) {
	//vector<Ty> col_vector(size, 0);
	Eigen::Matrix<Ty, Dynamic, 1> col_vector = Eigen::Matrix<Ty, Dynamic, 1>::Zero(size, 1);
	for (int i = 0; i < size; i++) {
		for (int j = nnzcol_ptr[i]; j < nnzcol_ptr[i + 1]; j++) {
			if (x == nnzcol_ind[j]) {
				col_vector(i) = data[j];
				break;
			}
		}
	}
	return col_vector;
}

template<class Ty>
Eigen::Matrix<Ty, Dynamic, 1> SparseQMatrix<Ty>::getCompressedEigenColVector(unsigned long x) {
	int length = nnzcol_ptr[x + 1] - nnzcol_ptr[x];
	Eigen::Matrix<Ty, Dynamic, 1> col_vector(length);
	for (int i = 0; i < size; i++) {
		for (int j = nnzcol_ptr[i]; j < nnzcol_ptr[i + 1]; j++) {
			if (x == nnzcol_ind[j]) {
				col_vector(i) = data[j];
			}
		}
	}
	return col_vector;
} ///////Problem!!!!

template<class Ty>
void Transfer2SparseQMat(vector<pair<int, Ty>>* sparse_data, int size, SparseQMatrix<Ty>& M) {
	vector<int> col_index;
	vector<int> col_ptr(size + 1, 0);
	vector<double> Values;
	int nnz_num = 0;
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < sparse_data[i].size(); j++) {
			col_index.emplace_back(sparse_data[i][j].first);
			Values.emplace_back(sparse_data[i][j].second);
		}
		nnz_num += sparse_data[i].size();
		col_ptr[i + 1] = nnz_num;
	}
	M.InitialSparseQMatrix(col_ptr.data(), col_index.data(), Values.data(), Values.size(), col_ptr.size() - 1);
}

template<class Ty>
void Transfer2SparseQMat(vector<vector<pair<Ty, size_t>>>& sparse_data, SparseQMatrix<Ty>& M) {
	int size = sparse_data.size();
	int nnz_num = 0;
	M.nnzcol_ptr[0] = nnz_num;
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < sparse_data[i].size(); j++) {
			M.data[nnz_num + j] = sparse_data[i][j].first;
			M.nnzcol_ind[nnz_num + j] = sparse_data[i][j].second;
		}
		nnz_num += sparse_data[i].size();
		M.nnzcol_ptr[i + 1] = nnz_num;
	}
}

template<class Ty>
void FastTransfer2SparseQMat(vector<pair<int, Ty>>* sparse_data, int size, SparseQMatrix<Ty>& M) {

	int nnz_num = 0;
	M.SetRowNum(size);
	M.nnzcol_ptr[0] = 0;
	for (int i = 0; i < size; i++) {
		nnz_num += sparse_data[i].size();
		M.nnzcol_ptr[i + 1] = nnz_num;
	}
	auto nnz = nnz_num;
	M.SetNNZ(nnz);
#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < sparse_data[i].size(); j++) {
			M.data[M.nnzcol_ptr[i] + j] = sparse_data[i][j].second;
			M.nnzcol_ind[M.nnzcol_ptr[i] + j] = sparse_data[i][j].first;
		}
	}
}

template<class Ty>
void FastTransfer2SparseQMat(vector<vector<pair<Ty, size_t>>>& sparse_data, SparseQMatrix<Ty>& M) {
	int nnz_num = 0;
	auto size = sparse_data.size();
	M.SetRowNum(size);
	M.nnzcol_ptr[0] = 0;
	for (int i = 0; i < size; i++) {
		nnz_num += sparse_data[i].size();
		M.nnzcol_ptr[i + 1] = nnz_num;
	}
	auto nnz = nnz_num;
	M.SetNNZ(nnz);
#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < sparse_data[i].size(); j++) {
			M.data[M.nnzcol_ptr[i] + j] = sparse_data[i][j].first;
			M.nnzcol_ind[M.nnzcol_ptr[i] + j] = sparse_data[i][j].second;
		}
	}
}

template<class Ty>
SparseQVector<Ty> SparseQMatrix<Ty>::getRowSparseQVector(unsigned long x) {
	SparseQVector<Ty> SpV;
	int nnz_num = nnzcol_ptr[x + 1] - nnzcol_ptr[x];
	SpV.SetNNZ(nnz_num);
	int count = 0;
	for (int i = nnzcol_ptr[x]; i < nnzcol_ptr[x + 1]; i++) {
		SpV.insert(nnzcol_ind[i], data[i], count);
		count++;
	}
	return SpV;
}

template<class Ty>
vector<vector<pair<Ty, size_t>>> Transfer2Triplet(SparseQMatrix<Ty>& M) {
	size_t size = M.size;
	vector<vector<pair<Ty, size_t>>> Triplet(size);
	auto data = M.data;
	auto nnz_ind = M.nnzcol_ind;
	auto nnz_ptr = M.nnzcol_ptr;
	for (int i = 0; i < size; i++) {
		for (int j = nnz_ptr[i]; j < nnz_ptr[i + 1]; j++) {
			Triplet[i].emplace_back(make_pair(data[j], nnz_ind[j]));
		}
	}
	return Triplet;
}

