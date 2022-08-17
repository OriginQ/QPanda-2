/*
 * This sparse vector class is coded by ZhuY!
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


using namespace Eigen;
using namespace std;

template<class Ty>
class SparseQVector {
public:
	unsigned long size;
	unsigned long nnz_num;
	Ty* data;

	unsigned long* nnz_ind;

	void InitialSparseQVector(int* nnz_index, Ty* val, int size_, int nnz);

	Ty operator()(unsigned long x) {
		return getValue(x);
	}

	SparseQVector(const SparseQVector<Ty>& m) {
		this->size = m.size;
		this->nnz_num = m.nnz_num;
		data = new Ty[nnz_num];
		nnz_ind = new unsigned long[nnz_num];
		for (int i = 0; i < nnz_num; i++) {
			data[i] = m.data[i];
			nnz_ind[i] = m.nnz_ind[i];
		}
	}

	void operator=(SparseQVector<Ty> m) {
		this->size = m.size;
		this->nnz_num = m.nnz_num;
		data = new Ty[nnz_num];
	    nnz_ind = new unsigned long[nnz_num];
		for (int i = 0; i < nnz_num; i++) {
			data[i] = m.data[i];
			nnz_ind[i]= m.nnz_ind[i];
		}
	}

	Ty getValue(unsigned long x);

	Ty getMaxValue();

	void SetNNZ(unsigned long nnz);

	void insert(unsigned long nnz_index, Ty val, unsigned long num);

	SparseQVector(void);
	~SparseQVector(void);
};

template<class Ty>
SparseQVector<Ty>::SparseQVector(void) {
	size = 0;
	nnz_num = 0;
	nnz_ind = nullptr;
	data = nullptr;
}

template<class Ty>
SparseQVector<Ty>::~SparseQVector() {
	delete[] data;
	delete[] nnz_ind;
}

template<class Ty>
Ty SparseQVector<Ty>::getValue(unsigned long x) {
	Ty value = 0;
	for (int i = 0; i < nnz_num; i++) {
		if (x == nnz_ind[i]) {
			value = data[i];
		}
	}
	return value;
}

template<class Ty>
void SparseQVector<Ty>::InitialSparseQVector(int* nnz_index, Ty* val, int size_, int nnz) {
	size = size_;
	nnz_num = nnz;
	nnz_ind = new unsigned long[nnz];
	data = new Ty[nnz];
	for (int i = 0; i < nnz_num; i++) nnz_ind[i] = nnz_index[i];
	for (int i = 0; i < nnz_num; i++) data[i] = val[i];
}

template<class Ty>
Ty SparseQVector<Ty>::getMaxValue() {
	Ty maxVal = -10e9;
	for (int i = 0; i < nnz_num; i++) {
		if (maxVal < data[i]) {
			maxVal = data[i];
		}
	}
	return maxVal;
}

template<class Ty>
void SparseQVector<Ty>::SetNNZ(unsigned long nnz) {
	nnz_num = nnz;
	nnz_ind = new unsigned long[nnz];
	data = new Ty[nnz];
}

template<class Ty>
void SparseQVector<Ty>::insert(unsigned long nnz_index, Ty val, unsigned long num) {
	nnz_ind[num] = nnz_index;
	data[num] = val;
}