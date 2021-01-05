/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "QPandaConfig.h"
#include "Core/VirtualQuantumProcessor/MPSQVM/MPSImplQPU.h"
#include "QPandaNamespace.h"
#include "Core/Utilities/Tools/Utils.h"
#include <algorithm>
#include <Core/Utilities/Tools/Uinteger.h>
#include <thread>
#include <map>
#include <iostream>
#include <sstream>
#ifdef USE_OPENMP
#include <omp.h>
#endif

#include<numeric>
#include "ThirdParty/Eigen/Eigen"

using namespace std;
using namespace Eigen;
USING_QPANDA

#define USE_DENSITY_MATRIX 

static uint64_t insert(const int iValue, const int &iQn1, const int &iQn2)
{
    if (iQn1 < iQn2)
    {
        uint64_t iMask1 = (1ull << iQn1) - 1;
        uint64_t iMask2 = (1ull << (iQn2 - 1)) - 1;
        int z = iValue & iMask1;
        int y = ~iMask1 & iValue & iMask2;
        int x = ~iMask2 & iValue;

        return ((x << 2) | (y << 1) | z);
    }
}

static void squeeze_qubits(const Qnum &original_qubits, Qnum &squeezed_qubits)
{
	Qnum sorted_qubits = original_qubits;

	sort(sorted_qubits.begin(), sorted_qubits.end());
	for (size_t i = 0; i < original_qubits.size(); i++)
	{
		for (size_t j = 0; j < sorted_qubits.size(); j++)
		{
			if (original_qubits[i] == sorted_qubits[j])
			{
				squeezed_qubits[i] = j;
				break;
			}
		}
	}
}

static cmatrix_t matrix_element_multiplication(const cmatrix_t &A, const cmatrix_t &B)
{
    size_t rows = A.rows();
    size_t cols = A.cols();

    cmatrix_t result(rows, cols);
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            result(i, j) = A(i, j) * B(i, j);
        }
    }

    return result;
}

static size_t reorder_qubits(const Qnum qubits, size_t index)
{
	size_t new_index = 0;

	int64_t current_pos = 0, current_val = 0, new_pos = 0, shift = 0;
	size_t num_qubits = qubits.size();
	for (size_t i = 0; i < num_qubits; i++)
	{
		current_pos = num_qubits - 1 - qubits[i];
		current_val = 0x1 << current_pos;
		new_pos = num_qubits - 1 - i;
		shift = new_pos - current_pos;
		if (index & current_val)
		{
			if (shift > 0)
				new_index += current_val << shift;
			else if (shift < 0)
				new_index += current_val >> -shift;
			else
				new_index += current_val;
		}
	}
	return new_index;
}

static size_t reverse_qubits(size_t num, size_t len)
{
	size_t sum = 0;
	for (size_t i = 0; i < len; ++i)
	{
		if ((num & 0x1) == 1)
			sum += 1ULL << (len - 1 - i);   // adding pow(2, len-1-i)

		num = num >> 1;

		if (num == 0)
			break;
	}
	return sum;
}

static bool is_ordered(const Qnum &qubits)
{
	bool ordered = true;
	if (qubits.size() == 1)
	{
		return ordered;
	}

	for (int index = 0; index < qubits.size() - 1; index++)
	{
		if (qubits[index] + 1 != qubits[index + 1])
		{
			ordered = false;
			break;
		}
	}
	return ordered;
}

static  std::vector<GateType> sort_paulis_by_qubits(const std::vector<GateType> &paulis, const Qnum &qubits)
{
	size_t min = UINT_MAX;
	size_t min_index = 0;

	std::vector<GateType> new_paulis;
	std::vector<size_t> temp_qubits = qubits;
	// find min_index, the next smallest index in qubits
	for (size_t i = 0; i < paulis.size(); i++)
	{
		min = temp_qubits[0];
		for (size_t qubit = 0; qubit < qubits.size(); qubit++)
		{
			if (temp_qubits[qubit] <= min)
			{
				min = temp_qubits[qubit];
				min_index = qubit;
			}
		}
		// select the corresponding pauli, and put it next in
		// the sorted vector
		new_paulis.push_back(paulis[min_index]);
		// make sure we don't select this index again by setting it to UINT_MAX
		temp_qubits[min_index] = UINT_MAX;
	}
	return new_paulis;
}

void MPSImplQPU::centralize_and_sort_qubits(const Qnum &qubits,
	Qnum &sorted_indices, Qnum &centralized_qubits)
{
	sorted_indices = qubits;
	size_t num_qubits = qubits.size();

	if (!is_ordered(qubits))
		sort(sorted_indices.begin(), sorted_indices.end());

	size_t n = sorted_indices.size();
	size_t mid_index = sorted_indices[(n - 1) / 2];
	size_t first = mid_index - (n - 1) / 2;
	centralized_qubits.resize(n);
	std::iota(std::begin(centralized_qubits), std::end(centralized_qubits), first);

	// fewer swaps
	mid_index = (centralized_qubits.size() - 1) / 2;
	for (size_t i = mid_index; i < sorted_indices.size(); i++)
		change_qubits_location(sorted_indices[i], centralized_qubits[i]);

	for (int i = mid_index - 1; i >= 0; i--)
		change_qubits_location(sorted_indices[i], centralized_qubits[i]);
}

bool MPSImplQPU::qubitMeasure(size_t qubit)
{
	return measure_one_collapsing(qubit);
}

QError MPSImplQPU::pMeasure(Qnum& qnum, prob_vec &mResult)
{
	size_t num_qubits = qnum.size();
	size_t length = 1ULL << num_qubits;

	Qnum new_qubits;
	Qnum sorted_indices;
	Qnum qubits(qnum.size());
	for (int i = 0; i < qnum.size(); i++)
		qubits[i] = get_qubit_index(qnum[i]);

	// calc probability
	MPSImplQPU temp;
	temp.initState(*this);

	temp.centralize_and_sort_qubits(qubits, sorted_indices, new_qubits);

	MPS_Tensor mps = temp.convert_qstate_to_mps_form(new_qubits.front(), new_qubits.back());

	prob_vec probs(length);
	cmatrix_t mat, mat_conj;
#pragma omp parallel for private(mat, mat_conj)
	for (int64_t i = 0; i < length; i++)
	{
		mat = mps.get_data(i);
		mat_conj = mat.conjugate();
		probs[i] = std::real((mat.array() * mat_conj.array()).sum());
	}

	// reorder all qubits
	Qnum squeezed_qubits(qubits.size());
	squeeze_qubits(qubits, squeezed_qubits);
	prob_vec new_probs(length);
	int new_index = 0;
#pragma omp parallel for private(new_index)
	for (int64_t i = 0; i < length; i++)
	{
		new_index = reorder_qubits(squeezed_qubits, i);
		new_probs[new_index] = probs[i];
	}

	// reverse all qubits
	mResult.resize(length);

#pragma omp parallel for 
	for (int64_t i = 0; i < length; i++)
		mResult[i] = new_probs[reverse_qubits(i, num_qubits)];

	return qErrorNone;
}

QError MPSImplQPU::initState(size_t head_rank, size_t rank_size, size_t qubit_num)
{
    if (m_init_state.empty())
    {
        m_qubits_num = qubit_num;

        cmatrix_t data0(1, 1), data1(1, 1);
        data0(0, 0) = 1.0;
        data1(0, 0) = 0.0;
        rvector_t initial_val(1);
        initial_val[0] = 1.0;

        m_qubits_tensor.clear();
        m_lambdas.clear();
        for (size_t i = 0; i < m_qubits_num - 1; i++)
        {
            m_qubits_tensor.push_back(MPS_Tensor(data0, data1));
            m_lambdas.push_back(initial_val);
        }
        m_qubits_tensor.push_back(MPS_Tensor(data0, data1));

        m_qubits_order.resize(qubit_num, 0);
        std::iota(m_qubits_order.begin(), m_qubits_order.end(), 0);

        m_qubits_location.resize(qubit_num, 0);
        std::iota(m_qubits_location.begin(), m_qubits_location.end(), 0);
    }
    else
    {
        cmatrix_t statevector_as_matrix(1, m_init_state.size());

        for (auto i = 0; i < m_init_state.size(); i++)
        {
            statevector_as_matrix(0, i) = m_init_state[i];
        }

        //cmatrix_t mat = Map<cmatrix_t>(m_init_state.data(), 1, m_init_state.size());
        initState_from_matrix(qubit_num, statevector_as_matrix);

        std::reverse(m_qubits_order.begin(), m_qubits_order.end());
        std::reverse(m_qubits_location.begin(), m_qubits_location.end());
    }
	

	return qErrorNone;
}

void MPSImplQPU::initState(const MPSImplQPU &other)
{
	if (this != &other)
	{
		m_qubits_num = other.m_qubits_num;
		m_qubits_tensor = other.m_qubits_tensor;
		m_lambdas = other.m_lambdas;
		m_qubits_order = other.m_qubits_order;
		m_qubits_location = other.m_qubits_location;
	}
}

QError MPSImplQPU::initState(size_t qubit_num,const QStat &state)
{
    m_init_state.clear();
    if (!state.empty())
    {
        double probs = .0;

        for (auto amplitude : state)
        {
            probs += std::norm(amplitude);
        }

        if (std::abs(probs - 1.) > 1e-6)
        {
            QCERR("state error");
            throw std::runtime_error("state error");
        }

        m_init_state.clear();
        m_init_state.resize(state.size());
        m_init_state.assign(state.begin(), state.end());
    }

    return qErrorNone;
}

void MPSImplQPU::initState_from_matrix(size_t num_qubits, const cmatrix_t &mat)
{
	m_qubits_tensor.clear();
	m_lambdas.clear();
	m_qubits_order.resize(num_qubits, 0);
	std::iota(m_qubits_order.begin(), m_qubits_order.end(), 0);
	m_qubits_location.resize(num_qubits, 0);
	std::iota(m_qubits_location.begin(), m_qubits_location.end(), 0);
	m_qubits_num = 0;

	// remaining_matrix is the matrix that remains after each iteration
	// It is initialized to the input statevector after reshaping
	cmatrix_t remaining_matrix, A;
	cmatrix_t U, V, reduce_U, reduce_V;
	rvector_t S, reduce_S;

	bool first_iter = true;
    
	for (size_t i = 0; i < num_qubits - 1; i++)
	{
		// prepare matrix for next iteration except for first iteration:
		if (first_iter)
			remaining_matrix = mat;
		else
			remaining_matrix = mul_v_by_s(reduce_V, reduce_S).adjoint();

		A.resize(remaining_matrix.rows() * 2, remaining_matrix.cols() / 2);
		A << remaining_matrix.leftCols(remaining_matrix.cols() / 2),
			remaining_matrix.rightCols(remaining_matrix.cols() / 2);

		// SVD
		JacobiSVD<cmatrix_t> svd(A, ComputeThinU | ComputeThinV);
		V = svd.matrixV();
		U = svd.matrixU();
		S = svd.singularValues();

		//cmatrix_t temp_A = U * S.asDiagonal() * V.adjoint();
		//bool is_valid_svd = A.isApprox(temp_A, 1e-9);

		size_t valid_size = 0;
		for (size_t i = 0; i < S.size(); ++i)
		{
			double val = S(i);
			if (val > 1e-9)
				valid_size++;
		}

		if (/*!is_valid_svd ||*/ !valid_size || U.hasNaN() || V.hasNaN() || S.hasNaN())
		{
			QCERR("svd  error");
			throw run_fail("svd  error");
		}

		reduce_U = U.leftCols(valid_size);
		reduce_S = S.head(valid_size);
		reduce_V = V.leftCols(valid_size);

		// update m_qubits_tensor  m_lambdas
		std::vector<cmatrix_t> left_data(2);
		left_data[0] = reduce_U.topRows(U.rows() / 2);
		left_data[1] = reduce_U.bottomRows(U.rows() / 2);

		MPS_Tensor left_gamma(left_data[0], left_data[1]);
		if (!first_iter)
			left_gamma.div_gamma_by_left_lambda(m_lambdas.back());

		m_qubits_tensor.push_back(left_gamma);
		m_lambdas.push_back(reduce_S);
		m_qubits_num++;

		first_iter = false;
	}

	// create the rightmost gamma and update m_qubits_tensor
	std::vector<cmatrix_t> right_data(2); 
	cmatrix_t reduce_dagger_V = reduce_V.adjoint();
	right_data[0] = reduce_dagger_V.leftCols(reduce_dagger_V.cols() / 2);
	right_data[1] = reduce_dagger_V.rightCols(reduce_dagger_V.cols() / 2);

	MPS_Tensor right_gamma(right_data[0], right_data[1]);
	m_qubits_tensor.push_back(right_gamma);
	m_qubits_num++;
}

QError MPSImplQPU::unitarySingleQubitGate(size_t qn, QStat& matrix, bool isConjugate, GateType)
{
	int dim = sqrt(matrix.size());
	cmatrix_t mat = cmatrix_t::Map(&matrix[0], dim, dim);
	if (isConjugate)
		mat.adjointInPlace();

	execute_one_qubit_gate(qn, mat);

	return qErrorNone;
}

QError MPSImplQPU::controlunitarySingleQubitGate(size_t qn, Qnum& qnum,
	QStat& matrix, bool isConjugate, GateType)
{
	qnum.push_back(qn);
	int target_dim = sqrt(matrix.size());
	int dim = 1ULL << qnum.size();

	cmatrix_t mat = cmatrix_t::Identity(dim, dim);
	int index = 0;
	for (int i = dim - target_dim; i < dim; i++)
	{
		for (int j = dim - target_dim; j < dim; j++)
		{
			mat(i, j) = matrix[index];
			index++;
		}
	}

	if (isConjugate)
		mat.adjointInPlace();

	if (qnum.size() == 2)
		execute_two_qubit_gate(qnum[0], qnum[1], mat);
	else
		execute_multi_qubit_gate(qnum, mat);

	return qErrorNone;
}

QError MPSImplQPU::unitaryDoubleQubitGate(size_t qn_0, size_t qn_1,
	QStat& matrix, bool isConjugate, GateType)
{
	int dim = sqrt(matrix.size());
	cmatrix_t mat = cmatrix_t::Map(&matrix[0], dim, dim);

	if (isConjugate)
		mat.adjointInPlace();

	execute_two_qubit_gate(qn_0, qn_1, mat);

	return qErrorNone;
}

QError MPSImplQPU::controlunitaryDoubleQubitGate(size_t qn_0, size_t qn_1, Qnum& qnum,
	QStat& matrix, bool isConjugate, GateType)
{
	qnum.push_back(qn_0);
	qnum.push_back(qn_1);
	int target_dim = sqrt(matrix.size());
	int dim = 1ULL << qnum.size();
	cmatrix_t mat = cmatrix_t::Identity(dim, dim);
	int index = 0;
	for (int i = dim - target_dim; i < dim; i++)
	{
		for (int j = dim - target_dim; j < dim; j++)
		{
			mat(i, j) = matrix[index];
			index++;
		}
	}

	if (isConjugate)
		mat.adjointInPlace();

	execute_multi_qubit_gate(qnum, mat);

	return qErrorNone;
}

QStat MPSImplQPU::getQState()
{
	QStat qstat_result;
	Qnum new_qubits;
	Qnum sorted_indices;
	Qnum qubits = m_qubits_location;
	size_t length = 1ULL << m_qubits_num;

	// calc probability
	MPSImplQPU temp;

	temp.initState(*this);
	temp.centralize_and_sort_qubits(qubits, sorted_indices, new_qubits);
	MPS_Tensor mps_vec = temp.convert_qstate_to_mps_form(new_qubits.front(), new_qubits.back());

	qstat_result.resize(length);
	// statevector is constructed in ascending order
#pragma omp parallel for 
	for (int64_t i = 0; i < length; i++)
		qstat_result[i] = mps_vec.get_data(i)(0, 0);

	QStat temp_qstat_result(length);
	// reorder all qubits
	Qnum squeezed_qubits(qubits.size());
	squeeze_qubits(qubits, squeezed_qubits);
	int new_index = 0;

#pragma omp parallel for private(new_index)
	for (int64_t i = 0; i < length; i++)
	{
		new_index = reorder_qubits(squeezed_qubits, i);
		temp_qstat_result[new_index] = qstat_result[i];
	}

	// reverse all qubits
	QStat return_qstat(length);
#pragma omp parallel for private(new_index)
	for (int64_t i = 0; i < length; i++)
		return_qstat[i] = temp_qstat_result[reverse_qubits(i, m_qubits_num)];

	return return_qstat;
}

QError MPSImplQPU::Reset(size_t qn)
{
	return qErrorNone;
}

void MPSImplQPU::change_qubits_location(size_t src, size_t dst)
{
	if (src == dst)
		return;

	if (src < dst)
	{
		for (size_t i = src; i < dst; i++)
			swap_qubits_location(i, i + 1);
	}
	else
	{
		for (size_t i = src; i > dst; i--)
			swap_qubits_location(i, i - 1);
	}
}

void MPSImplQPU::execute_one_qubit_gate(size_t qn, const cmatrix_t &mat)
{
	size_t index = get_qubit_index(qn);
	MPS_Tensor &tensor = m_qubits_tensor[index];
	tensor.apply_matrix(mat);
}

void MPSImplQPU::execute_two_qubit_gate(size_t qn_0, size_t qn_1, const cmatrix_t &mat)
{
	size_t index_A = get_qubit_index(qn_0);
	size_t index_B = get_qubit_index(qn_1);
	size_t A = index_A;

	// Move B to be right after or  before A  
	if (index_B > index_A + 1)
		change_qubits_location(index_B, index_A + 1);
	else if (index_A > 0 && index_B < index_A - 1)
		change_qubits_location(index_B, index_A - 1);

	bool swapped = false;
	if (index_B < index_A)
	{
		A = index_A - 1;
		swapped = true;
	}

	rvector_t left_lambda, right_lambda;

	rvector_t initial_val(1);
	initial_val[0] = 1.0;
	left_lambda = (A != 0) ? m_lambdas[A - 1] : initial_val;
	right_lambda = (A + 1 != m_qubits_num - 1) ? m_lambdas[A + 1] : initial_val;

	m_qubits_tensor[A].mul_gamma_by_left_lambda(left_lambda);
	m_qubits_tensor[A + 1].mul_gamma_by_right_lambda(right_lambda);

	MPS_Tensor temp = MPS_Tensor::contract(m_qubits_tensor[A], m_lambdas[A], m_qubits_tensor[A + 1]);

	temp.apply_matrix(mat, swapped);

	MPS_Tensor left_gamma, right_gamma;
	rvector_t lambda;
	MPS_Tensor::decompose(temp, left_gamma, lambda, right_gamma);
	left_gamma.div_gamma_by_left_lambda(left_lambda);
	right_gamma.div_gamma_by_right_lambda(right_lambda);
	m_qubits_tensor[A] = left_gamma;
	m_lambdas[A] = lambda;
	m_qubits_tensor[A + 1] = right_gamma;
}

void MPSImplQPU::execute_multi_qubit_gate(const Qnum &qubits, const cmatrix_t &mat)
{
	Qnum actual_qubits(qubits.size());
	for (int i = 0; i < qubits.size(); i++)
	{
		actual_qubits[i] = get_qubit_index(qubits[i]);
	}
	Qnum target_qubits = actual_qubits;

	if (!is_ordered(actual_qubits))
	{
		Qnum actual_indices(m_qubits_num);
		std::iota(std::begin(actual_indices), std::end(actual_indices), 0);
		target_qubits.resize(actual_qubits.size());

		// need to move all target qubits to be consecutive at the right end
		move_qubits_to_right_end(actual_qubits, target_qubits, actual_indices);
	}

	size_t num_qubits = target_qubits.size();
	size_t first = target_qubits.front();
	MPS_Tensor sub_tensor(convert_qstate_to_mps_form(first, first + num_qubits - 1));

	sub_tensor.apply_matrix(mat);

	cmatrix_t state_mat = sub_tensor.get_data(0);
	for (int i = 1; i < sub_tensor.get_data().size(); i++)
	{
		cmatrix_t temp(state_mat.rows(), state_mat.cols() + sub_tensor.get_data(i).cols());
		temp << state_mat, sub_tensor.get_data(i);
		state_mat = temp;
	}

	// We convert the matrix back into an MPS structure
	MPSImplQPU sub_MPS;
	sub_MPS.initState_from_matrix(num_qubits, state_mat);

	if (num_qubits == m_qubits_num)
	{
		m_qubits_tensor.clear();
		m_qubits_tensor = sub_MPS.m_qubits_tensor;
		m_lambdas = sub_MPS.m_lambdas;
	}
	else
	{
		// copy the sub_MPS back to the corresponding positions in the original MPS
		for (size_t i = 0; i < sub_MPS.m_qubits_num; i++)
			m_qubits_tensor[first + i] = sub_MPS.m_qubits_tensor[i];

		m_lambdas[first] = sub_MPS.m_lambdas[0];

		if (first > 0)
			m_qubits_tensor[first].div_gamma_by_left_lambda(m_lambdas[first - 1]);

		for (size_t i = 1; i < num_qubits - 1; i++)
			m_lambdas[first + i] = sub_MPS.m_lambdas[i];

		if (first + num_qubits - 1 < m_qubits_num - 1)
			m_qubits_tensor[first + num_qubits - 1].div_gamma_by_right_lambda(m_lambdas[first + num_qubits - 1]);
	}
}

void MPSImplQPU::move_qubits_to_right_end(const Qnum &qubits, Qnum &target_qubits, Qnum &actual_indices)
{
	size_t num_target_qubits = qubits.size();
	size_t num_moved = 0;
	size_t right_end = qubits[0];
	for (int i = 1; i < num_target_qubits; i++)
		right_end = std::max(qubits[i], right_end);

	for (int right_index = qubits.size() - 1; right_index >= 0; right_index--)
	{
		size_t next_right = qubits[right_index];
		for (size_t i = 0; i < actual_indices.size(); i++)
		{
			if (actual_indices[i] == next_right)
			{
				for (size_t j = i; j < right_end - num_moved; j++)
				{
					swap_qubits_location(j, j + 1);
					std::swap(actual_indices[j], actual_indices[j + 1]);
				}
				num_moved++;
				break;
			}
		}
	}

	// the target qubits are simply the rightmost qubits ending at right_end
	std::iota(std::begin(target_qubits), std::end(target_qubits), right_end + 1 - num_target_qubits);
}

cmatrix_t MPSImplQPU::mul_v_by_s(const cmatrix_t &mat, const rvector_t &lambda)
{
	rvector_t initial_val(1);
	initial_val[0] = 1.0;
	if (lambda.size() == 1 && lambda == initial_val) return mat;

	cmatrix_t res_mat(mat);
	int num_rows = mat.rows(), num_cols = mat.cols();

#pragma omp parallel for 
	for (int row = 0; row < num_rows; row++)
	{
		for (int col = 0; col < num_cols; col++)
			res_mat(row, col) = mat(row, col) * lambda[col];
	}
	return res_mat;
}

MPS_Tensor MPSImplQPU::convert_qstate_to_mps_form(size_t first_index, size_t last_index)
{
	MPS_Tensor temp = m_qubits_tensor[first_index];
	rvector_t left_lambda, right_lambda;
	rvector_t initial_val(1);
	initial_val[0] = 1.0;
	left_lambda = (first_index != 0) ? m_lambdas[first_index - 1] : initial_val;
	right_lambda = (last_index != m_qubits_num - 1) ? m_lambdas[last_index] : initial_val;

	temp.mul_gamma_by_left_lambda(left_lambda);

	if (first_index == last_index)
	{
		temp.mul_gamma_by_right_lambda(right_lambda);
		return temp;
	}

	for (size_t i = first_index + 1; i < last_index + 1; i++)
		temp = MPS_Tensor::contract(temp, m_lambdas[i - 1], m_qubits_tensor[i]);

	// now temp is a tensor of 2^n matrices of size 1X1
	temp.mul_gamma_by_right_lambda(right_lambda);
	return temp;
}

void MPSImplQPU::swap_qubits_location(size_t index_A, size_t index_B)
{
	size_t actual_A = index_A;
	size_t actual_B = index_B;
	if (actual_A > actual_B)
		std::swap(actual_A, actual_B);

	if (actual_A + 1 < actual_B)
	{
		size_t i;
		for (i = actual_A; i < actual_B; i++)
			swap_qubits_location(i, i + 1);

		for (i = actual_B - 1; i > actual_A; i--)
			swap_qubits_location(i, i - 1);

		return;
	}

	MPS_Tensor A = m_qubits_tensor[actual_A], B = m_qubits_tensor[actual_B];
	rvector_t left_lambda, right_lambda;
	rvector_t initial_val(1);
	initial_val[0] = 1.0;
	left_lambda = (actual_A != 0) ? m_lambdas[actual_A - 1] : initial_val;
	right_lambda = (actual_B != m_qubits_num - 1) ? m_lambdas[actual_B] : initial_val;

	m_qubits_tensor[actual_A].mul_gamma_by_left_lambda(left_lambda);
	m_qubits_tensor[actual_B].mul_gamma_by_right_lambda(right_lambda);
	MPS_Tensor temp = MPS_Tensor::contract(m_qubits_tensor[actual_A], m_lambdas[actual_A], m_qubits_tensor[actual_B]);

	temp.apply_swap();

	MPS_Tensor left_gamma, right_gamma;
	rvector_t lambda;
	MPS_Tensor::decompose(temp, left_gamma, lambda, right_gamma);
	left_gamma.div_gamma_by_left_lambda(left_lambda);
	right_gamma.div_gamma_by_right_lambda(right_lambda);
	m_qubits_tensor[actual_A] = left_gamma;
	m_lambdas[actual_A] = lambda;
	m_qubits_tensor[actual_B] = right_gamma;

	std::swap(m_qubits_order[index_A], m_qubits_order[index_B]);

	for (size_t i = 0; i < m_qubits_num; i++)
		m_qubits_location[m_qubits_order[i]] = i;
}

bool MPSImplQPU::measure_one_collapsing(size_t qubit)
{
	if (!is_ordered(m_qubits_location))
		move_all_qubits_to_sorted_ordering();

	return apply_measure(qubit);
}

std::vector<std::vector<size_t>> MPSImplQPU::measure_all_noncollapsing(Qnum qubits, int shots)
{
	std::map<std::string, unsigned int> results;

	MPSImplQPU temp;

	std::vector<size_t> single_result(qubits.size(), 0);
	std::vector<std::vector<size_t>> measure_result(shots, single_result);
	if (!is_ordered(m_qubits_location))
		move_all_qubits_to_sorted_ordering();

#pragma omp parallel for private(temp, single_result)
	for (int i = 0; i < shots; i++)
	{
		temp.initState((*this));
		single_result = temp.apply_measure(qubits);
		measure_result[i] = single_result;
	}

	return measure_result;
}

Qnum MPSImplQPU::apply_measure(Qnum qubits)
{
	if (!is_ordered(m_qubits_location))
		move_all_qubits_to_sorted_ordering();

	Qnum  outcome_vector(qubits.size());

	for (size_t i = 0; i < qubits.size(); i++)
		outcome_vector[i] = apply_measure(qubits[i]);

	return outcome_vector;
}

bool MPSImplQPU::apply_measure(size_t qubit)
{
	Qnum qubits_to_update;
	qubits_to_update.push_back(qubit);

	// step 1 - measure qubit 0 in Z basis

	double exp_val = real(expectation_value_pauli(qubits_to_update));

	// step 2 - compute probability for 0 or 1 result
	double prob0 = (1 + exp_val) / 2;
	double prob1 = 1 - prob0;

	// step 3 - randomly choose a measurement value for qubit 0

	double rnd = random_generator19937(0.0, 1.0);
	bool measurement;
	cmatrix_t measurement_matrix(2, 2);

	cmatrix_t zero_measure(2, 2);
	zero_measure << qcomplex_t(1, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0);

	cmatrix_t one_measure(2, 2);
	one_measure << qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(1, 0);

	if (rnd < prob0)
	{
		measurement = 0;
		measurement_matrix = zero_measure;
		measurement_matrix = measurement_matrix * (1 / sqrt(prob0));
	}
	else
	{
		measurement = 1;
		measurement_matrix = one_measure;
		measurement_matrix = measurement_matrix * (1 / sqrt(prob1));
	}

	auto qindex = get_qubit_index(qubits_to_update[0]);

	m_qubits_tensor[qindex].apply_matrix(measurement_matrix);

	//step 4 - propagate the changes to all qubits to the right
	cmatrix_t id_mat = cmatrix_t::Identity(4, 4);

	for (size_t i = qubit; i < m_qubits_num - 1; i++)
	{
		// no need to propagate if no entanglement
		if (m_lambdas[i].size() == 1)
			break;   

		execute_two_qubit_gate(i, i + 1, id_mat);
	}

	// and propagate the changes to all qubits to the left
	for (size_t i = qubit; i > 0; i--)
	{
		// no need to propagate if no entanglement
		if (m_lambdas[i - 1].size() == 1)
			break; 

		execute_two_qubit_gate(i - 1, i, id_mat);
	}
	return measurement;
}

void MPSImplQPU::move_all_qubits_to_sorted_ordering()
{
	// m_qubits_order can simply be initialized
	for (size_t left_index = 0; left_index < m_qubits_num; left_index++)
	{
		size_t min_index = left_index;
		for (size_t i = left_index + 1; i < m_qubits_num; i++)
		{
			if (m_qubits_order[i] == min_index)
			{
				min_index = i;
				break;
			}
		}

		// Move this qubit back to its original position
		for (size_t j = min_index; j > left_index; j--)
			swap_qubits_location(j, j - 1);
	}
}

qcomplex_t MPSImplQPU::expectation_value_pauli(const Qnum &qubits)
{
	Qnum internal_qubits(qubits.size());
	for (size_t i = 0; i < qubits.size(); i++)
		internal_qubits[i] = get_qubit_index(qubits[i]);

	Qnum extended_qubits = internal_qubits;

	const auto min = std::min_element(begin(internal_qubits), end(internal_qubits));
	const auto max = std::max_element(begin(internal_qubits), end(internal_qubits));
	size_t min_qubit = *min;
	size_t max_qubit = *max;

	// The number of qubits added  to extended_qubits
	size_t num_Is = 0;

	// Add all the additional qubits at the end of the vector of extended_qubits
	// The I matrices are added in expectation_value_pauli_internal, after they are reversed
	for (size_t i = min_qubit; i <= max_qubit; i++)
	{
		auto itr = std::find(internal_qubits.begin(), internal_qubits.end(), i);
		if (itr == internal_qubits.end())
		{
			extended_qubits.push_back(i);
			num_Is++;
		}
	}

	return expectation_value_pauli_internal(extended_qubits, { GateType::PAULI_Z_GATE }, min_qubit, max_qubit, num_Is);
}

qcomplex_t MPSImplQPU::expectation_value_pauli_internal(const Qnum &qubits, const std::vector<GateType> &matrices,
	size_t first_index, size_t last_index, size_t num_Is)
{
	// when computing the expectation value. We only have to sort the pauli matrices
	// to be in the same ordering as the qubits

	// Preliminary step - reverse the order of the matrices because
	// they are ordered in reverse to that of the qubits (in the interface)
	auto reversed_matrices = matrices;
	if(reversed_matrices.size() > 1)
		reverse(reversed_matrices.begin(), reversed_matrices.end());

	for (size_t i = 0; i < num_Is; i++)
		reversed_matrices.push_back(GateType::I_GATE);

	// sort the paulis according to the initial ordering of the qubits
	auto sorted_matrices = sort_paulis_by_qubits(reversed_matrices, qubits);

	auto gate = sorted_matrices[0];

	// Step 1 - multiply tensor of q0 by its left lambda
	MPS_Tensor left_tensor = m_qubits_tensor[first_index];

	if (first_index > 0)
		left_tensor.mul_gamma_by_left_lambda(m_lambdas[first_index - 1]);

	// The last gamma must be multiplied also by its right lambda.
	// Here we handle the special case that we are calculating exp val
	// on a single qubit
	// we need to mul every gamma by its right lambda
	if (first_index == last_index && first_index < m_qubits_num - 1)
		left_tensor.mul_gamma_by_right_lambda(m_lambdas[first_index]);

	// Step 2 - prepare the dagger of left_tensor
	MPS_Tensor left_tensor_dagger(left_tensor.get_data(0).adjoint(), left_tensor.get_data(1).adjoint());
	// Step 3 - Apply the gate to q0
	left_tensor.apply_pauli(gate);

	// Step 4 - contract Gamma0' with Gamma0 over dimensions a0 and i
	// Before contraction, Gamma0' has size a1 x a0 x i, Gamma0 has size i x a0 x a1
	// result = left_contract is a matrix of size a1 x a1
	cmatrix_t final_contract;
	MPS_Tensor::contract_2_dimensions(left_tensor_dagger, left_tensor, final_contract);
	for (size_t qubit_num = first_index + 1; qubit_num <= last_index; qubit_num++)
	{
		// Step 5 - multiply next Gamma by its left lambda (same as Step 1)
		// next gamma has dimensions a0 x a1 x i
		MPS_Tensor next_gamma = m_qubits_tensor[qubit_num];
		next_gamma.mul_gamma_by_left_lambda(m_lambdas[qubit_num - 1]);

		// Last qubit must be multiplied by rightmost lambda
		if (qubit_num == last_index && qubit_num < m_qubits_num - 1)
			next_gamma.mul_gamma_by_right_lambda(m_lambdas[qubit_num]);

		// Step 6 - prepare the dagger of the next gamma (same as Step 2)
		// next_gamma_dagger has dimensions a1' x a0' x i
		MPS_Tensor next_gamma_dagger(next_gamma.get_data(0).adjoint(), next_gamma.get_data(1).adjoint());

		// Step 7 - apply gate (same as Step 3)
		gate = sorted_matrices[qubit_num - first_index];
		next_gamma.apply_pauli(gate);

		// Step 8 - contract final_contract from previous stage with next gamma over a1
		// final_contract has dimensions a1 x a1, Gamma1 has dimensions a1 x a2 x i (where i=2)
		// result is a tensor of size a1 x a2 x i
		MPS_Tensor next_contract(final_contract * next_gamma.get_data(0), final_contract * next_gamma.get_data(1));

		// Step 9 - contract next_contract (a1 x a2 x i)
		// with next_gamma_dagger (i x a2 x a1) (same as Step 4)
		// here we need to contract across two dimensions: a1 and i
		// result is a matrix of size a2 x a2
		MPS_Tensor::contract_2_dimensions(next_gamma_dagger, next_contract, final_contract);

	}

	// Step 10 - contract over final matrix of size aN x aN
	// We need to contract the final matrix with itself
	// Compute this by taking the trace of final_contract
	qcomplex_t result = final_contract.trace();

	return result;
}


void MPSImplQPU::unitaryQubitGate(Qnum qubits, QStat matrix, bool isConjugate)
{
    auto length = 1ull << (1 << qubits.size());
    if (length != matrix.size())
    {
        QCERR("param error");
        throw run_fail("param error");
    }

    auto gate_type = GateType::GATE_UNDEFINED;
    if (1 == qubits.size())
    {
        unitarySingleQubitGate(qubits[0], matrix, isConjugate, gate_type);
    }
    else
    {
        unitaryDoubleQubitGate(qubits[0], qubits[1], matrix, isConjugate, gate_type);
    }
}

cmatrix_t MPSImplQPU::density_matrix(const Qnum &qubits)
{
    Qnum internal_qubits(qubits.size());
    for (size_t i = 0; i < qubits.size(); i++)
        internal_qubits[i] = get_qubit_index(qubits[i]);

    MPSImplQPU temp;
    temp.initState(*this);

    Qnum new_qubits;
    Qnum sorted_indices;
    centralize_and_sort_qubits(qubits, sorted_indices, new_qubits);

    MPS_Tensor mps_vec = temp.convert_qstate_to_mps_form(new_qubits.front(), new_qubits.back());

    size_t dims = mps_vec.get_dim();
    cmatrix_t rho(dims, dims);

    for (size_t i = 0; i < dims; i++) 
    {
        for (size_t j = 0; j < dims; j++)
        {
            auto A = mps_vec.get_data(i);

            auto B = mps_vec.get_data(j);
            auto conj_B = B.conjugate();
            
            auto mul_matrix = matrix_element_multiplication(A, conj_B);
            rho(i, j) = mul_matrix.sum();
        }
    }

    return rho;
}


double MPSImplQPU::single_expectation_value(const Qnum &qubits, const cmatrix_t &matrix)
{

#ifdef USE_DENSITY_MATRIX

    double P = 0.0;
    MPSImplQPU temp;
    temp.initState(*this);

    auto single_gate = Eigen_to_QStat(matrix);
    temp.unitarySingleQubitGate(qubits[0], single_gate, false, GateType::GATE_UNDEFINED);

    auto qstate = temp.getQState();

    for (size_t i = 0; i < qstate.size(); i++)
    {
        P += std::norm(qstate[i]);
    }

    return P;

#else

    Qnum internal_qubits(qubits.size());
    for (size_t i = 0; i < qubits.size(); i++)
        internal_qubits[i] = get_qubit_index(qubits[i]);

    Qnum reversed_qubits = qubits;
    std::reverse(reversed_qubits.begin(), reversed_qubits.end());

    cmatrix_t rho;

    Qnum target_qubits(qubits.size());
    if (is_ordered(reversed_qubits))
    {
        rho = density_matrix(reversed_qubits);
    }
    else
    {
        Qnum actual_indices(m_qubits_num);
        std::iota(std::begin(actual_indices), std::end(actual_indices), 0);

        MPSImplQPU temp;
        temp.initState(*this);
        temp.move_qubits_to_right_end(reversed_qubits, target_qubits, actual_indices);

        rho = temp.density_matrix(target_qubits);
    }

    qcomplex_t result = 0;
    for (size_t i = 0; i < matrix.rows(); i++)
    {
        for (size_t j = 0; j < matrix.rows(); j++)
        {
            result += matrix(i, j) * rho(j, i);
        }
    }

    return result.real();

#endif // !USE_DENSITY_MATRIX
}


double MPSImplQPU::double_expectation_value(const Qnum &qubits, const cmatrix_t &matrix)
{
    auto ctr_qubit = qubits[0];
    auto tar_qubit = qubits[1];

    auto gate = Eigen_to_QStat(matrix);

    if (qubits[1] > qubits[0])
    {
        gate = { gate[0] ,gate[2] ,gate[1] ,gate[3] ,
              gate[8] ,gate[10] ,gate[9] ,gate[11] ,
              gate[4] ,gate[6] ,gate[5] ,gate[7] ,
              gate[12] ,gate[14] ,gate[13] ,gate[15] };

        ctr_qubit = qubits[1];
        tar_qubit = qubits[0];
    }

    double P = 0.0;
    MPSImplQPU temp;
    temp.initState(*this);

    auto qstate = temp.getQState();

    size_t stat_size = 1ull << (m_qubits_num - 2);
    size_t length1 = 1ull << tar_qubit;
    size_t length2 = 1ull << ctr_qubit;

    for (size_t i = 0; i < stat_size; ++i)
    {
        size_t index00 = insert(i, tar_qubit, ctr_qubit);
        size_t index01 = index00 + length1;
        size_t index10 = index00 + length2;
        size_t index11 = index00 + length1 + length2;

        auto s00 = qstate[index00];
        auto s01 = qstate[index01];
        auto s10 = qstate[index10];
        auto s11 = qstate[index11];

        auto result_s00 = s00 * gate[0] + s01 * gate[1] + s10 * gate[2] + s11 * gate[3];
        auto result_s01 = s00 * gate[4] + s01 * gate[5] + s10 * gate[6] + s11 * gate[7];
        auto result_s10 = s00 * gate[8] + s01 * gate[9] + s10 * gate[10] + s11 * gate[11];
        auto result_s11 = s00 * gate[12] + s01 * gate[13] + s10 * gate[14] + s11 * gate[15];

        P += std::norm(result_s00);
        P += std::norm(result_s01);
        P += std::norm(result_s10);
        P += std::norm(result_s11);
    }

    return P;
}

double MPSImplQPU::expectation_value(const Qnum& qubits, const cmatrix_t& matrix)
{
    if (1 == qubits.size())
    {
        return single_expectation_value(qubits, matrix);
    }
    else if (2 == qubits.size())
    {
        return double_expectation_value(qubits, matrix);
    }
    else
    {
        QCERR("param error");
        throw run_fail("param error");
    }
}

qcomplex_t MPSImplQPU::pmeasure_bin_index(std::string str)
{
    Qnum qubits = m_qubits_location;

    auto char_to_bin = [](const char& val)
    {
        QPANDA_ASSERT(val != '0' && val != '1', "pmeasure_bin_index str error");
        return val != '0';
    };

    std::vector<MPS_Tensor> qubits_tensor = m_qubits_tensor;
    QPANDA_ASSERT(qubits_tensor.size() != str.size(), "pmeasure_bin_index str size error");

    rvector_t initial_val(1); 
    initial_val[0] = 1.0;
    m_qubits_tensor.front().mul_gamma_by_left_lambda(initial_val);
    m_qubits_tensor.back().mul_gamma_by_right_lambda(initial_val);

    if (1 == qubits_tensor.size())
    {
        return qubits_tensor[0].m_physical_index[char_to_bin(str[0])](0, 0);
    }

    for (size_t i = 0; i < qubits_tensor.size() - 1 ; i++)
    {
        qubits_tensor[i].mul_gamma_by_right_lambda(m_lambdas[i]);
    }

    cmatrix_t result = cmatrix_t::Identity(1, 1);

    std::reverse(str.begin(), str.end());
    for (auto i = 0; i < str.size(); ++i)
    { 
        int val = char_to_bin(str[m_qubits_location[i]]);

        result *= qubits_tensor[i].m_physical_index[val];
    }

    return result(0, 0);
}


qcomplex_t MPSImplQPU::pmeasure_dec_index(std::string str)
{
    uint128_t index(str.c_str());

    std::string bin_str = integerToBinary(index, m_qubits_num);

    return pmeasure_bin_index(bin_str);
}