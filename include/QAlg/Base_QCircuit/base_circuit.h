/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

Quanatum Fourier transform

*/

#ifndef  BASE_CIRCUIT_H
#define  BASE_CIRCUIT_H

#include "Core/Core.h"

QPANDA_BEGIN

/**
* @brief Quantum Fourier Transform
* @ingroup Base_QCircuit
* @param[in] Quantum bit vectors with ordinal numbers from high to low
* @note do notice the order of the bits in the quantum vector
*/
inline QCircuit QFT(QVec qvec)
{
	QCircuit  qft = CreateEmptyCircuit();
	for (auto i = 0; i < qvec.size(); i++)
	{
		qft << H(qvec[qvec.size() - 1 - i]);
		for (auto j = i + 1; j < qvec.size(); j++)
		{
			qft << CR(qvec[qvec.size() - 1 - j],
				qvec[qvec.size() - 1 - i], 2 * PI / (1 << (j - i + 1)));
		}
	}
	return qft;
}

/**
* @brief Encoding index-info(positive integers) into quantum states
* @ingroup Base_QCircuit
* @param[in] const size_t index-info(positive integers)
* @param[in] const QVec& target qubits
* @param[in] const size_t Last index information, only useful when b_merge is true
* @param[in] bool If b_merge is true, the lines of the current index will be merged with the circuit of the previous index, 
                  which can offset part of the quantum-gates, default is false
* @return the sub-circuit for current index
* @note
*/
inline QCircuit index_to_circuit(const size_t index, const QVec& index_qv, const size_t pre_index = 0, bool b_merge = false)
{
	size_t tmp_index = index;
	size_t tmp_pre_index = pre_index;
	if (!b_merge)
	{
		tmp_pre_index = 1;
	}

	QCircuit ret_cir;
	size_t data_qubits_cnt = index_qv.size();
	for (size_t i = 0; i < data_qubits_cnt; ++i)
	{
		if ((tmp_index % 2) != (tmp_pre_index % 2))
		{
			ret_cir << X(index_qv[i]);
		}

		tmp_index /= 2;

		if (b_merge)
		{
			tmp_pre_index /= 2;
		}
	}

	return ret_cir;
}

/**
* @brief Encoding data(size_t) into quantum states
* @ingroup Base_QCircuit
* @param[in] size_t data
* @param[in] const QVec& target qubits
* @return the sub-circuit for current data
* @note
*/
inline QCircuit data_to_circuit(size_t data, const QVec &data_qubits) {
	QCircuit data_cir;
	for (size_t i = 0; i < data_qubits.size(); ++i)
	{
		if (0 != data % 2)
		{
			data_cir << X(data_qubits[i]);
		}

		data /= 2;
	}

	return data_cir;
}

/**
* @brief Encoding array(size_t) into quantum states
* @ingroup Base_QCircuit
* @param[in] const std::vector<size_t>&
* @param[in] const QVec& index qubits
* @param[in] const QVec& data qubits
* @return the sub-circuit for current array
* @note
*/
inline QCircuit arry_to_cir(const std::vector<size_t>& arry, const QVec& index_qv, const QVec& data_qv) {
	QCircuit ret_cir;
	if (arry.size() == 0)
	{
		return ret_cir;
	}

	QCircuit tmp_index_cir = index_to_circuit(0, index_qv);
	for (size_t i = 0; i < arry.size(); )
	{
		QCircuit data_cir = data_to_circuit(arry[i], data_qv);
		data_cir.setControl(index_qv);
		ret_cir << tmp_index_cir << data_cir;

		const auto pre_index = i;
		tmp_index_cir = index_to_circuit(++i, index_qv, pre_index, true);
	}

	return ret_cir;
}

QPANDA_END

#endif