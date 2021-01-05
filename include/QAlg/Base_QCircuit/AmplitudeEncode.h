/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

amplitude encode

Author: XueCheng
*/

#ifndef  AMPLITUDE_ENCODE_H
#define  AMPLITUDE_ENCODE_H

#include "Core/Core.h"

QPANDA_BEGIN

/**
* @brief  amplitude encode
* @ingroup quantum Algorithm
* @param[in] QVec& Available qubits
* @param[in] std::vector<double>& the target datas which will be encoded to the quantum state
* @param[in] const bool Whether to carry out normalization verification
* @return the encoded circuit
* @note The coding data must meet normalization conditions.
*/
inline QCircuit amplitude_encode(QVec q, vector<double> data, const bool b_need_check_normalization = true)
{
	if (b_need_check_normalization)
	{
		const double max_precision = 1e-13;

		//check parameter b
		double tmp_sum = 0.0;
		//cout << "on amplitude_encode" << endl;
		for (const auto i : data)
		{
			//cout << i << ", ";
			tmp_sum += (i*i);
		}
		//cout << "tmp_sum = " << tmp_sum << endl;
		if (abs(1.0 - tmp_sum) > max_precision)
		{
			if (abs(tmp_sum) < max_precision)
			{
				QCERR("Error: The input vector b is zero.");
				return QCircuit();
			}

			//cout << "sum = " << abs(1.0 - tmp_sum) << endl;
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input vector b must satisfy the normalization condition.");
		}
	}

	if (data.size() > (1 << q.size()))
	{
		throw run_fail("Amplitude_encode parameter error.");
	}
	while (data.size() < (1 << q.size()))
	{
		data.push_back(0);
	}
	QCircuit qcir;
	double sum_0 = 0;
	double sum_1 = 0;
	size_t high_bit = (size_t)log2(data.size()) - 1;
	if (high_bit == 0)
	{
		if (data[0] > 1e-20 && data[1] > 1e-20)
		{
			qcir << RY(q[0], 2 * acos(data[0] / sqrt(data[0] * data[0] + data[1] * data[1])));
		}
		else if (data[0] > 1e-20 && data[1] < -1e-20)
		{
			qcir << RY(q[0], -2 * acos(data[0] / sqrt(data[0] * data[0] + data[1] * data[1])));
		}
		else if (data[0] < -1e-20 && data[1] > 1e-20)
		{
			qcir << RY(q[0], 2 * acos(data[0] / sqrt(data[0] * data[0] + data[1] * data[1])));
		}
		else if (data[0] < -1e-20 && data[1] < -1e-20)
		{
			qcir << RY(q[0], 2 * (2 * PI - acos(data[0] / sqrt(data[0] * data[0] + data[1] * data[1]))));
		}
		else if (abs(data[0]) < 1e-20 && data[1] > 1e-20)
		{
			qcir << RY(q[0], PI);
		}
		else if (abs(data[0]) < 1e-20 && data[1] < -1e-20)
		{
			qcir << RY(q[0], -PI);
		}
		else if (abs(data[1]) < 1e-20 && data[0] < -1e-20)
		{
			qcir << RY(q[0], 2 * PI);
		}
		else if (abs(data[0]) < 1e-20 && abs(data[1]) < 1e-20)
		{
			throw run_fail("Amplitude_encode error.");
		}
	}
	else
	{
		for (auto i = 0; i < (data.size() >> 1); i++)
		{
			sum_0 += data[i] * data[i];
			sum_1 += data[i + (data.size() >> 1)] * data[i + (data.size() >> 1)];
		}
		if (sum_0 + sum_1 > 1e-20)
			qcir << RY(q[high_bit], 2 * acos(sqrt(sum_0 / (sum_0 + sum_1))));
		else
		{
			throw run_fail("Amplitude_encode error.");
		}

		if (sum_0 > 1e-20)
		{
			QVec temp({ q[high_bit] });
			vector<double> vtemp(data.begin(), data.begin() + (data.size() >> 1));
			qcir << X(q[high_bit]) << amplitude_encode(q - temp, vtemp, false).control({ q[high_bit] }) << X(q[high_bit]);
		}
		if (sum_1 > 1e-20)
		{
			QVec temp({ q[high_bit] });
			vector<double> vtemp(data.begin() + (data.size() >> 1), data.end());
			qcir << amplitude_encode(q - temp, vtemp, false).control({ q[high_bit] });
		}
	}

	return qcir;
}

/**
* @brief  init superposition state
* @ingroup quantum Algorithm
* @param[in] QVec Available qubits
* @param[in] size_t the target data
* @return the circuit
* @note
*/
#if 0
inline QCircuit init_superposition_state(QVec q, size_t d){
	QCircuit qcir;

	size_t highest_bit = (size_t)log2(d - 1);

	if (d == (1 << (int)log2(d)))
	{
		for (auto i = 0; i < (int)log2(d); i++)
		{
			qcir << H(q[i]);
		}
	}
	else if (d == 3)
	{
		qcir << RY(q[1], 2 * acos(sqrt(2.0 / 3))) << X(q[1]);
		qcir << H(q[0]).control({ q[1] }) << X(q[1]);
	}
	else
	{
		qcir << RY(q[highest_bit], 2 * acos(sqrt((1 << highest_bit)*1.0 / d)));
		QCircuit qcir1;
		for (auto i = 0; i < highest_bit; i++)
		{
			qcir1 << H(q[i]);
		}
		qcir << X(q[highest_bit]) << qcir1.control({ q[highest_bit] }) << X(q[highest_bit]);

		size_t d1 = d - (1 << highest_bit);
		if (d > 1)
		{
			QCircuit qcir2 = init_superposition_state(q, d1);
			qcir2.setControl({ q[highest_bit] });
			qcir << qcir2;
		}
	}
	return qcir;
}
#endif
inline QCircuit init_superposition_state(QVec q, size_t d)
{
	QCircuit qcir;

	size_t highest_bit = (size_t)log2(d - 1);

	if (d == (1 << (int)log2(d)))
	{
		for (auto i = 0; i < (int)log2(d); i++)
		{
			qcir << H(q[i]);
		}
	}
	else if (d == 3)
	{
		qcir << RY(q[1], 2 * acos(sqrt(2.0 / 3))) << X(q[1]);
		qcir << H(q[0]).control({ q[1] }) << X(q[1]);
	}
	else
	{
		qcir << RY(q[highest_bit], 2 * acos(sqrt((1 << highest_bit)*1.0 / d)));
		QCircuit qcir1;
		for (auto i = 0; i < highest_bit; i++)
		{
			qcir1 << H(q[i]);
		}
		qcir << X(q[highest_bit]) << qcir1.control({ q[highest_bit] }) << X(q[highest_bit]);

		size_t d1 = d - (1 << highest_bit);
		if (d1 > 1)
		{
			QCircuit qcir2 = init_superposition_state(q, d1);
			qcir2.setControl({ q[highest_bit] });
			qcir << qcir2;
		}
	}
	return qcir;
}

QPANDA_END

#endif