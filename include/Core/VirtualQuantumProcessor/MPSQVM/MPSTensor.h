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

#ifndef _MPSTENSOR_H_
#define _MPSTENSOR_H_

#include "Core/Utilities/Tools/Utils.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include "Core/Utilities/Tools/QMatrixDef.h"
#include <iomanip>

QPANDA_BEGIN
class MPS_Tensor
{
public:
    std::vector<QMatrixXcd> m_physical_index;

public:
	MPS_Tensor() {}

	MPS_Tensor(const QMatrixXcd& data0, const QMatrixXcd& data1) 
	{
		m_physical_index.clear();
		m_physical_index.push_back(data0);
		m_physical_index.push_back(data1);
	}

    ~MPS_Tensor() {}

    //Get the dimension of the physical index of the tensor
    size_t get_dim() const
    {
        return m_physical_index.size();
    }

	std::vector <QMatrixXcd>  get_data() const 
	{ 
		return m_physical_index;
	}

    QMatrixXcd get_data(size_t i) const
	{ 
		return m_physical_index[i];
	}

    void apply_swap()
    {
        std::swap(m_physical_index[1], m_physical_index[2]);
    }

    void apply_matrix(const QMatrixXcd &mat, bool swapped = false)
    {
		if (mat.isIdentity())
			return;

		if (swapped)
			swap(m_physical_index[1], m_physical_index[2]);

		MPS_Tensor new_tensor;
		for (size_t i = 0; i < mat.rows(); i++)
			new_tensor.m_physical_index.push_back(mat(i, 0) * m_physical_index[0]);

		// add all other columns 
		for (size_t i = 0; i < mat.rows(); i++)
		{
			for (size_t j = 1; j < mat.cols(); j++)
			{
				new_tensor.m_physical_index[i] += mat(i, j) * m_physical_index[j];
			}
		}
		*this = new_tensor;
		if (swapped)
			swap(m_physical_index[1], m_physical_index[2]);
    }

    void mul_gamma_by_left_lambda(const QVectorXd &Lambda)
    {
        handle_gamma_by_lambda(Lambda, false,/*left*/ true /*mul*/);
    }

    void mul_gamma_by_right_lambda(const QVectorXd &Lambda)
    {
        handle_gamma_by_lambda(Lambda, true,/*right*/ true /*mul*/);
    }

    void div_gamma_by_left_lambda(const QVectorXd &Lambda)
    {
        handle_gamma_by_lambda(Lambda, false,/*left*/ false /*div*/);
    }

    void div_gamma_by_right_lambda(const QVectorXd &Lambda)
    {
        handle_gamma_by_lambda(Lambda, true,/*right*/ false /*div*/);
    }

    static MPS_Tensor contract(const MPS_Tensor &left_gamma, const QVectorXd &lambda, const MPS_Tensor &right_gamma);

    static void decompose(MPS_Tensor &temp, MPS_Tensor &left_gamma, QVectorXd &lambda, MPS_Tensor &right_gamma);

    static void contract_2_dimensions(const MPS_Tensor &left_gamma, const MPS_Tensor &right_gamma, QMatrixXcd &result);

    void apply_pauli(GateType gate)
	{
        switch (gate)
		{
		case GateType::PAULI_X_GATE:
            std::swap(m_physical_index[0], m_physical_index[1]);
            break;
		case GateType::PAULI_Y_GATE:
            m_physical_index[0] = m_physical_index[0] * qcomplex_t(0, 1);
            m_physical_index[1] = m_physical_index[1] * qcomplex_t(0, -1);
            std::swap(m_physical_index[0], m_physical_index[1]);
            break;
		case GateType::PAULI_Z_GATE:
            m_physical_index[1] = m_physical_index[1] * (-1.0);
            break;
		case GateType::I_GATE:
            break;
        default:
            throw std::invalid_argument("illegal gate for contract_with_self");
        }

    }

private:
    void handle_gamma_by_lambda(const QVectorXd &Lambda, bool right, /* or left */  bool mul    /* or div */);

};

QPANDA_END

#endif  //!_MPSTENSOR_H_
