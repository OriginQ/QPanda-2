#include "Core/VirtualQuantumProcessor/MPSQVM/MPSTensor.h"

void MPS_Tensor::handle_gamma_by_lambda(const rvector_t &Lambda,
    bool right, /* or left */
    bool mul    /* or div */)
{
    rvector_t initial_val(1);
    initial_val[0] = 1.0;
    if (Lambda.size() == 1 && Lambda == initial_val) return;

    size_t rows = m_physical_index[0].rows(), cols = m_physical_index[0].cols();

    for (size_t i = 0; i < m_physical_index.size(); i++)
    {
        for (size_t a1 = 0; a1 < rows; a1++)
        {
            for (size_t a2 = 0; a2 < cols; a2++)
            {
                size_t factor = right ? a2 : a1;

                if (mul)
                    m_physical_index[i](a1, a2) *= Lambda[factor];
                else
                    m_physical_index[i](a1, a2) /= Lambda[factor];
            }
        }
    }
}

MPS_Tensor MPS_Tensor::contract(const MPS_Tensor &left_gamma, 
	const rvector_t &lambda, const MPS_Tensor &right_gamma)
{
    MPS_Tensor result;
    MPS_Tensor new_left = left_gamma;
    new_left.mul_gamma_by_right_lambda(lambda);

    for (size_t i = 0; i < new_left.m_physical_index.size(); i++)
    {
        for (size_t j = 0; j < right_gamma.m_physical_index.size(); j++)
            result.m_physical_index.push_back(new_left.m_physical_index[i] * right_gamma.m_physical_index[j]);
    }

    return result;
}



void MPS_Tensor::contract_2_dimensions(const MPS_Tensor &left_gamma, 
	const MPS_Tensor &right_gamma, cmatrix_t &result)
{
	size_t left_rows = left_gamma.m_physical_index[0].rows();
	size_t left_columns = left_gamma.m_physical_index[0].cols();
	size_t left_size = left_gamma.get_dim();
	size_t right_rows = right_gamma.m_physical_index[0].rows();
	size_t right_columns = right_gamma.m_physical_index[0].cols();
	size_t right_size = right_gamma.get_dim();

	// left_columns/right_rows and left_size/right_size
	if (left_columns != right_rows)
		throw std::runtime_error("left_columns != right_rows");

	if (left_size != right_size)
		throw std::runtime_error("left_size != right_size");

	result = cmatrix_t::Zero(left_rows, right_columns);

	size_t omp_limit = left_rows * right_columns;

#pragma omp parallel for 
	for (int l_row = 0; l_row < left_rows; l_row++)
	{
		for (int r_col = 0; r_col < right_columns; r_col++)
		{
			for (int size = 0; size < left_size; size++)
			{
				for (int index = 0; index < left_columns; index++)
				{
					result(l_row, r_col) += left_gamma.m_physical_index[size](l_row, index) *
						right_gamma.m_physical_index[size](index, r_col);
				}
			}
		}
	}
}


void MPS_Tensor::decompose(MPS_Tensor &temp, MPS_Tensor &left_gamma, 
	rvector_t &lambda, MPS_Tensor &right_gamma)
{
	// reshape before SVD
	cmatrix_t temp1(temp.m_physical_index[0].rows(), temp.m_physical_index[0].cols() + temp.m_physical_index[1].cols());
	temp1 << temp.m_physical_index[0], temp.m_physical_index[1];
	cmatrix_t temp2(temp.m_physical_index[2].rows(), temp.m_physical_index[2].cols() + temp.m_physical_index[3].cols());
	temp2 << temp.m_physical_index[2], temp.m_physical_index[3];
	cmatrix_t A(temp1.rows() + temp2.rows(), temp1.cols());
	A << temp1, temp2;

	// use BDCSVD to SVD

	rvector_t S;
	cmatrix_t U, V;

	JacobiSVD<cmatrix_t> svd(A, ComputeThinU | ComputeThinV);
	V = svd.matrixV();
	U = svd.matrixU();
	S = svd.singularValues();

	cmatrix_t temp_A = U * S.asDiagonal() * V.adjoint();
	bool is_valid_svd = A.isApprox(temp_A, 1e-9);

	// reduce invalid data
	size_t valid_size = 0;
	for (size_t i = 0; i < S.size(); ++i)
	{
		double val = S(i);
		if (val > 1e-9)
			valid_size++;
	}

	if (/*!is_valid_svd ||*/ !valid_size || U.hasNaN() || V.hasNaN() || S.hasNaN() )
	{
		QCERR("svd  error");
		throw std::runtime_error("svd  error");
	}

    cmatrix_t reduce_U = U.leftCols(valid_size);
	rvector_t reduce_S = S.head(valid_size);
	cmatrix_t reduce_V = V.leftCols(valid_size);

	// calc Gamma and Lambda by  U S V 
	left_gamma.m_physical_index.resize(2);
	left_gamma.m_physical_index[0] = reduce_U.topRows(reduce_U.rows() / 2);
	left_gamma.m_physical_index[1] = reduce_U.bottomRows(reduce_U.rows() / 2);

	lambda = reduce_S;

	reduce_V.adjointInPlace();
	right_gamma.m_physical_index.resize(2);
	right_gamma.m_physical_index[0] = reduce_V.leftCols(reduce_V.cols() / 2);
	right_gamma.m_physical_index[1] = reduce_V.rightCols(reduce_V.cols() / 2);
}

