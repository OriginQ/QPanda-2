#include "Core/Utilities/Tools/QStatMatrix.h"
#include <cmath>
#include <string.h>
using namespace std;
USING_QPANDA

#define PRINT_TRACE 0
#if PRINT_TRACE
#define PTrace printf
#define PTraceMat(mat) (std::cout << (mat) << endl)
#else
#define PTrace
#define PTraceMat(mat)
#endif

bool QPanda::isPerfectSquare(int number)
{
    for(int i = 1; number > 0; i += 2)
    {
        number -= i;
    }
    return  0 == number;
}


QStat QPanda::operator+(const QStat &matrix_left, const QStat &matrix_right)
{
    if ((matrix_left.size() != matrix_right.size())    // insure dimension of the two matrixes is same
        || (!isPerfectSquare((int)matrix_left.size())))   
    {
        QCERR("QStat is illegal");
        throw invalid_argument("QStat is illegal");
    }

    int size = (int)matrix_left.size();
    QStat matrix_result(size, 0);

    for (int i = 0; i <size; i++)
    {
        matrix_result[i] = matrix_left[i] + matrix_right[i];
    }

    return matrix_result;
}


QStat QPanda::operator+(const QStat &matrix_left, const qcomplex_t value)
{
    if (!isPerfectSquare((int)matrix_left.size()))
    {
        QCERR("QStat is illegal");
        throw invalid_argument("QStat is illegal");
    }

    int size = (int)matrix_left.size();
    QStat matrix_result(size, 0);

    for (int i = 0; i < size; i++)
    {
        matrix_result[i] = matrix_left[i] + value;
    }

    return matrix_result;
}



QStat QPanda::operator+(const qcomplex_t value, const QStat &matrix_right)
{
    if (!isPerfectSquare((int)matrix_right.size()))
    {
        QCERR("QStat is illegal");
        throw invalid_argument("QStat is illegal");
    }

    int size = (int)matrix_right.size();
    QStat matrix_result(size, 0);

    for (int i = 0; i < size; i++)
    {
        matrix_result[i] = value + matrix_right[i];
    }

    return matrix_result;
}


QStat QPanda::operator-(const QStat &matrix_left, const QStat &matrix_right)
{
    if ((matrix_left.size() != matrix_right.size())  // insure dimension of the two matrixes is same
        || (!isPerfectSquare((int)matrix_left.size()))) 
    {
        QCERR("QStat is illegal");
        throw invalid_argument("QStat is illegal");
    }

    int size = (int)matrix_left.size();
    QStat matrix_result(size, 0);

    for (int i = 0; i < size; i++)
    {
        matrix_result[i] = matrix_left[i] - matrix_right[i];
    }

    return matrix_result;
}



QStat QPanda::operator-(const QStat &matrix_left, const qcomplex_t &value)
{
    if (!isPerfectSquare((int)matrix_left.size()))
    {
        QCERR("QStat is illegal");
        throw invalid_argument("QStat is illegal");
    }

    int size = (int)matrix_left.size();
    QStat matrix_result(size, 0);

    for (int i = 0; i < size; i++)
    {
        matrix_result[i] = matrix_left[i] - value;
    }

    return matrix_result;
}



QStat QPanda::operator-(const qcomplex_t &value, const QStat &matrix_right)
{
    if (!isPerfectSquare((int)matrix_right.size()))
    {
        QCERR("QStat is illegal");
        throw invalid_argument("QStat is illegal");
    }

    int size = (int)matrix_right.size();
    QStat matrix_result(size, 0);

    for (int i = 0; i < size; i++)
    {
        matrix_result[i] = value - matrix_right[i];
    }

    return matrix_result;
}



QStat QPanda::operator*(const QStat &matrix_left, const QStat &matrix_right)
{
    if ((matrix_left.size() != matrix_right.size())  // insure dimension of the two matrixes is same
        || (!isPerfectSquare((int)matrix_left.size())))
    {
        QCERR("QStat is illegal");
        throw invalid_argument("QStat is illegal");
    }

    int size = (int)matrix_left.size();
    QStat matrix_result(size, 0);
    int dimension = (int)sqrt(size);

    for (int i = 0; i < dimension; i++)
    {
        for (int j = 0; j < dimension; j++)
        {
            qcomplex_t temp = 0;
            for (int k = 0; k < dimension; k++)
            {
                temp += matrix_left[i*dimension + k] * matrix_right[k*dimension + j];
            }
            matrix_result[i*dimension + j] = temp;
        }
    }

    return matrix_result;
}


QStat QPanda::operator*(const QStat &matrix_left, const qcomplex_t &value)
{
    if (!isPerfectSquare((int)matrix_left.size()))
    {
        QCERR("QStat is illegal");
        throw invalid_argument("QStat is illegal");
    }

    int size = (int)matrix_left.size();
    QStat matrix_reslut(size, 0);

    for (int i = 0; i < size; i++)
    {
        matrix_reslut[i] = matrix_left[i] * value;
    }

    return matrix_reslut;
}


QStat QPanda::operator*(const qcomplex_t &value, const QStat &matrix_right)
{
    if (!isPerfectSquare((int)matrix_right.size()))
    {
        QCERR("QStat is illegal");
        throw invalid_argument("QStat is illegal");
    }

    int size =(int)matrix_right.size();
    QStat matrix_result(size, 0);

    for (int i = 0; i < size; i++)
    {
        matrix_result[i] = value * matrix_right[i];
    }

    return matrix_result;
}

std::ostream& QPanda::operator<<(std::ostream &out, QStat &mat) 
{
	out << matrix_to_string(mat) << endl;
	return out;
}

QStat QPanda::tensor(const QStat& leftMatrix, const QStat& rightMatrix)
{
	QStat result_matrix;

	//get rows  and columns
	double left_rows = sqrt(leftMatrix.size());
	double right_columns = sqrt(rightMatrix.size());

	result_matrix.resize(leftMatrix.size()*rightMatrix.size());
	int left_row = 0, left_column = 0, right_row = 0, right_column = 0, target_row = 0, target_column = 0;
	for (size_t left_index = 0; left_index < leftMatrix.size(); ++left_index)
	{
		for (size_t right_index = 0; right_index < rightMatrix.size(); ++right_index)
		{
			left_row = left_index / left_rows;
			left_column = left_index % ((int)left_rows);

			right_row = right_index / right_columns;
			right_column = right_index % ((int)right_columns);

			target_row = right_row + (left_row * right_columns);
			target_column = right_column + (left_column * right_columns);
			result_matrix[(target_row)*(left_rows*right_columns) + target_column] = (leftMatrix[left_index] * rightMatrix[right_index]);
		}
	}

	PTrace("ZhangMultip result: ");
	PTraceMat(result_matrix);
	return result_matrix;
}

int QPanda::partition(const QStat& srcMatrix, int partitionRowNum, int partitionColumnNum, blockedMatrix_t& blockedMat)
{
	blockedMat.m_vec_block.clear();

	PTrace("partition:\nsrcMatrix: ");
	PTraceMat(srcMatrix);

	size_t mat_size = srcMatrix.size();
	int src_mat_rows = sqrt(mat_size); // same to the Columns of the srcMatrix
	if ((0 != src_mat_rows % partitionRowNum) || (0 != src_mat_rows % partitionColumnNum))
	{
		QCERR_AND_THROW_ERRSTR(invalid_argument, "Error: Failed to partition.");
		return -1;
	}

	blockedMat.m_block_rows = partitionRowNum;
	blockedMat.m_block_columns = partitionColumnNum;

	int row_cnt_in_block = src_mat_rows / partitionRowNum;
	int col_cnt_in_block = src_mat_rows / partitionColumnNum;

	blockedMat.m_vec_block.resize(partitionRowNum*partitionColumnNum);
	for (size_t block_row = 0; block_row < partitionRowNum; ++block_row)
	{
		for (size_t block_col = 0; block_col < partitionColumnNum; ++block_col)
		{
			matrixBlock_t& block = blockedMat.m_vec_block[block_row*partitionColumnNum + block_col];
			block.m_row_index = block_row;
			block.m_column_index = block_col;

			for (size_t row_in_block = 0; row_in_block < row_cnt_in_block; row_in_block++)
			{
				for (size_t col_in_block = 0; col_in_block < col_cnt_in_block; col_in_block++)
				{
					int row_in_src_mat = block_row * row_cnt_in_block + row_in_block;
					int col_in_src_mat = block_col * col_cnt_in_block + col_in_block;
					block.m_mat.push_back(srcMatrix[row_in_src_mat*src_mat_rows + col_in_src_mat]);
				}
			}
		}
	}

	return 0;
}

int QPanda::blockMultip(const QStat& leftMatrix, const blockedMatrix_t& blockedMat, QStat& resultMatrix)
{
	if ((0 == leftMatrix.size()) || (blockedMat.m_vec_block.size() == 0))
	{
		QCERR_AND_THROW_ERRSTR(invalid_argument, "Error: parameter error.");
		return -1;
	}

	std::vector<matrixBlock_t> tmp_Block_Vec;
	tmp_Block_Vec.resize(blockedMat.m_vec_block.size());
	for (auto &itr : blockedMat.m_vec_block)
	{
		matrixBlock_t &tmp_block = tmp_Block_Vec[itr.m_row_index*(blockedMat.m_block_columns) + itr.m_column_index];
		tmp_block.m_row_index = itr.m_row_index;
		tmp_block.m_column_index = itr.m_column_index;
		tmp_block.m_mat = (QPanda::tensor(leftMatrix, itr.m_mat));
	}

	int row_cnt_in_block = sqrt(tmp_Block_Vec[0].m_mat.size());
	int col_cnt_in_block = row_cnt_in_block; //square matrix
	size_t block_index = 0;
	size_t item_in_block_index = 0;
	for (size_t block_row = 0; block_row < blockedMat.m_block_rows; block_row++)
	{
		for (size_t row_in_block = 0; row_in_block < row_cnt_in_block; row_in_block++)
		{
			for (size_t block_col = 0; block_col < blockedMat.m_block_columns; block_col++)
			{
				for (size_t col_in_block = 0; col_in_block < col_cnt_in_block; col_in_block++)
				{
					block_index = block_row * blockedMat.m_block_columns + block_col;
					item_in_block_index = row_in_block * col_cnt_in_block + col_in_block;
					resultMatrix.push_back(tmp_Block_Vec[block_index].m_mat[item_in_block_index]);
				}
			}
		}
	}

	PTrace("blockMultip result: ");
	PTraceMat(resultMatrix);

	return 0;
}

void QPanda::dagger(QStat &srcMat)
{
	//get  the rows and columns of the srcMat
	size_t mat_size = srcMat.size();
	int src_mat_rows = sqrt(mat_size); // same to the Columns of the srcMatrix
	int src_mat_columns = src_mat_rows;
	qcomplex_t tmp_val;
	for (size_t i = 0; i < src_mat_rows; ++i)
	{
		for (size_t j = i; j < src_mat_columns; ++j)
		{
			if (i == j)
			{
				srcMat[i*src_mat_columns + j].imag(-1 * (srcMat[i*src_mat_columns + j].imag()));
				continue;
			}

			tmp_val = srcMat[i*src_mat_columns + j];
			srcMat[i*src_mat_columns + j].real(srcMat[j*src_mat_columns + i].real());
			srcMat[i*src_mat_columns + j].imag(-1 * (srcMat[j*src_mat_columns + i].imag()));

			srcMat[j*src_mat_columns + i].real(tmp_val.real());
			srcMat[j*src_mat_columns + i].imag(-1 * (tmp_val.imag()));
		}
	}
}

#define COMPLEX_REAL_VAL_FORMAT ("%.03f") 
#define COMPLEX_IMAG_VAL_FORMAT ("%.03fi") 
string QPanda::matrix_to_string(const QStat& mat)
{
	int rows = 0;
	int columns = 0;
	rows = columns = sqrt(mat.size());
	string matrix_str = "\n";
	int index = 0;
	float imag_val = 0.0;
	float real_val = 0.0;
	const int max_width = 13;
	char output_buf[64] = "";
	std::string output_str;
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < columns; j++)
		{
			memset(output_buf, ' ', sizeof(output_buf));
			index = i * columns + j;
			imag_val = mat[index].imag();
			real_val = mat[index].real();
			if ((abs(real_val) < 0.000000001) || (abs(imag_val) < 0.000000001))
			{
				if ((abs(real_val) < 0.000000001) && (abs(imag_val) < 0.000000001))
				{
					snprintf(output_buf, sizeof(output_buf), " 0");
				}
				else if (abs(imag_val) < 0.000000001)
				{
					if (real_val < 0)
					{
						snprintf(output_buf, sizeof(output_buf), COMPLEX_REAL_VAL_FORMAT, (real_val));
					}
					else
					{
						snprintf(output_buf, sizeof(output_buf), (std::string(" ") + COMPLEX_REAL_VAL_FORMAT).c_str(), abs(real_val));
					}
				}
				else
				{
					//only imag_val
					if (imag_val < 0)
					{
						snprintf(output_buf, sizeof(output_buf), COMPLEX_IMAG_VAL_FORMAT, (imag_val));
					}
					else
					{
						snprintf(output_buf, sizeof(output_buf), (std::string(" ") + COMPLEX_IMAG_VAL_FORMAT).c_str(), abs(imag_val));
					}
				}
			}
			else if (imag_val < 0)
			{
				if (real_val < 0)
				{
					snprintf(output_buf, sizeof(output_buf), (std::string(COMPLEX_REAL_VAL_FORMAT) + COMPLEX_IMAG_VAL_FORMAT).c_str(), real_val, imag_val);
				}
				else
				{
					snprintf(output_buf, sizeof(output_buf), (std::string(" ") + COMPLEX_REAL_VAL_FORMAT + COMPLEX_IMAG_VAL_FORMAT).c_str(), abs(real_val), imag_val);
				}

			}
			else
			{
				if (real_val < 0)
				{
					snprintf(output_buf, sizeof(output_buf), (std::string(COMPLEX_REAL_VAL_FORMAT) + "+" + COMPLEX_IMAG_VAL_FORMAT).c_str(), real_val, imag_val);
				}
				else
				{
					snprintf(output_buf, sizeof(output_buf), (std::string(" ") + COMPLEX_REAL_VAL_FORMAT + "+" + COMPLEX_IMAG_VAL_FORMAT).c_str(), abs(real_val), imag_val);
				}
			}

			output_str = output_buf;
			size_t valLen = output_str.size();
			output_buf[valLen] = ' ';
			output_str = output_buf;
			output_str = output_str.substr(0, (max_width < valLen ? valLen : max_width) + 2);
			matrix_str.append(output_str);
		}
		matrix_str.append("\n");
	}

	return matrix_str;
}

int QPanda::mat_compare(const QStat& mat1, const QStat& mat2, const double precision /*= 0.000001*/)
{
	if (mat1.size() != mat2.size())
	{
		return -1;
	}

	qcomplex_t ratio; // constant value
	for (size_t i = 0; i < mat1.size(); ++i)
	{
		if ((abs(mat2.at(i).real() - 0.0) > precision) || (abs(mat2.at(i).imag() - 0.0) > precision))
		{
			ratio = mat1.at(i) / mat2.at(i);
			if (precision < (sqrt(ratio.real()*ratio.real() + ratio.imag()*ratio.imag()) - 1.0))
			{
				return -1;
			}
			break;
		}
	}

	qcomplex_t tmp_val;
	for (size_t i = 0; i < mat1.size(); ++i)
	{
		tmp_val = ratio * mat2.at(i);
		if ((abs(mat1.at(i).real() - tmp_val.real()) > precision) ||
			(abs(mat1.at(i).imag() - tmp_val.imag()) > precision))
		{
			return -1;
		}
	}

	return 0;
}