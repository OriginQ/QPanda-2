#include <cmath>
#include <string.h>
#include "ThirdParty/EigenUnsupported/Eigen/KroneckerProduct"
#include "Core/Utilities/Tools/QStatMatrix.h"

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

std::ostream& QPanda::operator<<(std::ostream &out, QStat mat) 
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

	PTrace("tensor result: ");
	PTraceMat(result_matrix);
	return result_matrix;
}


int QPanda::partition(qmatrix_t& srcMatrix, int partitionRowNum, int partitionColumnNum, blockedMatrix_t& blockedMat)
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
			block.m_mat.resize(partitionRowNum, partitionRowNum);
			for (size_t row_in_block = 0; row_in_block < row_cnt_in_block; row_in_block++)
			{
				for (size_t col_in_block = 0; col_in_block < col_cnt_in_block; col_in_block++)
				{
					int row_in_src_mat = block_row * row_cnt_in_block + row_in_block;
					int col_in_src_mat = block_col * col_cnt_in_block + col_in_block;
					block.m_mat(row_in_block, col_in_block)  =  srcMatrix(row_in_src_mat, col_in_src_mat);
				}
			}
		}
	}

	return 0;
}

int QPanda::blockMultip(qmatrix_t& leftMatrix, const blockedMatrix_t& blockedMat, qmatrix_t& resultMatrix)
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
		tmp_block.m_mat = kroneckerProduct(leftMatrix, itr.m_mat);
	}

	int row_cnt_in_block = sqrt(tmp_Block_Vec[0].m_mat.size());
	int col_cnt_in_block = row_cnt_in_block; //square matrix
	size_t block_index = 0;
	size_t item_in_block_index = 0;
	resultMatrix.resize(blockedMat.m_block_rows*row_cnt_in_block, blockedMat.m_block_columns*col_cnt_in_block);
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
					resultMatrix(block_row*row_cnt_in_block + row_in_block, block_col*col_cnt_in_block + col_in_block) =  tmp_Block_Vec[block_index].m_mat(row_in_block, col_in_block);
				}
			}
		}
	}

	PTrace("blockMultip result: ");
	PTraceMat(resultMatrix);

	return 0;
}

void QPanda::dagger(QStat &src_mat)
{
	//get  the rows and columns of the src_mat
	size_t mat_size = src_mat.size();
	int src_mat_rows = sqrt(mat_size); // same to the Columns of the src-Matrix
	int src_mat_columns = src_mat_rows;
	qcomplex_t tmp_val;
	for (size_t i = 0; i < src_mat_rows; ++i)
	{
		for (size_t j = i; j < src_mat_columns; ++j)
		{
			if (i == j)
			{
				src_mat[i*src_mat_columns + j].imag(-1 * (src_mat[i*src_mat_columns + j].imag()));
				continue;
			}

			tmp_val = src_mat[i*src_mat_columns + j];
			src_mat[i*src_mat_columns + j].real(src_mat[j*src_mat_columns + i].real());
			src_mat[i*src_mat_columns + j].imag(-1 * (src_mat[j*src_mat_columns + i].imag()));

			src_mat[j*src_mat_columns + i].real(tmp_val.real());
			src_mat[j*src_mat_columns + i].imag(-1 * (tmp_val.imag()));
		}
	}
}

QStat QPanda::dagger_c(const QStat &src_mat)
{
	auto tmp_mat = src_mat;
	dagger(tmp_mat);

	return tmp_mat;
}

static std::string double_to_string(const double d, const int precision = 8)
{
	std::ostringstream stream;
	stream.precision(precision);
	stream << d;
	return stream.str();
}

string QPanda::matrix_to_string(const QStat& mat, const int precision /*= 8*/)
{
	int rows = 0;
	int columns = 0;
	rows = columns = sqrt(mat.size());
	string matrix_str = "\n";
	int index = 0;
	float imag_val = 0.0;
	float real_val = 0.0;
	char output_buf[64] = "";

	//get max_width for every columns
	std::vector<size_t> columns_width_vec;
	for (size_t j = 0; j < columns; j++)
	{
		size_t tmp_max_width = 0;
		for (size_t i = 0; i < rows; i++)
		{
			index = i * columns + j;
			snprintf(output_buf, sizeof(output_buf), "(%-s, %-s)", double_to_string(mat[index].real(), precision).c_str(), double_to_string(mat[index].imag(), precision).c_str());
			const auto tmp_len = strlen(output_buf);
			if (tmp_len > tmp_max_width)
			{
				tmp_max_width = tmp_len;
			}
		}
		columns_width_vec.push_back(tmp_max_width);
	}

	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < columns; j++)
		{
			std::string output_str;
			memset(output_buf, 0, sizeof(output_buf));
			index = i * columns + j;
			snprintf(output_buf, sizeof(output_buf), "(%s, %s)", double_to_string(mat[index].real(), precision).c_str(), double_to_string(mat[index].imag(), precision).c_str());
			size_t valLen = strlen(output_buf);
			for (size_t m = 0; m < (columns_width_vec[j] - valLen + 2); ++m)
			{
				output_str += " ";
			}

			output_str += output_buf;
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

bool QPanda::operator==(const QStat &matrix_left, const QStat &matrix_right)
{
	return (0 == mat_compare(matrix_left, matrix_right, MAX_COMPARE_PRECISION));
}

bool QPanda::operator!=(const QStat &matrix_left, const QStat &matrix_right)
{
	return (0 != mat_compare(matrix_left, matrix_right, MAX_COMPARE_PRECISION));
}

EigenMatrixXc QPanda::QStat_to_Eigen(const QStat& src_mat)
{
	auto n = std::sqrt(src_mat.size());

	EigenMatrixXc eigen_matrix = EigenMatrixXc::Zero(n, n);
	for (auto rdx = 0; rdx < n; ++rdx)
	{
		for (auto cdx = 0; cdx < n; ++cdx)
		{
			eigen_matrix(rdx, cdx) = src_mat[rdx*n + cdx];
		}
	}

	return eigen_matrix;
}

QStat QPanda::Eigen_to_QStat(const EigenMatrixXc& eigen_mat)
{
	QStat q_mat;
	size_t rows = eigen_mat.rows();
	size_t cols = eigen_mat.cols();
	for (size_t i = 0; i < rows; ++i)
	{
		for (size_t j = 0; j < cols; ++j)
		{
			q_mat.push_back((qcomplex_t)(eigen_mat(i, j)));
		}
	}
	
	return q_mat;
}

bool QPanda::is_unitary_matrix(const QStat &circuit_matrix, const double precision /*= 0.000001*/)
{
	//double difference = 0.0;
	size_t matrix_dimension = sqrt(circuit_matrix.size());
	QStat tmp_matrix_dagger = dagger_c(circuit_matrix);
	const auto tmp_mat = tmp_matrix_dagger * circuit_matrix;
	//cout << "tmp_mat = " << tmp_mat << endl;
	QStat mat_I(circuit_matrix.size(), 0);
	for (size_t i = 0; i < matrix_dimension; ++i)
	{
		mat_I[i*matrix_dimension + i] = 1;
	}

	if (tmp_mat == mat_I)
	{
		return true;
	}

	return false;

	/*double trace = 0.0;
	for (size_t i = 0; i < matrix_dimension; ++i)
	{
		for (size_t j = 0; j < matrix_dimension; ++j)
		{
			trace += (tmp_matrix_dagger[i*matrix_dimension + j] * circuit_matrix[j*matrix_dimension + i]).real();
		}
	}

	difference = abs(1 - pow(trace / ((double)matrix_dimension), 2));

	return (difference < precision);*/
}