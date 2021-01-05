#include <math.h>
#include <EigenUnsupported/Eigen/KroneckerProduct>
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/Tools/MatrixDecomposition.h"
#include <chrono>
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include "QAlg/Base_QCircuit/AmplitudeEncode.h"

USING_QPANDA
using namespace std;
using namespace chrono;

#define PRINT_TRACE 0
#if PRINT_TRACE
#define PTrace printf
#define PTraceMat(mat) (std::cout << (mat) << endl)
#define PTraceCircuit(cir) (std::cout << cir << endl)
#else
#define PTrace
#define PTraceMat(mat)
#define PTraceCircuit(cir)
#endif

#define MAX_MATRIX_PRECISION 1e-10

using MatrixSequence = std::vector<MatrixUnit>;
using DecomposeEntry = std::pair<int, MatrixSequence>;

using ColumnOperator = std::vector<DecomposeEntry>;
using MatrixOperator = std::vector<ColumnOperator>;

using SingleGateUnit = std::pair<MatrixSequence, QStat>;

static void upper_partition(int order, MatrixOperator &entries)
{
	auto index = (int)std::log2(entries.size() + 1) - (int)std::log2(order) - 1;

	for (auto cdx = 0; cdx < order - 1; ++cdx)
	{
		for (auto rdx = 0; rdx < order - cdx - 1; ++rdx)
		{
			auto entry = entries[cdx][rdx];

			entry.first += order;
			entry.second[index] = MatrixUnit::SINGLE_P1;

			entries[cdx + order].emplace_back(entry);
		}
	}

    return;
}


static bool entry_requirement(const MatrixSequence& units, int udx, int cdx)
{
	int lj = ((cdx - 1) >> (udx - 1)) & 1;

	int M = 1;
	while (cdx)
	{
		cdx >>= 1;
		M += cdx ? 1 : 0;
	}

	//if 1 ≤ j ≤ m and cj = lj' = 1 , return true
	auto mat = units[units.size() - udx];
	return udx >= 1
		&& udx <= M
		&& lj
		&& mat == MatrixUnit::SINGLE_P1;
}

static bool steps_requirement(const MatrixSequence& units, int udx, int cdx)
{
	int M = 1;
	while (cdx)
	{
		cdx >>= 1;
		M += cdx ? 1 : 0;
	}

	//if j = n and none of cn...cm+1 is 1 , return true
	if (units.size() != udx)
	{
		return false;
	}
	else
	{
		auto iter = std::find(units.begin(), units.end() - M, MatrixUnit::SINGLE_P1);
		return (units.end() - M) == iter;
	}
}

static void under_partition(int order, MatrixOperator& entries)
{
	auto qubits = (int)std::log2(entries.size() + 1);

	for (auto cdx = 1; cdx < order; ++cdx)
	{
		if (cdx & 1)
		{
			for (auto rdx = 0; rdx < order; ++rdx)
			{
				auto value = entries[0][rdx + order - 1].first ^ cdx;
				auto entry = make_pair(value, entries[cdx - 1][rdx + order - cdx].second);

				entries[cdx].emplace_back(entry);
			}

			auto &units = entries[cdx].back().second;
			for (auto idx = 0; idx < (int)std::log2(order); ++idx)
			{
				units[qubits - idx - 1] = ((cdx >> idx) & 1) ?
					MatrixUnit::SINGLE_P1 : MatrixUnit::SINGLE_I2;
			}
		}
		else
		{
			for (auto rdx = 0; rdx < order; ++rdx)
			{
				auto range = (int)std::log2(order) + 1;
				auto refer = entries[0][rdx + order - 1].second;
				auto entry = entries[0][rdx + order - 1].first ^ cdx;

				MatrixSequence units(refer.begin() + qubits - range, refer.end());

				for (auto udx = 1; udx <= range; ++udx)  /*udx = j , cdx = L*/
				{
					bool steps_accord = steps_requirement(units, udx, cdx + 1);
					bool entry_accord = entry_requirement(units, udx, cdx + 1);

					units[range - udx] = steps_accord ? MatrixUnit::SINGLE_P1 :
						entry_accord ? MatrixUnit::SINGLE_P0 : units[range - udx];
				}

				for (auto idx = 0; idx < qubits - range; ++idx)
				{
					units.insert(units.begin(), MatrixUnit::SINGLE_I2);
				}

				entries[cdx].emplace_back(make_pair(entry, units));
			}

			auto refer_opt = entries[0][2 * order - 2].second;
			for (auto idx = 0; idx < qubits; ++idx)
			{
				if ((cdx >> idx) & 1)
				{
					refer_opt[qubits - idx - 1] = MatrixUnit::SINGLE_P1;
				}
			}

			entries[cdx].back().second = refer_opt;
		}
	}

    return;
}

static void controller(MatrixSequence &sequence, const EigenMatrix2c U2, EigenMatrixXc &matrix)
{
	EigenMatrix2c P0;
	EigenMatrix2c P1;
	EigenMatrix2c I2;

	P0 << Eigen::dcomplex(1, 0), Eigen::dcomplex(0, 0),
		  Eigen::dcomplex(0, 0), Eigen::dcomplex(0, 0);
	P1 << Eigen::dcomplex(0, 0), Eigen::dcomplex(0, 0),
		  Eigen::dcomplex(0, 0), Eigen::dcomplex(1, 0);
	I2 << Eigen::dcomplex(1, 0), Eigen::dcomplex(0, 0),
		  Eigen::dcomplex(0, 0), Eigen::dcomplex(1, 0);

	std::map<MatrixUnit, std::function<EigenMatrix2c()>> mapping =
	{
		{ MatrixUnit::SINGLE_P0, [&]() {return P0; } },
		{ MatrixUnit::SINGLE_P1, [&]() {return P1; } },
		{ MatrixUnit::SINGLE_I2, [&]() {return I2; } },
		{ MatrixUnit::SINGLE_V2, [&]() {return U2 - I2; } }
	};

	auto order = sequence.size();
	EigenMatrixXc Un = EigenMatrixXc::Identity(1, 1);
	EigenMatrixXc In = EigenMatrixXc::Identity(1ull << order, 1ull << order);

	for (const auto &val : sequence)
	{
		EigenMatrix2c M2 = mapping.find(val)->second();
		Un = Eigen::kroneckerProduct(Un, M2).eval();
	}

	matrix = In + Un;
    return;
}

static void recursive_partition(const EigenMatrixXc& sub_matrix, MatrixOperator &entries)
{
    Eigen::Index order = sub_matrix.rows();
    if (1 == order)
    {
        return;
    }
    else
    {
        EigenMatrixXc corner = sub_matrix.topLeftCorner(order / 2, order / 2);

        recursive_partition(corner, entries);

        upper_partition(order / 2, entries);
        under_partition(order / 2, entries);
    }

    return;
}

static void decomposition(EigenMatrixXc& matrix, MatrixOperator& entries, std::vector<SingleGateUnit>& cir_units)
{
	for (auto cdx = 0; cdx < entries.size(); ++cdx)
	{
		auto opts = entries[cdx].size();
		for (auto idx = 0; idx < opts; ++idx)
		{
			auto rdx = entries[cdx][idx].first;
			auto opt = entries[cdx][idx].second;

			if ((((abs(matrix(rdx, cdx).real()) < MAX_MATRIX_PRECISION) && (abs(matrix(rdx, cdx).imag()) < MAX_MATRIX_PRECISION)) && (idx != opts - 1)) ||
				(((abs(matrix(cdx + 1, cdx).real() - 1.0) < MAX_MATRIX_PRECISION) && (abs(matrix(cdx + 1, cdx).imag()) < MAX_MATRIX_PRECISION)) && (idx == opts - 1)))
			/*if ((EigenComplexT(0, 0) == matrix(rdx, cdx) && (idx != opts - 1)) ||
				(EigenComplexT(1, 0) == matrix(cdx + 1, cdx) && (idx == opts - 1)))*/
			{
				continue;
			}
			else
			{
				EigenMatrix2c C2; /*placeholder*/
				C2 << EigenComplexT(0, 1), EigenComplexT(0, 1),
					EigenComplexT(0, 1), EigenComplexT(0, 1);

				EigenMatrixXc Cn;
				controller(opt, C2, Cn);

				Qnum indices(2);
				for (Eigen::Index index = 0; index < (1ull << opt.size()); ++index)
				{
					if (Cn(rdx, index) != EigenComplexT(0, 0))
					{
						indices[index == rdx] = index;
					}  
				}

				EigenComplexT C0 = matrix(indices[0], cdx);  /*The entry to be eliminated */
				EigenComplexT C1 = matrix(indices[1], cdx);  /*The corresponding entry */

				EigenComplexT V11, V12, V21, V22;

				if (indices[0] < indices[1])
				{
					V11 = std::conj(C0) / std::sqrt(std::norm(C0) + std::norm(C1));
					V12 = std::conj(C1) / std::sqrt(std::norm(C0) + std::norm(C1));
					V21 = C1 / std::sqrt(std::norm(C0) + std::norm(C1));
					V22 = -C0 / std::sqrt(std::norm(C0) + std::norm(C1));
				}
				else
				{
					V11 = -C0 / std::sqrt(std::norm(C0) + std::norm(C1));
					V12 = C1 / std::sqrt(std::norm(C0) + std::norm(C1));
					V21 = std::conj(C1) / std::sqrt(std::norm(C0) + std::norm(C1));
					V22 = std::conj(C0) / std::sqrt(std::norm(C0) + std::norm(C1));
				}

				EigenMatrix2c V2;
				V2 << V11, V12, V21, V22;

				EigenMatrixXc Un;
				controller(opt, V2, Un);

				matrix = Un * matrix;

				QStat M2 = { (qcomplex_t)V11 ,(qcomplex_t)V12 ,(qcomplex_t)V21 ,(qcomplex_t)V22 };
				cir_units.insert(cir_units.begin(), std::make_pair(opt, M2));
			}
		}
	}

	EigenMatrix2c V2 = matrix.bottomRightCorner(2, 2);
	if(!V2.isApprox(EigenMatrixXc::Identity(2, 2), MAX_MATRIX_PRECISION))
	//if (EigenMatrixXc::Identity(2, 2) != V2)
	{
		QStat M2 = { (qcomplex_t)((EigenComplexT)1.0 / V2(0,0)), (qcomplex_t)(V2(0,1)),
					 (qcomplex_t)(V2(1,0)) , (qcomplex_t)((EigenComplexT)1.0 / V2(1,1))};

		auto entry = entries.back().back().second;
		cir_units.insert(cir_units.begin(), std::make_pair(entry, M2));
	}
}

static void initialize(EigenMatrixXc& matrix, MatrixOperator& entries)
{
    auto qubits = (int)std::log2(matrix.rows());

    MatrixSequence Cns(qubits, MatrixUnit::SINGLE_I2);
    Cns.back() = MatrixUnit::SINGLE_V2;
    entries.front().emplace_back(make_pair(1, Cns));

    ColumnOperator& column = entries.front();
    for (auto idx = 1; idx < qubits; ++idx)
    {
        size_t path = 1ull << idx;
        for (auto opt = 0; opt < (1 << idx) - 1; ++opt)
        {
            auto entry = column[opt].first;
            auto units = column[opt].second;

            // 1 : none of cn−1, . . . , c1 equals 1
            // * : otherwise
            auto iter = std::find(units.end() - idx, units.end(), MatrixUnit::SINGLE_P1);

            units[units.size() - 1 - idx] = (units.end() == iter) ?
                MatrixUnit::SINGLE_P1 : MatrixUnit::SINGLE_I2;

            column.emplace_back(make_pair(entry + path, units));
        }

        MatrixSequence Lns(qubits, MatrixUnit::SINGLE_I2);
        Lns[qubits - idx - 1] = MatrixUnit::SINGLE_V2;

        column.emplace_back(make_pair((1ull << idx), Lns));
    }

    return;
}

static void general_scheme(EigenMatrixXc& matrix, std::vector<SingleGateUnit>& cir_units)
{
	MatrixOperator entries;
	for (auto idx = 1; idx < matrix.cols(); ++idx)
	{
		ColumnOperator Co;
		entries.emplace_back(Co);
	}

	initialize(matrix, entries);
 	recursive_partition(matrix, entries);
	decomposition(matrix, entries, cir_units);

    return;
}

static void circuit_insert(QVec& qubits, std::vector<SingleGateUnit>& cir_units, QCircuit &circuit)
{
	std::sort(qubits.begin(), qubits.end(), [&](Qubit *a, Qubit *b)
	{
		return a->getPhysicalQubitPtr()->getQubitAddr()
			 < b->getPhysicalQubitPtr()->getQubitAddr();
	});

	auto rank = qubits.size();
	for (auto &val : cir_units)
	{
		QVec control;
		QCircuit cir;

		for (auto qdx = 0; qdx < rank; qdx++)
		{
			if (MatrixUnit::SINGLE_P0 == val.first[qdx])
			{
				cir << X(qubits[qdx]);
				control.emplace_back(qubits[qdx]);
			}
			else if (MatrixUnit::SINGLE_P1 == val.first[qdx])
			{
				control.emplace_back(qubits[qdx]);
			}
			else
			{}
		}

		for (auto qdx = 0; qdx < rank; qdx++)
		{
			if (MatrixUnit::SINGLE_V2 == val.first[qdx])
			{
				circuit << cir
					    << U4(val.second, qubits[qdx]).control(control).dagger()
						<< cir;

				break;
			}
		}
	}
}

/*******************************************************************
*                      class DiagonalMatrixDecompose
********************************************************************/
class DiagonalMatrixDecompose
{
public:
	DiagonalMatrixDecompose() {}
	~DiagonalMatrixDecompose() {}


	QCircuit decompose(const QVec& qubits, const QStat& src_mat)
	{
		//check param
		if (!is_unitary_matrix(src_mat))
		{
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: failed on HQRDecompose, the input matrix is not a unitary-matrix.");
		}

		const auto mat_dimension = sqrt(src_mat.size());
		const auto need_qubits_num = ceil(log2(mat_dimension));
		if (need_qubits_num > qubits.size())
		{
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: failed on HQRDecompose, no enough qubits.");
		}

		QCircuit decompose_result_cir;
		m_qubits = qubits;
		QVec controlqvec = qubits;
		controlqvec.pop_back();
		QStat tmp_mat22; //2*2 unitary matrix
		const size_t tmp_base_unitary_cnt = mat_dimension / 2;
		long pre_index = -1;
		for (size_t i = 0; i < tmp_base_unitary_cnt; ++i)
		{
			tmp_mat22.clear();
			const size_t tmp_row = (2 * i * mat_dimension) + (2 * i);
			tmp_mat22.push_back(src_mat[tmp_row]);
			tmp_mat22.push_back(src_mat[tmp_row + 1]);
			tmp_mat22.push_back(src_mat[tmp_row + mat_dimension]);
			tmp_mat22.push_back(src_mat[tmp_row + mat_dimension + 1]);
			QGate tmp_u4 = U4(tmp_mat22, qubits.back()).control(controlqvec);
			QGATE_SPACE::U4* p_gate = dynamic_cast<QGATE_SPACE::U4*>(tmp_u4.getQGate());
			if ((abs(p_gate->getAlpha()) < MAX_MATRIX_PRECISION)
				&& (abs(p_gate->getBeta()) < MAX_MATRIX_PRECISION)
				&& (abs(p_gate->getGamma()) < MAX_MATRIX_PRECISION)
				&& (abs(p_gate->getDelta()) < MAX_MATRIX_PRECISION))
			{
				continue;
			}

			if (0 == i)
			{
				QCircuit index_cir_zero = index_to_circuit(0, controlqvec);
				decompose_result_cir << index_cir_zero;
			}
			else
			{
				QCircuit index_cir = index_to_merge_circuit(i, pre_index, controlqvec);
				decompose_result_cir << index_cir;
			}

			decompose_result_cir << tmp_u4;
			pre_index = i;
		}

		return decompose_result_cir;
	}

protected:
	QCircuit index_to_circuit(size_t index, QVec& controlqvec)
	{
		QCircuit ret_cir;
		size_t data_qubits_cnt = controlqvec.size();
		for (size_t i = 0; i < data_qubits_cnt; ++i)
		{
			if (0 == index % 2)
			{
				ret_cir << X(controlqvec[data_qubits_cnt - i - 1]);
			}

			index /= 2;
		}

		return ret_cir;
	}

	QCircuit index_to_merge_circuit(size_t index, QVec& controlqvec)
	{
		if (0 == index)
		{
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: failed to build merge-index-circuit, the index must be >0.");
		}

		size_t pre_index = index - 1;
		QCircuit ret_cir;
		size_t data_qubits_cnt = controlqvec.size();
		for (size_t i = 0; i < data_qubits_cnt; ++i)
		{
			if ((index % 2) != (pre_index % 2))
			{
				ret_cir << X(controlqvec[data_qubits_cnt - i - 1]);
			}

			index /= 2;
			pre_index /= 2;
		}

		return ret_cir;
	}

	QCircuit index_to_merge_circuit(size_t index, long pre_index, QVec& controlqvec)
	{
		if (0 == index)
		{
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: failed to build merge-index-circuit, the index must be >0.");
		}

		size_t tmp_pre_index = pre_index;
		if (pre_index < 0)
		{
			tmp_pre_index = 1;
		}
		
		QCircuit ret_cir;
		size_t data_qubits_cnt = controlqvec.size();
		for (size_t i = 0; i < data_qubits_cnt; ++i)
		{
			if ((index % 2) != (tmp_pre_index % 2))
			{
				ret_cir << X(controlqvec[data_qubits_cnt - i - 1]);
			}

			index /= 2;

			if (pre_index > 0)
			{
				tmp_pre_index /= 2;
			}
		}

		return ret_cir;
	}

private:
	QVec m_qubits;
};

/*******************************************************************
*                      class HQRDecompose
* Householder QR-decompose
* refer to <Quantum circuits synthesis using Householder transformations>(https://arxiv.org/abs/2004.07710v1)
********************************************************************/
class HQRDecompose
{
public:
	HQRDecompose() 
		:m_dimension(0)
	{}
	~HQRDecompose() {}

	QCircuit decompose(QVec qubits, const EigenMatrixXc& src_mat)
	{
		//check param
		if (!src_mat.isUnitary(MAX_MATRIX_PRECISION))
		{
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: failed on HQRDecompose, the input matrix is not a unitary-matrix.");
		}

		
		const auto mat_dimension = src_mat.rows();
		const auto need_qubits_num = ceil(log2(mat_dimension));
		if (need_qubits_num > qubits.size())
		{
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: failed on HQRDecompose, no enough qubits.");
		}

		// do Householder QR_decompose
		m_dimension = mat_dimension;
		m_qubits = qubits;
		auto start = system_clock::now();
		Householder_QR_decompose(src_mat);

#if PRINT_TRACE
		auto end = system_clock::now();
		auto duration = duration_cast<microseconds>(end - start);
		PTrace("Total used: %f s",
			double(duration.count()) * microseconds::period::num / microseconds::period::den);
#endif

		return m_result_cir;
	}

protected:
	void print_vec(const QStat& vec)
	{
		for (auto i = 0; i < vec.size(); ++i)
		{
			printf("(%-g, %-g), ", vec[i].real(), vec[i].imag());
		}

		printf("\n");
	}

	void Householder_QR_decompose(EigenMatrixXc src_mat)
	{
		const auto& lines = m_dimension;
		QStat tmp_mat_R(m_dimension * m_dimension);
		QCircuit cir_Q;
		for (size_t cur_col = 0; cur_col < lines; ++cur_col)
		{
			//printf("On column %ld\n", cur_col);
			const auto cur_col_size = lines - cur_col;
			QStat cur_col_vec(cur_col_size);
			QStat cur_col_vec_dagger(cur_col_size);
			qstate_type norm = 0;
			for (size_t cur_row = cur_col; cur_row < lines; ++cur_row)
			{
				cur_col_vec[cur_row - cur_col] = -src_mat(cur_row, cur_col);
				norm += (cur_col_vec[cur_row - cur_col].real() * cur_col_vec[cur_row - cur_col].real() +
					cur_col_vec[cur_row - cur_col].imag() * cur_col_vec[cur_row - cur_col].imag());
			}

			norm = sqrt(norm);
			const double angle = arg(cur_col_vec[0]);
			cur_col_vec[0] -= exp(qcomplex_t(0, angle)); // ?
			//vec[0] = exp(complex_t(0, 1.0 * angle)) * (sqrt(vec[0].real() * vec[0].real() + vec[0].imag() * vec[0].imag()) - tmp_x);

			norm = 0.0;
			for (size_t i = 0; i < cur_col_size; ++i)
			{
				norm += (cur_col_vec[i].real() * cur_col_vec[i].real() + cur_col_vec[i].imag() * cur_col_vec[i].imag());
			}

			if (norm > 1e-7)
			{
				// vec Dagger  
				for (size_t i = 0; i < cur_col_size; ++i)
				{
					qstate_type real_v = cur_col_vec[i].real();
					qstate_type imag_v = cur_col_vec[i].imag();
					//cur_col_vec[i] = qcomplex_t(real_v, imag_v);
					cur_col_vec_dagger[i] = qcomplex_t(real_v, -imag_v);
				}
#if PRINT_TRACE
				printf("The vec:\n");
				print_vec(cur_col_vec);
				printf("The vec_dagger:\n");
				print_vec(cur_col_vec_dagger);
				printf(":::::::::::::::::::::::::::::\n");
#endif
				//sestavit matici P
				EigenMatrixXc matrix_p(cur_col_size, cur_col_size);
				for (size_t k = 0; k < cur_col_size; ++k)
				{
					for (size_t h = 0; h < cur_col_size; ++h)
					{
						if (k == h)
						{
							matrix_p(k, k) = qcomplex_t(1, 0) - qcomplex_t(2, 0) * cur_col_vec[k] * cur_col_vec_dagger[h] / norm;

						}
						else
						{
							matrix_p(k, h) = -qcomplex_t(2, 0) * cur_col_vec[k] * cur_col_vec_dagger[h] / norm;
						}
					}
				}
				PTrace("-----tmp matrixP:\n");
				PTraceMat(matrix_p);
				PTrace("tmp matrixP end -----------:\n");

				auto norm_sqr = sqrt(norm);
				for (size_t i = 0; i < cur_col_size; ++i)
				{
					cur_col_vec[i] /= norm_sqr;
				}
				cir_Q << build_cir_Pi(cur_col_vec);
				/*using testMat = Eigen::Matrix<qcomplex_t, -1, -1, Eigen::RowMajor>;
				testMat mat_test_1 = testMat::Map(&matrix_p[0], cur_col_size, cur_col_size);*/
				/*testMat mat_test_2 = testMat::Map(&src_mat[0], m_dimension, m_dimension);*/
				EigenMatrixXc mat_test_3 = src_mat.bottomRightCorner(m_dimension - cur_col, m_dimension - cur_col);
				matrix_p *= mat_test_3;
				//src_mat.block(cur_col, cur_col, m_dimension - cur_col, m_dimension - cur_col) *= matrix_p;
				//src_mat.bottomRightCorner(m_dimension - cur_col, m_dimension - cur_col) = matrix_p;
				src_mat.block(cur_col, cur_col, m_dimension - cur_col, m_dimension - cur_col) = matrix_p;
				PTrace("-----tmp matrixA:\n");
				PTraceMat(src_mat);
				PTrace("tmp matrixA end -----------:\n");
			}
		}

		//QCircuit last_cir_D = matrix_decompose(m_qubits, src_mat);
		QStat mat_r(src_mat.data(), src_mat.data() + src_mat.size());
		QCircuit last_cir_D = diagonal_matrix_decompose(m_qubits, mat_r);
		m_result_cir << last_cir_D << cir_Q.dagger();
	}

	QCircuit build_cir_Pi(const QStat& cur_col_vec)
	{
		QStat full_cur_vec(m_dimension - cur_col_vec.size(), qcomplex_t(0, 0));
		full_cur_vec.insert(full_cur_vec.end(), cur_col_vec.begin(), cur_col_vec.end());
		if (full_cur_vec.size() != m_dimension)
		{
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: current vector size error on HQRDecompose.");
		}

		QCircuit cir_swap_qubits;
		for (size_t i = 0; (i * 2) < (m_qubits.size() - 1); ++i)
		{
			cir_swap_qubits << SWAP(m_qubits[i], m_qubits[m_qubits.size() - 1 - i]);
		}

		std::vector<double> ui_mod(m_dimension);
		std::vector<double> ui_angle(m_dimension);
		double tatal = 0.0;
		for (size_t i = 0; i < m_dimension; ++i)
		{
			auto tmp_m = full_cur_vec[i].real() * full_cur_vec[i].real() + full_cur_vec[i].imag() * full_cur_vec[i].imag();
			tatal += tmp_m;
			ui_mod[i] = sqrt(tmp_m);
			ui_angle[i] = arg(full_cur_vec[i]);
		}

		QStat mat_d(m_dimension * m_dimension, qcomplex_t(0, 0));
		for (size_t i = 0; i < m_dimension; ++i)
		{
			mat_d[i + i * m_dimension] = exp(qcomplex_t(0, ui_angle[i]));
		}

		QCircuit cir_d = diagonal_matrix_decompose(m_qubits, mat_d);
		
		PTrace("cir_d:\n");
		PTraceCircuit(cir_d);

#if PRINT_TRACE
		const auto mat_test_d = getCircuitMatrix(cir_d);
		PTrace("mat_test_d:\n");
		PTraceCircuit(mat_test_d);
		if (mat_test_d == mat_d)
		{
			cout << "matrix decompose okkkkkkkkkkkkkkk" << endl;
		}
		else
		{
			cout << "ffffffffffffffffffffailed on matrix decompose." << endl;
		}
#endif

		QCircuit cir_y = build_cir_b(m_qubits, ui_mod);
		QCircuit cir_P;
		cir_P << cir_y << cir_swap_qubits  << cir_d << cir_swap_qubits;

		QCircuit cir_DG = zero_phase_shift_cir();
		QCircuit cir_pi;
		cir_pi << cir_swap_qubits  << cir_P.dagger() << cir_DG << cir_P << cir_swap_qubits;

		return cir_pi;
	}

	QCircuit zero_phase_shift_cir()
	{
		QCircuit cir_DG;
		QVec tmp_qubits = m_qubits;
		tmp_qubits.pop_back();
		cir_DG << applyQGate(m_qubits, X) << Z(m_qubits.back()).control(tmp_qubits) << applyQGate(m_qubits, X);

		return cir_DG;
	}

	QCircuit build_cir_b(QVec qubits, const std::vector<double>& b)
	{
		return amplitude_encode(qubits, b);
	}

private:
	QVec m_qubits;
	size_t m_dimension;
	QCircuit m_result_cir;
};

static QCircuit Householder_qr_matrix_decompose(QVec qubits, const EigenMatrixXc& src_mat)
{
	return HQRDecompose().decompose(qubits, src_mat);
}

/*******************************************************************
*                      public interface
********************************************************************/
QCircuit QPanda::matrix_decompose(QVec qubits, const QStat& src_mat, DecompositionMode de_mode/* = HOUSEHOLDER_QR*/)
{
	auto order = std::sqrt(src_mat.size());
	EigenMatrixXc tmp_mat = EigenMatrixXc::Map(&src_mat[0], order, order);

	return matrix_decompose(qubits, tmp_mat, de_mode);
}

QCircuit QPanda::matrix_decompose(QVec qubits, EigenMatrixXc& src_mat, DecompositionMode de_mode/* = HOUSEHOLDER_QR*/)
{
	if (!src_mat.isUnitary(MAX_MATRIX_PRECISION))
	{
		QCERR_AND_THROW_ERRSTR(invalid_argument, "Non-unitary matrix.");
	}

	if (qubits.size() != log2(src_mat.cols()))
	{
		QCERR_AND_THROW_ERRSTR(invalid_argument, "The qubits number is error.");
	}

	QCircuit output_circuit;
	switch (de_mode)
	{
	case HOUSEHOLDER_QR:
		output_circuit = Householder_qr_matrix_decompose(qubits, src_mat);
		break;

	default:
	{
		//QR decompose
		std::vector<SingleGateUnit> cir_units;
		general_scheme(src_mat, cir_units);
		circuit_insert(qubits, cir_units, output_circuit);
	}
		break;
	}
	
	return output_circuit;
}

QCircuit QPanda::diagonal_matrix_decompose(const QVec& qubits, const QStat& src_mat)
{
	return DiagonalMatrixDecompose().decompose(qubits, src_mat);
}
