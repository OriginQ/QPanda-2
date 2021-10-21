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

			if ((EigenComplexT(0, 0) == matrix(rdx, cdx) && (idx != opts - 1)) ||
				(EigenComplexT(1, 0) == matrix(cdx, cdx) && (idx == opts - 1)))
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
	if (EigenMatrixXc::Identity(2, 2) != V2)
	{
        QPANDA_ASSERT((V2(0, 0) * V2(1, 1)) == (V2(0, 1) * V2(1, 0)), "decomposition error on matrix.bottomRightCorner(2, 2)");

        qcomplex_t E0 = V2(1, 1) / ((V2(0, 0) * V2(1, 1)) - (V2(0, 1) * V2(1, 0)));
        qcomplex_t E1 = V2(0, 1) / ((V2(0, 1) * V2(1, 0)) - (V2(0, 0) * V2(1, 1)));
        qcomplex_t E2 = V2(1, 0) / ((V2(0, 1) * V2(1, 0)) - (V2(0, 0) * V2(1, 1)));
        qcomplex_t E3 = V2(0, 0) / ((V2(0, 0) * V2(1, 1)) - (V2(0, 1) * V2(1, 0)));

        QStat M2 = { E0 ,E1 ,E2 ,E3 };

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

static void circuit_insert(QVec& qubits, std::vector<SingleGateUnit>& cir_units, QCircuit &circuit,bool b_positive_seq)
{
	if (b_positive_seq) 
	{
		std::sort(qubits.begin(), qubits.end(), [&](Qubit *a, Qubit *b){
			return a->getPhysicalQubitPtr()->getQubitAddr()
				< b->getPhysicalQubitPtr()->getQubitAddr();
		});
	}
	else 
	{
		std::sort(qubits.begin(), qubits.end(), [&](Qubit *a, Qubit *b){
			return a->getPhysicalQubitPtr()->getQubitAddr()
				> b->getPhysicalQubitPtr()->getQubitAddr();
		});
	}
	
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

			tmp_mat22.clear();
			const size_t tmp_row = (2 * i * mat_dimension) + (2 * i);
			tmp_mat22.push_back(src_mat[tmp_row]);
			tmp_mat22.push_back(src_mat[tmp_row + 1]);
			tmp_mat22.push_back(src_mat[tmp_row + mat_dimension]);
			tmp_mat22.push_back(src_mat[tmp_row + mat_dimension + 1]);
			QGate tmp_u4 = U4(tmp_mat22, qubits.back()).control(controlqvec);
			QGATE_SPACE::U4* p_gate = dynamic_cast<QGATE_SPACE::U4*>(tmp_u4.getQGate());
			if ((abs(p_gate->getAlpha()) > MAX_MATRIX_PRECISION)
				|| (abs(p_gate->getBeta()) > MAX_MATRIX_PRECISION)
				|| (abs(p_gate->getGamma()) > MAX_MATRIX_PRECISION)
				|| (abs(p_gate->getDelta()) > MAX_MATRIX_PRECISION))
			{
				decompose_result_cir << tmp_u4;
			}

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
*                      public interface
********************************************************************/
QCircuit QPanda::matrix_decompose_qr(QVec qubits, const QStat& src_mat,const bool b_positive_seq)
{
	auto order = std::sqrt(src_mat.size());
	EigenMatrixXc tmp_mat = EigenMatrixXc::Map(&src_mat[0], order, order);

    return matrix_decompose_qr(qubits, tmp_mat, b_positive_seq);
}

QCircuit QPanda::matrix_decompose_qr(QVec qubits, EigenMatrixXc& src_mat,const bool b_positive_seq)
{
	if (!src_mat.isUnitary(MAX_MATRIX_PRECISION))
	{
		QCERR_AND_THROW_ERRSTR(invalid_argument, "Non-unitary matrix.");
	}

	if (qubits.size() != log2(src_mat.cols()))
	{
		QCERR_AND_THROW_ERRSTR(invalid_argument, "The qubits number is error or the input matrix is not a 2^n-dimensional matrix.");
	}

	QCircuit output_circuit;
    //QR decompose
    std::vector<SingleGateUnit> cir_units;
    general_scheme(src_mat, cir_units);
    circuit_insert(qubits, cir_units, output_circuit, b_positive_seq);
	
	return output_circuit;
}

QCircuit QPanda::diagonal_matrix_decompose(const QVec& qubits, const QStat& src_mat)
{
	return DiagonalMatrixDecompose().decompose(qubits, src_mat);
}
