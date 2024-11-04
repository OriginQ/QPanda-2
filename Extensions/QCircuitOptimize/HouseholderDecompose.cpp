#include <math.h>
#include <EigenUnsupported/Eigen/KroneckerProduct>
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/Tools/MatrixDecomposition.h"
#include <chrono>
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include "QAlg/Base_QCircuit/AmplitudeEncode.h"


#include "HouseholderDecompose.h"


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


#ifndef MAX_MATRIX_PRECISION
#define MAX_MATRIX_PRECISION 1e-10
#endif

/*******************************************************************
*                      class HQRDecompose
* Householder QR-decompose
* refer to <Quantum circuits synthesis using Householder transformations>(https://arxiv.org/abs/2004.07710v1)
********************************************************************/
class HQRDecompose
{
public:
    HQRDecompose()
        :m_dimension(0), m_b_positive_seq(true)
    {}
    ~HQRDecompose() {}

    QCircuit decompose(QVec qubits, const QMatrixXcd& src_mat, bool b_positive_seq)
    {
		if (b_positive_seq)
		{
			std::sort(qubits.begin(), qubits.end(), [&](Qubit *a, Qubit *b) {
				return a->get_phy_addr() < b->get_phy_addr();
			});
		}
		else
		{
			std::sort(qubits.begin(), qubits.end(), [&](Qubit *a, Qubit *b) {
				return a->get_phy_addr() > b->get_phy_addr();
			});
		}
		m_b_positive_seq = b_positive_seq;

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

    void Householder_QR_decompose(QMatrixXcd src_mat)
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

            if (norm > DBL_EPSILON)
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
                QMatrixXcd matrix_p(cur_col_size, cur_col_size);
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
                QMatrixXcd mat_test_3 = src_mat.bottomRightCorner(m_dimension - cur_col, m_dimension - cur_col);
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
		if (m_b_positive_seq) 
		{
			for (size_t i = 0; (i * 2) < (m_qubits.size() - 1); ++i){
				cir_swap_qubits << SWAP(m_qubits[i], m_qubits[m_qubits.size() - 1 - i]);
			}
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
        cir_P << cir_y << cir_swap_qubits << cir_d << cir_swap_qubits;

        QCircuit cir_DG = zero_phase_shift_cir();
        QCircuit cir_pi;
        cir_pi << cir_swap_qubits << cir_P.dagger() << cir_DG << cir_P << cir_swap_qubits;

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
	bool m_b_positive_seq;
};


QCircuit QPanda::matrix_decompose_householder(QVec qubits, const QStat& src_mat, bool b_positive_seq)
{
    auto order = std::sqrt(src_mat.size());
    QMatrixXcd tmp_mat = QMatrixXcd::Map(&src_mat[0], order, order);

    return matrix_decompose_householder(qubits, tmp_mat);
}

QCircuit QPanda::matrix_decompose_householder(QVec qubits, const QMatrixXcd& src_mat, bool b_positive_seq)
{
    if (!src_mat.isUnitary(MAX_MATRIX_PRECISION))
    {
        QCERR_AND_THROW_ERRSTR(invalid_argument, "Non-unitary matrix.");
    }

    if (qubits.size() != log2(src_mat.cols()))
    {
        QCERR_AND_THROW_ERRSTR(invalid_argument, "The qubits number is error or the input matrix is not a 2^n-dimensional matrix.");
    }

    QCircuit output_circuit = HQRDecompose().decompose(qubits, src_mat, b_positive_seq);
    return output_circuit;
}
