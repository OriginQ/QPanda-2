#include "Core/Utilities/Tools/Fidelity.h"
#include "algorithm"
#include "float.h"
#include "ThirdParty/Eigen/Eigen"
#include "ThirdParty/EigenUnsupported/Eigen/MatrixFunctions"
#include "ThirdParty/Eigen/src/Cholesky/LLT.h"

using namespace std;

using  qvector_t = Eigen::Matrix<qcomplex_t, 1, Eigen::Dynamic, Eigen::RowMajor>;
using  qmatrix_t = Eigen::Matrix<qcomplex_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

const double kEpsion = 1e-8;
using namespace Eigen;


static bool validity(const QStat &state)
{
    size_t size = state.size();
    QPANDA_RETURN(size < 1 || (size & (size - 1)), false);
    double sum = 0;
    for_each(state.begin(), state.end(), [&](const qcomplex_t &item) {
        sum += std::norm(item);
    });
    QPANDA_RETURN(abs(sum - 1) > kEpsion, false);

    return true;
}

static bool validity(const qmatrix_t &m)
{
    // trace == 1
    QPANDA_RETURN(abs(m.trace().real() - 1) > kEpsion, false);

    // is hermitian matrix
    bool is_equal = m.isApprox(m.transpose().conjugate(), kEpsion);
    QPANDA_RETURN(!is_equal, false);

    //is positive semidefinite
    Eigen::ComplexEigenSolver<qmatrix_t> es(m);
    qvector_t s = es.eigenvalues();

    for (size_t row = 0; row < s.rows(); row++)
    {
        for (size_t col = 0; col < s.cols(); col++)
        {
            auto item = s(row, col);
            QPANDA_RETURN(item.real() < -kEpsion || abs(item.imag()) > kEpsion, false);
        }
    }

    return true;
}

double QPanda::state_fidelity(const QStat &state1, const QStat &state2, bool validate /* = true */)
{
    if (validate)
    {
        QPANDA_ASSERT(state1.size() != state2.size() || !validity(state1) || !validity(state2),
            "Error: state fidelity");
    }

    qvector_t state_qvec1 = qvector_t::Map(&state1[0], state1.size());
    qvector_t state_qvec2 = qvector_t::Map(&state2[0], state2.size());
    auto state_qvec2_conj = state_qvec2.conjugate();

    return std::norm(state_qvec2_conj.dot(state_qvec1));
}

double QPanda::state_fidelity(const std::vector<QStat> &matrix1, const std::vector<QStat> &matrix2, bool validate /* = true */)
{
    auto sqrt_matrix = [](const qmatrix_t &m)->qmatrix_t
    {
        JacobiSVD<qmatrix_t> svd(m, ComputeThinU | ComputeThinV);
        qmatrix_t u = svd.matrixU();
        qmatrix_t v = svd.matrixV().transpose().conjugate();
        qvector_t s = svd.singularValues();

        qmatrix_t diag = s.asDiagonal();
        auto diag_sqrt = diag.sqrt();

        return u * diag_sqrt * v;
    };

    size_t dim1 = matrix1.size();
    size_t dim2 = matrix2.size();
    QPANDA_ASSERT(dim1 != dim2 || dim1 < 2, "Error: density matrix dim");
    qmatrix_t m1(dim1, dim1);
    qmatrix_t m2(dim2, dim2);

    for (size_t row = 0; row < dim1; row++)
    {
        QPANDA_ASSERT(matrix1[row].size() != dim1 || matrix2[row].size() != dim2, "Error: density matrix dim");
        m1.row(row) = qvector_t::Map(&matrix1[row][0], 1, dim1);
        m2.row(row) = qvector_t::Map(&matrix2[row][0], 1, dim2);
    }

    if (validate)
    {
        QPANDA_ASSERT(!validity(m1) || !validity(m2), "Error: density matrix is invalid");
    }

    auto sq1 = m1.sqrt();
    auto sq2 = m2.sqrt();
    auto sq1_sq2 = sq1 * sq2;

    // Nuclear Norm
    JacobiSVD<qmatrix_t> svd(sq1_sq2, ComputeThinU | ComputeThinV);
    qvector_t s = svd.singularValues();
    auto sum = s.sum();
    auto fid = sum * sum;

    return fid.real();
}

double QPanda::state_fidelity(const QStat &state, const std::vector<QStat> &matrix, bool validate /* = true */)
{
    if (validate)
    {
        QPANDA_ASSERT(!validity(state), "Error: state fidelity");
    }

    qvector_t v = qvector_t::Map(&state[0], state.size());
    size_t dim = matrix.size();
    QPANDA_ASSERT(state.size() != dim, "Error: state or matrix.");
    qmatrix_t m(dim, dim);

    for (size_t row = 0; row < dim; row++)
    {
        QPANDA_ASSERT(matrix[row].size() != dim, "Error: density matrix dim");
        m.row(row) = qvector_t::Map(&matrix[row][0], 1, dim);
    }

    auto state_mul_matrix = v.conjugate() * m;
    auto fid = state_mul_matrix.dot(v);

    return fid.real();
}

double QPanda::state_fidelity(const std::vector<QStat> &matrix, const QStat &state, bool validate /* = true */)
{
    return state_fidelity(state, matrix, validate);
}
