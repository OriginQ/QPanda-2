#include "Core/Utilities/Tools/QuantumStateTomography.h"
#include "Core/QuantumCircuit/QNodeDeepCopy.h"
#include "Core/Core.h"


#include "ThirdParty/EigenUnsupported/Eigen/KroneckerProduct"
#include "ThirdParty/Eigen/src/LU/FullPivLU.h"
#include "ThirdParty/Eigen/src/Cholesky/LDLT.h"
#include "ThirdParty/Eigen/src/LU/PartialPivLU.h"
#include "ThirdParty/Eigen/src/QR/HouseholderQR.h"


using namespace QPanda;
using namespace std;

const double kEpsion = 1e-8;

using namespace QPanda;
using namespace std;


static void _mle(QMatrixXcd &m)
{
    int64_t dim = m.rows();
    Eigen::ComplexEigenSolver<QMatrixXcd> es(m);
    QVectorXcd v = es.eigenvalues();
    auto s = es.eigenvectors();
    bool is_positive_semidefinite = true;

    for (int64_t i = 0; i < dim; i++)
    {
        if (v(0, i).real() > -kEpsion)
        {
            continue;
        }

        is_positive_semidefinite = false;
        auto tmp = v[i];
        v[i] = { 0, 0 };
        double left_v = dim - (i + 1);

        for (int64_t j = i + 1; j < dim; j++)
        {
            v[j] += tmp / left_v;
        }
    }

    QPANDA_OP(is_positive_semidefinite, return);
    m.setZero();
    for (int64_t i = 0; i < dim; i++)
    {
        QMatrixXcd left_cols = s.col(i);
        m += v(0, i) * (left_cols * left_cols.conjugate().transpose());
    }

    return;
}


QuantumStateTomography::QuantumStateTomography()
{
}

QuantumStateTomography::~QuantumStateTomography()
{

}



std::vector<QStat> QuantumStateTomography::exec(QuantumMachine *qm, size_t shots)
{
    m_prog_results.clear();
    m_prog_results.reserve(m_combine_progs.size());

    for (auto &prog : m_combine_progs)
    {
        auto result_count = qm->runWithConfiguration(prog, m_clist, shots);
        map<std::string, double> result_prob;
        for (auto &item : result_count)
        {
            result_prob.insert({ item.first, static_cast<double>(item.second) / shots });
        }

        m_prog_results.push_back(result_prob);
    }
    return caculate_tomography_density();
}


void QuantumStateTomography::set_qprog_results(size_t opt_num, const std::vector<std::map<std::string, double>>& results)
{
    m_prog_results = results;
    m_opt_num = opt_num;
    return;
}

std::vector<QStat> QuantumStateTomography::caculate_tomography_density()
{
    _get_s();
    QMatrixXcd gate_i(2, 2);
    QMatrixXcd gate_x(2, 2);
    QMatrixXcd gate_y(2, 2);
    QMatrixXcd gate_z(2, 2);

    gate_i << 1, 0, 0, 1;
    gate_x << 0, 1, 1, 0;
    gate_y << 0, qcomplex_t(0, -1), qcomplex_t(0, 1), 0;
    gate_z << 1, 0, 0, -1;

    vector<QMatrixXcd> kraus;
    kraus.push_back(gate_i);
    kraus.push_back(gate_x);
    kraus.push_back(gate_y);
    kraus.push_back(gate_z);

    auto uint_to_ary = [](size_t num, size_t ary, size_t bit)->vector<size_t>
    {
        vector<size_t> res(bit, 0);
        size_t bit_num = res.size() - 1;
        for (auto i = num; i > 0; i /= ary)
        {
            auto mod = i % ary;
            res[bit_num] = mod;
            bit_num--;
        }

        return res;
    };

    auto tensor_kraus = [&](size_t idx)->QMatrixXcd
    {
        auto ary = uint_to_ary(idx, 4, m_opt_num);
        QMatrixXcd ret_kraus = kraus[ary[0]];

        for (size_t i = 1; i < ary.size(); i++)
        {
            QMatrixXcd tmp = kroneckerProduct(ret_kraus, kraus[ary[i]]);
            ret_kraus.resize(tmp.rows(), tmp.cols());
            ret_kraus = tmp;
        }

        return ret_kraus;
    };

    size_t dim = 1ull << m_opt_num;
    QMatrixXcd eigen_density(dim, dim);
    eigen_density.setZero();

    for (size_t i = 0; i < m_s.size(); i++)
    {
        double fact = 1. / (1ull << m_opt_num);
        fact *= m_s[i];
        QMatrixXcd tmp_density = tensor_kraus(i);
        eigen_density += fact * tmp_density;
    }

    // positive semidefinite tranform
    _mle(eigen_density);
    vector<QStat> res_density(dim, QStat(dim, 0));

    for (int64_t row = 0; row < eigen_density.rows(); row++)
    {
        for (int64_t col = 0; col < eigen_density.cols(); col++)
        {
            res_density.at(row).at(col) = eigen_density(row, col);
        }
    }

    return res_density;
}


void QuantumStateTomography::_get_s()
{
    m_s.clear();
    auto uint_to_ary = [](size_t num, size_t ary, size_t bit)->vector<size_t>
    {
        vector<size_t> res(bit, 0);
        size_t bit_num = res.size() - 1;
        for (auto i = num; i > 0; i /= ary)
        {
            auto mod = i % ary;
            res[bit_num] = mod;
            bit_num--;
        }

        return res;
    };

    auto s_map_p = [](const vector<size_t> &idx_ary)->size_t
    {
        size_t p_idx = 0;
        int i = 0;
        for (size_t i = 0; i < idx_ary.size(); i++)
        {
            p_idx += (idx_ary[idx_ary.size() - i - 1] % 3) * pow(3, i);
        }

        return p_idx;
    };

    auto store_nozero_pos = [](const vector<size_t> &idx_ary)->vector<size_t>
    {
        vector<size_t> res;
        for (size_t i = 0; i < idx_ary.size(); i++)
        {
            if (idx_ary[i])
            {
                res.push_back(i);
            }
        }

        return res;
    };

    auto compute_s = [](const vector<size_t> &nonozore_pos, const map<string, double> &probs)
    {
        double s = 0;
        for (auto &p : probs)
        {
            int num_nozore = 0;
            for_each(nonozore_pos.begin(), nonozore_pos.end(), [&](const size_t &pos)
            {
                if ('0' != p.first[pos])
                {
                    num_nozore += 1;
                }
            });

            0 == num_nozore % 2 ? s += p.second : s -= p.second;
        }

        return s;
    };

    m_s.assign(1ull << (2 * m_opt_num), 0);
    for (size_t i = 0; i < m_s.size(); i++)
    {
        auto idx_ary = uint_to_ary(i, 4, m_opt_num);
        auto idx_p = s_map_p(idx_ary);
        vector<size_t> nozore_pos = store_nozero_pos(idx_ary);
        m_s[i] = compute_s(nozore_pos, m_prog_results[idx_p]);
    }

    return;
}


