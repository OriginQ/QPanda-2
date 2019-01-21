#include <iostream>
#include "OriginNelderMead.h"

namespace QPanda
{

    OriginNelderMead::OriginNelderMead() :
        m_rho(1),
        m_chi(2),
        m_psi(0.5),
        m_sigma(0.5),
        m_nonzdelt(0.05),
        m_zdelt(0.00025),
        m_fcalls(0),
        m_iter(0)
    {

    }

    void OriginNelderMead::exec()
    {
        init();

        Eigen::VectorXd xr;
        Eigen::VectorXd xe;
        Eigen::VectorXd xc;

        QResultPair fxr;
        QResultPair fxe;
        QResultPair fxc;

        bool do_shrink = false;

        // no-convergence test
        while ((m_fcalls < m_max_fcalls) && (m_iter < m_max_iter))
        {
            // the domain and function-value convergence test
            if (testTermination())
            {
                break;
            }

            calcCentroid();

            // Reflect
            xr = (1 + m_rho) * m_centroid - m_rho * m_sim.row(m_n).transpose();
            fxr = callFunc(xr);
            do_shrink = false;

            if (fxr.second < m_fsim[0])
            {
                // Expand
                xe = (1 + m_rho * m_chi) * m_centroid
                    - m_rho * m_chi * m_sim.row(m_n).transpose();
                fxe = callFunc(xe);

                if (fxe.second < fxr.second)
                {
                    m_sim.row(m_n) = xe;
                    m_key[m_n] = fxe.first;
                    m_fsim[m_n] = fxe.second;
                }
                else
                {
                    m_sim.row(m_n) = xr;
                    m_key[m_n] = fxr.first;
                    m_fsim[m_n] = fxr.second;
                }
            }
            else // m_fsim[0] <= fxr
            {
                if (fxr.second < m_fsim[m_n - 1])
                {
                    m_sim.row(m_n) = xr;
                    m_key[m_n] = fxr.first;
                    m_fsim[m_n] = fxr.second;
                }
                else // fxr >= m_fsim[m_n - 1]
                {
                    // Contraction
                    if (fxr.second < m_fsim[m_n])
                    {
                        // Outside Contraction
                        xc = (1 + m_psi * m_rho) * m_centroid
                            - m_psi * m_rho * m_sim.row(m_n).transpose();
                        fxc = callFunc(xc);

                        if (fxc <= fxr)
                        {
                            m_sim.row(m_n) = xc;
                            m_key[m_n] = fxc.first;
                            m_fsim[m_n] = fxc.second;
                        }
                        else
                        {
                            do_shrink = true;
                        }
                    }
                    else
                    {
                        // Inside Contraction
                        xc = (1 - m_psi) * m_centroid
                            + m_psi * m_sim.row(m_n).transpose();
                        fxc = callFunc(xc);

                        if (fxc.second < m_fsim[m_n])
                        {
                            m_sim.row(m_n) = xc;
                            m_key[m_n] = fxc.first;
                            m_fsim[m_n] = fxc.second;
                        }
                        else
                        {
                            do_shrink = true;
                        }
                    }

                    if (do_shrink)
                    {
                        // Shrink
                        for (size_t i = 1; i <= m_n; i++)
                        {
                            m_sim.row(i) = m_sim.row(0)
                                + m_sigma * (m_sim.row(i) - m_sim.row(0));

                            QResultPair result = callFunc(m_sim.row(i));
                            m_key[i] = result.first;
                            m_fsim[i] = result.second;
                        }
                    }
                }
            }

            sortData();

            m_iter++;

            dispResult();
        }
    }

    OptimizationResult OriginNelderMead::getResult()
    {
        if (m_fcalls >= m_max_fcalls)
        {
            m_result[DEF_MESSAGE] = DEF_OPTI_STATUS_MAX_FEV;
            std::cout << DEF_WARING + m_result[DEF_MESSAGE]
                << std::endl;
        }
        else if (m_iter >= m_max_iter)
        {
            m_result[DEF_MESSAGE] = DEF_OPTI_STATUS_MAX_ITER;
            std::cout << DEF_WARING + m_result[DEF_MESSAGE]
                << std::endl;
        }
        else
        {
            m_result[DEF_MESSAGE] = DEF_OPTI_STATUS_SUCCESS;
            dispResult();
        }

        m_result[DEF_VALUE] = std::to_string(m_fsim[0]);
        m_result[DEF_KEY] = m_key[0];

        return m_result;
    }

    bool OriginNelderMead::init()
    {
        m_n = m_optimized_para.size();
        if (0 == m_n)
        {
            throw std::string("Bad para.");
        }

        adaptFourPara();
        adaptTerminationPara();
        initialSimplex();

        m_result[DEF_MESSAGE] = DEF_OPTI_STATUS_CALCULATING;

        return true;
    }

    void OriginNelderMead::adaptFourPara()
    {
        if (m_adaptive)
        {
            m_rho = 1;
            m_chi = 1 + 2.0 / m_n;
            m_psi = 0.75 - 1 / (2.0*m_n);
            m_sigma = 1 - 1.0 / m_n;
        }
    }

    void OriginNelderMead::adaptTerminationPara()
    {
        if (size_t(DEF_UNINIT_INT) == m_max_iter)
        {
            m_max_iter = m_n * 200;
        }

        if (size_t(DEF_UNINIT_INT) == m_max_fcalls)
        {
            m_max_fcalls = m_n * 200;
        }
    }

    void OriginNelderMead::initialSimplex()
    {
        m_x0 = Eigen::VectorXd(m_n);
        memcpy(m_x0.data(),
            m_optimized_para.data(),
            m_optimized_para.size() * sizeof(m_optimized_para[0]));

        m_sim = Eigen::MatrixXd::Zero(m_n + 1, m_n);
        m_sim.row(0) = m_x0;

        for (size_t i = 0; i < m_n; i++)
        {
            Eigen::VectorXd y = m_x0;
            if (0 != y[i])
            {
                y[i] = (1 + m_nonzdelt)*y[i];
            }
            else
            {
                y[i] = m_zdelt;
            }

            m_sim.row(i + 1) = y;
        }

        m_key.resize(m_n + 1);
        m_fsim = Eigen::VectorXd::Zero(m_n + 1);
        for (size_t i = 0; i < m_n + 1; i++)
        {
            Eigen::VectorXd tmp = m_sim.row(i);
            QResultPair result = callFunc(tmp);
            m_key[i] = result.first;
            m_fsim[i] = result.second;
        }

        sortData();

        m_iter++;
    }

    QResultPair OriginNelderMead::callFunc(const Eigen::VectorXd &para)
    {
        m_fcalls++;

        vector_d optimized_para(para.data(), para.data() + para.size());
        return m_func(optimized_para);
    }

    bool OriginNelderMead::testTermination()
    {
        Eigen::MatrixXd tmp_sim = m_sim.block(1,
            0,
            m_sim.rows() - 1,
            m_sim.cols());
        for (size_t i = 0; i < m_n; i++)
        {
            tmp_sim.row(i) = (tmp_sim.row(i) - m_sim.row(0)).cwiseAbs();
        }

        Eigen::VectorXd tmp_fsim = m_fsim.block(1,
            0,
            m_fsim.rows() - 1,
            m_fsim.cols());

        for (size_t i = 0; i < m_n; i++)
        {
            tmp_fsim[i] = fabs(tmp_fsim[i] - m_fsim[0]);
        }

        return (tmp_sim.maxCoeff() <= m_xatol) && (tmp_fsim.maxCoeff() <= m_fatol);
    }

    void OriginNelderMead::calcCentroid()
    {
        m_centroid = Eigen::VectorXd::Zero(m_n);
        for (size_t i = 0; i < m_n; i++)
        {
            m_centroid += m_sim.row(i);
        }

        m_centroid /= m_n;
    }

    void OriginNelderMead::sortData()
    {
        std::vector<size_t> ind = sortVector(m_fsim);
        Eigen::MatrixXd tmp_sim = m_sim;
        vector_s tmp_key = m_key;
        for (size_t i = 0; i < ind.size(); i++)
        {
            m_key[i] = tmp_key[ind[i]];
            m_sim.row(i) = tmp_sim.row(ind[i]);
        } // sort so m_sim[0,:] has the lowest function value
    }

    std::vector<size_t> OriginNelderMead::sortVector(Eigen::VectorXd &vec)
    {
        std::multimap<double, size_t> mulit_map;
        size_t index = 0;

        size_t size = size_t(vec.size());
        for (size_t i = 0; i < size; i++)
        {
            mulit_map.insert(std::pair<double, size_t>(vec[i], index++));
        }

        std::sort(vec.data(), vec.data() + vec.size());

        std::vector<size_t> vec_original_index;
        for (size_t i = 0; i < size; i++)
        {
            auto map_iter = mulit_map.find(vec[i]);
            vec_original_index.push_back(map_iter->second);
            mulit_map.erase(map_iter);
        }

        return vec_original_index;
    }

    void OriginNelderMead::dispResult()
    {
        if (m_disp)
        {
            std::cout << m_result[DEF_MESSAGE] << std::endl;
            std::cout << "         Current function value: "
                << m_fsim[0] << std::endl;
            std::cout << "         Key: "
                << m_key[0] << std::endl;
            std::cout << "         Iterations: "
                << m_iter << std::endl;
            std::cout << "         Function evaluations: "
                << m_fcalls << std::endl;
        }
    }

}
