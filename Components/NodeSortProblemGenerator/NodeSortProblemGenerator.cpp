#include "Components/NodeSortProblemGenerator/NodeSortProblemGenerator.h"
#include <time.h>

#ifndef PI
#define PI 3.14159265358979
#endif

QPANDA_BEGIN

void NodeSortProblemGenerator::exec()
{
    if (m_graph.empty())
    {
        return;
    }

    std::vector<double> U_hat_vec;
    std::vector<double> D_hat_vec;

    calcGraphPara(m_graph, m_lambda1, U_hat_vec, D_hat_vec);
    m_pauli = genHamiltonian(m_graph, m_lambda2, m_lambda3, U_hat_vec, D_hat_vec);
    m_ansatz = genAnsatz(m_graph, U_hat_vec, D_hat_vec);
    m_linear_solver_result = genLinearSolverResult(
        m_graph,
        m_lambda2,
        m_lambda3,
        U_hat_vec,
        D_hat_vec,
        m_A,
        m_b);
}

void NodeSortProblemGenerator::calcGraphPara(
    const std::vector<std::vector<double>>& graph,
    double lambda,
    std::vector<double>& U_hat_vec,
    std::vector<double>& D_hat_vec) const
{
    int node_num = graph.size();
    std::map<int, std::vector<int>> input_map;
    std::map<int, std::vector<int>> output_map;
    std::vector<double> D_vec;
    D_vec.resize(node_num);
    memset(D_vec.data(), 0, sizeof(double) * D_vec.size());
    for (int i = 0; i < node_num; i++)
    {
        std::vector<int> input_node;
        std::vector<int> output_node;
        for (int j = 0; j < node_num; j++)
        {
            if (i == j)
            {
                continue;
            }

            if (fabs(graph[j][i]) > 1e-6)
            {
                D_vec[j] += graph[j][i];
                if (input_map.find(i) == input_map.end())
                {
                    input_map.insert(std::make_pair(i, std::vector<int>{ j }));
                }
                else
                {
                    input_map[i].push_back(j);
                }
            }

            if (fabs(graph[i][j]) > 1e-6)
            {
                D_vec[i] += graph[i][j];

                if (output_map.find(i) == output_map.end())
                {
                    output_map.insert(std::make_pair(i, std::vector<int>{ j }));
                }
                else
                {
                    output_map[i].push_back(j);
                }
            }
        }
    }

    std::vector<double> U_vec;
    U_vec.resize(node_num);
    memset(U_vec.data(), 0, sizeof(int) * U_vec.size());

    for (int i = 0; i < node_num; i++)
    {
        int R_hat = 0;
        int R = 0;
        auto i_iter = input_map.find(i);
        if (i_iter == input_map.end())
        {
            continue;
        }
        auto o_iter = output_map.find(i);
        if (o_iter == output_map.end())
        {
            continue;
        }

        std::vector<int> input = i_iter->second;

        for (int m = 0; m < input.size(); m++)
        {
            std::vector<int> output = o_iter->second;

            int ci = input[m];
            // remove current input point in output
            auto f_iter = std::find(output.begin(), output.end(), ci);
            if (f_iter != output.end())
            {
                output.erase(f_iter);
            }

            //R_hat += output.size();
            R_hat = input.size() * o_iter->second.size();

            auto ci_iter = output_map.find(ci);
            if (ci_iter != output_map.end())
            {
                auto ci_output = ci_iter->second;
                // remove current point in ci_output
                auto f_iter = std::find(ci_output.begin(), ci_output.end(), i);
                if (f_iter != ci_output.end())
                {
                    ci_output.erase(f_iter);
                }

                for (auto it : ci_output)
                {
                    // remove direct reached point in output
                    auto f_iter = std::find(output.begin(), output.end(), it);
                    if (f_iter != output.end())
                    {
                        output.erase(f_iter);
                    }

                    auto it_iter = output_map.find(it);
                    if (it_iter != output_map.end())
                    {
                        auto it_output = it_iter->second;
                        for (auto et : it_output)
                        {
                            // remove indirect reached point in output
                            auto f_iter = std::find(output.begin(), output.end(), et);
                            if (f_iter != output.end())
                            {
                                output.erase(f_iter);
                            }
                        }
                    }
                }
            }

            R += output.size();
        }

        if (R_hat != 0)
        {
            U_vec[i] = 1.0 * R * R / R_hat;
        }
    }

    std::vector<double> Uv_vec;
    Uv_vec.resize(node_num);

    std::vector<double> Dv_vec;
    Dv_vec.resize(node_num);
    for (int i = 0; i < node_num; i++)
    {
        double sum_delta_U = 0;
        double sum_delta_D = 0;
        auto i_iter = input_map.find(i);
        if (i_iter != input_map.end())
        {
            for (auto i_v : i_iter->second)
            {
                sum_delta_U +=
                    (graph[i_v][i] + graph[i][i_v]) / D_vec[i_v] *
                    (U_vec[i_v] - U_vec[i]) *
                    (graph[i_v][i] + graph[i][i_v]) / D_vec[i];
                sum_delta_D +=
                    (graph[i_v][i] + graph[i][i_v]) / D_vec[i_v] *
                    (D_vec[i_v] - D_vec[i]) *
                    (graph[i_v][i] + graph[i][i_v]) / D_vec[i];
            }
        }

        auto o_iter = output_map.find(i);
        if (o_iter != output_map.end())
        {
            for (auto o_v : o_iter->second)
            {
                sum_delta_U +=
                    (graph[o_v][i] + graph[i][o_v]) / D_vec[o_v] *
                    (U_vec[o_v] - U_vec[i]) *
                    (graph[o_v][i] + graph[i][o_v]) / D_vec[i];

                sum_delta_D +=
                    (graph[o_v][i] + graph[i][o_v]) / D_vec[o_v] *
                    (D_vec[o_v] - D_vec[i]) *
                    (graph[o_v][i] + graph[i][o_v]) / D_vec[i];
            }
        }

        Uv_vec[i] = lambda * sum_delta_U + U_vec[i];
        Dv_vec[i] = lambda * sum_delta_D + D_vec[i];
    }
    double Uv_max = *(std::max_element(Uv_vec.begin(), Uv_vec.end()));
    double Uv_min = *(std::min_element(Uv_vec.begin(), Uv_vec.end()));

    double Dv_max = *(std::max_element(Dv_vec.begin(), Dv_vec.end()));
    double Dv_min = *(std::min_element(Dv_vec.begin(), Dv_vec.end()));

    U_hat_vec.resize(node_num);
    D_hat_vec.resize(node_num);
    memset(U_hat_vec.data(), 0, sizeof(double) * node_num);
    memset(D_hat_vec.data(), 0, sizeof(double) * node_num);

    bool uzero_flag = false;
    if (fabs(Uv_max - Uv_min) < 1e-6)
    {
        uzero_flag = true;
    }
    bool dzero_flag = false;
    if (fabs(Dv_max - Dv_min) < 1e-6)
    {
        dzero_flag = true;
    }

    for (int i = 0; i < node_num; i++)
    {
        U_hat_vec[i] = uzero_flag ? 1 :
            1 + (Uv_vec[i] - Uv_min)* (node_num -1)/ (Uv_max - Uv_min);
        D_hat_vec[i] = dzero_flag ? 1 :
            1 + (Dv_vec[i] - Dv_min)* (node_num - 1) / (Dv_max - Dv_min);
        //U_hat_vec[i] = uzero_flag ? 1 :
        //    1 + (Uv_vec[i] - Uv_min) * (3 - 1) / (Uv_max - Uv_min);
        //D_hat_vec[i] = dzero_flag ? 1 :
        //    1 + (Dv_vec[i] - Dv_min) * (3 - 1) / (Dv_max - Dv_min);
    }
}

PauliOperator NodeSortProblemGenerator::genHamiltonian(
    const std::vector<std::vector<double>>& graph,
    double lambda_u, 
    double lambda_d, 
    const std::vector<double>& U_hat_vec, 
    const std::vector<double>& D_hat_vec) const
{
    if (U_hat_vec.empty() || D_hat_vec.empty())
    {
        return PauliOperator();
    }

    PauliOperator p;
    int node_num = U_hat_vec.size();
    for (int i = 0; i < node_num; i++)
    {
        for (int j = 0; j < node_num; j++)
        {
            if (i == j)
            {
                continue;
            }

            if (fabs(graph[i][j]) > 1e-6)
            {
                auto rest_len = lambda_u * (U_hat_vec[i] - U_hat_vec[j]) +
                    lambda_d * (D_hat_vec[i] - D_hat_vec[j]);

                PauliOperator p1(rest_len);
                p1 += PauliOperator(
                    "Z" + std::to_string(i),
                    0.5
                );
                p1 += PauliOperator(
                    "Z" + std::to_string(j),
                    -0.5
                );

                p += 0.5 * p1 * p1;
            }
        }
    }

    return p;
}

std::vector<AnsatzGate> NodeSortProblemGenerator::genAnsatz(
    const std::vector<std::vector<double>>& graph,
    const std::vector<double>& U_hat_vec, 
    const std::vector<double>& D_hat_vec) const
{
    if (U_hat_vec.empty() || D_hat_vec.empty())
    {
        return std::vector<AnsatzGate>();
    }

    std::vector<std::pair<int, std::vector<int>>> result_vec;
    int node_num = graph.size();
    std::map<int, std::vector<int>> output_map;
    for (int i = 0; i < node_num; i++)
    {
        std::vector<int> input_node;
        std::vector<int> output_node;
        for (int j = 0; j < node_num; j++)
        {
            if (i == j)
            {
                continue;
            }

            if (fabs(graph[i][j]) > 1e-6)
            {
                if (output_map.find(i) == output_map.end())
                {
                    output_map.insert(std::make_pair(i, std::vector<int>{ j }));
                }
                else
                {
                    output_map[i].push_back(j);
                }
            }
        }
    }

    std::vector<double> U_plus_D_vec;
    for (auto i = 0; i < U_hat_vec.size(); i++)
    {
        U_plus_D_vec.push_back(U_hat_vec[i] + D_hat_vec[i]);
    }

    while (output_map.size() > 0)
    {
        auto max_iter = output_map.begin();
        for (auto ot = output_map.begin(); ot != output_map.end(); ot++)
        {
            if (U_plus_D_vec[ot->first] > U_plus_D_vec[max_iter->first])
            {
                max_iter = ot;
            }
        }
        result_vec.push_back(*max_iter);
        auto value = max_iter->first;
        output_map.erase(max_iter);
        std::vector<std::map<int, std::vector<int>>::iterator> delete_vec;
        for (auto ot = output_map.begin(); ot != output_map.end(); ot++)
        {
            auto f_iter = std::find(ot->second.begin(), ot->second.end(), value);
            if (f_iter != ot->second.end())
            {
                ot->second.erase(f_iter);
            }

            if (ot->second.size() == 0)
            {
                delete_vec.push_back(ot);
            }
        }
        for (auto it : delete_vec)
        {
            output_map.erase(it);
        }
    }

    srand((int)time(0));
    std::vector<AnsatzGate> ansatz_vec;
    for (int i = 0; i < result_vec.size(); i++)
    {
        ansatz_vec.push_back({
            AnsatzGateType::AGT_NOT,
            result_vec[i].first });
    }

    for (int i = 0; i < result_vec.size(); i++)
    {
        for (int j = 0; j < result_vec[i].second.size(); j++)
        {
            ansatz_vec.push_back({
                AnsatzGateType::AGT_RY,
                result_vec[i].second[j],
                (rand() / double(RAND_MAX)) * 2 * PI,
                result_vec[i].first });
        }
    }

    for (int i = 0; i < result_vec.size(); i++)
    {
        ansatz_vec.push_back({
            AnsatzGateType::AGT_RX,
            result_vec[i].first,
            (rand() / double(RAND_MAX)) * 2 * PI,
            -1 });
    }

    return ansatz_vec;
}

Eigen::VectorXd NodeSortProblemGenerator::genLinearSolverResult(
    const std::vector<std::vector<double>>& graph,
    double lambda_u, 
    double lambda_d, 
    const std::vector<double>& U_hat_vec, 
    const std::vector<double>& D_hat_vec,
    Eigen::MatrixXd& A,
    Eigen::VectorXd& b) const
{
    if (graph.empty() || U_hat_vec.empty() || D_hat_vec.empty())
    {
        return Eigen::VectorXd(0);
    }

    int node_num = graph.size();
    A = Eigen::MatrixXd(node_num, node_num);
    for (int i = 0; i < node_num; i++) {
        for (int j = 0; j < node_num; j++) {
            if (i == j) {
                double tem = 0.0;
                for (int k = 0; k < node_num; k++) {
                    tem = tem + graph[k][j] + graph[j][k];
                }
                tem = tem - (graph[i][j] + graph[j][i]);
                A.row(i)[j] = tem;
            }
            else if (i > j) {
                A.row(i)[j] = -(graph[i][j] + graph[j][i]);
                A.row(j)[i] = -(graph[i][j] + graph[j][i]);
            }
        }
    }

    b = Eigen::VectorXd(node_num);
    for (int j = 0; j < node_num; j++) {
        b[j] = 0.0;
        for (int i = 0; i < node_num; i++) {
            auto M_ij = lambda_u * (U_hat_vec[i] - U_hat_vec[j]) +
                lambda_d * (D_hat_vec[i] - D_hat_vec[j]);
            auto M_ji = -1.0 * M_ij;
            b[j] = b[j] + graph[i][j] * M_ji - graph[j][i] * M_ij;
        }
    }

    Eigen::MatrixXd A_1 = pseudoinverse(A);
    Eigen::VectorXd sm = A_1 * b;

    return sm;
}

Eigen::MatrixXd NodeSortProblemGenerator::pseudoinverse(Eigen::MatrixXd matrix) const
{
    auto svd = matrix.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto& singularValues = svd.singularValues();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
        singularValuesInv(matrix.cols(), matrix.rows());
    singularValuesInv.setZero();
    double  pinvtoler = m_arbitary_cofficient;
    for (unsigned int i = 0; i < singularValues.size(); ++i) {
        if (singularValues(i) > pinvtoler)
            singularValuesInv(i, i) = 1.0 / singularValues(i);
        else
            singularValuesInv(i, i) = 0.0;
    }
    Eigen::MatrixXd pinvmat =
        svd.matrixV() * singularValuesInv * svd.matrixU().transpose();

    return pinvmat;
}

QPANDA_END