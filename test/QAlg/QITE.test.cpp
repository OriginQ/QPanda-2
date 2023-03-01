#include "gtest/gtest.h"
#include "Components/NodeSortProblemGenerator/NodeSortProblemGenerator.h"
#include "Utilities/Tools/Utils.h"
#include "QAlg/QITE/QITE.h"
#include "Core/Utilities/UnitaryDecomposer/MatrixUtil.h"

USING_QPANDA
using namespace std;

#ifdef QITE

int test4GraphOfQITE()
{
    std::vector<std::vector<double>> node7graph{
        // A  B  C  D  E  F  G
          {0, 1 ,0 ,0, 0, 0, 0},
          {1, 0 ,1 ,0, 0, 0, 0},
          {0, 1 ,0 ,1, 1, 1, 0},
          {0, 0 ,1 ,0, 1, 0, 1},
          {0, 0 ,1 ,1, 0, 1, 1},
          {0, 0 ,1 ,0, 1, 0, 1},
          {0, 0 ,0 ,1, 1, 1, 0}
    };

    std::vector<std::vector<double>> node9graph{
        //   O, A, B, C, D, E, F, G, H
            {0, 1, 0, 0, 0, 0, 0, 0, 0},
            {1, 0, 1, 0, 0, 0, 0, 0, 0},
            {0, 1, 0, 1, 1, 1, 0, 0, 0},
            {0, 0, 1, 0, 0, 1, 1, 0, 0},
            {0, 0, 1, 0, 0, 1, 0, 1, 0},
            {0, 0, 1, 1, 1, 0, 1, 1, 1},
            {0, 0, 0, 1, 0, 1, 0, 0, 1},
            {0, 0, 0, 0, 1, 1, 0, 0, 1},
            {0, 0, 0, 0, 0, 1, 1, 1, 0}
    };

    std::vector<std::vector<double>> node10graph{
        //   0  1  2  3  4  5  6  7  8  9  
            {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
            {1, 1, 0, 1, 0, 0, 0, 0, 0, 0},
            {0, 0, 1, 0, 1, 1, 1, 0, 0, 0},
            {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 1, 0, 0, 0, 1, 0, 1},
            {0, 0, 0, 0, 0, 0, 1, 0, 1, 0},
            {0, 0, 0, 0, 0, 0, 0, 1, 0, 1},
            {0, 0, 0, 0, 0, 0, 1, 0, 1, 0}
    };

    std::vector<std::vector<double>> node12graph{
        // 1  2  3  4  5  6  7  8  9 10 11 12
          {0, 1 ,0 ,0, 0, 1, 0, 0, 0, 0, 0, 0}, //1
          {1, 0 ,1 ,0, 0, 0, 0, 0, 0, 0, 0, 0}, //2
          {0, 1 ,0 ,1, 0, 0, 0, 0, 0, 0, 0, 0}, //3
          {0, 0 ,1 ,0, 1, 0, 0, 0, 0, 1, 0, 1}, //4
          {0, 0 ,0 ,1, 0, 1, 0, 0, 0, 0, 0, 1}, //5
          {1, 0 ,0 ,0, 1, 0, 1, 0, 0, 0, 1, 1}, //6
          {0, 0 ,0 ,0, 0, 1, 0, 1, 0, 0, 0, 0}, //7
          {0, 0 ,0 ,0, 0, 0, 1, 0, 1, 0, 0, 0}, //8
          {0, 0 ,0 ,0, 0, 0, 0, 1, 0, 1, 0, 0}, //9
          {0, 0 ,0 ,1, 0, 0, 0, 0, 1, 0, 1, 1}, //10
          {0, 0 ,0 ,0, 0, 1, 0, 0, 0, 1, 0, 1}, //11
          {0, 0 ,0 ,1, 1, 1, 0, 0, 0, 1, 1, 0}  //12

    };

    std::vector< std::vector<std::vector<double>>> graph_vec;

    //graph_vec.push_back(node7graph);
    //graph_vec.push_back(node9graph);
    graph_vec.push_back(node10graph);
    graph_vec.push_back(node12graph);

    for (int g = 0; g < graph_vec.size(); g++)
    {
        auto graph = graph_vec[g];
        auto node_num = graph.size();

        std::cout << node_num << " graph" << std::endl;

        NodeSortProblemGenerator problem;
        problem.setProblemGraph(graph);
        problem.exec();
        auto ansatz_vec = problem.getAnsatz();

        size_t cnt_num = 50;
        size_t iter_num = 100;
        size_t upthrow_num = 1;
        //double delta_tau = 2;
        //std::vector<double> delta_taus = { 0.01, 1, 5, 10, 100 };
        std::vector<double> delta_taus = { 2.6 };
        QITE::UpdateMode update_mode = QITE::UpdateMode::GD_DIRECTION;

        srand((int)time(0));

        for (auto delta_tau : delta_taus)
        {
            for (auto cnt = 0; cnt < cnt_num; cnt++)
            {
                std::string log_filename = std::to_string(node_num) + "_num_" +
                    std::to_string(cnt) + "_tau_" + std::to_string(delta_tau);

                std::fstream fout;
                fout.open(log_filename + "_init_para.txt", std::ios::out);
                for (auto i = 0; i < ansatz_vec.size(); i++)
                {
                    ansatz_vec[i].theta = (rand() / double(RAND_MAX)) * 2 * PI;
                    fout << ansatz_vec[i].theta << std::endl;
                }
                fout.close();

                QITE qite;
                qite.setHamiltonian(problem.getHamiltonian());
                qite.setAnsatzGate(ansatz_vec);
                qite.setIterNum(iter_num);
                qite.setDeltaTau(delta_tau);
                qite.setUpthrowNum(upthrow_num);
                qite.setParaUpdateMode(update_mode);
                qite.setLogFile(log_filename);
                auto ret = qite.exec();
                if (ret != 0)
                {
                    return ret;
                }
                qite.getResult();
            }
        }
    }

    return 0;
}

template<class T>
bool lessCmp(std::pair<int, T> p1, std::pair<int, T> p2)
{
    return p1.second < p2.second;
}

template<class T>
std::vector<std::pair<int, T>> quickSort(const std::vector<T>& vec)
{
    std::vector<std::pair<int, T>> sort_vec;
    for (int i = 0; i < vec.size(); i++)
    {
        sort_vec.push_back(std::make_pair(i, vec[i]));
    }

    std::sort(sort_vec.begin(), sort_vec.end(), lessCmp<T>);

    return sort_vec;
}

void testAccurateResult()
{
    std::vector<std::vector<double>> node7graph{
        // A  B  C  D  E  F  G
          {0, 1 ,0 ,0, 0, 0, 0},
          {1, 0 ,1 ,0, 0, 0, 0},
          {0, 1 ,0 ,1, 1, 1, 0},
          {0, 0 ,1 ,0, 1, 0, 1},
          {0, 0 ,1 ,1, 0, 1, 1},
          {0, 0 ,1 ,0, 1, 0, 1},
          {0, 0 ,0 ,1, 1, 1, 0}
    };

    std::vector<std::vector<double>> node9graph{
        //   O, A, B, C, D, E, F, G, H
            {0, 1, 0, 0, 0, 0, 0, 0, 0},
            {1, 0, 1, 0, 0, 0, 0, 0, 0},
            {0, 1, 0, 1, 1, 1, 0, 0, 0},
            {0, 0, 1, 0, 0, 1, 1, 0, 0},
            {0, 0, 1, 0, 0, 1, 0, 1, 0},
            {0, 0, 1, 1, 1, 0, 1, 1, 1},
            {0, 0, 0, 1, 0, 1, 0, 0, 1},
            {0, 0, 0, 0, 1, 1, 0, 0, 1},
            {0, 0, 0, 0, 0, 1, 1, 1, 0}
    };

    std::vector<std::vector<double>> node10graph{
        //   0  1  2  3  4  5  6  7  8  9  
            {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
            {1, 1, 0, 1, 0, 0, 0, 0, 0, 0},
            {0, 0, 1, 0, 1, 1, 1, 0, 0, 0},
            {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 1, 0, 0, 0, 1, 0, 1},
            {0, 0, 0, 0, 0, 0, 1, 0, 1, 0},
            {0, 0, 0, 0, 0, 0, 0, 1, 0, 1},
            {0, 0, 0, 0, 0, 0, 1, 0, 1, 0}
    };

    std::vector<std::vector<double>> node12graph{
        // 1  2  3  4  5  6  7  8  9 10 11 12
          {0, 1 ,0 ,0, 0, 1, 0, 0, 0, 0, 0, 0}, //1
          {1, 0 ,1 ,0, 0, 0, 0, 0, 0, 0, 0, 0}, //2
          {0, 1 ,0 ,1, 0, 0, 0, 0, 0, 0, 0, 0}, //3
          {0, 0 ,1 ,0, 1, 0, 0, 0, 0, 1, 0, 1}, //4
          {0, 0 ,0 ,1, 0, 1, 0, 0, 0, 0, 0, 1}, //5
          {1, 0 ,0 ,0, 1, 0, 1, 0, 0, 0, 1, 1}, //6
          {0, 0 ,0 ,0, 0, 1, 0, 1, 0, 0, 0, 0}, //7
          {0, 0 ,0 ,0, 0, 0, 1, 0, 1, 0, 0, 0}, //8
          {0, 0 ,0 ,0, 0, 0, 0, 1, 0, 1, 0, 0}, //9
          {0, 0 ,0 ,1, 0, 0, 0, 0, 1, 0, 1, 1}, //10
          {0, 0 ,0 ,0, 0, 1, 0, 0, 0, 1, 0, 1}, //11
          {0, 0 ,0 ,1, 1, 1, 0, 0, 0, 1, 1, 0}  //12

    };

    std::vector< std::vector<std::vector<double>>> graph_vec;

    graph_vec.push_back(node7graph);
    graph_vec.push_back(node9graph);
    graph_vec.push_back(node10graph);
    graph_vec.push_back(node12graph);

    for (int g = 0; g < graph_vec.size(); g++)
    {
        auto graph = graph_vec[g];
        auto node_num = graph.size();

        std::cout << node_num << " graph" << std::endl;

        NodeSortProblemGenerator problem;
        problem.setProblemGraph(graph);
        problem.exec();

        auto vec = transPauliOperatorToVec(problem.getHamiltonian());
        int min_index = 0;
        auto min = vec[min_index];
        for (int i = 0; i < vec.size(); i++)
        {
            if (min > vec[i])
            {
                min = vec[i];
                min_index = i;
            }
        }
        auto sort_result = quickSort(vec);
        for (int i = 0; i < sort_result.size(); i++)
        {
            if (i >= 10)
            {
                break;
            }
            std::cout << dec2bin(sort_result[i].first, graph.size()) << "(" << sort_result[i].first << "), " << sort_result[i].second << std::endl;
        }

        std::cout << "min_index:" << min_index << ",  value:" << vec[min_index] << std::endl;
    }
}

void testQiteInterface()
{
    std::vector<std::vector<double>> graph{
        // A  B  C  D  E  F  G
          {0, 1 ,0 ,0, 0, 0, 0},
          {1, 0 ,1 ,0, 0, 0, 0},
          {0, 1 ,0 ,1, 1, 1, 0},
          {0, 0 ,1 ,0, 1, 0, 1},
          {0, 0 ,1 ,1, 0, 1, 1},
          {0, 0 ,1 ,0, 1, 0, 1},
          {0, 0 ,0 ,1, 1, 1, 0}
    };

    NodeSortProblemGenerator problem;
    problem.setProblemGraph(graph);
    problem.exec();
    auto hamiltonina = problem.getHamiltonian();
    auto ansatz_vec = problem.getAnsatz();

    size_t cnt_num = 1;
    size_t iter_num = 100;
    size_t upthrow_num = 3;
    double delta_tau = 2.6;
    QITE::UpdateMode update_mode = QITE::UpdateMode::GD_DIRECTION;

    auto result = qite(hamiltonina, ansatz_vec, iter_num, "", update_mode, upthrow_num, delta_tau);

    /*for (auto& i : result)
    {
        if (i.second > 1e-3)
        {
            std::cout << i.first << "\t" << i.second << std::endl;
        }
    }*/
}
#endif // QITE

bool testToyDemo()
{
    QITE qite;
    qite.setHamiltonian(transVecToPauliOperator(std::vector<int>{1, 2, 3, 0}));

    std::vector<AnsatzGate> vec{
        {AnsatzGateType::AGT_RX, 1, 0.5},
        {AnsatzGateType::AGT_RY, 0, 0.5, 1}
    };
    qite.setAnsatzGate(vec);
    qite.setParaUpdateMode(QITE::UpdateMode::GD_DIRECTION);
    qite.setIterNum(1);
    qite.setUpthrowNum(1);
    qite.exec();
    //qite.getResult();
    //a = qite.getResult();
    if (qite.getResult().begin()->second < 0.9)
        return false;
    else 
        return true;
}

#include <EigenUnsupported/Eigen/KroneckerProduct>
void QITE_Test()
{
    auto qnumber = 4;
    auto time = 1/12; 

    auto strike_price = 100;
    auto market_rate = 0.0;
    auto volatility = 0.2;
    auto call_type = true;
    auto s_min = 50;
    auto s_max = 150;
    auto x_min = std::log(s_min);
    auto x_max = std::log(s_max);

    auto a = (double)1 / 2 - market_rate / (volatility * volatility);
    auto b = (double)-1 / 2 * a * a - market_rate / (volatility * volatility);

    auto states = 16;

    prob_vec t = { 0.618167, -2.668739, -0.559011, -0.355224, -1.516136, -1.153081, -0.695832, -2.56526, 2.060167,
         0.876335, 0.614217, 6.756333, 3.059823, -0.355224, -0.156125, 10.995574, 0.199099, 0.355224 };

     auto ansatz_vec = { AnsatzGate(AnsatzGateType::AGT_X, 3),
                         AnsatzGate(AnsatzGateType::AGT_H, 1),
                         AnsatzGate(AnsatzGateType::AGT_H, 2),
                         AnsatzGate(AnsatzGateType::AGT_H, 0),
                         AnsatzGate(AnsatzGateType::AGT_RY, 3, 3.142),
                         AnsatzGate(AnsatzGateType::AGT_RY, 2, 4.173),
                         AnsatzGate(AnsatzGateType::AGT_RY, 1, 1.392),
                         AnsatzGate(AnsatzGateType::AGT_RY, 0, 3.713),  
                         AnsatzGate(AnsatzGateType::AGT_RY, 2, 2.399, 3),
                         AnsatzGate(AnsatzGateType::AGT_RY, 1, 0.935, 2),
                         AnsatzGate(AnsatzGateType::AGT_RY, 0, 2.196, 1),
                         AnsatzGate(AnsatzGateType::AGT_RY, 3, 5.014),
                         AnsatzGate(AnsatzGateType::AGT_RY, 2, 2.736),
                         AnsatzGate(AnsatzGateType::AGT_RY, 1, 1.477),
                         AnsatzGate(AnsatzGateType::AGT_RY, 0, 4.472),
                         AnsatzGate(AnsatzGateType::AGT_RY, 2, 3.415, 3),
                         AnsatzGate(AnsatzGateType::AGT_RY, 1, 6.283, 2),
                         AnsatzGate(AnsatzGateType::AGT_RY, 0, 4.244, 1),
                         AnsatzGate(AnsatzGateType::AGT_RY, 3, 4.711),
                         AnsatzGate(AnsatzGateType::AGT_RY, 2, 0.717),
                         AnsatzGate(AnsatzGateType::AGT_RY, 1, 1.741),
                         AnsatzGate(AnsatzGateType::AGT_RY, 0, 1.158),
                         AnsatzGate(AnsatzGateType::AGT_RY, 2, 2.531, 3),
                         AnsatzGate(AnsatzGateType::AGT_RY, 1, 5.705, 2),
                         AnsatzGate(AnsatzGateType::AGT_RY, 0, 3.525, 1),
                         AnsatzGate(AnsatzGateType::AGT_RY, 3, 4.582),
                         AnsatzGate(AnsatzGateType::AGT_RY, 2, 2.465),
                         AnsatzGate(AnsatzGateType::AGT_RY, 1, 0.098),
                         AnsatzGate(AnsatzGateType::AGT_RY, 0, 5.018) };

    auto iter_num = 50;
    auto delta_tau = (double)0.09 / 600;
    auto update_mode = QITE::UpdateMode::GD_VALUE;

    EigenMatrixX  h_hat = QMatrixXd::Zero(states, states);
    auto  delta_x = (x_max - x_min) / (states - 1);

    for (auto i = 0; i < 16; ++i)
    {
        if (i == 0 || i == (states - 1))
            h_hat(i, i) = -b;
        else
        {
            h_hat(i, i - 1) = (double)0.5 / (delta_x * delta_x);
            h_hat(i, i) = (double)-1 / (delta_x * delta_x);
            h_hat(i, i + 1) = (double)0.5 / (delta_x * delta_x);
        }
    }
    cout << h_hat << endl;

    auto qite = QITE();
    qite.setAnsatzGate(ansatz_vec);

    auto machine = initQuantumMachine(CPU);
     
    qite.setPauliMatrix(machine, h_hat);
    //qite.setHamiltonian(com);
    qite.setIterNum(iter_num);
    qite.setDeltaTau(delta_tau);
    qite.setParaUpdateMode(update_mode);
    auto ret = qite.exec(false);
    //auto result = qite.get_exec_result(false, false);
    auto result = qite.get_all_exec_result(false, false);
}

void decompose_test()
{
    EigenMatrixX  h_hat = QMatrixXd::Zero(8, 8);
    auto  delta_x = 9;

    for (auto i = 0; i < 8; ++i)
    {
        if (i == 0 || i == 7)
            h_hat(i, i) = -0.8;
        else
        {
            h_hat(i, i - 1) = (double)0.5 / (delta_x * delta_x);
            h_hat(i, i) = (double)-1 / (delta_x * delta_x);
            h_hat(i, i + 1) = (double)0.5 / (delta_x * delta_x);
        }
    }

    cout << h_hat << endl;

    auto machine = initQuantumMachine();
    machine->init();

    PualiOperatorLinearCombination opt;
    matrix_decompose_paulis(machine, h_hat, opt);

    QMatrixXcd mat1 = QMatrixXcd::Zero(8, 8);
    for (auto& val : opt)
    {
        cout << "pauli circuit coefficient : " << val.first << endl;
        cout << "pauli circuit : " << endl;
        cout << val.second << endl;
        auto matrix = getCircuitMatrix(val.second, true);

        mat1 += (val.first * QStat_to_Eigen(matrix));
    }

    std::cout << h_hat - mat1.real() << endl;
    return;
}

void decompose_test1()
{
    EigenMatrixX matrix = EigenMatrixX::Identity(4,4);
    matrix(3) = 3.24;
    matrix(1) = 9.24;
    matrix(0) = 6.2;

    auto machine = initQuantumMachine();
    machine->init();
    auto q = machine->qAllocMany(4);

    PualiOperatorLinearCombination opt;
    matrix_decompose_paulis({q[2],q[3]}, matrix, opt);

    auto a = pauli_combination_replace(opt, machine, "Z", "S");
    for (auto val:a)
    {
        cout << val.first << endl;
        cout << val.second << endl;
    }

    return;
}


TEST(QITE, test1)
{
    decompose_test1();
    bool test_val = false;
    //testAccurateResult();
    test_val = testToyDemo();
}