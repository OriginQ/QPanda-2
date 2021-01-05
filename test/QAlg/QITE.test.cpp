#include "gtest/gtest.h"
#include "Components/NodeSortProblemGenerator/NodeSortProblemGenerator.h"
#include "Utilities/Tools/Utils.h"
#include "QAlg/QITE/QITE.h"

USING_QPANDA
using namespace std;

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

    for (auto& i : result)
    {
        if (i.second > 1e-3)
        {
            std::cout << i.first << "\t" << i.second << std::endl;
        }
    }
}

void testToyDemo()
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
    qite.getResult();
}

TEST(QITE, test1)
{
    //EXPECT_EQ(test4GraphOfQITE(), 0);
    //testAccurateResult();
    testToyDemo();
    //testQiteInterface();
}