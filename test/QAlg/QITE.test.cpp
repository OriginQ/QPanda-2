#include "gtest/gtest.h"
#include "Components/NodeSortProblemGenerator/NodeSortProblemGenerator.h"
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
    //graph_vec.push_back(node12graph);

    for (int g = 0; g < graph_vec.size(); g++)
    {
        auto graph = graph_vec[g];
        auto node_num = graph.size();

        std::cout << node_num << " graph" << std::endl;

        NodeSortProblemGenerator problem;
        problem.setProblemGraph(graph);
        problem.exec();
        auto ansatz_vec = problem.getAnsatz();

        size_t cnt_num = 1;
        size_t iter_num = 100;
        size_t upthrow_num = 3;
        double delta_tau = 2.6;
        QITE::UpdateMode update_mode = QITE::UpdateMode::GD_DIRECTION;

        srand((int)time(0));
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
            auto ret =  qite.exec();
            if (ret != 0)
            {
                return ret;
            }
            qite.getResult();
        }
    }

    return 0;
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

TEST(QITE, test1)
{
    //EXPECT_EQ(test4GraphOfQITE(), 0);
    testQiteInterface();
}