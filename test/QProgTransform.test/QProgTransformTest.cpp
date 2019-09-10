#include <map>
#include "QPanda.h"
#include <algorithm>  
#include "gtest/gtest.h"
#include "Core/QPanda.h"
#include "Core/Utilities/QNodeDeepCopy.h"
#include "Core/Utilities/QProgToDAG/GraphMatch.h"
#include "ThirdParty/uintwide/generic_template_uintwide_t.h"
#include "ThirdParty/Eigen/Dense"

using namespace std;
USING_QPANDA

TEST(MatrixXi, EigenTest)
{
    throw exception();
    Eigen::ArrayXXf a(2, 2);

    AdjacencyMatrix temp= AdjacencyMatrix::Zero(2,4);

    temp(0, 0) = 1;
    cout << temp << endl;

    cout << temp.row(0) << endl;
    cout << temp.row(1) << endl;

    cout << temp.rows() << endl;
    cout << temp.row(0).minCoeff() << endl;
    cout << temp.row(1).minCoeff() << endl;
    cout << temp.cols() << endl;
}

TEST(MatrixXi, Eigen)
{
    auto qvm = initQuantumMachine();
    auto q = qvm->allocateQubits(5);
    auto c = qvm->allocateCBits(5);

/*
    0----H-----RX-----Z--- -CZ---------------
                            |
    1----H-----X------H---------------------

    2----RY-----------------H---------------
                      |
    3-----------------CZ-----------RZ-------
               |            |
    4----H-----CZ-----H--- -CZ-----Y------RX--

    0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
    0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

*/
    auto prog = QProg();
    prog << H(q[0]) << H(q[1]) << RY(q[2], PI / 2) << H(q[4])
         << RX(q[0], PI / 2) << X(q[1]) << CZ(q[3], q[4])
         << Z(q[0]) << H(q[1]) << CZ(q[2], q[3]) << H(q[4])
         << CZ(q[1], q[0]) << H(q[2]) << CZ(q[3], q[4])
         << RZ(q[3], PI / 2) << Y(q[4])
         << RX(q[4], PI / 2);

    //std::cout << transformQProgToOriginIR(prog, qvm) << endl << endl;

    auto query_cir = QCircuit();
    query_cir << H(q[4]) << CZ(q[3], q[4]) << H(q[4]);
    //query_cir << H(q[4]);

    auto replace_cir = QCircuit();
    //replace_cir << T(q[4]) << S(q[4]) << Z1(q[4]);
    replace_cir << RY(q[4],-PI/2) << CZ(q[3], q[4]) << RY(q[4], PI / 2);

    QNodeMatch dag_match;
    TopologincalSequence graph_seq;
    dag_match.getMainGraphSequence(prog, graph_seq);

    TopologincalSequence query_seq;
    dag_match.getQueryGraphSequence(query_cir, query_seq);

    MatchVector result;
    cout << dag_match.graphQuery(graph_seq, query_seq, result) << endl;

    for (auto val : result)
    {
        cout << "match : " << endl;
        for (auto i = 0; i < val.size(); ++i)
        {
            cout << "layer " << i << ": ";
            for (auto node : val[i])
            {
                cout << node.m_node_type << "(" << node.m_vertex_num << ")" << " ";
            }
            cout << endl;
        }
    }

    QProg update_prog;
    dag_match.graphReplace(query_cir, replace_cir, result, graph_seq, update_prog, qvm);

    std::cout << std::endl << transformQProgToOriginIR(update_prog, qvm);

    qvm->directlyRun(prog);
    auto result1 = qvm->getQState();

    qvm->directlyRun(update_prog);
    auto result2 = qvm->getQState();

    cout << endl;
    for (int i = 0; i < 31; ++i)
    {
        cout << result1[i] << " £º" << result2[i] << endl;
    }

    getchar();
}


TEST(QNodeDeepCopy, deepCopy)
{
    throw exception();

    auto machine = initQuantumMachine(QMachineType::CPU);
    auto q = machine->allocateQubits(20);
    auto c = machine->allocateCBits(20);

    auto prog = QProg();
    auto cir = QCircuit();
    auto cir1 = QCircuit();


    cir << Y(q[2]) << H(q[2])<<CNOT(q[0],q[1])<<cir1;


    auto while_prog= CreateWhileProg(c[1], &cir);

    auto cprpg = ClassicalProg(c[0]);

    auto me = Measure(q[1], c[1]); 
       
    prog << cprpg << me << while_prog;

    std::cout << transformQProgToOriginIR(prog, machine) << endl;

    auto temp = deepCopy(prog);
    prog.clear();

    std::cout << transformQProgToOriginIR(temp, machine) << endl;
    std::cout << transformQProgToOriginIR(prog, machine) << endl;

    destroyQuantumMachine(machine);
    getchar();
}

TEST(QProgTransform, QRunesToQProg)
{
    throw exception();
    auto qvm = initQuantumMachine();
    auto q = qvm->allocateQubits(5);
    //qvm->allocateQubits(2);
    auto prog = CreateEmptyQProg();

    auto h1 = H(q[0]);
    auto h2 = H(q[0]);

    transformQRunesToQProg("D:\\w1", prog, qvm);

    cout << transformQProgToOriginIR(prog,qvm) << endl;

    destroyQuantumMachine(qvm);

    getchar();
}

int main(int argc, char **argv)
{
    testing::GTEST_FLAG(catch_exceptions) = 1;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}