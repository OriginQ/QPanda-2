#include <map>
#include <cstdlib>
#include <sstream>
#include <string>
#include <complex>
#include <algorithm>
#include <regex>
#include <ctime>
#include "Core/Core.h"
#include "gtest/gtest.h"
#include "Core/QuantumCircuit/QNodeDeepCopy.h"
#include "Core/Utilities/Tools/MatrixDecomposition.h"
#include "Core/VirtualQuantumProcessor/MPSQVM/MPSQVM.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/GraphMatch.h"
#include "Core/Utilities/Tools/JsonConfigParam.h"
#include "Core/Utilities/Tools/QCircuitGenerator.h"


#include "Extensions/Extensions.h"
#ifdef USE_EXTENSION

using namespace std;
USING_QPANDA

static bool test_vf1_0()
{
    auto qvm = CPUQVM();
    qvm.init();
    auto q = qvm.qAllocMany(2);
    auto c = qvm.cAllocMany(2);
    auto circuit = QCircuit();
    auto prog = QProg();
    auto circuit1 = QCircuit();
    auto prog1 = QProg();
    const double cost = std::cos(0.5 * (PI / 2));
    const double sint = std::sin(0.5 * (PI / 2));
    // RYY GATE MATRIX
    QStat ryy_matrix =
    {
        qcomplex_t(cost, 0), qcomplex_t(0,0), qcomplex_t(0,0), qcomplex_t(0,sint),
        qcomplex_t(0,0), qcomplex_t(cost, 0), qcomplex_t(0,-sint), qcomplex_t(0,0),
        qcomplex_t(0,0), qcomplex_t(0,-sint), qcomplex_t(cost, 0), qcomplex_t(0,0),
        qcomplex_t(0,sint), qcomplex_t(0,0), qcomplex_t(0,0), qcomplex_t(cost, 0)
    };
    // RXX GATE MATRIX
    QStat rxx_matrix =
    {
        qcomplex_t(cost, 0), qcomplex_t(0,0), qcomplex_t(0,0), qcomplex_t(0,-sint),
        qcomplex_t(0,0), qcomplex_t(cost, 0), qcomplex_t(0,-sint), qcomplex_t(0,0),
        qcomplex_t(0,0), qcomplex_t(0,-sint), qcomplex_t(cost, 0), qcomplex_t(0,0),
        qcomplex_t(0,-sint), qcomplex_t(0,0), qcomplex_t(0,0), qcomplex_t(cost, 0)
    };
    const qcomplex_t i(0., 1.);
    const qcomplex_t exp_p = std::exp(i * 0.5 * (PI / 2));
    const qcomplex_t exp_m = std::exp(-i * 0.5 * (PI / 2));
    // RZZ GATE MATRIX
    QStat rzz_matrix =
    {
        exp_p, qcomplex_t(0,0), qcomplex_t(0,0), qcomplex_t(0,0),
        qcomplex_t(0,0), exp_m, qcomplex_t(0,0), qcomplex_t(0,0),
        qcomplex_t(0,0), qcomplex_t(0,0), exp_m, qcomplex_t(0,0),
        qcomplex_t(0,0), qcomplex_t(0,0), qcomplex_t(0,0), exp_p
    };
    // RZX GATE MATRIX
    QStat rzx_matrix =
    {
        qcomplex_t(cost, 0), qcomplex_t(0,0), qcomplex_t(0,-sint), qcomplex_t(0,0),
        qcomplex_t(0,0), qcomplex_t(cost, 0), qcomplex_t(0,0), qcomplex_t(0,sint),
        qcomplex_t(0,-sint), qcomplex_t(0,0), qcomplex_t(cost, 0), qcomplex_t(0,0),
        qcomplex_t(0,0), qcomplex_t(0,sint), qcomplex_t(0,0), qcomplex_t(cost, 0)
    };
    circuit << QOracle(q, rxx_matrix)
        << QOracle(q, ryy_matrix)
        << QOracle(q, rzz_matrix)
        << QOracle(q, rzx_matrix);
    prog << circuit;

    circuit1 << RXX(q[0], q[1], PI / 2)
        << RYY(q[0], q[1], PI / 2)
        << RZZ(q[0], q[1], PI / 2)
        << RZX(q[0], q[1], PI / 2);
    prog1 << circuit1;
    //prog1 << H(q[0]) << H(q[1]);
    cout << prog1 << endl;
    auto prog_text = convert_qprog_to_originir(prog1, &qvm);
    cout << prog_text << endl;
    auto ir_prog = convert_originir_string_to_qprog(prog_text, &qvm);
    cout << ir_prog << endl;
    /*auto result = qvm.probRunDict(prog, q);
    auto result1 = qvm.probRunDict(prog1, q);*/
    std::cout << "QOracle run result: " << std::endl;

    /*for (auto res : result)
    {
        cout << res.first << ", " << res.second << endl;
    }

    std::cout << "ryy gate run result: " << std::endl;
    for (auto res : result1)
    {
        cout << res.first << ", " << res.second << endl;
    }*/
    return true;
}

TEST(RxxRyyRzzRzxGate, test1)
{
    bool test_val = false;
    try
    {
        test_val = test_vf1_0();

    }
    catch (const std::exception& e)
    {
        cout << "Got a exception: " << e.what() << endl;
    }
    catch (...)
    {
        cout << "Got an unknow exception: " << endl;
    }

    //ASSERT_TRUE(test_val);

    //cout << "VF2 test over, press Enter to continue." << endl;
    //getchar();
}

#endif