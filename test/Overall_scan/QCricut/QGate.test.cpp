#if 0
#include "QPanda.h"
#include "gtest/gtest.h"
#include "Core/Utilities/Tools/QStatMatrix.h"

//using namespace std;
USING_QPANDA


TEST(QGateInfaceTest,test) {
    auto qvm = initQuantumMachine();
    auto qubits = qvm->allocateQubits(4);
    auto cbits = qvm->allocateCBits(4);
    QProg prog;
    QCircuit circuit;
    QVec qvec{ qubits[0], qubits[1] };
    
    auto HGate =  H(qubits[1]);
    auto XGate = X(qubits[1]);
  

    HGate = QGate(HGate);
    XGate = QGate(dynamic_pointer_cast<AbstractQGateNode>(XGate.getImplementationPtr()));

    circuit.pushBackNode(dynamic_pointer_cast<QNode>(HGate.getImplementationPtr()));
    circuit << XGate;

 
    EXPECT_EQ( 2 ,prog.getNodeType());
    EXPECT_EQ(1, circuit.getNodeType());
    EXPECT_EQ(0, XGate.getNodeType());


    int before = qubits.size();
    int num = HGate.getQuBitVector(qubits);
    int after = qubits.size();
    EXPECT_EQ(num , after - before);


    auto XCopy = XGate.control(qvec);  
    //cout << "qvec size: " << qvec.size() << endl;
    EXPECT_EQ(qvec.size(), XCopy.getControlQubitNum());

    XGate.setControl(qubits);
    EXPECT_EQ(qubits.size(), XGate.getControlQubitNum());
    EXPECT_FALSE( XGate.isDagger());
   
    XGate.clear_control();
    EXPECT_EQ(0, XGate.getControlQubitNum());

   
    HGate.setDagger(false);                      
    EXPECT_FALSE(HGate.isDagger());              

    auto HCopy = HGate.dagger();
    EXPECT_TRUE( HCopy.isDagger());

    HGate.setDagger(true);                      
    EXPECT_TRUE(HGate.isDagger());

}

TEST(QGateNodeFactorytest,test) {

    auto qvm = initQuantumMachine();
    auto qubits = qvm->allocateQubits(4);

    auto XGate = QGateNodeFactory::getInstance()->getGateNode("X", qubits);
    EXPECT_EQ(0, XGate.getNodeType());
    EXPECT_FALSE( XGate.isDagger() );
}

/// RxxRyyRzzRzxGate.cpp
TEST(RxxRyyRzzRzxGate, test) {
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
    ///cout << prog1 << endl;
    //auto prog_text = convert_qprog_to_originir(prog1, &qvm);
    // << prog_text << endl;
    //auto ir_prog = convert_originir_string_to_qprog(prog_text, &qvm);
    //cout << ir_prog << endl;
    /*auto result = qvm.probRunDict(prog, q);
    auto result1 = qvm.probRunDict(prog1, q);*/
    //std::cout << "QOracle run result: " << std::endl;

    /*for (auto res : result)
    {
        cout << res.first << ", " << res.second << endl;
    }

    std::cout << "ryy gate run result: " << std::endl;
    for (auto res : result1)
    {
        cout << res.first << ", " << res.second << endl;
    }*/


}

TEST(QGateOverload, test)
{
    auto machine = CPUQVM();
    machine.init();

    auto q = machine.qAllocMany(4);
    auto c = machine.cAllocMany(4);


    //mul
    auto h_gate = H(q[0]);
    auto rx_gate = RX(q[0], 6);
    auto rz_gate = RZ(q[0], 9).dagger();

    auto plus_0 = h_gate * rx_gate * rz_gate;
    auto exp_3 = rx_gate.exp(3);

    auto matrix_0 = get_unitary(QProg(h_gate));
    auto matrix_1 = get_unitary(QProg(rx_gate));
    auto matrix_2 = get_unitary(QProg(rz_gate));
    auto matrix_4 = get_unitary(QProg(exp_3));

    //auto test_matrix_0 = matrix_0 * matrix_1 * matrix_2;
    auto test_matrix_0 = matrix_2 * matrix_1 * matrix_0;

    std::cout << matrix_4 << std::endl;

    auto circuit = QCircuit();
    circuit << h_gate << rx_gate << rz_gate;
    auto test_matrix_1 = get_unitary(QProg(circuit));

    auto circuit_exp = QCircuit();
    circuit_exp << rx_gate << rx_gate << rx_gate;
    auto test_matrix_2 = get_unitary(QProg(circuit_exp));

    std::cout << test_matrix_1 << std::endl;
    std::cout << test_matrix_2 << std::endl;

    EXPECT_EQ(test_matrix_0, test_matrix_1);
    EXPECT_EQ(matrix_4, test_matrix_2);
}

#endif