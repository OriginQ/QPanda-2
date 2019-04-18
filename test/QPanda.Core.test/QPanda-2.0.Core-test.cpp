#include <iostream>
#include <limits>
#include "Utilities/OriginCollection.h"
#include "Core/VirtualQuantumProcessor/NoiseQPU/NoiseModel.h"
#include "Core/VirtualQuantumProcessor/CPUImplQPU.h"
#include "QPanda.h"
#include "gtest/gtest.h"
USING_QPANDA
using namespace std;
TEST(ClassicalConditionTest, testClassicalConditionADD)
{    
    CPUQVM* m_qvm = new CPUQVM();
    m_qvm->init();
    auto c1 = m_qvm->allocateCBit();
    auto c2 = m_qvm->allocateCBit();

    c1.setValue(10);
    c2.setValue(20);
    ASSERT_EQ(c1.eval(),10);
    ASSERT_EQ(c2.eval(),20);
    // cc3 is classicalCondition
    auto cc3 = c1+c2;
    ASSERT_EQ(cc3.eval(), c1.eval()+ c2.eval());
    // cc4 is classicalCondition
    auto cc4 = c1+10;
    ASSERT_EQ(cc4.eval(), c1.eval()+10);

    // cc5 is classicalCondition
    auto cc5 = 10+c2;
    ASSERT_EQ(cc5.eval(), c2.eval()+10);

    auto prog = QProg();
    auto m_prog = CreateEmptyQProg();
    prog<<(c1=c1+1)<<(c2=c2+c1+cc3+cc4);
    auto qwhile = CreateWhileProg(c1<11,&prog);
    m_prog<<qwhile;
    m_qvm->directlyRun(prog);
    ASSERT_EQ(c1.eval(),11);
    ASSERT_EQ(c2.eval(),83);
    
    m_qvm->finalize();
    delete m_qvm;
}

TEST(ClassicalConditionTest, testClassicalConditionSUB)
{    
    CPUQVM* m_qvm = new CPUQVM();
    m_qvm->init();
    auto c1 = m_qvm->allocateCBit();
    auto c2 = m_qvm->allocateCBit();

    c1.setValue(10);
    c2.setValue(20);

    // cc3 is classicalCondition
    auto cc3 = c2-c1;
    ASSERT_EQ(cc3.eval(),10);
    // cc4 is classicalCondition
    auto cc4 = c1-10;
    ASSERT_EQ(cc4.eval(),0);
    // cc5 is classicalCondition
    auto cc5 = 20-c2;
    ASSERT_EQ(cc5.eval(),0);

    QProg prog;
    auto m_prog = CreateEmptyQProg();
    prog<<(c1=c1+1)<<(c2=c2-c1-cc3-cc4);
    auto qwhile = CreateWhileProg(c1<11,&prog);
    m_prog<<qwhile;
    m_qvm->directlyRun(prog);
    ASSERT_EQ(c1.eval(),11);
    ASSERT_EQ(c2.eval(),-1);
    m_qvm->finalize();
    delete m_qvm;
}

TEST(ClassicalConditionTest, testClassicalConditionMUL)
{    
    CPUQVM* m_qvm = new CPUQVM();
    m_qvm->init();
    auto c1 = m_qvm->allocateCBit();
    auto c2 = m_qvm->allocateCBit();

    c1.setValue(10);
    c2.setValue(20);

    // cc3 is classicalCondition
    auto cc3 = c2*c1;
    ASSERT_EQ(cc3.eval(),200);
    // cc4 is classicalCondition
    auto cc4 = c1*10;
    ASSERT_EQ(cc4.eval(),100);
    // cc5 is classicalCondition
    auto cc5 = 20*c2;
    ASSERT_EQ(cc5.eval(),400);

    auto prog = QProg();
    auto m_prog = CreateEmptyQProg();
    prog<<(c1=c1+1)<<(c2=c2*c1*cc3*cc4);
    auto qwhile = CreateWhileProg(c1<11,&prog);
    m_prog<<qwhile;
    m_qvm->directlyRun(prog);
    std::cout <<c2.eval()<<std::endl;
    ASSERT_EQ(c1.eval(),11);
    ASSERT_EQ(c2.eval(),5324000);
    m_qvm->finalize();
    delete m_qvm;
}

TEST(ClassicalConditionTest, testClassicalConditionDIV)
{    
    CPUQVM* m_qvm = new CPUQVM();
    m_qvm->init();
    auto c1 = m_qvm->allocateCBit();
    auto c2 = m_qvm->allocateCBit();

    c1.setValue(10);
    c2.setValue(20);

    // cc3 is classicalCondition
    auto cc3 = c2/c1;
    ASSERT_EQ(cc3.eval(),2);
    // cc4 is classicalCondition
    //ASSERT_THROW(auto cc4 = c1/0,std::invalid_argument);
    // cc5 is classicalCondition
    auto cc5 = 20/c2;
    ASSERT_EQ(cc5.eval(),1);

    auto prog = QProg();
    auto m_prog = CreateEmptyQProg();
    prog<<(c1=c1+1)<<(c2 = c2 + 2)<<(c2=c2/c1*(c1/11));
    auto qwhile = CreateWhileProg(c1<11,&prog);
    m_prog<<qwhile;
    m_qvm->directlyRun(prog);
    ASSERT_EQ(c1.eval(),11);
    ASSERT_EQ(c2.eval(),2);
    m_qvm->finalize();
    delete m_qvm;
}

TEST(QVecTest,test)
{
    init();
    auto prog = QProg();

    //vector<Qubit*> qvec;
    auto qvec = qAllocMany(5);
    auto cvec = cAllocMany(2);
    cvec[1].setValue(0);
    cvec[0].setValue(0);
    auto prog_in = QProg();
    prog_in<<(cvec[1]=cvec[1]+1)<<H(qvec[cvec[0]])<<(cvec[0]=cvec[0]+1);
    auto qwhile = CreateWhileProg(cvec[1]<5,&prog_in); 
    prog<<qwhile;
    directlyRun(prog);
    auto result =PMeasure_no_index(qvec);
    for(auto & aiter : result)
    {
        std::cout<<aiter<<std::endl;
    }
    finalize();
}

QCircuit testCIR(QVec qvec)
{
    auto cir = QCircuit();
    QVec qvec2;
    qvec2.push_back(qvec[0]);
    qvec2.push_back(qvec[1]);
    cir << X(qvec[2]).control(qvec2);
    return cir;
}

TEST(CirCuitTest, test)
{
    init();
    auto qvec = qAllocMany(6);
    auto prog = QProg();
    auto c1 = QCircuit();
    auto c2 = QCircuit();

    auto c3 = QCircuit();
    c1 << X(qvec[0])<<X(qvec[1]);
    c2 <<c1
        <<testCIR({ qvec[0],qvec[1],qvec[2] })
       << testCIR({ qvec[0],qvec[1],qvec[3] })
       << testCIR({ qvec[0],qvec[1],qvec[4] });
    c3 << c2;
    prog << c3;

    auto result = probRunDict(prog, qvec, -1);
    for (auto &aiter : result)
    {
        std::cout << aiter.first << " : " << aiter.second << endl;
    }

    finalize();
}

TEST(OriginCollectionTest,CreateTest)
{
    OriginCollection test("./test");
    test={"key","value","value2"};


    test.insertValue("444", 0.89898,0.454543);
    test.insertValue(555, 0.89898,"akjsdhjahd");

    std::vector<int > cc = { 1, 2, 3, 4 };
    test.insertValue(666, 0.89898, cc );
    
    std::vector<std::string> key_n = {"key","value2"};
    test.insertValue(key_n, "888", 122);
    test.insertValue(key_n, 6564465, 345);
    
    auto value =test.getValue("value");
    test.write();

    
    for(auto & aiter : value)
    {
        std::cout<<aiter<<std::endl;
    }

    OriginCollection test2("test2");
    test2 = { "key","value" };
    std::map<std::string, bool> a;
    a.insert(std::make_pair("c0", true));
    a.insert(std::make_pair("c1", true));
    a.insert(std::make_pair("c2", true));
    for (auto aiter : a)
    {
        test2.insertValue( aiter.first, aiter.second);
    }

    std::cout << test2.getJsonString() << std::endl;
}

TEST(QProgTransformQuil, QUIL)
{

    auto qvm = initQuantumMachine();
    auto qubits = qvm->allocateQubits(4);
    auto cbits = qvm->allocateCBits(4);
    QProg prog;
    QCircuit circuit;

    circuit << RX(qubits[0], PI / 6) << H(qubits[1]) << Y(qubits[2])
        << iSWAP(qubits[2], qubits[3]);
    prog << circuit << MeasureAll(qubits, cbits);
    auto result = runWithConfiguration(prog, cbits, 10000);
    for (auto &aiter : result)
    {
        std::cout << aiter.first << " : " << aiter.second << endl;
    }
    auto quil = transformQProgToQuil(prog,qvm);
    std::cout << quil << std::endl;
    destroyQuantumMachine(qvm);
    return;
}

//TEST(NoiseMachineTest, test)
//{
//    NoiseQVM qvm;
//    qvm.init();
//    auto qvec = qvm.allocateQubits(2);
//    auto cvec = qvm.allocateCBits(2);
//    auto prog = QProg();
//    prog << X(qvec[0])
//       << X(qvec[1])
//       <<Measure(qvec[0],cvec[0])
//       << Measure(qvec[1],cvec[1]);
//    rapidjson::Document doc;
//    doc.Parse("{}");
//    auto &alloc = doc.GetAllocator();
//    doc.AddMember("shots", 1000, alloc);
//    auto result = qvm.runWithConfiguration(prog, cvec, doc);
//    for (auto &aiter : result)
//    {
//        std::cout << aiter.first << " : " << aiter.second << endl;
//    }
//    auto state = qvm.getQState();
//    for (auto &aiter : state)
//    {
//        std::cout << aiter << endl;
//    }
//    qvm.finalize();
//}

double noisyRabiOscillation(double omega_d, double delta, size_t time, double sample = 1000)
{
    rapidjson::Document doc;
    doc.Parse("{}");
    auto & alloc = doc.GetAllocator();
    vector<std::vector<std::string>> m_gates_matrix = { {"X","Y","Z",
                        "T","S","H",
                        "RX","RY","RZ",
                        "U1" },
                       { "CNOT" } };
    for (auto a : m_gates_matrix[MetadataGateType::METADATA_SINGLE_GATE])
    {
        Value value(rapidjson::kArrayType);
        value.PushBack(NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR, alloc);
        value.PushBack(5.0, alloc);
        value.PushBack(2.0, alloc);
        value.PushBack(0.03, alloc);
        doc.AddMember(Value().SetString(a.c_str(), alloc).Move(), value, alloc);
    }
    for (auto a : m_gates_matrix[MetadataGateType::METADATA_DOUBLE_GATE])
    {
        Value value(rapidjson::kArrayType);
        value.PushBack(NOISE_MODEL::DEPHASING_KRAUS_OPERATOR, alloc);
        value.PushBack(0.0001, alloc);
        doc.AddMember(Value().SetString(a.c_str(), alloc).Move(), value, alloc);
    }
    NoiseQVM qvm;
    qvm.init(doc);
    auto qvec = qvm.allocateQubits(1);
    auto cvec = qvm.allocateCBits(1);
    double theta = asin(-omega_d / sqrt(omega_d*omega_d + delta * delta));
    double coef = -sqrt(omega_d*omega_d + delta * delta)*0.2;
    auto prog = QProg();
    prog << RY(qvec[0], theta);
    for (size_t i = 0; i < time; i++)
    {
        prog << RZ(qvec[0], coef);
    }
    prog << RY(qvec[0], -theta);
    prog << Measure(qvec[0], cvec[0]);
    rapidjson::Document doc1;
    doc1.Parse("{}");
    auto &alloc1 = doc1.GetAllocator();
    doc1.AddMember("shots", 1000, alloc1);
    auto result = qvm.runWithConfiguration(prog, cvec, doc1);
    qvm.finalize();
    return result["1"]*1.0 / sample;
}

TEST(NoiseMachineTest, rabiOscillation)
{
    std::vector<double> prob;
    for (size_t i = 0; i < 200; i++)
    {
        prob.push_back(noisyRabiOscillation(2, 0, i));
        std::cout << i << std::endl;
    }
    for (auto i : prob)
    {
        std::cout << i << ",";
    }
    getchar();
}
TEST(NoiseMachineTest, damping)
{
    NoiseQVM qvm;
    qvm.init();
    auto qvec = qvm.allocateQubits(1);
    auto cvec = qvm.allocateCBits(1);
    bool result;
    int times;
    std::vector<double> outcome;
    size_t trials = 200;
    size_t length = 200;
    rapidjson::Document doc;
    doc.Parse("{}");
    auto &alloc = doc.GetAllocator();
    doc.AddMember("shots", 1000, alloc);

    for (size_t i = 0; i < length; i += 10)
    {
        auto prog = QProg();
        prog << X(qvec[0]);
        for (size_t j = 0; j < i; j++)
        {
            prog << RX(qvec[0],0);
        }
        prog << Measure(qvec[0], cvec[0]);
        auto result = qvm.runWithConfiguration(prog, cvec, doc);
        prog.clear();
        std::cout << result["0"] << " , " ;
    }
    qvm.finalize();
}

