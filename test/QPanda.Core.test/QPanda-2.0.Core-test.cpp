#include <iostream>
#include <limits>
#include "ClassicalCondition.test.h"
#include "Utilities/OriginCollection.h"

#include "QPanda.h"
USING_QPANDA
using namespace std;
TEST_F(ClassicalConditionTest, testClassicalConditionADD)
{    
    auto c1 = m_qvm->Allocate_CBit();
    auto c2 = m_qvm->Allocate_CBit();

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
    directlyRun(prog);
    ASSERT_EQ(c1.eval(),11);
    ASSERT_EQ(c2.eval(),83);
    
}

TEST_F(ClassicalConditionTest, testClassicalConditionSUB)
{    
    auto c1 = m_qvm->Allocate_CBit();
    auto c2 = m_qvm->Allocate_CBit();

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
    directlyRun(prog);
    ASSERT_EQ(c1.eval(),11);
    ASSERT_EQ(c2.eval(),-1);
}

TEST_F(ClassicalConditionTest, testClassicalConditionMUL)
{    
    auto c1 = m_qvm->Allocate_CBit();
    auto c2 = m_qvm->Allocate_CBit();

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
    directlyRun(prog);
    std::cout <<c2.eval()<<std::endl;
    ASSERT_EQ(c1.eval(),11);
    ASSERT_EQ(c2.eval(),5324000);
}

TEST_F(ClassicalConditionTest, testClassicalConditionDIV)
{    
    auto c1 = m_qvm->Allocate_CBit();
    auto c2 = m_qvm->Allocate_CBit();

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
    directlyRun(prog);

    ASSERT_EQ(c1.eval(),11);
    ASSERT_EQ(c2.eval(),2);
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

    init();
    auto qubits = qAllocMany(4);
    auto cbits = cAllocMany(4);
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
    auto quil = qProgToQuil(prog);
    std::cout << quil << std::endl;

    finalize();
    return;
}

TEST(NoiseMachineTest, test)
{
    NoiseQVM qvm;
    qvm.init();
    auto qvec = qvm.Allocate_Qubits(2);
    auto cvec = qvm.Allocate_CBits(2);
    auto prog = QProg();
    prog << X(qvec[0])
       << X(qvec[1])
       <<Measure(qvec[0],cvec[0])
       << Measure(qvec[1],cvec[1]);
    rapidjson::Document doc;
    doc.Parse("{}");
    auto &alloc = doc.GetAllocator();
    doc.AddMember("shots", 1000, alloc);
    auto result = qvm.runWithConfiguration(prog, cvec, doc);
    for (auto &aiter : result)
    {
        std::cout << aiter.first << " : " << aiter.second << endl;
    }

    qvm.finalize();
}
