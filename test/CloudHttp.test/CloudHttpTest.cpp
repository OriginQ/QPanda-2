#ifdef USE_CURL
#include "QPanda.h"
#include "gtest/gtest.h"
#include "QuantumMachine/QCloudMachine.h"
using namespace std;
using namespace rapidjson;
USING_QPANDA

TEST(CloudHttpTest, Post)
{
    auto QCM = new QCloudMachine();
    QCM->init();
    auto qlist = QCM->allocateQubits(10);
    auto clist = QCM->allocateCBits(10);
    auto qprog = QProg();
    auto qprog1 = QProg();

    for_each(qlist.begin(), qlist.end(), [&](Qubit *val) { qprog << H(val); });
    qprog << CZ(qlist[1], qlist[5])
        << CZ(qlist[3], qlist[7])
        << CZ(qlist[0], qlist[4])
        << RZ(qlist[7], PI / 4)
        << RX(qlist[5], PI / 4)
        << RX(qlist[4], PI / 4)
        << RY(qlist[3], PI / 4)
        << CZ(qlist[2], qlist[6])
        << RZ(qlist[3], PI / 4)
        << RZ(qlist[8], PI / 4)
        << CZ(qlist[9], qlist[5])
        << RY(qlist[2], PI / 4)
        << RZ(qlist[9], PI / 4)
        << CZ(qlist[2], qlist[3]);

    /******Test PMeasure***********/
    rapidjson::Document doc;
    doc.SetObject();
    rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();

    doc.AddMember("BackendType", QMachineType::CPU, allocator);
    doc.AddMember("token", "3CD107AEF1364924B9325305BF046FF3", allocator);
    std::cout << QCM->probRunDict(qprog, qlist, doc) << endl;;

    /******Test Measure***********/
    rapidjson::Document doc1;
    doc1.SetObject();
    rapidjson::Document::AllocatorType &allocator1 = doc1.GetAllocator();

    doc1.AddMember("BackendType", QMachineType::CPU, allocator1);
    doc1.AddMember("RepeatNum", 1000, allocator1);
    doc1.AddMember("token", "3CD107AEF1364924B9325305BF046FF3", allocator1);
    std::cout << QCM->runWithConfiguration(qprog, doc1) << endl;

    //QCM->getResult("1904261513203828");
    QCM->finalize();

    getchar();
}

#endif // USE_CURL


