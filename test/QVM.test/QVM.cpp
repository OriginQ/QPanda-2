#include "QPanda.h"
#include "gtest/gtest.h"
#include "time.h"
#include "Core/QuantumMachine/PartialAmplitudeQVM.h"
#include "Core/QuantumMachine/SingleAmplitudeQVM.h"
using namespace std;
USING_QPANDA

double getStateProb(complex<double> val)
{
    return val.real()*val.real() + val.imag()*val.imag();
}


TEST(QVM, TotalAmplitudeQVM)
{
    throw exception();
    auto machine = initQuantumMachine(CPU_SINGLE_THREAD);
    auto qlist = machine->allocateQubits(10);
    auto clist = machine->allocateCBits(10);

    auto prog = QProg();
    for_each(qlist.begin(), qlist.end(), [&](Qubit *val) { prog << H(val); });
    prog << CZ(qlist[1], qlist[5])
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
         << CR(qlist[2], qlist[7], PI / 2);


    machine->directlyRun(prog);

    auto res = machine->getQState();
    machine->finalize();

    for (int i = 0; i < 12; ++i)
    {
        cout << i << " : " << res[i] << endl;
    }

    cout << "--------------" << endl;

}

TEST(QVM, PartialAmplitudeQVM)
{
    throw exception();
    auto machine = new PartialAmplitudeQVM();
    machine->init();
    //auto qlist = machine->allocateQubits(10);
    //auto clist = machine->allocateCBits(10);

    //auto prog = QProg();
    //for_each(qlist.begin(), qlist.end(), [&](Qubit *val) { prog << H(val); });
    //prog << CZ(qlist[1], qlist[5])
    //    << CZ(qlist[3], qlist[7])
    //    << CZ(qlist[0], qlist[4])
    //    << RZ(qlist[7], PI / 4)
    //    << RX(qlist[5], PI / 4)
    //    << RX(qlist[4], PI / 4)
    //    << RY(qlist[3], PI / 4)
    //    << CZ(qlist[2], qlist[6])
    //    << RZ(qlist[3], PI / 4)
    //    << RZ(qlist[8], PI / 4)
    //    << CZ(qlist[9], qlist[5])
    //    << RY(qlist[2], PI / 4)
    //    << RZ(qlist[9], PI / 4)
    //    << CR(qlist[2], qlist[7], PI / 2);
    //machine->run(prog);
    
    machine->run("C:\\Users\\QuantumBYLZ061902\\Desktop\\QRunes生成器\\QRunes70_30.txt");


    /*Test getQStat*/
    /*auto res0 = machine->getQStat();
    cout << res0.find("0000000000")->first << " : "
         << res0.find("0000000000")->second << endl;
    cout << res0.find("0000000001")->first << " : "
         << res0.find("0000000001")->second << endl;
    cout << res0.find("0000000010")->first << " : "
         << res0.find("0000000010")->second << endl;*/
    //cout << res.find("0000000011")->first << " : "
    //     << res.find("0000000011")->second << endl;
    //cout << res.find("0000000100")->first << " : "
    //     << res.find("0000000100")->second << endl;
    //cout << res.find("0000000101")->first << " : "
    //     << res.find("0000000101")->second << endl;

    //cout << "---------------------" << endl;

    //auto res1 = machine->PMeasure("8");
    //for (auto val : res1)
    //{
    //    std::cout << val.first << " : " << val.second << endl;
    //}

    //cout << "---------------------" << endl;

    //QVec qv = { qlist[1],qlist[2],qlist[3] ,qlist[4] ,qlist[5] ,qlist[6] ,qlist[7] ,qlist[8],qlist[9] };
    //auto res2 = machine->PMeasure(qv, "8");
    //for (auto val : res2)
    //{
    //    std::cout << val.first << " : " << val.second << endl;
    //}

  /*  cout << "---------------------" << endl;

   auto res3 = machine->getProbDict(qlist, "8");
   for (auto val : res3)
   {
       std::cout << val.first << " : " << val.second << endl;
   }*/

    //cout << "---------------------" << endl;

    //auto res4 = machine->getProbDict(qv, "8");
    //for (auto val : res4)
    //{
    //    std::cout << val.first << " : " << val.second << endl;
    //}

    //cout << "---------------------" << endl;

    //cout << machine->PMeasure_bin_index("0000000100") << endl;
    //cout << machine->PMeasure_dec_index("0") << endl;
    //cout << "---------------------" << endl;

    //std::vector<std::string> set = { "0000000000","0000000001","0000000100" };
    //auto res = machine->PMeasureSubSet(prog, set);


    //for (auto val : res)
    //{
    //    std::cout << val.first << " : " << val.second << endl;
    //}


    machine->finalize();
    getchar();
}


TEST(QVM, SingleAmplitudeQVM)
{
    throw exception();

    auto machine = new SingleAmplitudeQVM();
    machine->init();
    auto qlist = machine->allocateQubits(10);
    auto clist = machine->allocateCBits(10);

    auto prog = QProg();
    for_each(qlist.begin(), qlist.end(), [&](Qubit *val) { prog << H(val); });
    prog << CZ(qlist[1], qlist[5])
        << CZ(qlist[3], qlist[5])
        << CZ(qlist[2], qlist[4])
        << CZ(qlist[3], qlist[7])
        << CZ(qlist[0], qlist[4])
        << RY(qlist[7], PI / 2)
        << RX(qlist[8], PI / 2)
        << RX(qlist[9], PI / 2)
        << CR(qlist[0], qlist[1], PI)
        << CR(qlist[2], qlist[3], PI)
        << RY(qlist[4], PI / 2)
        << RZ(qlist[5], PI / 4)
        << RX(qlist[6], PI / 2)
        << RZ(qlist[7], PI / 4)
        << CR(qlist[8], qlist[9], PI)
        << CR(qlist[1], qlist[2], PI)
        << RY(qlist[3], PI / 2)
        << RX(qlist[4], PI / 2)
        << RX(qlist[5], PI / 2)
        << CR(qlist[9], qlist[1], PI)
        << RY(qlist[1], PI / 2)
        << RY(qlist[2], PI / 2)
        << RZ(qlist[3], PI / 4)
        << CR(qlist[7], qlist[8], PI);

    machine->run(prog);

    cout << "-------getQStat()---------" << endl;

    auto res = machine->getQStat();
    cout << res.find("0000000000")->first << " : "
        << res.find("0000000000")->second << endl;
    cout << res.find("0000000001")->first << " : "
        << res.find("0000000001")->second << endl;
    cout << res.find("0000000010")->first << " : "
        << res.find("0000000010")->second << endl;
    cout << res.find("0000000011")->first << " : "
        << res.find("0000000011")->second << endl;


    cout << "-------PMeasure(select_max)-------" << endl;

    auto res1 = machine->PMeasure("8");
    for (auto val : res1)
    {
        std::cout << val.first << " : " << val.second << endl;
    }

    cout << "-----PMeasure(QVec,select_max)-------" << endl;

    QVec qv = { qlist[1],qlist[2],qlist[3] ,qlist[4] ,qlist[5] ,qlist[6] ,qlist[7] ,qlist[8],qlist[9] };
    auto res2 = machine->PMeasure(qv, "8");
    for (auto val : res2)
    {
        std::cout << val.first << " : " << val.second << endl;
    }

    cout << "----getProbDict(QVec,select_max)-----" << endl;

    auto res3 = machine->getProbDict(qlist, "8");
    for (auto val : res3)
    {
        std::cout << val.first << " : " << val.second << endl;
    }


    auto res4 = machine->getProbDict(qv, "8");
    for (auto val : res4)
    {
        std::cout << val.first << " : " << val.second << endl;
    }

    machine->finalize();
    getchar();
}

int main(int argc, char **argv) 
{
    testing::GTEST_FLAG(catch_exceptions) = 1;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
