#include <iostream>
#include "QPanda.h"
#include <iostream>
#include "gtest/gtest.h"
#include "QPanda.h"
#include <map>
#include <cstdlib>
#include <sstream>
#include <string>
#include <algorithm>  

using namespace std;
USING_QPANDA

const string sQRunesPath("D:\\QRunes");

TEST(QProgTransform, QASM)
{

    init();

    auto prog = CreateEmptyQProg();

    auto cir = CreateEmptyCircuit();

    auto q0 = qAlloc();
    auto q1 = qAlloc();
    auto q2 = qAlloc();
    auto c0 = cAlloc();

    cir.setDagger(true);

    auto h1 = H(q1);
    h1.setDagger(true);


    prog << CZ(q0, q2) << H(q1) << CZ(q1, q2) << X(q2) << h1<<RX(q1,2/PI);

    cout << qProgToQASM(prog);

    finalize();


    getchar();
}




TEST(QProgTransform, QRunesToQprog)
{
    exit(0);

    init();

    auto prog = CreateEmptyQProg();

    qRunesToQProg("D:\\QRunes", prog);

    cout << qProgToQRunes(prog);

    finalize();

    getchar();
}


TEST(QProgTransform, QRunes)
{

    exit(0);

    init();

    auto prog = CreateEmptyQProg();
    auto cir = CreateEmptyCircuit();


    auto qubit = qAllocMany(6);
    auto cbit = cAllocMany(2);

    auto q0 = qAlloc();

    getchar();
}


