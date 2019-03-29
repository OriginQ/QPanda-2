#include <map>
#include "QPanda.h"
#include <algorithm>  
#include "gtest/gtest.h"
using namespace std;
USING_QPANDA

TEST(QProgTransform, QASM)
{
    init();

    auto prog = CreateEmptyQProg();
    auto cir = CreateEmptyCircuit();

    auto q0 = qAlloc();
    auto q1 = qAlloc();
    auto q2 = qAlloc();
    auto c0 = cAlloc();

    auto qlist = qAllocMany(2);

    cir << Y(q2) << H(q2);
    cir.setControl(qlist);

    auto h1 = H(q1);
    h1.setDagger(true);

    prog << H(q1) << X(q2) << h1 << RX(q1, 2 / PI) << cir << CR(q1, q2, PI / 2);

    cout << qProgToQRunes(prog);

    finalize();
    getchar();
}

TEST(QProgTransform, QRunesToQprog)
{
    init();
    auto prog = CreateEmptyQProg();

    qRunesToQProg("D:\\QRunes", prog);

    cout << qProgToQRunes(prog) << endl;

    finalize();

    getchar();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
