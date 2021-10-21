#include "gtest/gtest.h"
#include "Core/Utilities/QProgInfo/Visualization/DrawLatex.h"
#include "QPanda.h"
#include <fstream>

USING_QPANDA

TEST(DrawLatex, output_latex_source)
{
    auto qvm = initQuantumMachine(QMachineType::CPU);
    auto q = qvm->qAllocMany(3);
    auto c = qvm->cAllocMany(3);
    QProg prog;
    QCircuit cir1, cir2;

    // 构建量子程序
    auto gate = S(q[1]);
    gate.setDagger(true);
    cir1 << H(q[0]).control(q[1]) << S(q[2]) << CNOT(q[0], q[1]) << CZ(q[1], q[2]) << gate;
    cir1.setDagger(true);
    cir2 << cir1 << H(q) << CU(1, 2, 3, 4, q[0], q[2])<< BARRIER(q) << S(q[2]) << CR(q[2], q[1], PI / 2) << SWAP(q[1], q[2]);
    cir2.setDagger(true);
    prog << cir2 << MeasureAll(q, c);

    // 输出latex文本
    std::string latex_str = draw_qprog(prog, PIC_TYPE::LATEX);
    std::fstream f("latex_out_test.tex", std::ios_base::out);
    f << latex_str;
    f.close();

    // 打印字符画
    std::cout << prog << std::endl;
    destroyQuantumMachine(qvm);
}