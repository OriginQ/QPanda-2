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


    auto gate = S(q[1]);
    gate.setDagger(true);
    cir1 << H(q[0]).control(q[1]) << S(q[2]) << CNOT(q[0], q[1]) << CZ(q[1], q[2]) << gate;
    cir1.setDagger(true);
    cir2 << cir1 << H(q) << CU(1, 2, 3, 4, q[0], q[2]) << BARRIER(q) << S(q[2]) << CR(q[2], q[1], PI / 2) << SWAP(q[1], q[2]);
    cir2.setDagger(true);
    prog << cir2 << MeasureAll(q, c);

    QStat matrix;
    for (size_t i = 0; i < 4; i++)
    {
        for (size_t j = 0; j < 4; j++)
        {
            matrix.push_back({ 1 });
        }
    }
    QProg prog2;
    prog2 << cir2 << oracle(q, "Test") << QDouble(q[1], q[2], matrix) << Toffoli(q[0], q[1], q[2]) << MeasureAll(q, c);

  
    std::string latex_str = draw_qprog(prog, PIC_TYPE::LATEX);
    std::fstream f0("latex_out_test.tex", std::ios_base::out);
    f0 << latex_str;
    f0.close();

  
    latex_str = draw_qprog(prog2, PIC_TYPE::LATEX);
    std::fstream f1("latex_out_test2.tex", std::ios_base::out);
    f1 << latex_str;
    f1.close();

    latex_str = draw_qprog_with_clock(prog, PIC_TYPE::LATEX);
    std::fstream f2("latex_out_test_with_time.tex", std::ios_base::out);
    f2 << latex_str;
    f2.close();

    
    std::string text_pic = draw_qprog(prog);

#if defined(WIN32) || defined(_WIN32)
    text_pic = fit_to_gbk(text_pic);
    text_pic = Utf8ToGbkOnWin32(text_pic.c_str());
#endif
    std::cout << text_pic << std::endl;
    text_pic = draw_qprog(prog2);

#if defined(WIN32) || defined(_WIN32)
    text_pic = fit_to_gbk(text_pic);
    text_pic = Utf8ToGbkOnWin32(text_pic.c_str());
#endif
    std::cout << text_pic << std::endl;

    std::string text_pic_with_clock = draw_qprog_with_clock(prog);
#if defined(WIN32) || defined(_WIN32)
    text_pic_with_clock = fit_to_gbk(text_pic_with_clock);
    text_pic_with_clock = Utf8ToGbkOnWin32(text_pic_with_clock.c_str());
#endif
    std::cout << text_pic_with_clock << std::endl;
    destroyQuantumMachine(qvm);
}