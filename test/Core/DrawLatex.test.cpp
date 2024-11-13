#include "gtest/gtest.h"
#include "Core/Utilities/QProgInfo/Visualization/DrawLatex.h"
#include "QPanda.h"
#include <fstream>

USING_QPANDA

TEST(DrawLatex, output_latex_source)
{
    auto qvm = initQuantumMachine(QMachineType::CPU);
    auto q = qvm->qAllocMany(3);
    auto q1 = qvm->qAllocMany(3);
    auto c = qvm->cAllocMany(3);
    QProg prog;
    QCircuit cir1, cir2;

    auto gate = S(q[1]);
    gate.setDagger(true);
    cir1 << H(q[0]).control(q[1]) << S(q[2]) << CNOT(q[0], q[1]) << CZ(q[1], q[2]) << gate;
    cir1.setDagger(true);
    cir2 << cir1 << H(q) << BARRIER(q) << CU(1, 2, 3, 4, q[0], q[2]) << BARRIER(q[0]) << BARRIER(q1[1]) << S(q1[1]) << CR(q[2], q[1], PI / 2) << SWAP(q[1], q[2]);
    cir2.setDagger(true);
    prog << cir2 << MeasureAll(q, c);

    QStat matrix;
    for (size_t i = 0; i < 4; i++)
    {
        for (size_t j = 0; j < 4; j++)
        {
            matrix.push_back({1});
        }
    }
    QProg prog2;
    prog2 << cir2 << oracle(q, "Test") << QDouble(q[1], q[2], matrix) << Toffoli(q[0], q[1], q[2]) << MeasureAll(q, c);

    std::string latex_str = draw_qprog(prog, PIC_TYPE::LATEX);
    std::fstream f0("latex_out_test.tex", std::ios_base::out);
    f0 << latex_str;
    f0.close();

    latex_str = draw_qprog(prog2, PIC_TYPE::LATEX, true);
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

    auto latex_drawer = std::make_shared<LatexMatrix>();
    latex_drawer->set_logo("TestLogo");
    latex_drawer->insert_gate({2, 3}, {}, 3, LATEX_GATE_TYPE::GENERAL_GATE, "MAJ");
    latex_drawer->insert_barrier({ 1, 3 }, 3);
    latex_drawer->insert_gate({3, 4}, {}, 3, LATEX_GATE_TYPE::GENERAL_GATE, "MAJ");
    latex_drawer->insert_reset(4, 4);
    latex_drawer->set_label({{0, "a_0"}, {1, "a_1"}}, {{2, "c_2"}});
    latex_drawer->set_label({{0, "s_0"}, {1, "s_1"}}, {}, "", false);
    latex_str = latex_drawer->str();

    std::fstream f_test("test.tex", std::ios_base::out);
    f_test << latex_str;
    f_test.close();
}


TEST(DrawLatex, test1) {
    CPUQVM qm;
    qm.init();

    auto q = qm.qAllocMany(14);
    auto c = qm.cAllocMany(14);

    QProg prog;
    prog << H(q[0])
        << CNOT(q[0], q[1])
        << CNOT(q[1], q[2])
        << CNOT(q[2], q[3])
        << CNOT(q[3], q[4])
        << CNOT(q[4], q[5])
        << CNOT(q[5], q[6])
        << CNOT(q[6], q[7])
        << CNOT(q[7], q[8])
        << CNOT(q[8], q[9])
        << CNOT(q[9], q[10])
        << CNOT(q[10], q[11])
        << CNOT(q[11], q[12])
        << CNOT(q[12], q[13])
        << MeasureAll(q, c)
        ;
    std::cout << "text格式：" << endl;
    std::cout << prog << endl;
    std::cout << "latex格式：" << endl;
    auto latex = draw_qprog(prog, QPanda::PIC_TYPE::LATEX);
    std::cout << latex << endl;
}

TEST(DrawLatex, test2) {
    CPUQVM qm;
    qm.init();

    //auto q = qm.qAllocMany(14);
    //auto c = qm.cAllocMany(14);

    std::string ir = R"(QINIT 2
CREG 2
H q[1]
MEASURE q[1],c[1])";
    QProg prog = convert_originir_string_to_qprog(ir, &qm);

    std::cout << "text格式：" << endl;
    std::cout << prog << endl;
    std::cout << "latex格式：" << endl;
    //auto latex = draw_qprog(prog, QPanda::PIC_TYPE::LATEX, "latex.tex", true);
    auto latex = draw_qprog(prog, QPanda::PIC_TYPE::LATEX);
    std::cout << latex << endl;
}

bool test_draw_text_without_params() {
    std::cout << "test_draw_text_without_params()" << endl;
    CPUQVM qm;
    qm.init();
    auto qbits = qm.qAllocMany(3);
    QProg prog;
    prog << P(qbits[0], 3.14);
    std::cout << prog << endl;
    auto text_pic_str = draw_qprog(prog, PIC_TYPE::TEXT);

#if defined(WIN32) || defined(_WIN32)
        text_pic_str = fit_to_gbk(text_pic_str);
        text_pic_str = Utf8ToGbkOnWin32(text_pic_str.c_str());
#endif
    std::cout <<"res:\n"<< text_pic_str <<"\n"<< endl;

    std::string expected_str =
        "\n"
        "          ┌─┐ \n"//<blank><blank><blank><blank><blank><blank><blank><blank><blank><blank>┌─┐<blank>
        "q_0:  |0>─┤P├ \n"//q_0:<blank><blank>|0>─┤P├<blank>
        "          └─┘ \n"//<blank><blank><blank><blank><blank><blank><blank><blank><blank><blank>└─┘<blank>
        " c :   / ═\n"      //<blank>c<blank>:<blank><blank><blank>/<blank>═
        "          \n"       //<blank><blank><blank><blank><blank><blank><blank><blank><blank><blank>
        "\n";
    std::cout << std::to_string(expected_str == text_pic_str)<<"\n";
    return expected_str == text_pic_str;
}

bool test_draw_text_with_params() {
    std::cout << "test_draw_text_with_params()" << endl;
    CPUQVM qm;
    qm.init();
    auto qbits = qm.qAllocMany(3);
    QProg prog;
    prog << P(qbits[0], 3.14);
    
    std::cout << prog << endl;
    auto text_pic_str = draw_qprog(prog,PIC_TYPE::TEXT, false, true, 100, "",
        NodeIter(), NodeIter());
#if defined(WIN32) || defined(_WIN32)
    text_pic_str = fit_to_gbk(text_pic_str);
    text_pic_str = Utf8ToGbkOnWin32(text_pic_str.c_str());
#endif
    std::cout << "res:\n" << text_pic_str << "\n" << endl;
    std::string expected_res =
        "\n"
        "          ┌───────────┐ \n"       //< blank > <blank><blank><blank><blank><blank><blank><blank><blank><blank>┌───────────┐<blank>
        "q_0:  |0>─┤P(3.140000)├ \n"                 //q_0:<blank><blank> | 0 > ─┤P(3.140000)├<blank>
        "          └───────────┘ \n"       //<blank><blank><blank><blank><blank><blank><blank><blank><blank><blank>└───────────┘<blank>
        " c :   / ═\n"                                 //<blank>c<blank>:<blank><blank><blank> / <blank>═
        "          \n"                                  //<blank><blank><blank><blank><blank><blank><blank><blank><blank><blank>
        "\n"
        ;
    return expected_res == text_pic_str;
}

bool test_draw_latex_without_params() {
    std::cout << "test_draw_latex_without_params()" << endl;
    CPUQVM qm;
    qm.init();
    auto qbits = qm.qAllocMany(3);
    QProg prog;
    prog << P(qbits[0], 3.14);
    std::cout << prog << endl;
    auto latex = draw_qprog(prog, PIC_TYPE::LATEX);
#if defined(WIN32) || defined(_WIN32)
    latex = fit_to_gbk(latex);
    latex = Utf8ToGbkOnWin32(latex.c_str());
#endif
    std::cout << "res:\n" << latex << "\n" << endl;
    for (auto& ch : latex) {
        if (ch == ' ') {
            std::cout << "<blank>";
        }
        else if (ch == '\n') {
            std::cout << "<enter>\n";
        }
        else {
            std::cout << ch;
        }
    }
    std::string expected_str =
        ////
        "\\documentclass[border=2px]{standalone}\n"//\documentclass[border=2px]{standalone}<enter>
        "\n"//<enter>
        "\\usepackage[braket, qm]{qcircuit}\n"//\usepackage[braket,<blank>qm]{qcircuit}<enter>
        "\\usepackage{graphicx}\n"//\usepackage{graphicx}<enter>
        "\n"//<enter>
        "\\begin{document}\n"//\begin{document}<enter>
        "\\scalebox{1.0}{\n"//\scalebox{1.0}{<enter>
        "\\Qcircuit @C = 1.0em @R = 0.5em @!R{ \\\\\n"//\Qcircuit<blank>@C<blank>=<blank>1.0em<blank>@R<blank>=<blank>0.5em<blank>@!R{<blank>\\<enter>
        "\\nghost{q_{0}\\ket{0}}&\\lstick{q_{0}\\ket{0}}&\\gate{\\mathrm{P}}&\\rstick{}\\qw&\\nghost{}\\\\\n"//\nghost{q_{0}\ket{0}}&\lstick{q_{0}\ket{0}}&\gate{\mathrm{P}}&\rstick{}\qw&\nghost{}\\<enter>
        "\\\\ }}\n"//\\<blank>}}<enter>
        "\\end{document}\n"//\end{document}<enter>
        ;
    std::cout << std::to_string(expected_str == latex)<<"\n";
    return expected_str == latex;
}

bool test_draw_latex_with_params() {
    std::cout << "test_draw_latex_with_params()" << endl;
    CPUQVM qm;
    qm.init();
    auto qbits = qm.qAllocMany(3);
    QProg prog;
    prog << P(qbits[0], 3.14);

    std::cout << prog << endl;
    auto latex = draw_qprog(prog, PIC_TYPE::LATEX, false, true, 100, "",
        NodeIter(), NodeIter());
#if defined(WIN32) || defined(_WIN32)
    latex = fit_to_gbk(latex);
    latex = Utf8ToGbkOnWin32(latex.c_str());
#endif
    std::cout << "res:\n" << latex << "\n" << endl;
    for (auto& ch : latex) {
        if (ch == ' ') {
            std::cout << "<blank>";
        }
        else if (ch == '\n') {
            std::cout << "<enter>\n";
        }
        else {
            std::cout << ch;
        }
    }
    std::cout << latex << endl;
    std::string expected_res =
        ////
        "\\documentclass[border=2px]{standalone}\n"//\documentclass[border = 2px]{ standalone }<enter>
        "\n"//<enter>
        "\\usepackage[braket, qm]{qcircuit}\n"//\usepackage[braket, <blank>qm]{ qcircuit }<enter>
        "\\usepackage{graphicx}\n"//\usepackage{ graphicx }<enter>
        "\n"//<enter>
        "\\begin{document}\n"//\begin{ document }<enter>
        "\\scalebox{1.0}{\n"//\scalebox{ 1.0 }{<enter>
        "\\Qcircuit @C = 1.0em @R = 0.5em @!R{ \\\\\n"//\Qcircuit<blank>@C < blank >= < blank>1.0em<blank>@R < blank >= < blank>0.5em<blank>@!R{ <blank>\\<enter>
        "\\nghost{q_{0}\\ket{0}}&\\lstick{q_{0}\\ket{0}}&\\gate{\\mathrm{P}\\,\\mathrm{(3.140000)}}&\\rstick{}\\qw&\\nghost{}\\\\\n"//\nghost{q_{0}\ket{0}}&\lstick{q_{0}\ket{0}}&\gate{\mathrm{P}\,\mathrm{(3.140000)}}&\rstick{}\qw & \nghost{}\\<enter>
        "\\\\ }}\n"//\\<blank> }}<enter>
        "\\end{document}\n"//\end{ document }<enter>
        ;
    return expected_res == latex;
    return true;
}

bool test_draw_text_with_clock_without_params() {
    std::cout << "test_draw_text_with_clock_without_params()" << endl;
    CPUQVM qm;
    qm.init();
    auto qbits = qm.qAllocMany(3);
    QProg prog;
    prog << P(qbits[0], 3.14)<<P(qbits[0],6.28);
    std::cout << prog << endl;

    auto text_pic_str = draw_qprog_with_clock(prog, PIC_TYPE::TEXT);

#if defined(WIN32) || defined(_WIN32)
    text_pic_str = fit_to_gbk(text_pic_str);
    text_pic_str = Utf8ToGbkOnWin32(text_pic_str.c_str());
#endif
    std::cout << "res:\n" << text_pic_str << "\n" << endl;
    return true;
}

bool test_draw_text_with_clock_with_params() {
    std::cout << "test_draw_text_with_clock_with_params()" << endl;
    CPUQVM qm;
    qm.init();
    auto qbits = qm.qAllocMany(3);
    QProg prog;
    prog << P(qbits[0], 3.14) << P(qbits[0], 6.28);
    std::cout << prog << endl;

    auto text_pic_str = draw_qprog_with_clock(prog, PIC_TYPE::TEXT,"QPandaConfig.json",false,true);

#if defined(WIN32) || defined(_WIN32)
    text_pic_str = fit_to_gbk(text_pic_str);
    text_pic_str = Utf8ToGbkOnWin32(text_pic_str.c_str());
#endif
    std::cout << "res:\n" << text_pic_str << "\n" << endl;
    return true;
}

bool test_draw_latex_with_clock_without_params() {
    std::cout << "test_draw_latex_without_clock_without_params()" << endl;
    CPUQVM qm;
    qm.init();
    auto qbits = qm.qAllocMany(3);
    QProg prog;
    prog << P(qbits[0], 3.14) << P(qbits[0], 6.28);
    std::cout << prog << endl;

    auto text_pic_str = draw_qprog_with_clock(prog, PIC_TYPE::LATEX);

#if defined(WIN32) || defined(_WIN32)
    text_pic_str = fit_to_gbk(text_pic_str);
    text_pic_str = Utf8ToGbkOnWin32(text_pic_str.c_str());
#endif
    std::cout << "res:\n" << text_pic_str << "\n" << endl;
    return true;
}

bool test_draw_latex_with_clock_with_params() {
    std::cout << "test_draw_latex_with_clock_with_params()" << endl;
    CPUQVM qm;
    qm.init();
    auto qbits = qm.qAllocMany(3);
    QProg prog;
    prog << P(qbits[0], 3.14) << P(qbits[0], 6.28);
    std::cout << prog << endl;

    auto text_pic_str = draw_qprog_with_clock(prog, PIC_TYPE::LATEX, "QPandaConfig.json", false, true);

#if defined(WIN32) || defined(_WIN32)
    text_pic_str = fit_to_gbk(text_pic_str);
    text_pic_str = Utf8ToGbkOnWin32(text_pic_str.c_str());
#endif
    std::cout << "res:\n" << text_pic_str << "\n" << endl;
    return true;
}

TEST(DrawLatex, test3) {
    bool res = true;
    res = res && test_draw_text_without_params();
    res = res && test_draw_text_with_params();
    res = res && test_draw_latex_without_params();
    res = res && test_draw_latex_with_params();
    res = res && test_draw_text_with_clock_without_params();
    res = res && test_draw_text_with_clock_with_params();
    res = res && test_draw_latex_with_clock_without_params();
    res = res && test_draw_latex_with_clock_with_params();

    assert(res);
}