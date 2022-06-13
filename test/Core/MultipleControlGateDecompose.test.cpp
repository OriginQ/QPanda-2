#include <iostream>
#include "QPanda.h"
#include "gtest/gtest.h"
#include "Core/Utilities/Tools/MultiControlGateDecomposition.h"

USING_QPANDA
using namespace std;

bool state_compare(const QStat& state1, const QStat& state2)
{
    QPANDA_RETURN(state1.size() != state2.size(), false);

    for (auto i = 0; i < state1.size(); ++i)
    {
        if (std::fabs(state1[i].real() - state2[i].real()) > 1e-6)
            return false;
        if (std::fabs(state1[i].imag() - state2[i].imag()) > 1e-6)
            return false;
    }

    return true;
}

bool decompose_compare()
{
    CPUQVM machine;
    machine.init();

    auto q = machine.qAllocMany(6);
    auto c = machine.cAllocMany(6);

    QProg prog;
    //prog << H(q[1]) << H(q[2]) << CR(q[0], q[1], PI / 5);
    prog << CR(q[0], q[1], PI / 2).control({ q[2] });

    auto ldd = ldd_decompose(prog);
    //decompose_multiple_control_qgate(prog, &machine, "QPandaConfig.json", true);

    //machine.directlyRun(prog);
    //auto state = machine.getQState();

    //machine.directlyRun(ldd);
    //auto ldd_state = machine.getQState();

    //for (auto val : state)
    //    std::cout << val << endl;

    //std::cout << "================" << endl;

    //for (auto val : ldd_state)
     //   std::cout << val << endl;

    std::cout << convert_qprog_to_originir(prog, &machine) << endl;
    std::cout << convert_qprog_to_originir(ldd, &machine) << endl;
    std::cout << "================" << endl;

    std::cout << ldd << endl;


    std::cout << "getQGateNum(prog) " << getQGateNum(prog) << endl;
    std::cout << "getQGateNum(ldd) " << getQGateNum(ldd) << endl;
    return true;
    //return state_compare(state, ldd_state);
}


bool ldd_decompose_test_1()
{
    CPUQVM machine;
    machine.init();

    auto q = machine.qAllocMany(6);
    auto c = machine.cAllocMany(6);

    QProg prog;
    //prog << H(q[1]) << H(q[2]) << CR(q[0], q[1], PI / 5);
    prog  << CR(q[0], q[1], PI / 5);

    auto ldd = ldd_decompose(prog);
    decompose_multiple_control_qgate(prog, &machine, "QPandaConfig.json", false);

    machine.directlyRun(prog);
    auto state = machine.getQState();

    machine.directlyRun(ldd);
    auto ldd_state = machine.getQState();

    for (auto val : state)
        std::cout << val << endl;

    std::cout << "================" << endl;

    for (auto val : ldd_state)
        std::cout << val << endl;

    std::cout << prog << endl;
    std::cout << ldd << endl;

    return state_compare(state, ldd_state);
}

bool ldd_decompose_test_2()
{
    CPUQVM machine;
    machine.init();

    auto q = machine.qAllocMany(3);
    auto c = machine.cAllocMany(3);

    QProg prog;
    //prog << H(q[0]) << H(q[2]);
    prog << X(q[2]).control({ q[0] ,q[1] });
    //    << Z(q[0]).control({ q[1],q[2] });

    //prog << H(q[0]) << iSWAP(q[1], q[0]) << MeasureAll(q, c);
    //auto result = machine.runWithConfiguration(prog, c, 1000);

    std::cout << "src prog:" << prog << std::endl;

    const auto mat_1 = getCircuitMatrix(prog);

    auto ldd_prog = ldd_decompose(prog);
    decompose_multiple_control_qgate(prog, &machine, "QPandaConfig.json", false);

    std::cout << "after ldd decompose_multiple_control_gate prog:" << ldd_prog << std::endl;

    const auto mat_2 = getCircuitMatrix(ldd_prog);

    if (mat_1 == mat_2)
    {
        std::cout << "The multi-control gate was successfully decomposed." << std::endl;
        return true;
    }

    std::cout << "Decompose error !" << std::endl;
    return false;
}

bool ucry_decompose_test_2()
{
    CPUQVM machine;
    machine.init();

    auto q = machine.qAllocMany(4);
    auto c = machine.cAllocMany(4);

    QProg prog;

    prob_vec params;
    for (auto i = 0; i < (1ull << 3); ++i)
    {
        params.emplace_back(random_generator19937());
    }

    auto ucry = ucry_circuit({ q[0], q[1], q[2] }, q[3], params);

    cout << ucry << endl;

    auto ucry_result = ucry_decompose({ q[0], q[1], q[2] }, q[3], params);

    cout << ucry_result << endl;

    //auto ldd_prog = ldd_decompose(prog);
    //decompose_multiple_control_qgate(prog, &machine, "QPandaConfig.json", false);

    //std::cout << "after ldd decompose_multiple_control_gate prog:" << ldd_prog << std::endl;

    const auto mat_1 = getCircuitMatrix(ucry);
    const auto mat_2 = getCircuitMatrix(ucry_result);

    cout << mat_1 << endl;
    cout << mat_2 << endl;


    if (mat_1 == mat_2)
    {
        std::cout << "The multi-control gate was successfully decomposed." << std::endl;
        return true;
    }

    std::cout << "Decompose error !" << std::endl;
    return false;
}

bool ucry_decompose_test_1()
{
    CPUQVM machine;
    machine.init();

    auto q = machine.qAllocMany(3);
    auto c = machine.cAllocMany(3);

    QProg prog;

    prob_vec params;
    for (auto i = 0; i < (1ull << 2); ++i)
    {
        params.emplace_back(random_generator19937());
    }

    auto ucry = ucry_circuit({ q[0], q[1] }, q[2], params);

    cout << ucry << endl;

    auto ucry_result = ucry_decompose({ q[0], q[1] }, q[2], params);

    cout << ucry_result << endl;

    //auto ldd_prog = ldd_decompose(prog);
    //decompose_multiple_control_qgate(prog, &machine, "QPandaConfig.json", false);

    //std::cout << "after ldd decompose_multiple_control_gate prog:" << ldd_prog << std::endl;

    const auto mat_1 = getCircuitMatrix(ucry);
    const auto mat_2 = getCircuitMatrix(ucry_result);

    cout << mat_1 << endl;
    cout << mat_2 << endl;


    if (mat_1 == mat_2)
    {
        std::cout << "The multi-control gate was successfully decomposed." << std::endl;
        return true;
    }

    std::cout << "Decompose error !" << std::endl;
    return false;
}

bool ucry_decompose_test_3()
{
    CPUQVM machine;
    machine.init();

    auto q = machine.qAllocMany(2);
    auto c = machine.cAllocMany(2);

    QProg prog;

    prob_vec params;
    for (auto i = 0; i < (1ull << 1); ++i)
    {
        params.emplace_back(random_generator19937());
    }

    auto ucry = ucry_circuit({ q[0] }, q[1], params);

    cout << ucry << endl;

    auto ucry_result = ucry_decompose({ q[0] }, q[1], params);

    cout << ucry_result << endl;

    //auto ldd_prog = ldd_decompose(prog);
    //decompose_multiple_control_qgate(prog, &machine, "QPandaConfig.json", false);

    //std::cout << "after ldd decompose_multiple_control_gate prog:" << ldd_prog << std::endl;

    const auto mat_1 = getCircuitMatrix(ucry);
    const auto mat_2 = getCircuitMatrix(ucry_result);

    cout << mat_1 << endl;
    cout << mat_2 << endl;


    if (mat_1 == mat_2)
    {
        std::cout << "The multi-control gate was successfully decomposed." << std::endl;
        return true;
    }

    std::cout << "Decompose error !" << std::endl;
    return false;
}

TEST(MultipleControlGateDecompose, test)
{
	bool test_val = true;
	try
	{
        //test_val = test_val && ucry_decompose_test_1();
        //test_val = test_val && ucry_decompose_test_2();
        //test_val = test_val && ucry_decompose_test_3();
        //test_val = test_val && ldd_decompose_test_2();
        test_val = test_val && decompose_compare();
        //test_val = test_val && ldd_decompose_test_2();
	}
	catch (const std::exception& e)
	{
        QCERR(e.what());
    }
    catch (...)
    {
        QCERR("unknow exception");
	}

	ASSERT_TRUE(test_val);
    std::cout << "MultipleControlGateDecompose test pass" << std::endl;
}