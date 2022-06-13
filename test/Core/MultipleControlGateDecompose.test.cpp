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

bool ldd_decompose_test_1()
{
    CPUQVM machine;
    machine.init();

    auto q = machine.qAllocMany(6);
    auto c = machine.cAllocMany(6);

    QProg prog;
    prog << H(q[1]) << H(q[2]) << Z(q[0]).control({ q[1],q[2] });

    auto ldd = ldd_decompose(prog);

    machine.directlyRun(prog);
    auto state = machine.getQState();

    machine.directlyRun(ldd);
    auto ldd_state = machine.getQState();

    for (auto val : state)
        cout << val << endl;

    cout << "================" << endl;

    for (auto val : ldd_state)
        cout << val << endl;

    //cout << prog << endl;
    //cout << ldd << endl;

    return state_compare(state, ldd_state);
}

bool ldd_decompose_test_2()
{
    CPUQVM machine;
    machine.init();

    auto q = machine.qAllocMany(6);
    auto c = machine.cAllocMany(6);

    QProg prog;
    prog << H(q[1]) << H(q[2])
        << Z(q[0]).control({ q[1],q[2] });

    cout << "src prog:" << prog << endl;

    const auto mat_1 = getCircuitMatrix(prog);

    auto ldd_prog = ldd_decompose(prog);

    cout << "after ldd decompose_multiple_control_gate prog:" << ldd_prog << endl;

    const auto mat_2 = getCircuitMatrix(ldd_prog);

    if (mat_1 == mat_2)
    {
        cout << "The multi-control gate was successfully decomposed." << endl;
        return true;
    }

    cout << "Decompose error !" << endl;
    return false;
}


TEST(MultipleControlGateDecompose, test)
{
	bool test_val = true;
	try
	{
        test_val = test_val && ldd_decompose_test_1();
        test_val = test_val && ldd_decompose_test_2();
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