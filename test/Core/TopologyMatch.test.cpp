#include "QPanda.h"
#include "gtest/gtest.h"
#include "Core/Utilities/QProgTransform/TopologyMatch.h"
#include "Core/Utilities/QProgInfo/QuantumMetadata.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"

using namespace std;

USING_QPANDA

static bool test_topology_match_1()
{
	CPUQVM qvm;
	qvm.init();
	auto q = qvm.allocateQubits(6);
	auto c = qvm.allocateCBits(6);
	auto srcprog = QProg();
	QGate h_gate = H(q[2]);
	QGate cu_gate = CU(1, 2, 3, 4, q[2], q[0]);
	h_gate.setDagger(1);
	srcprog << CNOT(q[1], q[5])
		<< CNOT(q[0], q[2])
		<< CNOT(q[1], q[2])
		<< CNOT(q[1], q[4])
		<< CZ(q[0], q[1])
		<< X(q[3])
		<< Y(q[4])
		<< H(q[5])
		<< h_gate
		<< CNOT(q[1], q[3])
		<< CNOT(q[0], q[5])
		<< CNOT(q[1], q[2])
		<< cu_gate
		<< MeasureAll(q, c);

	auto src_result = qvm.runWithConfiguration(srcprog, c, 2048);

	auto outprog = topology_match(srcprog, q, &qvm);

	auto matched_prog_result = qvm.runWithConfiguration(outprog, c, 2048);

    for (const auto& i : src_result)
    {
        if (abs((long)i.second < 50))
            continue;
        if (abs((long)i.second - (long)matched_prog_result.at(i.first)) > 50) {
            return false;
        }
    }

	return true;
}

TEST(TopologyMatch, test1)
{
	bool test_val = false;
	try
	{
		test_val = test_topology_match_1();
	}
	catch (const std::exception& e)
	{
		std::cout << "Got a exception: " << e.what() << endl;
		test_val = false;
	}
	catch (...)
	{
		std::cout << "Got an unknow exception: " << endl;
		test_val = false;
	}

	ASSERT_TRUE(test_val);
}