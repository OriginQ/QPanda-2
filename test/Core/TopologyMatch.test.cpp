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
        if (std::abs((long)i.second) < 80)
            continue;
        if (std::abs((long)i.second - (long)matched_prog_result.at(i.first)) > 80) {
            return false;
        }
    }

	return true;
}

static bool test_topology_match()
{
	CPUQVM qvm;
	qvm.init();
	auto q = qvm.allocateQubits(6);
	auto c = qvm.allocateCBits(6);

	std::string originir = "QINIT 6\nCREG 6\nH q[0]\nH q[1]\nH q[2]\nH q[3]\nH q[4]\nX q[5]\nX q[0]\nX q[4]\nH q[5]\nCNOT q[0],q[5]\nX q[0]\nCNOT q[1],q[5]\nH q[0]\nH q[1]\nCNOT q[2],q[5]\nH q[2]\nCNOT q[3],q[5]\nH q[3]\nCNOT q[4],q[5]\nX q[4]\nH q[4]\nMEASURE q[0],c[0]\nMEASURE q[1],c[1]\nMEASURE q[2],c[2]\nMEASURE q[3],c[3]\nMEASURE q[4],c[4]\nMEASURE q[5],c[5]";
	std::string originir0 = "QINIT 6\nCREG 6\nH q[0]\nH q[1]\nH q[2]\nH q[3]\nH q[4]\nX q[5]\nX q[0]\nX q[4]\nH q[5]\nCNOT q[0],q[5]\nX q[0]\nCNOT q[1],q[5]\nH q[0]\nH q[1]\nCNOT q[2],q[5]\nH q[2]\nCNOT q[3],q[5]\nH q[3]\nCNOT q[4],q[5]\nX q[4]\nH q[4]\nMEASURE q[0],c[0]\nMEASURE q[1],c[1]\nMEASURE q[2],c[2]\nMEASURE q[3],c[3]\nMEASURE q[4],c[4]";
    auto prog = convert_originir_string_to_qprog(originir, &qvm);
    auto prog0 = convert_originir_string_to_qprog(originir0, &qvm);

	auto simple_topo = get_circuit_topo(prog);
	auto simple_topo0 = get_circuit_topo(prog0);
	return simple_topo == simple_topo0;
}

TEST(TopologyMatch, test1)
{
	bool test_val = false;
	try
	{
		//test_val = test_topology_match_1();
		test_val = test_topology_match();
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
