#include "QPanda.h"
#include "gtest/gtest.h"
#include "Core/Utilities/QProgTransform/TopologyMatch.h"
#include "Core/Utilities/QProgInfo/QuantumMetadata.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"

using namespace std;

USING_QPANDA

TEST(TopologyMatch, test)
{
	auto qvm = initQuantumMachine();
	auto q = qvm->allocateQubits(16);
	auto c = qvm->allocateCBits(16);
	auto srcprog = QProg();
	QGate h_gate = H(q[12]);
	QGate cu_gate =  CU(1, 2, 3, 4, q[12], q[10]);
	h_gate.setDagger(1);
	srcprog << CNOT(q[1], q[9])
		<< CNOT(q[0], q[2])
		<< CNOT(q[1], q[12])
		<< CNOT(q[12], q[9])
		<< CZ(q[10], q[14])
		<< X(q[13])
		<< Y(q[14])
		<< H(q[15])
		<< h_gate
		<< CNOT(q[11], q[13])
		<< CNOT(q[10], q[15])
		<< CNOT(q[10], q[12])
		<< cu_gate;
	
	auto outprog = topology_match(srcprog, q, qvm);

	std::cout << transformQProgToOriginIR(outprog, qvm) << std::endl;

	std::cout << outprog << endl;
	getchar();
	return;
}