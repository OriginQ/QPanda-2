#include "QPanda.h"
#include "gtest/gtest.h"
#include "Core/Utilities/QProgTransform/TopologyMatch.h"
#include "Core/Utilities/QProgInfo/QuantumMetadata.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"

using namespace std;

USING_QPANDA

const std::string excepted_val = R"(QINIT 16
CREG 16
CNOT q[1],q[0]
CNOT q[2],q[3]
CZ q[5],q[4]
X q[7]
H q[12]
CNOT q[15],q[0]
H q[15]
H q[0]
CNOT q[15],q[0]
H q[15]
H q[0]
CNOT q[15],q[0]
Y q[4]
CNOT q[6],q[7]
H q[5]
H q[12]
CNOT q[12],q[5]
H q[12]
H q[5]
CNOT q[1],q[0]
CNOT q[12],q[5]
H q[12]
H q[5]
CNOT q[12],q[5]
H q[12]
H q[5]
CNOT q[12],q[5]
H q[0]
H q[15]
CNOT q[15],q[0]
H q[15]
H q[0]
CNOT q[12],q[13]
H q[12]
H q[13]
CNOT q[12],q[13]
H q[12]
H q[13]
CNOT q[12],q[13]
DAGGER
H q[0]
ENDDAGGER
CNOT q[13],q[14]
H q[13]
H q[14]
CNOT q[13],q[14]
H q[13]
H q[14]
CNOT q[13],q[14]
CNOT q[15],q[0]
H q[15]
H q[0]
CNOT q[15],q[0]
H q[15]
H q[0]
CNOT q[15],q[0]
H q[14]
H q[15]
CNOT q[15],q[14]
H q[15]
H q[14]
CU q[15],q[14],(1,2,3,4))";
TEST(TopologyMatch, test)
{
	auto qvm = new CPUQVM();
	qvm->init();
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

	qvm->directlyRun(srcprog);
	auto r1 = qvm->PMeasure_no_index(q);

	auto outprog = topology_match(srcprog, q, qvm);

	qvm->directlyRun(outprog);
	auto r2 = qvm->PMeasure_no_index(q);

	string actual_val = transformQProgToOriginIR(outprog, qvm);

	//std::cout << transformQProgToOriginIR(outprog, qvm) << std::endl;

	//std::cout << outprog << endl;
	//getchar();
	ASSERT_EQ(actual_val, excepted_val);
	return;
}