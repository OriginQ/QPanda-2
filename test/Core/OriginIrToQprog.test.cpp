#include "gtest/gtest.h"
#include <iostream>
#include <string>
#include "QPanda.h"
#include <time.h>

USING_QPANDA
using namespace std;

#ifndef PRECISION
#define PRECISION 0.000001
#endif // !PRECISION



const std::string nativeGateIR = R"(QINIT 3
CREG 3
P q[0],(2.14)
H q[0]
H q[1]
H q[2]
X q[0]
Y q[1]
Z q[2]
X1 q[1]
Y1 q[2]
Z1 q[0]
RX q[0],(1.570796)
RY q[1],(1.570796)
RZ q[2],(1.570796)
T q[2]
S q[0]
U1 q[0],(1.570796)
U2 q[0],(1.570796,-3.141593)
U3 q[0],(1.570796,4.712389,1.570796)
U4 q[1],(3.141593,4.712389,1.570796,-3.141593)
RPhi q[0],(1.570796,-3.141593)
CU q[1],q[0],(1.57,0,1.57,3.14)
CNOT q[0],q[1]
CZ q[0],q[1]
SWAP q[0],q[1]
CR q[0],q[1],(1.570796)
ISWAP q[0],q[1]
RXX q[0],q[1],(1.570796)
RYY q[0],q[1],(1.570796)
RZZ q[0],q[1],(1.570796)
RZX q[0],q[1],(1.570796)
TOFFOLI  q[0],q[1],q[2]
MEASURE q[0],c[0]
MEASURE q[1],c[1]
MEASURE q[2],c[2]
)";




static bool test_IR_to_Qprog_fun_1()
{
	auto qvm = CPUQVM();
	qvm.init();
	// 申请寄存器并初始化
	QVec sqv = qvm.qAllocMany(3);

	// QASM转换量子程序
	auto machine = initQuantumMachine(QMachineType::CPU);
	QVec out_qv;
	std::vector<ClassicalCondition> out_cv;
	QProg prog = convert_originir_string_to_qprog(nativeGateIR, machine, out_qv, out_cv);
	cout << prog << endl;
	//virtual_z_transform(prog, machine);

	//cout << prog << endl;

	destroyQuantumMachine(machine);
	if (prog.get_qgate_num() > 0)
	{
		return true;
	}
	else
	{
		return false;
	}
}


TEST(OriginIrToQprog, test1)
{
	bool test_actual = true;
	try
	{
		test_actual = test_actual && test_IR_to_Qprog_fun_1();
		// test_actual = test_actual && test_prog_to_qasm_2();
	}

	catch (const std::exception& e)
	{
		cout << "Got a exception: " << e.what() << endl;
	}
	catch (...)
	{
		cout << "Got an unknow exception: " << endl;
	}

	ASSERT_TRUE(test_actual);
}






