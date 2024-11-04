#include "gtest/gtest.h"
#include <iostream>
#include <string>
#include <time.h>
#include <vector>

#include "include/QPanda.h"

using namespace QPanda;
using namespace std;

#ifndef PRECISION
#define PRECISION 0.000001
#endif // !PRECISION

namespace QASMToQProg_TEST {
	std::string SX_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
creg c[2];
SX q[1];
sx q[0];
)";

	std::string SXdg_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
creg c[2];
SXdg q[1];
sxdg q[0];
)";

	std::string ISWAP_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
iswap q[0],q[1];
)";

	std::string DCX_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
DCX q[0],q[1];
dcx q[0],q[1];
)";

	std::string CP_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
CP(-PI/2) q[0],q[1];
cp(-PI/2) q[0],q[1];
)";

	std::string CS_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
CS q[0],q[1];
cs q[0],q[1];
)";

	std::string CSdg_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
CSdg q[0],q[1];
csdg q[0],q[1];
)";

	std::string CCZ_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
CCZ q[0],q[1],q[2];
ccz q[0],q[1],q[2];
)";

	std::string ECR_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
ECR q[0],q[1];
ecr q[0],q[1];
)";//目前对于;;这样的语句报错

	std::string R_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
R(3.14,3.15) q[0];
r(3.14,3.15) q[0];
)";

	std::string XXMinusYY_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
XXMinusYY(3.14,3.15) q[0],q[1];
xx_minus_yy(3.14,3.15) q[0],q[1];
)";

	std::string XXPlusYY_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
XXPlusYY(3.14,3.15) q[0],q[1];
xx_plus_yy(3.14,3.15) q[0],q[1];
)";

	std::string V_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
V q[0];
v q[0];
)";

	std::string W_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
W q[0];
w q[0];
)";


	std::string CCX_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[4] q;

toffoli q[0], q[1], q[2];
TOFFOLI q[0], q[1], q[2];
ccx q[0], q[1], q[2];
CCX q[0], q[1], q[2];
)";

	std::string CH_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[4] q;

ch q[0], q[1];
CH q[0], q[1];
)";

	std::string CNOT_CZ_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[4] q;

cx q[0], q[1];
CX q[0], q[1];
cnot q[0], q[1];
CNOT q[0], q[1];

cz q[0], q[1];
CZ q[1], q[0];
)";

	std::string CRX_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;
crx(3.14) q[0], q[1];
CRX(3.14) q[0], q[1];
)";

	std::string CRY_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;
cry(3.14) q[0], q[1];
CRY(3.14) q[0], q[1];
)";

	std::string CRZ_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;
crz(3.14) q[0], q[1];
CRZ(3.14) q[0], q[1];
)";

	std::string CSWAP_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[3] q;

cswap q[0], q[1], q[2];
CSWAP q[0], q[1], q[2];
)";

	std::string CSX_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[4] q;

csx q[0], q[1];
CSX q[0], q[1];
)";

	std::string CU_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[4] q;

cu(3.14, 3.15, 3.16, 3.17) q[1], q[2];
CU(3.14, 3.15, 3.16, 3.17) q[1], q[2];
)";

	std::string CU1_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[4] q;

cp(3.14)q[0], q[1];
cu1(3.14)q[0], q[1];
CP(3.14)q[0], q[1];
CU1(3.14)q[0], q[1];
cphase(3.14)q[0], q[1];
CPHASE(3.14)q[0], q[1];
)";

	std::string CU3_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[4] q;

cu3(3.14, 3.15, 3.16) q[1], q[2];
CU3(3.14, 3.15, 3.16) q[1], q[2];
)";

	std::string CY_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[4] q;

cy q[0], q[1];
CY q[0], q[1];
)";

	std::string C3SQRTX_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[4] q;

c3sqrtx q[0], q[1], q[2], q[3];
C3SQRTX q[0], q[1], q[2], q[3];
)";

	std::string C3X_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[4] q;

c3x q[0], q[1], q[2], q[3];
C3X q[0], q[1], q[2], q[3];
)";

	std::string C4X_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[5] q;

c4x q[0], q[1], q[2], q[3], q[4];
C4X q[0], q[1], q[2], q[3], q[4];
)";

	std::string H_I_X_Y_Z_S_T_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[4] q;
h q[0];
H q[1];

i q[0];
id q[1];
I q[2];
u0 q[3];

x q[0];
X q[1];

y q[0];
Y q[1];

z q[0];
Z q[1];

s q[0];
S q[1];

t q[0];
T q[1];
)";

	std::string RCCX_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[3] q;
rccx q[0], q[1], q[2];
RCCX q[0], q[1], q[2];
)";


	std::string RC3X_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[4] q;
rc3x q[0], q[1], q[2], q[3];
RC3X q[0], q[1], q[2], q[3];
)";

	std::string RXX_RYY_RZZ_RZX_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[4] q;

rxx(3.14) q[0], q[1];
RXX(3.15) q[0], q[1];

ryy(3.14) q[0], q[1];
RYY(3.15) q[0], q[1];

rzz(3.14) q[0], q[1];
RZZ(3.15) q[0], q[1];

rzx(3.14) q[0], q[1];
RZX(3.15) q[0], q[1];
)";

	std::string RX_RY_RZ_P_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[4] q;

rx(3.14) q[0];
RX(3.15) q[1];

ry(3.14) q[0];
RY(3.15) q[1];

rz(3.14) q[0];
RZ(3.15) q[1];

p(3.14) q[0];
P(3.15) q[0];
u1(3.16) q[0];
U1(3.17) q[0];
phase(3.18) q[0];
)";

	std::string SDG_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;

sdg q[0];
SDG q[0];
Sdg q[0];
)";

	std::string SWAP_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[4] q;

swap q[0], q[1];
SWAP q[1], q[0];
)";


	std::string TDG_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;

tdg q[0];
TDG q[0];
Tdg q[0];
)";

	std::string U2_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[4] q;

u2(3.14, 3.15) q[0];
U2(3.14, 3.15) q[0];
)";

	std::string U3_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[4] q;

u(3.14, 3.15, 3.16) q[0];
u3(3.14, 3.15, 3.16) q[0];
U(3.14, 3.15, 3.16) q[0];
U3(3.14, 3.15, 3.16) q[0];
)";

	// quantum instructions

	std::string RESET_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[3] q;
reset q[0];
)";

	std::string MEASURE_qasm = R"(
qubit[4] q;
bit[4] c;
x q[2];
c[2] = measure q[2];
)";

	std::string BARRIER_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[5] q;
barrier q[0],q[1],q[2],q[3];
)";

	// Classical expr
	std::string Classical_expr_qasm = R"(
OPENQASM 3.0;
include "stdgates.inc";

qubit[5] q;
bit[2] c;
rz(pi - 5) q[0];
c[1] = measure q[0]; 
)";


	static bool test_qasm2originir2Qprog(const std::string& qasm_str)
	{
		std::cout << "### from qasm: " << qasm_str << std::endl;
		std::cout << "### to originir:\n" << QuantumComputation::fromQASM(qasm_str).toOriginIR() << std::endl;
		auto qvm = CPUQVM();
		qvm.init();
		QProg prog = convert_qasm_string_to_qprog(qasm_str, &qvm);
		std::cout << "### qprog from qasm:" << std::endl;
		cout << prog << endl;


		///*if (!prog.is_empty())
		//{
		//	return true;
		//}
		//else
		//{
		//	return false;
		//}*/
		return true;
	}



	// 其它操作 originir

	std::string RESET_ir = R"(QINIT 4
CREG 4
RESET q[0]

)";

	std::string MEASURE_ir = R"(QINIT 4
CREG 4
X q[2]
MEASURE q[2],c[2]

)";

	std::string BARRIER_ir = R"(QINIT 4
CREG 4
BARRIER q[0], q[1],q[2]

)";

	//算术表达式 originir
	std::string ClassEXp_ir = R"(QINIT 2
CREG 7
c[0]=5
c[2]=35
c[1]=25
c[2]=c[0]+c[1]
c[3]=c[0]-c[1]
c[4]=c[0]*c[1]
c[5]=c[0]/c[1]
c[6]=c[1]/c[2]

)";

	std::string gate_ir = R"(QINIT 3
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


	static bool test_originir2Qprog(const std::string& originir_str)
	{
		std::cout << "### originir_str:" << originir_str << std::endl;
		auto qvm = CPUQVM();
		qvm.init();
		std::cout << "### qprog from originir_str:" << std::endl;
		QProg prog = QPanda::convert_originir_string_to_qprog(originir_str, &qvm);
		cout << prog << endl;


		if (!prog.is_empty())
		{
			return true;
		}
		else
		{
			return false;
		}
	}



	TEST(QASMToQprog, StandardGate)
	{

		bool test_actual = true;
		try
		{

			test_actual = test_actual && test_qasm2originir2Qprog(SX_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(SXdg_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(ISWAP_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(DCX_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(CS_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(CCZ_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(ECR_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(R_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(CSdg_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(XXMinusYY_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(XXPlusYY_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(V_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(W_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(CCX_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(CH_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(CNOT_CZ_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(CRX_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(CRY_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(CRZ_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(CSWAP_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(CSX_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(CU_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(CU1_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(CU3_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(CY_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(C3SQRTX_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(C3X_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(C4X_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(H_I_X_Y_Z_S_T_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(RCCX_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(RC3X_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(RXX_RYY_RZZ_RZX_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(RX_RY_RZ_P_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(SDG_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(SWAP_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(TDG_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(U2_qasm);

			test_actual = test_actual && test_qasm2originir2Qprog(U3_qasm);

		}

		catch (const std::exception& e)
		{
			cout << "Got a exception: " << e.what() << endl;
		}
		catch (...)
		{
			cout << "Got an unknow exception: " << endl;
		}
		test_actual = test_actual && test_qasm2originir2Qprog(Classical_expr_qasm);

		ASSERT_TRUE(test_actual);
	}

	TEST(QORIGINIRToQprog, Instructions)
	{

		bool test_actual = true;
		try
		{

			test_actual = test_actual && test_originir2Qprog(RESET_ir);
			test_actual = test_actual && test_originir2Qprog(MEASURE_ir);
			test_actual = test_actual && test_originir2Qprog(BARRIER_ir);
			test_actual = test_actual && test_originir2Qprog(gate_ir);
			test_actual = test_actual && test_originir2Qprog(ClassEXp_ir);
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


	TEST(QASMToQprog, Instructions)
	{

		bool test_actual = true;
		try
		{

			test_actual = test_actual && test_qasm2originir2Qprog(MEASURE_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(RESET_qasm);
			test_actual = test_actual && test_qasm2originir2Qprog(BARRIER_qasm);


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



	TEST(QORIGINIRToQprog, SuanshuInstructions)
	{

		bool test_actual = true;
		try
		{
			test_actual = test_actual && test_originir2Qprog(ClassEXp_ir);
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
};


bool test_convert_qasm_to_qprog() {

	std::string qasm_str = QASMToQProg_TEST::U3_qasm;
	std::ofstream ofs("test_qasm2qprog.txt");
	if (!ofs) {
		std::cerr << "Unable to open file";
		return 1; // 返回错误代码  
	}
	std::cout << "qasm_str:\n" << qasm_str << std::endl;
	ofs << qasm_str << std::endl;

	ofs.close();
	auto qvm = CPUQVM();
	qvm.init();
	QVec qv;
	std::vector<ClassicalCondition> cv;
	QProg prog = convert_qasm_to_qprog("test_qasm2qprog.txt", &qvm,qv,cv);
	std::cout << prog << std::endl;

	
	return true;
}

TEST(QASMToQprog, file2qprog) {
	test_convert_qasm_to_qprog();
}
