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

static QCircuit build_U_fun(QVec qubits)
{
	QCircuit cir_u;
	cir_u << T(qubits[0]);
	//cir_u << U1(qubits[0], 2.0*PI / 3.0);
	return cir_u;
}

static bool test_QProgToQASM_fun_1()
{
	auto machine = initQuantumMachine(CPU);
	size_t first_register_qubits_cnt = 3;
	QVec first_register = machine->allocateQubits(first_register_qubits_cnt);

	size_t second_register_qubits_cnt = 1;
	QVec second_register = machine->allocateQubits(second_register_qubits_cnt);

	//QCircuit cir_qpe = build_QPE_circuit(first_register, second_register, build_U_fun);
	//cout << cir_qpe << endl;
	QProg qpe_prog;
	//qpe_prog << X(second_register[0]) << cir_qpe;
	qpe_prog << CU(1.2345, 3, 4, 5, first_register[0], first_register[1]);

	//cout << (qpe_prog) << endl;
	auto mat_src = getCircuitMatrix(qpe_prog);
	//cout << "mat_src: " << endl << mat_src << endl;

	write_to_qasm_file(qpe_prog, machine, "qpe.qasm");

	QVec qasm_qubit;
	std::vector<ClassicalCondition> cv;
	QProg qasm_prog = convert_qasm_to_qprog("qpe.qasm", machine, qasm_qubit, cv);
	//cout << (qasm_prog) << endl;

	auto mat_end = getCircuitMatrix(qasm_prog);
	//cout << "mat_end: " << endl << mat_end << endl;


	auto result1 = probRunDict(qasm_prog, { qasm_qubit[0], qasm_qubit[1], qasm_qubit[2] });
	/*for (auto &val : result1)
	{
		val.second = abs(val.second) < PRECISION ? 0.0 : val.second;
	}*/

	/*std::cout << "QPE result:" << endl;
	for (auto &val : result1)
	{
		std::cout << val.first << ", " << val.second << std::endl;
	}*/
	destroyQuantumMachine(machine);
	if (mat_src == mat_end)
	{
		return true;
	}
	else
	{
		return false;
	}
}

bool test_prog_to_qasm_2()
{
	std::string filename = "testfile.txt";
	std::ofstream os(filename);
	os << R"(
        // test QASM file 
        OPENQASM 2.0; 
        include "qelib1.inc"; 
        qreg q[3];
        creg c[3];
        x q[0];
        x q[1];
        z q[2];
        h q[0];
        tdg q[1];
        )";

	os.close();
	auto machine = initQuantumMachine(QMachineType::CPU);
	QProg prog = convert_qasm_to_qprog(filename, machine);
	//cout << "src prog:" << prog << endl;
	auto mat1 = getCircuitMatrix(prog);
	single_gate_optimizer(prog, QCircuitOPtimizerMode::Merge_U3);
	//cout << "U3 prog:" << prog << endl;
	auto mat2 = getCircuitMatrix(prog);
	if (mat1 != mat2)
	{
		return false;
	}

	//cout << "prog_to_originir:" << endl << convert_qprog_to_originir(prog, machine) << endl; 
	//cout << "prog:" << prog << endl;
	//std::cout << convert_qprog_to_qasm(prog, machine) << std::endl;  
	destroyQuantumMachine(machine);

	return true;
	//return convert_qprog_to_qasm(prog, machine);
}

TEST(QProgToQASM, test1)
{
	bool test_actual = true;
	try
	{
		test_actual = test_actual && test_QProgToQASM_fun_1();
		test_actual = test_actual && test_prog_to_qasm_2();
	}

	catch (const std::exception& e)
	{
		cout << "Got a exception: " << e.what() << endl;
	}
	catch (...)
	{
		cout << "Got an unknow exception: " << endl;
	}

	/*std::string excepted_val = R"(QINIT 3
CREG 3
U3 q[0],(1.5707963,3.1415927,3.1415927)
U3 q[1],(3.1415927,-1.8269113,2.1000796)
U3 q[2],(0,3.1415927,0))";*/
	ASSERT_TRUE(test_actual);
}


const std::string UserDefinedeGateIR = R"(QINIT 7
# here is an annotation line
CREG 7
QGATE N0 a,b
# here is an annotation line
H a
T b
ENDQGATE
QGATE N1 a,b
U1 a,(1.570796)
U2 b,(1.570796,-3.141593)
ENDQGATE
QGATE N2 a,b
U3 a,(1.570796,4.712389,1.570796)
U4 b,(3.141593,4.712389,1.570796,-3.141593)
ENDQGATE
QGATE N3 a,b
RX a,(0.785398)
RY b,(0.785398)
ENDQGATE
QGATE N4 a,b
RZ a,(0.785398)
CNOT b,a
ENDQGATE
QGATE N5 a,b,d
ISWAP a,b
SQISWAP d,b
ENDQGATE
QGATE N6 a,b,d
SWAP a,b
TOFFOLI d,a,b
CR d,b,(1.570796)
CZ d,b
ENDQGATE
# here is an annotation line
N0 q[0],q[1]
N1 q[0],q[1]
N2 q[0],q[1]
N3 q[0],q[1]
N4 q[0],q[1]
# here is an annotation line
N5 q[2],q[0],q[3]
N6 q[1],q[0],q[2]
MEASURE q[0],c[0]
# here is an annotation line
MEASURE q[1],c[1]
MEASURE q[2],c[2]
MEASURE q[3],c[3]
)";


const std::string UserDefinedeGateIR0 = R"(QINIT 10
CREG 10
QGATE abcder a,b
H a
H b
ENDQGATE
QGATE N5eeee a,b
abcder a,b
H a
ENDQGATE
QGATE cddasf a,b,d,e,f,g,h,i,j,k
H a
H b
H d
H e
H f
H g
H h
H i
H j
H k
ENDQGATE
H q[0]
SQISWAP q[2],q[1]
SWAP q[3],q[4]
H q[6]
CR q[9],q[8],(1.570796)
S q[0]
H q[2]
BARRIER q[3]
H q[9]
N5eeee q[0],q[1]
N5eeee q[3],q[4]
CR q[9],q[6],(0.392699)
H q[0]
BARRIER q[3]
CR q[5],q[4],(1.570796)
CR q[1],q[0],(1.570796)
H q[3]
CR q[6],q[4],(0.785398)
BARRIER q[7]
CR q[2],q[3],(1.570796)
H q[4]
S q[5]
Y q[2]
DAGGER
H q[5]
ENDDAGGER
CNOT q[4],q[5]
CR q[2],q[0],(0.785398)
H q[4]
CNOT q[5],q[6]
H q[1]
CNOT q[4],q[5]
abcder q[1],q[2]
CNOT q[5],q[6]
H q[5]
CNOT q[5],q[6]
CNOT q[6],q[7]
MEASURE q[0],c[0]
MEASURE q[1],c[1]
MEASURE q[2],c[2]
MEASURE q[3],c[3]
MEASURE q[4],c[4]
MEASURE q[5],c[5]
MEASURE q[6],c[6]
MEASURE q[7],c[7]
MEASURE q[8],c[8]
MEASURE q[9],c[9]
)";


const std::string QPE_IR = R"(QINIT 3
# here is an annotation line
CREG 2
# here is an annotation line
H q[2]
H q[0]
H q[1]
CONTROL q[1]
RX q[2],(-3.141593)
ENDCONTROL
CONTROL q[0]
# here is an annotation line
RX q[2],(-3.141593)
RX q[2],(-3.141593)
ENDCONTROL
# here is an annotation line
DAGGER
H q[1]
CR q[0],q[1],(1.570796)
H q[0]
ENDDAGGER
MEASURE q[0],c[0]
# here is an annotation line
MEASURE q[1],c[1]
)";



TEST(QProgToQASM, test_UserDefinedeGate)
{
	auto machine = initQuantumMachine(QMachineType::CPU);
	QVec out_qv;
	std::vector<ClassicalCondition> out_cv;

	std::cout << "The user-defined-gate (base OriginIR): " << std::endl;
	std::cout << UserDefinedeGateIR << std::endl;
	std::cout << "============================================" << std::endl;

	std::cout << "The user-defined-gate (TransformToOriginIR): " << std::endl;

	// OriginIR转换量子程序
	QProg out_prog = convert_originir_string_to_qprog(UserDefinedeGateIR, machine);

	// 量子程序转换OriginIR，打印并对比转换结果
	std::cout <<
		convert_qprog_to_originir(out_prog, machine)
		<< std::endl;

	std::cout << "============================================" << std::endl;

	std::cout << "The user-defined-gate (QProg): " << std::endl;
	std::cout << out_prog
		<< std::endl;
	destroyQuantumMachine(machine);

}

TEST(QProgToQASM, qprog2pyquil)
{
	init();
	QuantumMachine* machine = initQuantumMachine(CPU);
	auto qvec = qAllocMany(4);
	auto cbits = cAllocMany(4);
	auto circuit = createEmptyCircuit();
	auto prog = createEmptyQProg();
	circuit << H(qvec[0]) << CNOT(qvec[0], qvec[1]);

	circuit << CNOT(qvec[1], qvec[2]) << SWAP(qvec[2], qvec[3]);

	prog
		<< H(qvec[0])
		<< T(qvec[0])
		<< S(qvec[0])
		<< X(qvec[1])
		<< Y(qvec[1])
		<< Z(qvec[1])
		<< X1(qvec[2])
		<< Y1(qvec[2])
		<< Z1(qvec[2])
		<< RX(qvec[3], PI)
		<< RY(qvec[3], PI)
		<< RZ(qvec[3], PI)
		<< U1(qvec[0], PI)
		<< U2(qvec[0], PI, PI / 2)
		<< U3(qvec[0], PI, PI / 2, PI / 4)
		<< U4(qvec[0], PI, PI / 2, PI / 4, PI / 4)
		<< RX(qvec[0], PI)
		<< RY(qvec[0], PI)
		<< RZ(qvec[0], PI)
		<< SWAP(qvec[0], qvec[1])
		<< iSWAP(qvec[2], qvec[3])
		<< Toffoli(qvec[0], qvec[1], qvec[2])
		<< CR(qvec[0], qvec[1], PI)
		<< CZ(qvec[0], qvec[1])
		<< Measure(qvec[0], cbits[0])
		<< Measure(qvec[1], cbits[1])
		<< Measure(qvec[2], cbits[2])
		<< Measure(qvec[3], cbits[3]);

	std::string a = convert_qprog_to_quil(prog, machine);
	//write_to_native_quil_file(prog, machine, "E:\\q.txt");

	std::cout << (a);
	finalize();
}