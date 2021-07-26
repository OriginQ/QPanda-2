#include "gtest/gtest.h"
#include "QPanda.h"
#include <iostream>
#include <vector>
#include <stdio.h>
#include <memory>

#define PI 3.1415926
#define PRINT_TRACE 0

USING_QPANDA
using namespace std;
class QVMInit
{
public:
	QVMInit(QMachineType t = QMachineType::CPU)
		:m_qvm_type(t)
	{
		on_create();
	}
	~QVMInit() {
		finalize();
	}

	QVec allocate_qubits(size_t size) { return qAllocMany(size); }
	vector<ClassicalCondition> allocate_class_bits(size_t size) { return cAllocMany(size); }

protected:
	void on_create() {
		init(QMachineType::CPU);
	}

private:
	const QMachineType m_qvm_type;
};

#define QVM_INITIALIZE(qbit_num, cbit_num) QVMInit tmp_qvm; auto q = tmp_qvm.allocate_qubits(qbit_num); auto c = tmp_qvm.allocate_class_bits(cbit_num);

TEST(JudgeTwoNodeIterIsSwappable, nonGateNode)
{
	QVM_INITIALIZE(8, 8);

	QProg prog;
	QCircuit circuit3;
	circuit3 << X(q[2]) << RZ(q[1], PI / 2) << Y(q[0]) << CNOT(q[3], q[5]) << S(q[4]) << H(q[2]) << S(q[2]);

	prog << CNOT(q[0], q[3]) << RZ(q[1], PI / 2) << S(q[1]) << H(q[1]) << circuit3 << Z(q[0]) << H(q[1]).dagger() << SWAP(q[3], q[0]) << CNOT(q[0], q[3]).dagger()
		<< RY(q[3], PI / 6) << CNOT(q[0], q[1])
		<< MeasureAll(q, c);

	NodeIter itr1 = prog.getFirstNodeIter();
	NodeIter itr2 = itr1;
	++itr2;
	++itr2;
	++itr2;
	++itr2;

	bool bCouldBeExchanged = isSwappable(prog, itr1, itr2);
	/*if (bCouldBeExchanged)
	{
		std::cout << "could be exchanged." << std::endl;
	}
	else
	{
		std::cout << "could NOT be exchanged." << std::endl;
	}*/

	ASSERT_TRUE(!bCouldBeExchanged);
}

TEST(JudgeTwoNodeIterIsSwappable, measureAndResetNode)
{
	QVM_INITIALIZE(8, 8);

	QProg prog;
	QCircuit circuit3;

	circuit3 << X(q[2]) << RZ(q[1], PI / 2) << Y(q[0]) << CNOT(q[3], q[5]) << S(q[4]) << H(q[2]) << S(q[2]);

	prog << CNOT(q[0], q[3]) << RZ(q[1], PI / 2) << Reset(q[0]) << I(q[3]) << circuit3 << Z(q[0]) << H(q[1]).dagger() << SWAP(q[3], q[0]) << CNOT(q[0], q[3]).dagger()
		<< RY(q[3], PI / 6) << CNOT(q[0], q[1])
		<< MeasureAll(q, c);

	NodeIter itr1 = prog.getFirstNodeIter();
	NodeIter itr2 = itr1;
	++itr2;
	++itr2;
	++itr2;
	bool bCouldBeExchanged = isSwappable(prog, itr1, itr2);
	/*if (bCouldBeExchanged)
	{
		std::cout << "could be exchanged." << std::endl;
	}
	else
	{
		std::cout << "could NOT be exchanged." << std::endl;
	}*/

	ASSERT_TRUE(!bCouldBeExchanged);
}

TEST(JudgeTwoNodeIterIsSwappable, controlTest)
{
	QVM_INITIALIZE(8, 8);

	QProg prog;
	QProg prog3;
	QCircuit circuit3;
	QCircuit circuit4;

	QVec control_vec;
	control_vec.push_back(q[1]);
	auto gate1 = Y(q[5]);
	gate1.setControl(control_vec);

	circuit4 << S(q[3]) << H(q[c[4]]) << RZ(q[1], PI / 2).dagger() << CNOT(q[2], q[1]);
	circuit3 << X(q[2]) << RZ(q[1], PI / 2) << Y(q[0]) << CNOT(q[3], q[5]) << S(q[4]) << circuit4 << H(q[2]) << S(q[2]);
	circuit3.setDagger(true);
	//circuit3.setControl(control_vec);

	QProg branch_true;
	QProg branch_false;
	branch_true << I(q[1]) << CNOT(q[1], q[5]) << S(q[4]) << CNOT(q[1], q[5]).dagger() << Y(q[5]) << gate1;
	branch_false << H(q[1]) << CNOT(q[1], q[4]) << CNOT(q[5], q[4]);

	auto qif = CreateIfProg(c[1] > 5, branch_true, branch_false);


	prog << CNOT(q[0], q[3]) << CNOT(q[1], q[2]) << T(q[3]) << circuit3 << qif
		<< RY(q[3], PI / 6) << CNOT(q[0], q[1])
		<< MeasureAll(q, c);

	NodeIter itr1 = branch_true.getFirstNodeIter();
	NodeIter itr2 = branch_true.getLastNodeIter();

	bool bCouldBeExchanged = isSwappable(prog, itr1, itr2);
	/*if (bCouldBeExchanged)
	{
		std::cout << "could be exchanged." << std::endl;
	}
	else
	{
		std::cout << "could NOT be exchanged." << std::endl;
	}*/

	ASSERT_TRUE(bCouldBeExchanged);
}

TEST(JudgeTwoNodeIterIsSwappable, inDifferentSubProg)
{
	QVM_INITIALIZE(8, 8);

	QProg prog;
	QCircuit circuit3;

	QVec control_vec;
	control_vec.push_back(q[1]);
	auto gate1 = Y(q[5]);
	gate1.setControl(control_vec);

	circuit3 << X(q[2]) << RZ(q[1], PI / 2) << Y(q[0]) << CNOT(q[3], q[5]) << S(q[4]) << H(q[2]) << S(q[2]) << CNOT(q[0], q[3]).dagger();
	//circuit3.setDagger(true);

	prog << CNOT(q[0], q[3]) << circuit3 << Z(q[0]) << H(q[1]).dagger() << SWAP(q[3], q[0]) << CNOT(q[0], q[3]).dagger()
		<< RZ(q[1], PI / 2).dagger() << CNOT(q[1], q[2]) << T(q[3]) << circuit3
		<< RY(q[3], PI / 6) << CNOT(q[0], q[1])
		<< MeasureAll(q, c);

	NodeIter itr1 = prog.getFirstNodeIter();
	NodeIter itr2 = circuit3.getLastNodeIter();

	bool bCouldBeExchanged = isSwappable(prog, itr1, itr2);
	/*if (bCouldBeExchanged)
	{
		std::cout << "could be exchanged." << std::endl;
	}
	else
	{
		std::cout << "could NOT be exchanged." << std::endl;
	}*/

	ASSERT_TRUE(!bCouldBeExchanged);
}

TEST(JudgeTwoNodeIterIsSwappable, flowCtrlNodeTest)
{
	QVM_INITIALIZE(8, 8);

	QProg prog;
	QProg prog2;
	QProg prog3;
	QCircuit circuit3;
	QCircuit circuit4;

	QVec control_vec;
	control_vec.push_back(q[1]);
	auto gate1 = Y(q[5]);
	gate1.setControl(control_vec);

	circuit4 << S(q[3]) << H(q[c[4]]) << RZ(q[1], PI / 2).dagger() << CNOT(q[2], q[1]);
	circuit3 << X(q[2]) << RZ(q[1], PI / 2) << Y(q[0]) << CNOT(q[3], q[5]) << S(q[4]) << circuit4 << H(q[2]) << S(q[2]);
	//circuit3 = circuit4.dagger();
	circuit3.setDagger(true);
	//circuit3.setControl(control_vec);

	QProg branch_true;
	QProg branch_false;
	branch_true << H(q[1]) << CNOT(q[1], q[5]) << S(q[4]) <</* (c[1] = c[1] + 1) <<*/ H(q[5]) << Y(q[1]) << gate1;
	branch_false << H(q[1]) << CNOT(q[1], q[4]) << CNOT(q[5], q[4]);

	auto qif = CreateIfProg(c[1] > 5, branch_true, branch_false);

	QProg prog_in;
	prog_in << CNOT(q[4], q[5]) << c[0] << H(q[4]) << /*(c[0] = c[0] + 1) <<*/ T(q[1]);
	auto qwhile = CreateWhileProg(c[0] < 3, prog_in);

	prog2 << circuit3 << Z(q[0]) << H(q[1]) << Measure(q[2], c[0]) << CNOT(q[2], q[3]) << H(q[4]) << qwhile << S(q[1]) << RZ(q[1], PI / 2) << Y(q[2]) << prog3;

	prog << CNOT(q[0], q[3]) << I(q[0]) << RZ(q[1], PI / 2) << qwhile << S(q[1]) << qif << H(q[1]) << circuit3 << Z(q[0]) << H(q[1]).dagger() << SWAP(q[3], q[0]) << CNOT(q[0], q[3]).dagger()
		<< RZ(q[1], PI / 2).dagger() << prog2 << CNOT(q[1], q[2]) << T(q[3]) << circuit3 << qif
		<< RY(q[3], PI / 6) << CNOT(q[0], q[1])
		<< MeasureAll(q, c);

	NodeIter itr1 = prog.getFirstNodeIter();
	NodeIter itr2 = itr1;
	++itr2;
	++itr2;
	++itr2;
	++itr2;
	++itr2;
	++itr2;
	++itr2;
	++itr2;
	++itr2;
	++itr2;
	++itr2;
	bool bCouldBeExchanged = isSwappable(prog, itr1, itr2);
	/*if (bCouldBeExchanged)
	{
		std::cout << "could be exchanged." << std::endl;
	}
	else
	{
		std::cout << "could NOT be exchanged." << std::endl;
	}*/

	ASSERT_TRUE(bCouldBeExchanged);
}

TEST(JudgeTwoNodeIterIsSwappable, nestingTest)
{
	QVM_INITIALIZE(8, 8);

	QProg prog;
	QProg prog2;
	QProg prog3;
	QCircuit circuit3;
	QCircuit circuit4;

	QVec control_vec;
	control_vec.push_back(q[1]);
	auto gate1 = Y(q[5]);
	gate1.setControl(control_vec);

	circuit4 << S(q[3]) << H(q[c[4]]) << RZ(q[1], PI / 2).dagger() << CNOT(q[2], q[1]);
	circuit3 << X(q[2]) << RZ(q[1], PI / 2) << Y(q[0]) << CNOT(q[3], q[5]) << S(q[4]) << circuit4 << H(q[2]) << S(q[2]);
	circuit3.setDagger(true);
	//circuit3.setControl(control_vec);

	QProg branch_true;
	QProg branch_false;
	branch_true << H(q[1]) << CNOT(q[1], q[5]) << S(q[4]) <</* (c[1] = c[1] + 1) <<*/ H(q[5]) << Y(q[1]) << gate1;
	branch_false << H(q[1]) << CNOT(q[1], q[4]) << CNOT(q[5], q[4]);

	auto qif = CreateIfProg(c[1] > 5, branch_true, branch_false);

	QProg prog_in;
	prog_in << CNOT(q[4], q[5]) << c[0] << H(q[4]) << /*(c[0] = c[0] + 1) <<*/ T(q[1]);
	auto qwhile = CreateWhileProg(c[0] < 3, prog_in);

	prog2 << circuit3 << Z(q[0]) << H(q[1]) << Measure(q[2], c[0]) << CNOT(q[2], q[3]) << H(q[4]) << qwhile << S(q[1]) << RZ(q[1], PI / 2) << Y(q[2]) << prog3;

	prog << CNOT(q[0], q[3]) << qwhile << S(q[1]) << qif << H(q[1]) << circuit3 << Z(q[0]) << H(q[1]).dagger() << CNOT(q[0], q[3]).dagger()
		<< RZ(q[1], PI / 2).dagger() << prog2 << CNOT(q[1], q[2]) << T(q[3]) << circuit3 << qif
		<< RY(q[3], PI / 6) << CNOT(q[0], q[1])
		<< MeasureAll(q, c);

	NodeIter itr1 = prog.getFirstNodeIter();
	NodeIter itr2 = itr1;
	++itr2;
	++itr2;
	++itr2;
	++itr2;
	++itr2;
	++itr2;
	++itr2;
	++itr2;

	bool bCouldBeExchanged = isSwappable(prog, itr1, itr2);
	/*if (bCouldBeExchanged)
	{
		std::cout << "could be exchanged." << std::endl;
	}
	else
	{
		std::cout << "could NOT be exchanged." << std::endl;
	}*/

	ASSERT_TRUE(bCouldBeExchanged);
}

TEST(JudgeTwoNodeIterIsSwappable, judgeMatrix)
{
	QVM_INITIALIZE(8, 8);
	QProg prog;
	QProg prog2;
	QCircuit circuit3;
	QCircuit circuit4;

	QVec control_vec;
	control_vec.push_back(q[1]);

	auto gate1 = Y(q[5]);
	gate1.setControl(control_vec);

	circuit4 << S(q[3]) << H(q[c[4]]) << RZ(q[0], PI / 2).dagger() << CNOT(q[2], q[5]);
	circuit3 << X(q[2]) << RZ(q[0], PI / 2) << Y(q[0]) << CNOT(q[3], q[5]) << S(q[4]) << circuit4 << H(q[2]) << S(q[2]);
	circuit3.setDagger(true);
	circuit3.setControl(control_vec);

	QProg prog_in;
	prog_in << CNOT(q[4], q[5]) << c[0] << H(q[4]) << /*(c[0] = c[0] + 1) <<*/ T(q[1]);
	auto qwhile = CreateWhileProg(c[0] < 3, prog_in);

	prog2 << circuit3 << Z(q[0]) << H(q[1]);

	prog << CNOT(q[0], q[3]) << I(q[0]) << RZ(q[1], PI / 2) << qwhile << S(q[1]) << H(q[1]) << circuit3 
		<< Z(q[0]) << H(q[1]).dagger() << SWAP(q[3], q[0]) << CNOT(q[0], q[3]).dagger()
		<< MeasureAll(q, c);

	NodeIter itr1 = circuit3.getFirstNodeIter();
	NodeIter itr2 = circuit3.getLastNodeIter();

	bool bCouldBeExchanged = isSwappable(prog, itr1, itr2);
	/*if (bCouldBeExchanged)
	{
		std::cout << "could be exchanged." << std::endl;
	}
	else
	{
		std::cout << "could NOT be exchanged." << std::endl;
	}*/

	ASSERT_TRUE(!bCouldBeExchanged);
}

TEST(JudgeTwoNodeIterIsSwappable, doubleGateTest)
{
	QVM_INITIALIZE(8, 8);

	QProg prog;
	QProg prog3;
	QCircuit circuit3;

	circuit3 << X(q[2]) << S(q[1]) << H(q[2]) << S(q[2]) << Z(q[0]);
	circuit3.setDagger(true);

	QProg branch_true;
	QProg branch_false;
	branch_true << H(q[2]) << CNOT(q[2], q[5]) << S(q[4]) << (c[1] = c[1] + 1) << H(q[5]) << Y(q[2]);
	branch_false << H(q[4]) << CNOT(q[5], q[4]);

	auto qif = CreateIfProg(c[1] > 5, branch_true, branch_false);

	QProg prog_in;
	prog_in << CNOT(q[4], q[5]) << c[0] << H(q[4]);
	auto qwhile = CreateWhileProg(c[0] < 3, prog_in);

	QVec control_vec;
	control_vec.push_back(q[1]);	

	auto gate2 = CNOT(q[0], q[3]).dagger();
	gate2.setControl(control_vec);
	prog << CNOT(q[0], q[3]) << I(q[0]) << RZ(q[1], PI / 2) << qwhile << qif << H(q[1]) << S(q[1]) /*<< S(q[1]).dagger()*/ 
		<< circuit3 << Z(q[0]) << H(q[1]).dagger() << RZ(q[1], PI / 2).dagger() << gate2
		<< MeasureAll(q, c);

	NodeIter itr1 = prog.getFirstNodeIter();

#if PRINT_TRACE
	printAllNodeType(QProg(*itr1));
	auto porg111 = QProg(*itr1);
	porg111 << I(q[1]);
	auto mat_tmp1 = getCircuitMatrix(porg111);
	cout << mat_tmp1 << endl;
#endif

	NodeIter itr2 = itr1;
	++itr2;
	++itr2;
	++itr2;
	++itr2;
	++itr2;
	++itr2;
	++itr2;
	++itr2;
	++itr2;
	++itr2;
	++itr2;

#if PRINT_TRACE
	printAllNodeType(QProg(*itr2));
	auto mat_tmp2 = getCircuitMatrix(QProg(*itr2));
	cout << mat_tmp2 << endl;
#endif

	bool bCouldBeExchanged = isSwappable(prog, itr1, itr2);
	/*if (bCouldBeExchanged)
	{
		std::cout << "could be exchanged." << std::endl;
	}
	else
	{
		std::cout << "could NOT be exchanged." << std::endl;
	}*/

	ASSERT_TRUE(bCouldBeExchanged);
}

TEST(JudgeTwoNodeIterIsSwappable, doubleGateTest2)
{
	QVM_INITIALIZE(8, 8);

	QProg prog;
	QProg prog3;
	QCircuit circuit3;

	circuit3 << X(q[2]) << S(q[1]) << H(q[2]) << S(q[2]) << Z(q[0]);
	circuit3.setDagger(true);

	QProg branch_true;
	QProg branch_false;
	branch_true << H(q[2]) << CNOT(q[2], q[5]) << S(q[4]) << (c[1] = c[1] + 1) << H(q[5]) << Y(q[2]);
	branch_false << H(q[4]) << CNOT(q[5], q[4]);

	auto qif = CreateIfProg(c[1] > 5, branch_true, branch_false);

	QProg prog_in;
	prog_in << CNOT(q[4], q[5]) << c[0] << H(q[4]);
	auto qwhile = CreateWhileProg(c[0] < 3, prog_in);

	QVec control_vec;
	control_vec.push_back(q[1]);

	auto gate2 = CNOT(q[0], q[3]).dagger();
	gate2.setControl(control_vec);
	prog << CNOT(q[0], q[3]) << I(q[0]) << RZ(q[1], PI / 2) << qwhile << qif << H(q[1]) << S(q[1])
		<< circuit3 << Z(q[0]) << H(q[1]).dagger() << RZ(q[1], PI / 2).dagger() << gate2
		<< MeasureAll(q, c);

	NodeIter itr1 = prog.getFirstNodeIter();

	NodeIter itr2 = prog.getLastNodeIter();
	--itr2;

	bool bCouldBeExchanged = isSwappable(prog, itr1, itr2);
	/*if (bCouldBeExchanged)
	{
		std::cout << "could be exchanged." << std::endl;
	}
	else
	{
		std::cout << "could NOT be exchanged." << std::endl;
	}*/
	//the function return value's type is bool.
	ASSERT_TRUE(bCouldBeExchanged);
	//cout << " JudgeTwoNodeIterIsSwappable tests over." << endl;
}