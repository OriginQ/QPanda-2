#include "gtest/gtest.h"
#include "QPanda.h"
#include <iostream>
#include <vector>
#include <stdio.h>
#include <memory>

#define PI 3.1415926

#define PRINT_TRACE 0

USING_QPANDA

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

TEST(QProgToMatrix, setControlTest)
{
	QVM_INITIALIZE(8, 8);

	bool result = true;

	if (0)
	{
		QCircuit cir1;
		QVec control_vec;
		control_vec.push_back(q[0]);
		auto test_gate = CNOT(q[1], q[2]);
		test_gate.setControl(control_vec);
		cir1 << test_gate;

#if PRINT_TRACE
		auto mat = getCircuitMatrix(cir1);
		cout << "matrix of cir3:" << endl;
		cout << mat << endl;
#endif // PRINT_TRACE
	}

	if (1)
	{
		QCircuit cir2;
		QVec control_vec2;
		control_vec2.push_back(q[1]);
		control_vec2.push_back(q[7]);
		auto test_gate = CNOT(q[6], q[3]);
		test_gate.setControl(control_vec2);
		cir2 << test_gate << I(q[4]) << I(q[5]);
		auto mat = getCircuitMatrix(cir2);

#if PRINT_TRACE
		cout << cir2 << endl;
		cout << "matrix of cir2:" << endl;
		cout << mat << endl;
#endif

		QCircuit cir3;
		auto test_gate2 = CNOT(q[6], q[7]);
		control_vec2.clear();
		control_vec2.push_back(q[4]);
		control_vec2.push_back(q[5]);
		test_gate2.setControl(control_vec2);
		cir3 << SWAP(q[3], q[7]) << SWAP(q[3], q[5]) << SWAP(q[1], q[4]) << test_gate2 << SWAP(q[1], q[4]) << SWAP(q[3], q[5]) << SWAP(q[3], q[7]) << I(q[1]);
		auto mat3 = getCircuitMatrix(cir3);

#if PRINT_TRACE
		cout << cir3 << endl;
		cout << "matrix of cir3:" << endl;
		cout << mat3 << endl;
#endif
		if (0 == mat_compare(mat, mat3))
		{
			cout << "test 2 ok." << endl;
		}
		else
		{
			cout << "test 2 failed." << endl;
			result = false;
		}
	}

	if (1)
	{
		QCircuit cir2;
		QVec control_vec2;
		control_vec2.push_back(q[4]);
		control_vec2.push_back(q[5]);
		auto test_gate = CNOT(q[6], q[7]);
		test_gate.setControl(control_vec2);
		cir2 << SWAP(q[3], q[7]) << SWAP(q[1], q[4]) << SWAP(q[5], q[7]) << test_gate;
		auto mat = getCircuitMatrix(cir2);

#if PRINT_TRACE
		cout << cir2 << endl;
		cout << "matrix of cir2:" << endl;
		cout << mat << endl;
#endif

		QCircuit cir3;
		auto test_gate2 = CNOT(q[6], q[7]);
		control_vec2.clear();
		control_vec2.push_back(q[4]);
		control_vec2.push_back(q[5]);
		test_gate2.setControl(control_vec2);
		cir3 << SWAP(q[1], q[4]) << SWAP(q[3], q[5]) << SWAP(q[3], q[7]) << test_gate;
		auto mat3 = getCircuitMatrix(cir3);

#if PRINT_TRACE
		cout << cir3 << endl;
		cout << "matrix of cir3:" << endl;
		cout << mat3 << endl;
#endif

		if (0 == mat_compare(mat, mat3))
		{
			cout << "test 3 ok." << endl;
		}
		else
		{
			cout << "test 3 failed." << endl;
			result = false;
		}
	}

	if (1)
	{
		QCircuit cir2;
		QVec control_vec2;
		control_vec2.push_back(q[2]);
		control_vec2.push_back(q[5]);
		auto test_gate = SWAP(q[3], q[1]);
		test_gate.setControl(control_vec2);
		cir2 << test_gate << I(q[4]);
		auto mat = getCircuitMatrix(cir2);

#if PRINT_TRACE
		cout << cir2 << endl;
		cout << "matrix of cir2:" << endl;
		cout << mat << endl;
#endif

		QCircuit cir3;
		auto test_gate2 = SWAP(q[4], q[5]);
		control_vec2.clear();
		control_vec2.push_back(q[2]);
		control_vec2.push_back(q[3]);
		test_gate2.setControl(control_vec2);
		cir3 << SWAP(q[3], q[5]) << SWAP(q[1], q[4]) << test_gate2 << SWAP(q[1], q[4]) << SWAP(q[3], q[5]);
		auto mat3 = getCircuitMatrix(cir3);

#if PRINT_TRACE
		cout << cir3 << endl;
		cout << "matrix of cir3:" << endl;
		cout << mat3 << endl;
#endif

		if (0 == mat_compare(mat, mat3))
		{
			cout << "test 4 ok." << endl;
		}
		else
		{
			cout << "test 4 failed." << endl;
			result = false;
		}
	}

	if (1)
	{
		QCircuit cir2;
		QVec control_vec2;
		control_vec2.push_back(q[3]);
		auto test_gate = CNOT(q[0], q[1]);
		test_gate.setControl(control_vec2);
		cir2 << test_gate << I(q[2]);
		cout << cir2 << endl;
		auto mat = getCircuitMatrix(cir2);
#if PRINT_TRACE
		cout << "matrix of cir2:" << endl;
		cout << mat << endl;
#endif
		QCircuit cir3;
		auto test_gate2 = CNOT(q[2], q[3]);
		control_vec2.clear();
		control_vec2.push_back(q[1]);
		test_gate2.setControl(control_vec2);
		cir3 << SWAP(q[0], q[2]) << SWAP(q[3], q[1]) << test_gate2 << SWAP(q[1], q[3]) << SWAP(q[0], q[2]);
		cout << cir3 << endl;
		auto mat3 = getCircuitMatrix(cir3);
#if PRINT_TRACE
		cout << "matrix of cir3:" << endl;
		cout << mat3 << endl;
#endif

		if (0 == mat_compare(mat, mat3))
		{
			cout << "test 5 ok." << endl;
		}
		else
		{
			cout << "test 5 failed." << endl;
			result = false;
		}
	}

	ASSERT_TRUE(result);
}

TEST(QProgToMatrix, circuitTest)
{
	QVM_INITIALIZE(8, 8);

	float delta1 = PI;
	float delta2 = PI / 2;
	float delta3 = PI / 3;
	float delta4 = PI / 4;
	float delta5 = PI / 5;

	QStat mat1;
	QStat mat2;

	bool result = true;
	{
		QProg prog1;
		QProg prog2;
		prog1 << RZ(q[1], PI / 2) << (H(q[1])) << (CNOT(q[0], q[1])) << (H(q[1]));
		prog2 << (H(q[1])) << (CNOT(q[0], q[1])) << (H(q[1])) << RZ(q[1], PI / 2);
		mat1 = getCircuitMatrix(prog1);
		mat2 = getCircuitMatrix(prog2);
		if (0 == mat_compare(mat1, mat2))
		{
			cout << "test cir1 matrix ok." << endl;
		}
		else
		{
			cout << "test cir1 matrix failed." << endl;
			result = false;
		}
	}

	{
		QProg prog1;
		QProg prog2;
		prog1 << RZ(q[0], PI / 2) << (CNOT(q[0], q[1])) << (H(q[1]));
	    prog2 << (CNOT(q[0], q[1])) << RZ(q[0], PI / 2) << (H(q[1]));
		mat1 = getCircuitMatrix(prog1);
		mat2 = getCircuitMatrix(prog2);
		if (0 == mat_compare(mat1, mat2))
		{
			cout << "test cir2 matrix ok." << endl;
		}
		else
		{
			cout << "test cir2 matrix failed." << endl;
			result = false;
		}
	}
	
	{
		QProg prog1;
		QProg prog2;
		prog1 << RZ(q[1], delta1) << (CNOT(q[0], q[1])) << RZ(q[1], delta2) << (CNOT(q[0], q[1])) << RZ(q[0], delta3) << RZ(q[1], delta4) << CNOT(q[1], q[0]);
		prog2 << (CNOT(q[0], q[1])) << RZ(q[1], delta2) << (CNOT(q[0], q[1])) << RZ(q[0], delta3) << RZ(q[1], delta4 + delta1) << CNOT(q[1], q[0]);
		mat1 = getCircuitMatrix(prog1);
		mat2 = getCircuitMatrix(prog2);
		if (0 == mat_compare(mat1, mat2))
		{
			cout << "test cir3 matrix ok." << endl;
		}
		else
		{
			cout << "test cir3 matrix failed." << endl;
			result = false;
		}
	}

	{
		QProg prog1;
		QProg prog2;
		prog1 << (CNOT(q[0], q[2])) << RZ(q[2], delta1) << (CNOT(q[1], q[2])) << RZ(q[2], delta2) << (CNOT(q[0], q[2])) << RZ(q[2], delta3) << (CNOT(q[1], q[2]));
		prog2 << (CNOT(q[1], q[2])) << RZ(q[2], delta3) << (CNOT(q[0], q[2])) << RZ(q[2], delta2) << (CNOT(q[1], q[2])) << RZ(q[2], delta1) << (CNOT(q[0], q[2]));

		mat1 = getCircuitMatrix(prog1);
		mat2 = getCircuitMatrix(prog2);
		if (0 == mat_compare(mat1, mat2))
		{
			cout << "test cir4 matrix ok." << endl;
		}
		else
		{
			cout << "test cir4 matrix failed." << endl;
			result = false;
		}
	}

	{
		QProg prog1;
		QProg prog2;
		prog1 << (CNOT(q[0], q[2])) << RZ(q[2], delta1) << (CNOT(q[1], q[2])) << RZ(q[2], delta2) << (CNOT(q[1], q[2]));
		prog2 << (CNOT(q[1], q[2])) << (CNOT(q[0], q[2])) << RZ(q[2], delta2) << (CNOT(q[1], q[2])) << RZ(q[2], delta1);

		mat1 = getCircuitMatrix(prog1);
		mat2 = getCircuitMatrix(prog2);
		if (0 == mat_compare(mat1, mat2))
		{
			cout << "test cir5 matrix ok." << endl;
		}
		else
		{
			cout << "test cir5 matrix failed." << endl;
			result = false;
		}
	}

	{
		QProg prog1;
		QProg prog2;
		prog1 << (CNOT(q[0], q[1])) << RZ(q[1], delta2) << (CNOT(q[0], q[1])) << RZ(q[0], delta3) << RZ(q[1], delta1 + delta4) << (CNOT(q[1], q[0]));
		prog2 << RZ(q[0], delta3) << RZ(q[1], delta1 + delta4) << (CNOT(q[1], q[0])) << RZ(q[0], delta2);

		mat1 = getCircuitMatrix(prog1);
		mat2 = getCircuitMatrix(prog2);
		if (0 == mat_compare(mat1, mat2))
		{
			cout << "test cir6 matrix ok." << endl;
		}
		else
		{
			cout << "test cir6 matrix failed." << endl;
			result = false;
		}
	}

	{
		QProg prog1;
		QProg prog2;
		prog1 << H(q[0]) << T(q[0]) << S(q[0]) << S(q[0]) << H(q[1]);
		prog2 << H(q[1]) << H(q[0]) << T(q[0]) << S(q[0]) << S(q[0]);

		mat1 = getCircuitMatrix(prog1);
		mat2 = getCircuitMatrix(prog2);
		if (0 == mat_compare(mat1, mat2))
		{
			cout << "test cir7 matrix ok." << endl;
		}
		else
		{
			cout << "test cir7 matrix failed." << endl;
			result = false;
		}
	}

	{
		QProg prog1;
		QProg prog2;
		prog1 << (CNOT(q[2], q[1])) << (CNOT(q[2], q[3])) << (CNOT(q[1], q[3])) << RZ(q[3], delta1) << (CNOT(q[1], q[3])) << RZ(q[3], delta2) << (CNOT(q[0], q[3])) <<
			RZ(q[3], delta3) << (CNOT(q[2], q[3])) << (CNOT(q[1], q[3])) << RZ(q[3], delta4) << (CNOT(q[0], q[3])) << (CNOT(q[1], q[3]));
		prog2 << (CNOT(q[2], q[3])) << RZ(q[3], delta2) << (CNOT(q[0], q[3])) << RZ(q[3], delta3) << (CNOT(q[1], q[3])) << RZ(q[3], delta4) << (CNOT(q[0], q[3])) <<
			(CNOT(q[2], q[3])) << RZ(q[3], delta1) << (CNOT(q[1], q[3])) << (CNOT(q[2], q[1]));

		mat1 = getCircuitMatrix(prog1);
		mat2 = getCircuitMatrix(prog2);
		if (0 == mat_compare(mat1, mat2))
		{
			cout << "test cir8 matrix ok." << endl;
		}
		else
		{
			cout << "test cir8 matrix failed." << endl;
			result = false;
		}
	}

	{
		QProg prog1;
		QProg prog2;
		prog1 << (CNOT(q[2], q[1])) << (CNOT(q[1], q[3])) << (CNOT(q[2], q[3])) << RZ(q[3], delta1) << (CNOT(q[1], q[3])) << RZ(q[3], delta2) << (CNOT(q[0], q[3])) <<
			RZ(q[3], delta3) << (CNOT(q[2], q[3])) << (CNOT(q[1], q[3])) << RZ(q[3], delta4) << (CNOT(q[0], q[3])) << (CNOT(q[1], q[3]));
		prog2 << (CNOT(q[2], q[3])) << RZ(q[3], delta2) << (CNOT(q[0], q[3])) << RZ(q[3], delta3) << (CNOT(q[1], q[3])) << RZ(q[3], delta4) << (CNOT(q[0], q[3])) <<
			(CNOT(q[2], q[3])) << RZ(q[3], delta1) << (CNOT(q[1], q[3])) << (CNOT(q[2], q[1]));

		mat1 = getCircuitMatrix(prog1);
		mat2 = getCircuitMatrix(prog2);
		if (0 == mat_compare(mat1, mat2))
		{
			cout << "test cir9 matrix ok." << endl;
		}
		else
		{
			cout << "test cir9 matrix failed." << endl;
			result = false;
		}
	}

	{
		QProg prog1;
		QProg prog2;
		prog1 << (CNOT(q[2], q[1])) << (CNOT(q[0], q[3])) << (CNOT(q[1], q[3])) << RZ(q[3], delta1) << (CNOT(q[1], q[3])) << (CNOT(q[2], q[3])) << RZ(q[3], delta2) <<
			CNOT(q[0], q[3]) << RZ(q[3], delta3) << (CNOT(q[1], q[3])) << RZ(q[3], delta4);
		prog2 << (CNOT(q[2], q[3])) << RZ(q[3], delta3) << (CNOT(q[0], q[3])) << RZ(q[3], delta2) << (CNOT(q[1], q[3])) << RZ(q[3], delta1) << (CNOT(q[0], q[3])) <<
			(CNOT(q[2], q[3])) << RZ(q[3], delta4) << (CNOT(q[2], q[1]));

		mat1 = getCircuitMatrix(prog1);
		mat2 = getCircuitMatrix(prog2);
		if (0 == mat_compare(mat1, mat2))
		{
			cout << "test cir10 matrix ok." << endl;
		}
		else
		{
			cout << "test cir10 matrix failed." << endl;
			result = false;
		}
	}

	{
		QProg prog1;
		QProg prog2;
		prog1 << (CNOT(q[2], q[1])) << RZ(q[3], delta1) << (CNOT(q[2], q[3])) << RZ(q[3], delta2) << (CNOT(q[1], q[3])) << RZ(q[3], delta3) << (CNOT(q[1], q[3])) <<
			(CNOT(q[0], q[3])) << RZ(q[3], delta4) << (CNOT(q[2], q[3])) << (CNOT(q[1], q[3])) << RZ(q[3], delta5) << (CNOT(q[0], q[3])) << (CNOT(q[1], q[3]));
		prog2 << RZ(q[3], delta1) << (CNOT(q[1], q[3])) << RZ(q[3], delta3) << (CNOT(q[2], q[3])) << (CNOT(q[0], q[3])) << RZ(q[3], delta5) << (CNOT(q[1], q[3])) <<
			RZ(q[3], delta4) << (CNOT(q[0], q[3])) << RZ(q[3], delta2) << (CNOT(q[2], q[3])) << (CNOT(q[2], q[1]));

		mat1 = getCircuitMatrix(prog1);
		mat2 = getCircuitMatrix(prog2);
		if (0 == mat_compare(mat1, mat2))
		{
			cout << "test cir11 matrix ok." << endl;
		}
		else
		{
			cout << "test cir11 matrix failed." << endl;
			result = false;
		}
	}

	{
		QProg prog1;
		QProg prog2;
		prog1 << (CNOT(q[2], q[1])) << (CNOT(q[0], q[3])) << (CNOT(q[1], q[3])) << RZ(q[3], delta1) << (CNOT(q[1], q[3])) << (CNOT(q[2], q[3])) << RZ(q[3], delta2) <<
			(CNOT(q[0], q[3])) << RZ(q[3], delta3) << (CNOT(q[1], q[3])) << RZ(q[3], delta4) << (CNOT(q[2], q[3])) << (CNOT(q[1], q[3]));
		prog2 << (CNOT(q[2], q[3])) << RZ(q[3], delta3) << (CNOT(q[0], q[3])) << RZ(q[3], delta2) << (CNOT(q[1], q[3])) << RZ(q[3], delta1) << (CNOT(q[0], q[3])) <<
			(CNOT(q[2], q[3])) << RZ(q[3], delta4) << (CNOT(q[1], q[3])) << (CNOT(q[2], q[1]))/* << Reset(q[2])*/;

		mat1 = getCircuitMatrix(prog1);
		mat2 = getCircuitMatrix(prog2);
		if (0 == mat_compare(mat1, mat2))
		{
			cout << "test cir12 matrix ok." << endl;
		}
		else
		{
			cout << "test cir12 matrix failed." << endl;
			result = false;
		}
	}

	ASSERT_TRUE(result);
}

TEST(QProgToMatrix, swapTest)
{
	QVM_INITIALIZE(8, 8);

	bool result = true;

	{
		QCircuit cir2;
		QVec control_vec2;
		control_vec2.push_back(q[3]);
		control_vec2.push_back(q[1]);
		auto test_gate = SWAP(q[5], q[4]);
		test_gate.setControl(control_vec2);
		cir2 << test_gate << I(q[2]);
		auto mat = getCircuitMatrix(cir2);

#if PRINT_TRACE
		cout << cir2 << endl;
		cout << "matrix of cir2:" << endl;
		cout << mat << endl;
#endif

		QCircuit cir3;
		auto test_gate2 = SWAP(q[4], q[5]);
		control_vec2.clear();
		control_vec2.push_back(q[2]);
		control_vec2.push_back(q[3]);
		test_gate2.setControl(control_vec2);
		cir3 << SWAP(q[1], q[2]) << test_gate2 << SWAP(q[1], q[2]);
		auto mat3 = getCircuitMatrix(cir3);

#if PRINT_TRACE
		cout << cir3 << endl;
		cout << "matrix of cir3:" << endl;
		cout << mat3 << endl;
#endif

		if (0 == mat_compare(mat, mat3))
		{
			cout << "test 4 ok." << endl;
		}
		else
		{
			cout << "test 4 failed." << endl;
			result = false;
		}
	}

	ASSERT_TRUE(result);
}

bool test_getProgMatrix1()
{
	auto qvm = initQuantumMachine(QMachineType::CPU);
	auto q = qvm->allocateQubits(8);
	auto c = qvm->allocateCBits(8);

	QProg prog1;
	QProg prog2;

	/*prog1 << (CNOT(q[0], q[3])).control(q[1]) << I(q[2]);
	prog2 << (CNOT(q[0], q[3])).control(q[2]) << I(q[1]);*/

	/*prog1 << RX(q[0], PI/6.0) << RX(q[0], PI / 7.0) << RX(q[0], PI / 3.0);
	prog2 << RX(q[0], (27.0 * PI) / 42.0);*/

	/*prog1 << RY(q[0], PI / 6.0) << RY(q[0], PI / 7.0) << RY(q[0], PI / 3.0);
	prog2 << RY(q[0], (27.0 * PI) / 42.0);*/

	prog1 << RZ(q[0], PI / 6.0) << RZ(q[0], PI / 7.0) << RZ(q[0], PI / 3.0);
	prog2 << RZ(q[0], (27.0 * PI) / 42.0);

	QStat result_mat1 = getCircuitMatrix(prog1);
	QStat result_mat2 = getCircuitMatrix(prog2);

	cout << "result_mat1" << result_mat1 << endl;
	cout << "result_mat2" << result_mat2 << endl;
	if (result_mat1 != result_mat2)
	{
		cout << "test failed." << endl;
	}

	destroyQuantumMachine(qvm);
	return true;
}

bool test_getProgMatrix_2()
{
	auto qvm = initQuantumMachine(QMachineType::CPU);
	auto q = qvm->allocateQubits(8);
	auto c = qvm->allocateCBits(8);

	QProg prog1;

	prog1 << H(q[0]) << I(q[1]);

	QStat result_mat1 = getCircuitMatrix(prog1/*, true*/);

	cout << "result_mat1" << result_mat1 << endl;

	destroyQuantumMachine(qvm);
	return true;
}

TEST(QProgToMatrix, test1)
{
	bool test_val = false;
	try
	{
		//test_val = test_getProgMatrix1();
		test_val = test_getProgMatrix_2();
	}
	catch (const std::exception& e)
	{
		cout << "Got a exception: " << e.what() << endl;
	}
	catch (...)
	{
		cout << "Got an unknow exception: " << endl;
	}

	cout << "QProgToMatrix test over, press Enter to continue." << endl;
	getchar();

	ASSERT_TRUE(test_val);
}