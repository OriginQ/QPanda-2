#include "gtest/gtest.h"
#include "QAlg/IQAE/IterativeQuantumAmplitudeEstimation.h"
#include "QPanda.h"


USING_QPANDA
using namespace std;

double run_circuit(QCircuit& cir,QVec& qvec) {
	int qnumber = qvec.size();
	auto qvm = initQuantumMachine(CPU);
	auto qubits = qvm->qAllocMany(qnumber);
	auto cbits = qvm->cAllocMany(4);
	auto prog = createEmptyQProg();
	prog << cir;
	auto result = probRunDict(prog, qubits[qnumber - 1]);
	double aim_res;

	for (auto& aiter : result)
	{
		cout << aiter.first << " : " << aiter.second << endl;
		aim_res = aiter.second;
	}
	
	cout << "=================================================" << endl;
	finalize();
	return aim_res;
}

bool test1()
{
	cout << "=====================  test 1  ====================" << endl;
	init();
	QuantumMachine* machine = initQuantumMachine(CPU);
	auto qvec = machine->qAllocMany(1);
	QCircuit cir = createEmptyCircuit();
	cir << H(qvec[0]);
	double _sour = run_circuit(cir, qvec);
	double iqae_res = iterative_amplitude_estimation(cir, qvec, 0.0001, 0.01);
	/* the epsilon of the result checking is 0.01 on the condition that the alg`s epsilon is 0.0001 ! */
	if (abs(_sour - iqae_res) > 0.1) 
	{
		return false;
	}
	cout << "the result of Iterative Quantum Amplitude Estimation is: " << iqae_res << endl << endl;
	return true;
}


bool test2()
{
	cout << "=====================  test 2  ====================" << endl;
	init();
	QuantumMachine* machine = initQuantumMachine(CPU);
	auto qvec = machine->qAllocMany(2);
	QCircuit cir = createEmptyCircuit();
	cir << RY(qvec[0], PI / 6.0) << X(qvec[1]).control(qvec[0]);
	double _sour = run_circuit(cir, qvec);
	double iqae_res = iterative_amplitude_estimation(cir, qvec, 0.0001, 0.01);
	/* the epsilon of the result checking is 0.01 on the condition that the alg`s epsilon is 0.0001 ! */
	if (abs(_sour - iqae_res) > 0.1)
	{
		return false;
	}
	cout << "the result of Iterative Quantum Amplitude Estimation is: " << iqae_res << endl << endl;
	return true;
}

bool test3()
{
	cout << "=====================  test 3  ====================" << endl;
	init();
	QuantumMachine* machine = initQuantumMachine(CPU);
	auto qvec = machine->qAllocMany(3);
	QCircuit cir = createEmptyCircuit();
	cir << H(qvec[0]) << RY(qvec[1], PI / 6.0) << RY(qvec[2], PI / 6.0);
	double _sour = run_circuit(cir, qvec);
	double iqae_res = iterative_amplitude_estimation(cir, qvec, 0.0001, 0.01);
	/* the epsilon of the result checking is 0.01 on the condition that the alg`s epsilon is 0.0001 ! */
	if (abs(_sour - iqae_res) > 0.1)
	{
		return false;
	}
	cout << "the result of Iterative Quantum Amplitude Estimation is: " << iqae_res << endl << endl;
	return true;

}

bool test4()
{
	cout << "=====================  test 4  ====================" << endl;
	init();
	QuantumMachine* machine = initQuantumMachine(CPU);
	auto qvec = machine->qAllocMany(4);
	QCircuit cir = createEmptyCircuit();
	cir << RY(qvec[3], PI / 6.0)
		<< X(qvec[1]).control(qvec[0]) 
		<< RY(qvec[2], PI / 6.0).control(qvec[0])
		<< RY(qvec[3], PI / 6.0).control(qvec[1]);
	double _sour = run_circuit(cir, qvec);
	double iqae_res = iterative_amplitude_estimation(cir, qvec, 0.0001, 0.01);
	/* the epsilon of the result checking is 0.01 on the condition that the alg`s epsilon is 0.0001 ! */
	if (abs(_sour - iqae_res) > 0.1)
	{
		return false;
	}
	cout << "the result of Iterative Quantum Amplitude Estimation is: " << iqae_res << endl << endl;
	return true;
}

TEST(IQAE, test)
{
    bool test_val = false;
    try
    {
		test_val = test1();
		test_val = test_val && test2();
		test_val = test_val && test3();
		test_val = test_val && test4();
    }
    catch (const std::exception& e)
    {
        cout << "Got a exception: " << e.what() << endl;
    }
    catch (...)
    {
        cout << "Got an unknow exception: " << endl;
    }

    ASSERT_TRUE(test_val);
    cout << endl;
}