
#include "Core/Utilities/Tools/Utils.h"
#include "Core/Utilities/QPandaNamespace.h"
#include "QAlg/DJ_Algorithm/DJ_Algorithm.h"
#include "Core/Core.h"
#include <vector>
#include<cmath>

using namespace std;
USING_QPANDA

QProg Deutsch_Jozsa_algorithm(vector<Qubit*> qubit_vector,
    Qubit* qubit2, 
    vector<ClassicalCondition> cbit_vector,
    DJ_Oracle oracle) {

    auto prog = CreateEmptyQProg();
    //Firstly, create a circuit container

	prog << X(qubit2);
    prog << apply_QGate(qubit_vector, H) << H(qubit2);
    // Perform Hadamard gate on all qubits

    prog << oracle(qubit_vector, qubit2);

    // Finally, Hadamard the first qubit and measure it
    prog << apply_QGate(qubit_vector, H) << MeasureAll(qubit_vector, cbit_vector);
    return prog;
}

QProg QPanda::deutschJozsaAlgorithm(vector<bool> boolean_function,QuantumMachine * qvm, DJ_Oracle oracle)
{
	if (boolean_function.size()== 0)
	{
		QCERR("param error");
		throw invalid_argument("param error");
	}
	auto size = (size_t)(log(boolean_function.size())/log(2));
	if (size == 0)
	{
		QCERR("param error");
		throw invalid_argument("param error");
	}
	auto qvec = qvm->allocateQubits(size);
	auto qubit = qvm->allocateQubit();
	auto cvec = qvm->allocateCBits(size);

	QProg prog;
	prog << Deutsch_Jozsa_algorithm(qvec, qubit, cvec, oracle);
	return prog;
}

