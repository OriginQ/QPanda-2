#include "Core/Core.h"
#include "Core/Utilities/Tools/Utils.h"
#include "QAlg/Grover/QuantumWalkGroverAlg.h"

USING_QPANDA
using namespace std;

QCircuit QPanda::build_coin_circuit(QVec &coin_qubits, QVec &index_qubits, QCircuit cir_mark)
{
	QCircuit coin_cir = QCircuit();
	auto coin_h_cir = apply_QGate(coin_qubits, H);

	cir_mark.setControl(coin_qubits);
	QCircuit mark_cir;
	mark_cir << apply_QGate(coin_qubits, X) << cir_mark << apply_QGate(coin_qubits, X);

	const size_t coin_qubits_size = coin_qubits.size();
	if (coin_qubits_size != index_qubits.size())
	{
		QCERR_AND_THROW_ERRSTR(runtime_error, "Error: error size of coin_qubits.");
	}

	QCircuit swap_cir;
	for (size_t i = 0; i < coin_qubits_size; ++i)
	{
		swap_cir << SWAP(coin_qubits[i], index_qubits[i]);
	}

	coin_cir << coin_h_cir << mark_cir << coin_h_cir << swap_cir;

	return coin_cir;
}

QProg QPanda::quantum_walk_alg(QCircuit cir_oracle,
	QCircuit cir_coin,
	const QVec &index_qubits,
	const QVec &ancilla_qubits,
	size_t repeat) 
{
	QProg quantum_walk_prog;

	//initial state prepare
	QCircuit circuit_prepare = apply_QGate(index_qubits, H);
	quantum_walk_prog << circuit_prepare;

	//anclilla qubits
	//quantum_walk_prog << X(ancilla_qubits.front()) << H(ancilla_qubits.front()) << X(ancilla_qubits.back());
	quantum_walk_prog << X(ancilla_qubits.back()) << H(ancilla_qubits.back());

	//repeat oracle
	for (size_t i = 0; i < repeat; ++i)
	{
		quantum_walk_prog << cir_oracle << cir_coin;
	}

	return quantum_walk_prog;
}