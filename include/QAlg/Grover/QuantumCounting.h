#ifndef  _QUANTUM_COUNTING_H
#define  _QUANTUM_COUNTING_H

#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Tools/Utils.h"
#include "Core/Utilities/QPandaNamespace.h"
#include <vector>

QPANDA_BEGIN

#define PRINT_TRACE 0
#if PRINT_TRACE
#define PTrace printf
#define PTraceMat(mat) (std::cout << (mat) << endl)
#define PTraceQCircuit(string, cir) (std::cout << string << endl << cir << endl)
#define PTraceQCirMat(string, cir) {auto m = getCircuitMatrix(cir); std::cout << string << endl << m << endl;}
#else
#define PTrace
#define PTraceMat(mat)
#define PTraceMat(string, cir)
#define PTraceQCirMat(string, cir)
#endif

class AbstractQuantumCounting
{
public:
	virtual size_t qu_counting() = 0;
};

class QuantumCounting : public AbstractQuantumCounting
{
public:
	QuantumCounting(QuantumMachine *qvm, QCircuit cir_oracle, QCircuit cir_diffusion, 
		const QVec &data_index_qubits, const QVec &ancilla_qubits)
		:m_qvm(*qvm)
		, m_cir_oracle(cir_oracle)
		, m_cir_diffusion(cir_diffusion)
		, m_data_index_qubits(data_index_qubits)
		, m_ancilla_qubits(ancilla_qubits)
		, m_index_qubit_size(data_index_qubits.size())
	{}
	~QuantumCounting() {}

	size_t qu_counting() override {
		auto qu_count_prog = build_qu_counting_prog();

		//PTraceQCircuit("qu_count_prog", qu_count_prog);

		auto result = probRunDict(qu_count_prog, m_work_qubits);

		std::cout << "the result" << std::endl;
		size_t target_result_index = 0;
		double max_val = 0.0;
		size_t i = 0;
		for (const auto& aiter : result)
		{
			if (aiter.second > max_val)
			{
				max_val = aiter.second;
				target_result_index = i;
			}
			++i;
			std::cout << aiter.first << " : " << aiter.second << std::endl;
		}
		std::cout << "result end" << std::endl;

		double v = pow(2, m_index_qubit_size);
		double theta = ((double)target_result_index / v)*PI * 2;
		double M = v * (pow(sin(theta / 2), 2));
		double rs = v - M;
		std::cout << "counting result: " << rs << std::endl;

		return floor(rs);
	}

	QProg build_qu_counting_prog() {
		m_work_qubits = m_qvm.allocateQubits(m_index_qubit_size);

		/*PTraceQCircuit("m_cir_oracle", m_cir_oracle);

		PTraceQCircuit("m_cir_diffusion", m_cir_diffusion);*/

		auto prog = QProg();
		prog << apply_QGate(m_work_qubits, H);
		prog << apply_QGate(m_data_index_qubits, H);
		for (const auto qubit : m_ancilla_qubits)
		{
			prog << X(qubit) << H(qubit);
		}
		//PTraceQCircuit("prog", prog);

		QCircuit cir_G = QCircuit();
		cir_G << m_cir_oracle << m_cir_diffusion;
		//PTraceQCircuit("cir_G", cir_G);

		QVec control_qubits_vec;
		size_t index_cnt = 0;
		for (const auto work_qubit : m_work_qubits)
		{
			control_qubits_vec.clear();
			control_qubits_vec.push_back(work_qubit);
			auto temp = cir_G.control(control_qubits_vec);
			for (size_t i = 0; i < pow(2, index_cnt); i++)
			{
				prog << temp;
			}
			++index_cnt;
		}

		prog << QFTdagger(m_work_qubits);

		return prog;
	}

	QCircuit QFT(std::vector<Qubit*> qvec){
		QCircuit  qft = CreateEmptyCircuit();
		for (auto i = 0; i < qvec.size(); i++)
		{
			qft << H(qvec[qvec.size() - 1 - i]);
			for (auto j = i + 1; j < qvec.size(); j++)
			{
				qft << CR(qvec[qvec.size() - 1 - j],
					qvec[qvec.size() - 1 - i], 2 * PI / (1 << (j - i + 1)));
			}
		}
		return qft;
	}

	QCircuit QFTdagger(std::vector<Qubit*> qvec){
		QCircuit  qft = QFT(qvec);
		qft.setDagger(true);
		return qft;
	}

private:
	QuantumMachine &m_qvm;
	QCircuit m_cir_oracle;
	QCircuit m_cir_diffusion;
	const size_t m_index_qubit_size;
	const QVec &m_data_index_qubits;
	const QVec &m_ancilla_qubits;
	QVec m_work_qubits;
};

QPANDA_END

#endif