#include "QHamiltonian.h"
#include "../../Utility/Utilities.h"

namespace QPanda
{
	QCircuit simulateZTerm(
		const std::vector<Qubit*> &qubit_vec,
		double coef,
		double t)
	{
		QCircuit circuit;
		if (0 == qubit_vec.size())
		{
			return circuit;
		}
		else if (1 == qubit_vec.size())
		{
			circuit << QGateNodeFactory::getInstance()->getGateNode(
				"RZ", qubit_vec[0], -coef*t);
		}
		else
		{
			circuit << parity_check_circuit(qubit_vec);
			circuit << QGateNodeFactory::getInstance()->getGateNode(
				"RZ", qubit_vec[qubit_vec.size() - 1], -coef * t);
			circuit << parity_check_circuit(qubit_vec);
		}

		return circuit;
	}

	QCircuit simulateOneTerm(
		const std::vector<Qubit*> &qubit_vec,
		const QTerm &hamiltonian_term,
		double coef,
		double t)
	{
		QCircuit circuit;
		if ((0 == qubit_vec.size()) ||
			(0 == hamiltonian_term.size()))
		{
			return circuit;
		}

		QCircuit transform;
		std::vector<Qubit*> tmp_vec;
		auto iter = hamiltonian_term.begin();
		for (; iter != hamiltonian_term.end(); iter++)
		{
			auto key = iter->first;
			auto value = iter->second;

			char ch = toupper(value);
			switch (ch)
			{
			case 'X':
				transform << QGateNodeFactory::getInstance()->getGateNode(
					"H", qubit_vec[key]);
				break;
			case 'Y':
				transform << QGateNodeFactory::getInstance()->getGateNode(
					"RX", qubit_vec[key], Q_PI_2);
				break;
			case 'Z':
				break;
			default:
				throw std::string("bad hamiltonian item.");
				break;
			}
			tmp_vec.emplace_back(qubit_vec[key]);
		}

		circuit << transform;
		circuit << simulateZTerm(tmp_vec, coef, t);
		circuit << transform.dagger();

		return circuit;
	}

	QCircuit simulateHamiltonian(
		const std::vector<Qubit*> &qubit_vec,
		const QHamiltonian &hamiltonian,
		double t,
		size_t slices)
	{
		QCircuit circuit;
		if ((0 == qubit_vec.size()) || 
			(0 == hamiltonian.size()) ||
			(0 == slices))
		{
			return circuit;
		}

		for (auto i = 0; i < slices; i++)
		{
			for (auto j = 0; j < hamiltonian.size(); i++)
			{
				auto item = hamiltonian[j];
				circuit << simulateOneTerm(qubit_vec, 
					                       item.first, 
					                       item.second, 
					                       t / slices);
			}
		}

		return circuit;
	}

	QCircuit simulatePauliZHamiltonian(
		const std::vector<Qubit*>& qubit_vec, 
		const QHamiltonian & hamiltonian, 
		double t)
	{
		QCircuit circuit;

		for (auto j = 0; j < hamiltonian.size(); j++)
		{
			std::vector<Qubit*> tmp_vec;
			auto item = hamiltonian[j];
			auto map = item.first;

			for (auto iter = map.begin(); iter != map.end(); iter++)
			{
				if ('Z' != iter->second)
				{
					throw std::string("Bad pauliZ Hamiltonian.");
				}

				tmp_vec.push_back(qubit_vec[iter->first]);
			}

			if (!tmp_vec.empty())
			{
				circuit << simulateZTerm(tmp_vec, item.second, t);
			}
		}

		return circuit;
	}

	QCircuit applySingleGateToAll(
		const std::string &gate, 
		const std::vector<Qubit*> &qubit_vec)
	{
		QCircuit circuit;
		for_each(qubit_vec.begin(), qubit_vec.end(), [&](Qubit* qbit)
		{
			circuit << QGateNodeFactory::getInstance()->getGateNode(gate, qbit);
		});

		return circuit;
	}

	void applySingleGateToAll(
		const std::string & gate, 
		const std::vector<Qubit*>& qubit_vec, 
		QCircuit & circuit)
	{
		for_each(qubit_vec.begin(), qubit_vec.end(), [&](Qubit* qbit)
		{
			circuit << QGateNodeFactory::getInstance()->getGateNode(gate, qbit);
		});
	}

	QCircuit ising_model(
		const std::vector<Qubit*> &qubit_vec, 
		const QGraph &graph, 
		const vector_d &gamma)
	{
		QCircuit circuit;

		for (size_t i = 0; i < gamma.size(); i++)
		{
			QCircuit qcirc = QCircuit();
			for_each(graph.begin(), graph.end(), [&](const QGraphItem &item)
			{
				qcirc << QGateNodeFactory::getInstance()->getGateNode(
					"CNOT",
					qubit_vec[item.first],
					qubit_vec[item.second]
				);
				qcirc << QGateNodeFactory::getInstance()->getGateNode(
					"RZ",
					qubit_vec[item.second],
					2 * gamma[i] * item.weight
				);
				qcirc << QGateNodeFactory::getInstance()->getGateNode(
					"CNOT",
					qubit_vec[item.first],
					qubit_vec[item.second]
				);
			});

			circuit << qcirc;
		}

		return circuit;
	}

	QCircuit pauliX_model(
		const std::vector<Qubit*> &qubit_vec, 
		const vector_d &beta)
	{
		QCircuit circuit;

		for (size_t i = 0; i < beta.size(); i++)
		{
			QCircuit qcirc = QCircuit();
			for_each(qubit_vec.begin(), qubit_vec.end(), [&](Qubit* qbit)
			{
				qcirc << QGateNodeFactory::getInstance()->getGateNode(
					"RX",
					qbit,
					2 * beta[i]
				);

				circuit << qcirc;
			});
		}

		return circuit;
	}
}