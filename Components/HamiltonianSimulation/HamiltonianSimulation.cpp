#include "Core/Core.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/Utilities/Tools/Utils.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/Compiler/QProgToQASM.h"
#include "Components/HamiltonianSimulation/HamiltonianSimulation.h"

namespace QPanda
{
    QCircuit simulateZTerm(
        std::vector<Qubit*> &qubit_vec,
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
            circuit << RZ(qubit_vec[0], 2 * coef*t);
        }
        else
        {
            circuit << parityCheckCircuit(qubit_vec);
            circuit << RZ(qubit_vec[qubit_vec.size() - 1], 2 * coef * t);
            circuit << parityCheckCircuit(qubit_vec);
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
                transform << H(qubit_vec[key]);
                break;
            case 'Y':
                transform << RX(qubit_vec[key], Q_PI_2);
                break;
            case 'Z':
                break;
            default:
                std::string err = "bad hamiltonian item.";
                std::cout << err << std::endl;
                QCERR(err);
                throw std::runtime_error(err);
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
            for (auto j = 0; j < hamiltonian.size(); j++)
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
                    std::string err = "Bad pauliZ Hamiltonian.";
                    std::cout << err << std::endl;
                    throw err;
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
        for_each(qubit_vec.begin(), qubit_vec.end(), [&](Qubit* qubit)
        {
			circuit << QGateNodeFactory::getInstance()->getGateNode(gate, { qubit });
        });

        return circuit;
    }

    void applySingleGateToAll(
        const std::string & gate, 
        const std::vector<Qubit*>& qubit_vec, 
        QCircuit & circuit)
    {
        for_each(qubit_vec.begin(), qubit_vec.end(), [&](Qubit* qubit)
        {
				circuit << QGateNodeFactory::getInstance()->getGateNode(gate, { qubit });
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
                qcirc << CNOT(qubit_vec[item.first],
					qubit_vec[item.second]);
                qcirc << RZ(qubit_vec[item.second],
					2 * gamma[i] * item.weight);
				qcirc << CNOT(qubit_vec[item.first],
					qubit_vec[item.second]);
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
			for_each(qubit_vec.begin(), qubit_vec.end(), [&](Qubit* qubit)
			{
				qcirc << RX(qubit,2 * beta[i]);

                circuit << qcirc;
            });
        }

        return circuit;
    }


    QMatrixXcd expMat(
        const qcomplex_t& conf,
        const QMatrixXcd& Mat,
        double number)
    {
        QMatrixXcd expmatrix(Mat.rows(), Mat.cols());
        expmatrix.setZero();
        int i = 0;
        qcomplex_t conf_new{ conf };
        QMatrixXcd mat_val;
        /*if (number)
        {
            conf_new = conf * PI;
        }
        else
        {
            conf_new = conf;
        }*/
        conf_new = conf * number;
        mat_val = conf_new * Mat;
        QMatrixXcd mx(mat_val.rows(), mat_val.cols());
        mx.setIdentity();
        /*Accuracy control of this complex matrix is prone to insufficient accuracy, 
        so use the number to control the -while loop */
        do
        {
            expmatrix += mx;
            ++i;
            mx *= (mat_val / i);
        } while (i < 1024);
        return expmatrix;
    }
}

USING_QPANDA
using namespace std;
using namespace Eigen;


QOperator::QOperator()
{
    QCircuit tmp;
    m_pQuantumCircuitOperator = tmp;
}


QOperator::QOperator(QGate& gate)
{
    QCircuit tmp;
    tmp << gate;
    m_pQuantumCircuitOperator = tmp;
}

QOperator::QOperator(QCircuit& circuit)
{
    QCircuit tmp;
    tmp << circuit;
    m_pQuantumCircuitOperator = tmp;
}

QStat QOperator::get_matrix()
{
    QProg prog(this->m_pQuantumCircuitOperator);
    return getCircuitMatrix(prog);
}


std::string QOperator::to_instruction(std::string ir_type)
{
    std::string ir_instruction = "";
    auto machine = initQuantumMachine(QMachineType::CPU);
    QProg prog(this->m_pQuantumCircuitOperator);
    if (ir_type == "OriginIR")
    {
        ir_instruction = transformQProgToOriginIR(prog, machine);
    }
    else if (ir_type == "Quil")
    {
        ir_instruction = transformQProgToQuil(prog, machine);
    }
    else if (ir_type == "QASM")
    {
        ir_instruction = convert_qprog_to_qasm(prog, machine);
    }
    else
    {
        QCERR_AND_THROW(runtime_error, "Input IR Type ERROR");
    }
    return ir_instruction;
}

