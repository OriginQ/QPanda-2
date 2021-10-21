/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

//#include "QProgram.h"

#include "Core/Core.h"
#include "Core/Utilities/QProgInfo/ConfigMap.h"
#include "Core/QuantumCircuit/ClassicalProgram.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QuantumMeasure.h"
#include "Core/QuantumCircuit/QReset.h"
using namespace QGATE_SPACE;
using namespace std;
USING_QPANDA
QProg  QPanda::CreateEmptyQProg()
{
	QProg temp;
	return temp;
}

QProg QPanda::createEmptyQProg()
{
	QProg temp;
	return temp;
}

QProg::QProg(std::shared_ptr<AbstractQuantumProgram> node)
{
	if (!node)
	{
		QCERR("node is null shared_ptr");
		throw invalid_argument("node is null shared_ptr");
	}

	m_quantum_program = node;
}

QProg::QProg()
{
	auto class_name = ConfigMap::getInstance()["QProg"];
	auto qprog = QuantumProgramFactory::getInstance().getQuantumQProg(class_name);
	m_quantum_program.reset(qprog);
}

QProg::QProg(const QProg &old_qprog)
{
	m_quantum_program = old_qprog.m_quantum_program;
}

QProg::QProg(QProg &other)
{
	m_quantum_program = other.m_quantum_program;
}


QProg::QProg(std::shared_ptr<QNode> pnode)
	:QProg()
{
	if (!pnode)
	{
		throw std::runtime_error("node is null");
	}
	m_quantum_program->pushBackNode(pnode);
}

QProg::QProg(ClassicalCondition &node)
	:QProg()
{
	ClassicalProg tmp(node);
	m_quantum_program->pushBackNode(dynamic_pointer_cast<QNode>(tmp.getImplementationPtr()));
}

size_t QProg::get_used_qubits(QVec& qubit_vector)
{
	if (!m_quantum_program)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}
	return m_quantum_program->get_used_qubits(qubit_vector);
}

size_t QProg::get_max_qubit_addr()
{
	if (!m_quantum_program)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}
	return m_quantum_program->get_max_qubit_addr();
}

size_t QProg::get_used_cbits(std::vector<ClassicalCondition>& cbit_vector)
{
	if (!m_quantum_program)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}
	return m_quantum_program->get_used_cbits(cbit_vector);
}

size_t QProg::get_qgate_num()
{
	if (!m_quantum_program)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}
	return m_quantum_program->get_qgate_num();
}

bool  QProg::is_measure_last_pos()
{
	if (!m_quantum_program)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}
	return m_quantum_program->is_measure_last_pos();
}

std::map<Qubit*, bool>  QProg::get_measure_pos()
{
	if (!m_quantum_program)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}
	return m_quantum_program->get_measure_pos();
}

std::vector<std::pair<Qubit*, ClassicalCondition>> QProg::get_measure_qubits_cbits()
{
	if (!m_quantum_program)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}
	return m_quantum_program->get_measure_qubits_cbits();
}

QProg::~QProg()
{
	m_quantum_program.reset();
}

std::shared_ptr<AbstractQuantumProgram> QProg::getImplementationPtr()
{
	return m_quantum_program;
}

void QProg::pushBackNode(std::shared_ptr<QNode> node)
{
	if (!node)
	{
		QCERR("node is null");
		throw runtime_error("node is null");
	}
	m_quantum_program->pushBackNode(node);
}

NodeIter QProg::getFirstNodeIter()
{
	if (!m_quantum_program)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}

	return m_quantum_program->getFirstNodeIter();
}

NodeIter  QProg::getLastNodeIter()
{
	if (!m_quantum_program)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}

	return m_quantum_program->getLastNodeIter();
}

NodeIter QProg::getEndNodeIter()
{
	if (!m_quantum_program)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}

	return m_quantum_program->getEndNodeIter();
}

NodeIter QProg::getHeadNodeIter()
{
	if (!m_quantum_program)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}

	return m_quantum_program->getHeadNodeIter();
}

NodeIter QProg::insertQNode(const NodeIter & iter,shared_ptr<QNode> node)
{
	if (nullptr == node)
	{
		QCERR("node is nullptr");
		throw runtime_error("node is nullptr");
	}

	if (!m_quantum_program)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}

	return m_quantum_program->insertQNode(iter, node);
}

NodeIter QProg::deleteQNode(NodeIter & iter)
{
	return m_quantum_program->deleteQNode(iter);
}

NodeType QProg::getNodeType() const
{
	if (!m_quantum_program)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}

	return dynamic_pointer_cast<QNode>(m_quantum_program)->getNodeType();
}

void QProg::clear()
{
	if (!m_quantum_program)
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}

	return m_quantum_program->clear();
}

template <>
QProg & QProg::operator<<<ClassicalCondition>(ClassicalCondition cc)
{
	ClassicalProg temp(cc);
	pushBackNode(dynamic_pointer_cast<QNode>(temp.getImplementationPtr()));
	return *this;
}

OriginProgram::~OriginProgram()
{
}

OriginProgram::OriginProgram()
{
}

NodeType OriginProgram::getNodeType() const
{
	return m_node_type;
}

bool OriginProgram::check_insert_node_type(std::shared_ptr<QNode> node)
{
	if (nullptr == node.get())
	{
		QCERR("node is null");
		throw std::runtime_error("node is null");
	}
	const NodeType t = node->getNodeType();
	switch (t)
	{
	case GATE_NODE:
	{
		QVec temp_qv;
		auto qgate_node = std::dynamic_pointer_cast<AbstractQGateNode>(node);
		qgate_node->getQuBitVector(temp_qv);
		m_used_qubit_vector += temp_qv;
		temp_qv.clear();
		qgate_node->getControlVector(temp_qv);
		m_used_qubit_vector += temp_qv;
		m_qgate_num++;

		for (auto qbit : temp_qv)
		{
			if (m_last_measure.find(qbit) != m_last_measure.end())
				m_last_measure[qbit] = false;

		}

	}
	break;
	case CIRCUIT_NODE:
	{
		QVec temp_qv;
		auto qcircuit_node = std::dynamic_pointer_cast<AbstractQuantumCircuit>(node);
		qcircuit_node->get_used_qubits(temp_qv);
		m_used_qubit_vector += temp_qv;
		temp_qv.clear();
		qcircuit_node->getControlVector(temp_qv);
		m_used_qubit_vector += temp_qv;

		m_qgate_num += qcircuit_node->get_qgate_num();
	}
	break;
	case PROG_NODE:
	{
		QVec temp_qv;
		auto qprog_node = std::dynamic_pointer_cast<AbstractQuantumProgram>(node);
		qprog_node->get_used_qubits(temp_qv);
		m_used_qubit_vector += temp_qv;
		m_qgate_num += qprog_node->get_qgate_num();
		
		auto child_measure_pos = qprog_node->get_measure_pos();

		for (auto iter : child_measure_pos)
		{
			if (m_last_measure.find(iter.first) != m_last_measure.end())
				m_last_measure[iter.first] &= iter.second;
			else
				m_last_measure[iter.first] = iter.second;
		}

		std::vector<ClassicalCondition> cbits_vect;
		qprog_node->get_used_cbits(cbits_vect);

		auto aiter = m_used_cbit_vector.begin();

		for (auto aiter = cbits_vect.begin(); aiter != cbits_vect.end(); aiter++)
		{
			auto biter = m_used_cbit_vector.begin();
			for (; biter != m_used_cbit_vector.end(); biter++)
			{
				if ((*aiter).getExprPtr()->getCBit()->get_addr()
					== (*biter).getExprPtr()->getCBit()->get_addr())
				{
					break;
				}
			}

			if (biter == m_used_cbit_vector.end())
			{
				m_used_cbit_vector.push_back(*aiter);
			}
		}

		for (const auto & iter : qprog_node->get_measure_qubits_cbits())
		{
			m_mea_qubits_cbits.push_back(iter);
		}
	}
	break;
	case MEASURE_GATE:
	{
		auto qmea_node = std::dynamic_pointer_cast<AbstractQuantumMeasure>(node);
		auto qbit = qmea_node->getQuBit();
		if (m_last_measure.find(qbit) == m_last_measure.end())
		{
			m_last_measure[qbit] = true;
		}

		QVec qv(qmea_node->getQuBit());
		m_used_qubit_vector += qv;

		auto iter = m_used_cbit_vector.begin();
		for (; iter != m_used_cbit_vector.end(); iter++)
		{
			if ((*iter).getExprPtr()->getCBit()->get_addr()
				== qmea_node->getCBit()->get_addr())
			{
				break;
			}
		}
		if (iter == m_used_cbit_vector.end())
		{
			m_used_cbit_vector.push_back(qmea_node->getCBit());
		}

		m_mea_qubits_cbits.push_back({ qmea_node->getQuBit() , qmea_node->getCBit() });
	}
	break;
	case QIF_START_NODE:
	{
		auto qctrl_node = std::dynamic_pointer_cast<AbstractControlFlowNode>(node);
		check_insert_node_type(qctrl_node->getTrueBranch());
		if(qctrl_node->getFalseBranch())
			check_insert_node_type(qctrl_node->getFalseBranch());
	}
    case WHILE_START_NODE:
    {
        auto qctrl_node = std::dynamic_pointer_cast<AbstractControlFlowNode>(node);
        check_insert_node_type(qctrl_node->getTrueBranch());
    }
	break;
	case	RESET_NODE:
	{
		auto qreset_node = std::dynamic_pointer_cast<AbstractQuantumReset>(node);
		m_used_qubit_vector += QVec(qreset_node->getQuBit());
	}
	break;
	case CLASS_COND_NODE:
	case	QWAIT_NODE:
		break;

	default:
		throw qcircuit_construction_fail("bad node type");
	}

	return true;
}

size_t OriginProgram::get_used_qubits(QVec& qubit_vector)
{
	for (auto aiter : m_used_qubit_vector)
	{
		qubit_vector.push_back(aiter);
	}
	return m_used_qubit_vector.size();
}

size_t OriginProgram::get_max_qubit_addr()
{
	size_t max_addr = 0;
	for (auto aiter : m_used_qubit_vector)
	{
		if (aiter->get_phy_addr() > max_addr)
			max_addr = aiter->get_phy_addr();
	}
	return max_addr;
}

size_t OriginProgram::get_used_cbits(std::vector<ClassicalCondition>& cbit_vector)
{
	for (auto aiter : m_used_cbit_vector)
	{
		cbit_vector.push_back(aiter);
	}
	return m_used_cbit_vector.size();
}

size_t OriginProgram::get_qgate_num()
{
	return m_qgate_num;
}

bool OriginProgram::is_measure_last_pos()
{
	for (auto iter : m_last_measure)
	{
		if (iter.second == false)
			return  false;
	}

	return true;
}

REGISTER_QPROGRAM(OriginProgram);

void QuantumProgramFactory::registClass(string name, CreateQProgram method)
{
	if ((name.size() <= 0) || (nullptr == method))
	{
		QCERR("Unknown internal error");
		throw runtime_error("Unknown internal error");
	}
	m_qprog_map.insert(pair<string, CreateQProgram>(name, method));
}

AbstractQuantumProgram * QuantumProgramFactory::getQuantumQProg(std::string& name)
{
	if (name.size() <= 0)
	{
		QCERR("param error");
		throw runtime_error("param error");
	}
	auto aiter = m_qprog_map.find(name);
	if (aiter != m_qprog_map.end())
	{
		return aiter->second();
	}
	return nullptr;
}

