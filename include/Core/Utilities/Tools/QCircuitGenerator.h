#ifndef QCIRCUIT_GENERATOR_H
#define QCIRCUIT_GENERATOR_H

#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumMachine/QVec.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include <memory>

QPANDA_BEGIN

class QCircuitGenerator
{
	struct CircuitNode
	{
		std::string m_op;
		std::vector<uint32_t> m_target_q;
		std::vector<uint32_t> m_control_q;
		bool m_is_dagger;
		std::vector<std::string> m_angle;

		CircuitNode()
			:m_is_dagger(false)
		{}

		CircuitNode(std::string op, const std::vector<uint32_t>& target_q,
			const std::vector<std::string>& angle, const std::vector<uint32_t>& control_q, bool is_dagger)
			:m_op(op), m_target_q(target_q), m_control_q(control_q)
			, m_is_dagger(is_dagger), m_angle(angle)
		{}
	};

public:
	using CircuitNodeRef = std::shared_ptr<QCircuitGenerator::CircuitNode>;
	using Ref = std::shared_ptr<QCircuitGenerator>;

public:
	QCircuitGenerator() {}
	~QCircuitGenerator() {}

	void set_param(const QVec& qubits, const std::vector<double>& angle) {
		m_qubit = qubits;
		if (angle.size() > 0){
			m_angle_vec = angle;
		}
	}

	CircuitNodeRef build_cir_node(std::string op, const std::vector<uint32_t>& target_q,
		const std::vector<std::string>& angle = {}, const std::vector<uint32_t>& control_q = {}, bool is_dagger = false) {
		return std::make_shared<CircuitNode>(op, target_q, angle, control_q, is_dagger);
	}
	QCircuit get_cir();
	void append_cir_node(CircuitNodeRef node) { m_cir_node_vec.push_back(node); }
	void append_cir_node(std::string op, const std::vector<uint32_t>& target_q,
		const std::vector<std::string>& angle = {}, const std::vector<uint32_t>& control_q = {}, bool is_dagger = false) {
		m_cir_node_vec.push_back(build_cir_node(op, target_q, angle, control_q, is_dagger)); }

	const std::vector<CircuitNodeRef>& get_cir_node_vec() const { return m_cir_node_vec; }
	uint32_t get_circuit_width() { return m_circuit_width; }
	void set_circuit_width(uint32_t w) { m_circuit_width = w; }
protected:
	double angle_str_to_double(const std::string& angle_str);

private:
	QVec m_qubit;
	std::vector<double> m_angle_vec;
	std::vector<CircuitNodeRef> m_cir_node_vec;
	uint32_t m_circuit_width;
};


QPANDA_END
#endif // QCIRCUIT_GENERATOR_H