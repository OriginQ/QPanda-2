#include "Core/VirtualQuantumProcessor/PartialAmplitude/PartialAmplitudeGraph.h"
using namespace std;
USING_QPANDA

static constexpr double _SQ2 = 1 / 1.4142135623731;
static constexpr double _PI = 3.14159265358979;

static void _H(QGateNode &node, QPUImpl *qpu)
{
    QStat _h = { _SQ2, _SQ2, _SQ2, -_SQ2 };
    qpu->unitarySingleQubitGate(node.qubits[0], _h, node.is_dagger, GateType::HADAMARD_GATE);
}

static void _T(QGateNode &node, QPUImpl *qpu)
{
    QStat _t = { 1, 0, 0, qcomplex_t(_SQ2, _SQ2) };
    qpu->unitarySingleQubitGate(node.qubits[0], _t, node.is_dagger, GateType::T_GATE);
}

static void _S(QGateNode &node, QPUImpl *qpu)
{
    QStat _s = { 1, 0, 0, qcomplex_t(0, 1) };
    qpu->unitarySingleQubitGate(node.qubits[0], _s, node.is_dagger, GateType::S_GATE);
}

static void _X(QGateNode &node, QPUImpl *qpu)
{
    QStat _x = { 0, 1, 1, 0 };
    qpu->unitarySingleQubitGate(node.qubits[0], _x, node.is_dagger, GateType::PAULI_X_GATE);
}

static void _Y(QGateNode &node, QPUImpl *qpu)
{
    QStat _y = { 0, qcomplex_t(0, -1), qcomplex_t(0, 1), 0 };
    qpu->unitarySingleQubitGate(node.qubits[0], _y, node.is_dagger, GateType::PAULI_Y_GATE);
}

static void _Z(QGateNode &node, QPUImpl *qpu)
{
    QStat _z = { 1, 0, 0, -1 };
    qpu->unitarySingleQubitGate(node.qubits[0], _z, node.is_dagger, GateType::PAULI_Z_GATE);
}

static void _P0(QGateNode &node, QPUImpl *qpu)
{
    QStat _p0 = { 1, 0, 0, 0 };
    qpu->unitarySingleQubitGate(node.qubits[0], _p0, node.is_dagger, GateType::P0_GATE);
}

static void _P1(QGateNode &node, QPUImpl *qpu)
{
    QStat _p1 = { 0, 0, 0, 1 };
    qpu->unitarySingleQubitGate(node.qubits[0], _p1, node.is_dagger, GateType::P1_GATE);
}

static void _X1(QGateNode &node, QPUImpl *qpu)
{
    QStat _x1 = { _SQ2, qcomplex_t(0, -_SQ2), qcomplex_t(0, -_SQ2), _SQ2 };
    qpu->unitarySingleQubitGate(node.qubits[0], _x1, node.is_dagger, GateType::HADAMARD_GATE);
}

static void _Y1(QGateNode &node, QPUImpl *qpu)
{
    QStat _y1 = { _SQ2, -_SQ2, _SQ2, _SQ2 };
    qpu->unitarySingleQubitGate(node.qubits[0], _y1, node.is_dagger, GateType::HADAMARD_GATE);
}

static void _Z1(QGateNode &node, QPUImpl *qpu)
{
    QStat _z1 = { qcomplex_t(-_SQ2, _SQ2), 0, 0, qcomplex_t(_SQ2, _SQ2) };
    qpu->unitarySingleQubitGate(node.qubits[0], _z1, node.is_dagger, GateType::HADAMARD_GATE);
}

static void _P00(QGateNode &node, QPUImpl *qpu)
{
    QStat _p00 = { 1, 0, 0, 0,
                   0, 1, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 0 };
    qpu->unitaryDoubleQubitGate(node.qubits[1], node.qubits[0], _p00, node.is_dagger, GateType::P00_GATE);
}

static void _P11(QGateNode &node, QPUImpl *qpu)
{
    QStat _p11 = { 0, 0, 0, 0,
                   0, 0, 0, 0,
                   0, 0, 0, 0,
                   0, 0, 0, 1 };
    qpu->unitaryDoubleQubitGate(node.qubits[1], node.qubits[0], _p11, node.is_dagger, GateType::P11_GATE);
}

static void _CNOT(QGateNode &node, QPUImpl *qpu)
{
    QStat _cnot = { 1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 0, 1,
                    0, 0, 1, 0 };
    qpu->unitaryDoubleQubitGate(node.qubits[1], node.qubits[0], _cnot, node.is_dagger, GateType::CNOT_GATE);
}

static void _CZ(QGateNode &node, QPUImpl *qpu)
{
    QStat _cz = { 1, 0, 0, 0,
                  0, 1, 0, 0,
                  0, 0, 1, 0,
                  0, 0, 0, -1 };
    qpu->unitaryDoubleQubitGate(node.qubits[1], node.qubits[0], _cz, node.is_dagger, GateType::CZ_GATE);
}

static void _SWAP(QGateNode &node, QPUImpl *qpu)
{
    QStat _swap = { 1, 0, 0, 0,
        0, 0, 1, 0,
        0, 1, 0, 0,
        0, 0, 0, 1 };
    qpu->unitaryDoubleQubitGate(node.qubits[1], node.qubits[0], _swap, node.is_dagger, GateType::SWAP_GATE);
}

static void _SQISWAP(QGateNode &node, QPUImpl *qpu)
{
    QStat _sqiswap = { 1, 0, 0, 0,
                       0, _SQ2, qcomplex_t(0, _SQ2), 0,
                       0, qcomplex_t(0, _SQ2), _SQ2, 0,
                       0, 0, 0, 1 };
    qpu->unitaryDoubleQubitGate(node.qubits[1], node.qubits[0], _sqiswap, node.is_dagger, GateType::SQISWAP_GATE);
}

static void _ISWAP(QGateNode &node, QPUImpl *qpu)
{
    QStat _iswap = {1 , 0, 0, 0,
                    0, 0, qcomplex_t(0, 1), 0,
                    0, qcomplex_t(0, 1), 0, 0,
                    0, 0, 0, 1 };
    qpu->unitaryDoubleQubitGate(node.qubits[1], node.qubits[0], _iswap, node.is_dagger, GateType::ISWAP_GATE);
}

static void _RX(QGateNode &node, QPUImpl *qpu)
{
    auto theta = node.params[0];
    QStat _rx = { qcomplex_t(cos(theta / 2), 0), 
                  qcomplex_t(0, -sin(theta / 2)),
                  qcomplex_t(0, -sin(theta / 2)),
                  qcomplex_t(cos(theta / 2), 0) };

    qpu->unitarySingleQubitGate(node.qubits[0], _rx, node.is_dagger, GateType::RX_GATE);
}

static void _RY(QGateNode &node, QPUImpl *qpu)
{
    auto theta = node.params[0];
    QStat _ry = { qcomplex_t(cos(theta / 2), 0), 
                  qcomplex_t(-sin(theta / 2), 0),
                  qcomplex_t(sin(theta / 2), 0),
                  qcomplex_t(cos(theta / 2), 0) };

    qpu->unitarySingleQubitGate(node.qubits[0], _ry, node.is_dagger, GateType::RY_GATE);
}

static void _RZ(QGateNode &node, QPUImpl *qpu)
{
    auto theta = node.params[0];
    QStat _rz = { exp(qcomplex_t(0, -theta / 2)), 
                  0,
                  0,
                  exp(qcomplex_t(0, theta / 2)) };

    qpu->unitarySingleQubitGate(node.qubits[0], _rz, node.is_dagger, GateType::RZ_GATE);
}

static void _U1(QGateNode &node, QPUImpl *qpu)
{
    auto theta = node.params[0];
    QStat _u1 = { 1, 
                  0,
                  0,
                  exp(qcomplex_t(0, theta)) };

    qpu->unitarySingleQubitGate(node.qubits[0], _u1, node.is_dagger, GateType::U1_GATE);
}

static void _U2(QGateNode &node, QPUImpl *qpu)
{
    auto phi = node.params[0];
    auto lambda = node.params[1];

    auto alpha = (phi + lambda) / 2;
    auto beta = phi;
    auto gamma = PI / 2;
    auto delta = lambda;

    QStat _u2(4);

    auto coefficient = static_cast<qstate_type>(sqrt(2) / 2);
    _u2[0] = 1 * coefficient;
    _u2[1].real(static_cast<qstate_type>(-cos(lambda)) * coefficient);
    _u2[1].imag(static_cast<qstate_type>(-sin(lambda)) * coefficient);

    _u2[2].real(static_cast<qstate_type>(cos(phi)) * coefficient);
    _u2[2].imag(static_cast<qstate_type>(sin(phi)) * coefficient);
    _u2[3].real(static_cast<qstate_type>(cos(phi + lambda)) * coefficient);
    _u2[3].imag(static_cast<qstate_type>(sin(phi + lambda)) * coefficient);

    qpu->unitarySingleQubitGate(node.qubits[0], _u2, node.is_dagger, GateType::U2_GATE);
}

static void _U3(QGateNode &node, QPUImpl *qpu)
{
    auto theta = node.params[0];
    auto phi = node.params[1];
    auto lambda = node.params[2];

    QStat _u3(4);

	const auto _v1 = (qstate_type)(std::cos(theta / 2.0));
	const auto _v2 = (qstate_type)(std::sin(theta / 2.0));

    _u3[0] = _v1;
    _u3[1] = -std::exp(qcomplex_t(0, lambda)) * _v2;
    _u3[2] = std::exp(qcomplex_t(0, phi)) * _v2;
    _u3[3] = std::exp(qcomplex_t(0, phi + lambda)) * _v1;

    qpu->unitarySingleQubitGate(node.qubits[0], _u3, node.is_dagger, GateType::U3_GATE);
}

static void _U4(QGateNode &node, QPUImpl *qpu)
{
    auto alpha = node.params[0];
    auto beta = node.params[1];
    auto gamma = node.params[2];
    auto delta = node.params[3];

    QStat _u4;

    QStat matrix;
    _u4.emplace_back(qcomplex_t(cos(alpha - beta / 2 - delta / 2)*cos(gamma / 2),
        sin(alpha - beta / 2 - delta / 2)*cos(gamma / 2)));
    _u4.emplace_back(qcomplex_t(-cos(alpha - beta / 2 + delta / 2)*sin(gamma / 2),
        -sin(alpha - beta / 2 + delta / 2)*sin(gamma / 2)));
    _u4.emplace_back(qcomplex_t(cos(alpha + beta / 2 - delta / 2)*sin(gamma / 2),
        sin(alpha + beta / 2 - delta / 2)*sin(gamma / 2)));
    _u4.emplace_back(qcomplex_t(cos(alpha + beta / 2 + delta / 2)*cos(gamma / 2),
        sin(alpha + beta / 2 + delta / 2)*cos(gamma / 2)));

    qpu->unitarySingleQubitGate(node.qubits[0], _u4, node.is_dagger, GateType::U4_GATE);
}

static void _CR(QGateNode &node, QPUImpl *qpu)
{
    auto theta = node.params[0];
    QStat _cr = { 1, 0, 0, 0,
                  0, 1, 0, 0,
                  0, 0, 1, 0,
                  0, 0, 0, exp(qcomplex_t(0, theta)) };

    qpu->unitaryDoubleQubitGate(node.qubits[1],node.qubits[0], _cr, node.is_dagger, GateType::CPHASE_GATE);
}

static void _TOFFOLI(QGateNode &node, QPUImpl *qpu)
{
    QStat _x = { 0, 1, 1, 0 };
    Qnum control_qubits = { node.qubits[1],node.qubits[2] };
    qpu->controlunitarySingleQubitGate(node.qubits[0], control_qubits, _x, node.is_dagger, GateType::TOFFOLI_GATE);
}


PartialAmplitudeGraph::PartialAmplitudeGraph()
{
    m_function_mapping[GateType::HADAMARD_GATE] = _H;
    m_function_mapping[GateType::T_GATE] = _T;
    m_function_mapping[GateType::S_GATE] = _S;

    m_function_mapping[GateType::P0_GATE] = _P0;
    m_function_mapping[GateType::P1_GATE] = _P1;

    m_function_mapping[GateType::PAULI_X_GATE] = _X;
    m_function_mapping[GateType::PAULI_Y_GATE] = _Y;
    m_function_mapping[GateType::PAULI_Z_GATE] = _Z;

    m_function_mapping[GateType::X_HALF_PI] = _X1;
    m_function_mapping[GateType::Y_HALF_PI] = _Y1;
    m_function_mapping[GateType::Z_HALF_PI] = _Z1;

    m_function_mapping[GateType::P00_GATE] = _P00;
    m_function_mapping[GateType::P11_GATE] = _P11;

    m_function_mapping[GateType::RX_GATE] = _RX;
    m_function_mapping[GateType::RY_GATE] = _RY;
    m_function_mapping[GateType::RZ_GATE] = _RZ;

    m_function_mapping[GateType::U1_GATE] = _U1;
    m_function_mapping[GateType::U2_GATE] = _U2;
    m_function_mapping[GateType::U3_GATE] = _U3;
    m_function_mapping[GateType::U4_GATE] = _U4;

    m_function_mapping[GateType::CNOT_GATE] = _CNOT;
    m_function_mapping[GateType::CZ_GATE] = _CZ;
    m_function_mapping[GateType::SWAP_GATE] = _SWAP;
    m_function_mapping[GateType::SQISWAP_GATE] = _SQISWAP;
    m_function_mapping[GateType::ISWAP_GATE] = _ISWAP;
    m_function_mapping[GateType::CPHASE_GATE] = _CR;

    m_function_mapping[GateType::TOFFOLI_GATE] = _TOFFOLI;

    m_key_map.insert(make_pair(GateType::CNOT_GATE, GateType::PAULI_X_GATE));
    m_key_map.insert(make_pair(GateType::CZ_GATE, GateType::PAULI_Z_GATE));
    m_key_map.insert(make_pair(GateType::CPHASE_GATE, GateType::U1_GATE));
    m_key_map.insert(make_pair(GateType::TOFFOLI_GATE, GateType::CNOT_GATE));
}

void PartialAmplitudeGraph::computing_graph(const cir_type &prog_map, std::shared_ptr<QPUImpl> simulator)
{
    QPANDA_ASSERT(nullptr == simulator, "nullptr == simulator");

    for (auto val : prog_map)
    {
        auto iter = m_function_mapping.find(val.gate_type);
        if (iter == m_function_mapping.end())
        {
            QCERR("Error");
            throw invalid_argument("Error");
        }
        else
        {
            iter->second(val, simulator.get());
        }
    }
}

void PartialAmplitudeGraph::traversal(std::vector<QGateNode> &circuit)
{
	for (size_t i = 0; i < circuit.size(); ++i)
	{
		auto iter = m_key_map.find(circuit[i].gate_type);
		if (m_key_map.end() != iter)
		{
			if (GateType::TOFFOLI_GATE == circuit[i].gate_type)
			{
				if (is_corss_node(circuit[i].qubits[1], circuit[i].qubits[0]) &&
					is_corss_node(circuit[i].qubits[2], circuit[i].qubits[0]))
				{
					std::vector<QGateNode> P0_Cir = circuit;
                    std::vector<QGateNode> P1_Cir = circuit;

					QGateNode P0_Node = { P00_GATE , circuit[i].is_dagger, std::vector<uint32_t>(2)};
					P0_Node.qubits[1] = circuit[i].qubits[1];
					P0_Node.qubits[0] = circuit[i].qubits[2];
					P0_Cir[i] = P0_Node;

					QGateNode P1_Node1 = { P11_GATE , circuit[i].is_dagger, std::vector<uint32_t>(2)};
					P1_Node1.qubits[1] = circuit[i].qubits[1];
					P1_Node1.qubits[0] = circuit[i].qubits[2];

					QGateNode P1_Node2 = { PAULI_X_GATE ,circuit[i].is_dagger, std::vector<uint32_t>(1)};
					P1_Node2.qubits[0] = circuit[i].qubits[0];

					P1_Cir[i] = P1_Node2;
					P1_Cir.emplace(P1_Cir.begin() + i, P1_Node1);

					traversal(P0_Cir);
					traversal(P1_Cir);

					split_circuit(P0_Cir);
					split_circuit(P1_Cir);
					break;
				}
				else if (is_corss_node(circuit[i].qubits[1], circuit[i].qubits[2]))
				{
					std::vector<QGateNode> P0_Cir = circuit;
					std::vector<QGateNode> P1_Cir = circuit;

					if (is_corss_node(circuit[i].qubits[1], circuit[i].qubits[0]))
					{
                        QGateNode P0_Node = { P0_GATE , circuit[i].is_dagger, std::vector<uint32_t>(1)};
						P0_Node.qubits[0] = circuit[i].qubits[1];
						P0_Cir[i] = P0_Node;

						QGateNode P1_Node1 = { P1_GATE , circuit[i].is_dagger, std::vector<uint32_t>(1)};
						P1_Node1.qubits[0] = circuit[i].qubits[1];

						QGateNode P1_Node2 = { CNOT_GATE ,circuit[i].is_dagger, std::vector<uint32_t>(2)};
						P1_Node2.qubits[0] = circuit[i].qubits[0];
						P1_Node2.qubits[1] = circuit[i].qubits[2];

						P1_Cir[i] = P1_Node2;
						P1_Cir.emplace(P1_Cir.begin() + i, P1_Node1);
					}
					else
					{
						QGateNode P0_Node = { P0_GATE , circuit[i].is_dagger, std::vector<uint32_t>(1)};
						P0_Node.qubits[0] = circuit[i].qubits[2];
						P0_Cir[i] = P0_Node;

						QGateNode P1_Node1 = { P1_GATE , circuit[i].is_dagger, std::vector<uint32_t>(1)};
						P1_Node1.qubits[0] = circuit[i].qubits[2];

						QGateNode P1_Node2 = { CNOT_GATE ,circuit[i].is_dagger, std::vector<uint32_t>(2)};
						P1_Node2.qubits[0] = circuit[i].qubits[0];
						P1_Node2.qubits[1] = circuit[i].qubits[1];

						P1_Cir[i] = P1_Node2;
						P1_Cir.emplace(P1_Cir.begin() + i, P1_Node1);
					}

					traversal(P0_Cir);
					traversal(P1_Cir);

					split_circuit(P0_Cir);
					split_circuit(P1_Cir);
					break;
				}
				else
				{}
			}
			else
			{
				if (is_corss_node(circuit[i].qubits[1], circuit[i].qubits[0]))
				{
					vector<QGateNode> P0_Cir = circuit;
					vector<QGateNode> P1_Cir = circuit;

					QGateNode P0_Node = { P0_GATE , circuit[i].is_dagger, std::vector<uint32_t>(1)};
					P0_Node.qubits[0] = circuit[i].qubits[1];
					P0_Cir[i] = P0_Node;

                    QGateNode P1_Node1 = { P1_GATE , circuit[i].is_dagger, std::vector<uint32_t>(1)};
					P1_Node1.qubits[0] = circuit[i].qubits[1];

					QGateNode P1_Node2 = { iter->second ,circuit[i].is_dagger, std::vector<uint32_t>(1)};
                    P1_Node2.params = circuit[i].params;
					P1_Node2.qubits[0] = circuit[i].qubits[0];

					P1_Cir[i] = P1_Node2;
					P1_Cir.emplace(P1_Cir.begin() + i, P1_Node1);

					traversal(P0_Cir);
					traversal(P1_Cir);

					split_circuit(P0_Cir);
					split_circuit(P1_Cir);
					break;
				}
			}
		}
	}
}


void PartialAmplitudeGraph::split_circuit(std::vector<QGateNode> &circuit)
{
	bool is_operater{ true };
	for (auto val : circuit)
	{
		auto iter = m_key_map.find(val.gate_type);
		if (iter != m_key_map.end())
		{
			if (GateType::TOFFOLI_GATE == val.gate_type)
			{
				if ((is_corss_node(val.qubits[1], val.qubits[0])) ||
					(is_corss_node(val.qubits[1], val.qubits[2])) ||
					(is_corss_node(val.qubits[0], val.qubits[2])))
				{
					is_operater = false;
					break;
				}
			}
			else
			{
				if (is_corss_node(val.qubits[0], val.qubits[1]))
				{
					is_operater = false;
					break;
				}
			}
		}
	}

	if (is_operater)
	{
		std::vector<QGateNode> upper_graph, under_graph;
		for (auto val : circuit)
		{
			QGateNode node = { val.gate_type,val.is_dagger };
			switch (val.gate_type)
			{
			case GateType::P0_GATE:
			case GateType::P1_GATE:
			case GateType::HADAMARD_GATE:
			case GateType::T_GATE:
			case GateType::S_GATE:
			case GateType::PAULI_X_GATE:
			case GateType::PAULI_Y_GATE:
			case GateType::PAULI_Z_GATE:
			case GateType::X_HALF_PI:
			case GateType::Y_HALF_PI:
			case GateType::Z_HALF_PI:
			{
                node.qubits.resize(1);

				if (val.qubits[0] < (m_qubit_num / 2))
				{
					node.qubits[0] = val.qubits[0];
					upper_graph.emplace_back(node);
				}
				else
				{
					node.qubits[0] = val.qubits[0] - (m_qubit_num / 2);
					under_graph.emplace_back(node);
				}
			}
			break;

            case GateType::U1_GATE:
            case GateType::U2_GATE:
            case GateType::U3_GATE:
            case GateType::U4_GATE:
			case GateType::RX_GATE:
			case GateType::RY_GATE:
            case GateType::RZ_GATE:
			{
                node.qubits.resize(1);
                node.params = val.params;

				if (val.qubits[0] < (m_qubit_num / 2))
				{
					node.qubits[0] = val.qubits[0];
					upper_graph.emplace_back(node);
				}
				else
				{
					node.qubits[0] = val.qubits[0] - (m_qubit_num / 2);
					under_graph.emplace_back(node);
				}
			}
			break;

			case GateType::CNOT_GATE:
			case GateType::CZ_GATE:
			case GateType::ISWAP_GATE:
			case GateType::SWAP_GATE:
			case GateType::SQISWAP_GATE:
			case GateType::P00_GATE:
			case GateType::P11_GATE:
			{
                node.qubits.resize(2);

				if ((val.qubits[0] <= (m_qubit_num / 2)) &&
					(val.qubits[1] <= (m_qubit_num / 2)))
				{
					node.qubits[0] = val.qubits[0];
					node.qubits[1] = val.qubits[1];
					upper_graph.emplace_back(node);
				}
				else
				{
					node.qubits[0] = val.qubits[0] - (m_qubit_num / 2),
						node.qubits[1] = val.qubits[1] - (m_qubit_num / 2);
					under_graph.emplace_back(node);
				}
			}
			break;

			case GateType::CPHASE_GATE:
			{
                node.qubits.resize(2);
                node.params = val.params;

				if ((val.qubits[0] <= (m_qubit_num / 2)) &&
					(val.qubits[1] <= (m_qubit_num / 2)))
				{
					node.qubits[0] = val.qubits[0];
					node.qubits[1] = val.qubits[1];
					upper_graph.emplace_back(node);
				}
				else
				{
					node.qubits[0] = val.qubits[0] - (m_qubit_num / 2),
					node.qubits[1] = val.qubits[1] - (m_qubit_num / 2);
					under_graph.emplace_back(node);
				}
			}
			break;

			case GateType::TOFFOLI_GATE:
			{
                node.qubits.resize(3);

				if ((val.qubits[0] <= (m_qubit_num / 2)) &&
					(val.qubits[1] <= (m_qubit_num / 2)) &&
					(val.qubits[2] <= (m_qubit_num / 2)))
				{
					node.qubits[0] = val.qubits[0];
					node.qubits[1] = val.qubits[1];
					node.qubits[2] = val.qubits[2];
					upper_graph.emplace_back(node);
				}
				else if ((val.qubits[0] > (m_qubit_num / 2)) &&
					(val.qubits[1] > (m_qubit_num / 2)) &&
					(val.qubits[2] > (m_qubit_num / 2)))
				{
					node.qubits[0] = val.qubits[0] - (m_qubit_num / 2),
					node.qubits[1] = val.qubits[1] - (m_qubit_num / 2);
					node.qubits[2] = val.qubits[2] - (m_qubit_num / 2);
					under_graph.emplace_back(node);
				}
				else
				{
					QCERR("Toffoli spilt error");
					throw run_fail("Toffoli spilt error");
				}
			}
			break;

			default:
			{
				QCERR("UnSupported QGate Node");
				throw undefine_error("QGate");
			}
			break;
			}
		}

		std::vector<cir_type> circuit_vec = { upper_graph ,under_graph };
		m_sub_graph.emplace_back(circuit_vec);
    }
}

bool PartialAmplitudeGraph::is_corss_node(size_t ctr, size_t tar)
{
	if (ctr == tar)
	{
		QCERR("Control qubit is equal to target qubit");
		throw run_fail("Control qubit is equal to target qubit");
	}

	return ((ctr >= m_qubit_num / 2) && (tar < m_qubit_num / 2)) ||
		   ((tar >= m_qubit_num / 2) && (ctr < m_qubit_num / 2));
}
