#include "VirtualZTransfer.h"
#include "Core/Utilities/QProgInfo/MetadataValidity.h"
#include "Core/Utilities/QProgTransform/TransformDecomposition.h"
#include "Core/Utilities/Tools/QProgFlattening.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include "Core/Utilities/Tools/QCircuitOptimize.h"
#include "Core/Utilities/Tools/ArchGraph.h"

using namespace std;

USING_QPANDA

#define PRINT_TRACE 0
#if PRINT_TRACE
#define PTrace printf
#define PTraceMat(mat) (std::cout << (mat) << std::endl)
#define PTraceCircuit(cir) (std::cout << cir << std::endl)
#else
#define PTrace
#define PTraceMat(mat)
#define PTraceCircuit(cir)
#endif

/*******************************************************************
*                 class QProgCrosstalkCompensation
********************************************************************/
class QProgCrosstalkCompensation
{
	struct QCrosstalkAngle {
		uint32_t m_qubit;
		double m_crosstalk_angle;

		QCrosstalkAngle()
			:m_qubit(0), m_crosstalk_angle(0)
		{}
	};

public:
	QProgCrosstalkCompensation(const std::string& config_data = CONFIG_PATH)
		:m_qubit_size(0)
	{
		init(config_data);
	}
	~QProgCrosstalkCompensation() {}

	void init(const std::string& config_data = CONFIG_PATH) {
		m_config_reader.load_config(config_data);
		if (!(read_adjacent_matrix())) {
			QCERR_AND_THROW_ERRSTR(runtime_error, "Error: failed to read Crosstalk_config.");
		}

		read_high_frequency_qubit();
		read_compensate_angle();
	}

	QProg do_crosstalk_compensation(QProg prog) {
		QVec qv;
		get_all_used_qubits(prog, qv);
		std::map<uint32_t, Qubit*> i_to_qubit;
		for (const auto q : qv) {
			i_to_qubit.emplace(q->get_phy_addr(), q);
		}

		QProg new_prog;
#if 1
		m_layer_info = prog_layer(prog);
#if PRINT_TRACE
		PTrace("On do_crosstalk_compensation, do layer:\n");
		auto text_pic_str = draw_qprog(prog, m_layer_info);
#if defined(WIN32) || defined(_WIN32)
		text_pic_str = fit_to_gbk(text_pic_str);
		text_pic_str = Utf8ToGbkOnWin32(text_pic_str.c_str());
#endif
		cout << text_pic_str << endl;

#endif // PRINT_TRACE

		get_all_measure_node();

		for (auto& single_layer : m_layer_info)
		{
			for (auto& seq_node_item : single_layer)
			{
				auto& n = seq_node_item.first;
				auto measure_pos = find_measure(n->m_iter);
				if (m_measure_buf.end() != measure_pos)
				{
					new_prog.pushBackNode(std::dynamic_pointer_cast<QNode>((deepCopy(measure_pos->second)).getImplementationPtr()));
					m_measure_buf.erase(measure_pos);
				}

				if (DAGNodeType::MEASURE == n->m_type)
				{
					//do nothing
				}
				else if (DAGNodeType::RESET == n->m_type)
				{
					std::shared_ptr<AbstractQuantumReset> p_reset = std::dynamic_pointer_cast<AbstractQuantumReset>(*(n->m_iter));
					QReset tmp_reset_node(p_reset);
					new_prog.pushBackNode(std::dynamic_pointer_cast<QNode>((deepCopy(tmp_reset_node)).getImplementationPtr()));
				}
				else
				{
					std::shared_ptr<AbstractQGateNode> p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(*(n->m_iter));
					QGate tmp_gate_node(p_gate);
					new_prog.pushBackNode(std::dynamic_pointer_cast<QNode>((deepCopy(tmp_gate_node)).getImplementationPtr()));

					if (CZ_GATE == n->m_type)
					{
						QVec gate_qubits;
						p_gate->getQuBitVector(gate_qubits);

						//std::vector<int> need_compensate_qubits; //adjacent qubit
#if 0
						for (auto tmp_gate_qubit : gate_qubits)
						{
							const size_t index = tmp_gate_qubit->get_phy_addr();
							//bool b_high_frequency_qubit = false;
							for (const auto& tmp_high_qubit : m_high_frequency_qubits)
							{
								if (tmp_high_qubit == index)
								{
									//need_compensate_qubits.push_back(tmp_high_qubit);
									//get adjacent qubit
									for (int a = 0; a < m_qubit_topo_matrix[index].size(); ++a)
									{
										if (0 != m_qubit_topo_matrix[index][a])
										{
											if ((gate_qubits[0]->get_phy_addr() != a) && (gate_qubits[1]->get_phy_addr() != a))
											{
												need_compensate_qubits.push_back(a);
											}
										}
									}

									//b_high_frequency_qubit = true;
								}
							}

							/*if (b_high_frequency_qubit)
							{
								break;
							}*/
						}
#endif
						std::pair<size_t, size_t> used_qubit_pair = std::make_pair(gate_qubits[0]->get_phy_addr(), gate_qubits[1]->get_phy_addr());
						auto _compensate_angle_itr = m_compensate_angle.find(used_qubit_pair);
						if (m_compensate_angle.end() == _compensate_angle_itr)
						{
							const auto _tmp = used_qubit_pair.first;
							used_qubit_pair.first = used_qubit_pair.second;
							used_qubit_pair.second = _tmp;
							_compensate_angle_itr = m_compensate_angle.find(used_qubit_pair);
							if (m_compensate_angle.end() == _compensate_angle_itr)
							{
								QCERR("Error: There is no corresponding crosstalk compensation information, error qubit pair: ("
									<< gate_qubits.front()->get_phy_addr() << ", " << gate_qubits.back()->get_phy_addr() << ")");
								continue;
							}
						}

						const auto& cresstalk_qubits = _compensate_angle_itr->second;
						for (const auto& crosstalk_angle_item : cresstalk_qubits)
						{
							if (i_to_qubit.find(crosstalk_angle_item.m_qubit) != i_to_qubit.end()) {
								new_prog << RZ(i_to_qubit.at(crosstalk_angle_item.m_qubit), crosstalk_angle_item.m_crosstalk_angle);
							}
						}
					}
				}
			}
		}

		for (auto measure_node : m_measure_buf)
		{
			new_prog.pushBackNode(std::dynamic_pointer_cast<QNode>((deepCopy(measure_node.second)).getImplementationPtr()));
		}

#if PRINT_TRACE
		PTrace("On do_crosstalk_compensation, new prog:\n");
		PTraceCircuit(new_prog);
#endif // PRINT_TRACE
#endif
		return new_prog;
	}

	void get_all_measure_node() {
		for (auto& single_layer : m_layer_info)
		{
			for (auto& seq_node_item : single_layer)
			{
				auto& n = seq_node_item.first;
				if (DAGNodeType::MEASURE == n->m_type)
				{
					std::shared_ptr<AbstractQuantumMeasure> p_measure = std::dynamic_pointer_cast<AbstractQuantumMeasure>(*(n->m_iter));
					QMeasure tmp_measure_node(p_measure);
					const auto& next_node_vec = seq_node_item.second;
					if (next_node_vec.empty() || (nullptr == next_node_vec[0]))
					{
						m_measure_buf.emplace_back(NodeIter(), tmp_measure_node);
					}
					else
					{
						m_measure_buf.emplace_back(next_node_vec[0]->m_iter, tmp_measure_node);
					}
				}
			}
		}
	}

	std::list<std::pair<NodeIter, QMeasure>>::iterator find_measure(const NodeIter& target_itr) {
		for (auto itr = m_measure_buf.begin(); itr != m_measure_buf.end(); ++itr)
		{
			if (target_itr == itr->first)
			{
				return itr;
			}
		}

		return m_measure_buf.end();
	}

	void read_compensate_angle() {
		m_compensate_angle.clear();
		//auto& virtual_z_config = get_virtual_z_config();
		const auto& virtual_z_config = get_crosstalk_config();
		/*auto& doc = m_config_reader.get_root_element();
		if (!(doc.HasMember(VIRTUAL_Z_CONFIG)))
		{
			QCERR_AND_THROW(init_fail, "Error: virtual_Z_config error.");
		}

		const auto& virtual_z_config = doc[VIRTUAL_Z_CONFIG];*/

		if (!(virtual_z_config.HasMember(COMPENSATE_ANGLE)))
		{
			QCERR("Failed to read_compensate_angle.");
			return;
		}

		auto& compensate_angle_conf = virtual_z_config[COMPENSATE_ANGLE];
		std::string qubit_str;
		string qubit_1;
		string qubit_2;
		for (rapidjson::Value::ConstMemberIterator iter = compensate_angle_conf.MemberBegin(); iter != compensate_angle_conf.MemberEnd(); ++iter)
		{
			std::string str_key = iter->name.GetString();
			size_t start_pos = str_key.find_first_of('(') + 1;
			qubit_str = str_key.substr(start_pos);
			qubit_1 = qubit_str.substr(0, qubit_str.find_first_of(','));
			start_pos = qubit_str.find_first_of(',') + 1;
			qubit_2 = qubit_str.substr(start_pos, qubit_str.find_first_of(')') - start_pos);
			auto qubis_pair = std::make_pair(stoi(qubit_1), stoi(qubit_2));

			const auto& _crosstalk_angle = compensate_angle_conf[str_key.c_str()];
			std::vector<QCrosstalkAngle> cross_angle_vec;
			for (rapidjson::Value::ConstMemberIterator iter = _crosstalk_angle.MemberBegin();
				iter != _crosstalk_angle.MemberEnd(); ++iter)
			{
				cross_angle_vec.emplace_back(QCrosstalkAngle());
				cross_angle_vec.back().m_qubit = atoi(iter->name.GetString());
				if (!iter->value.IsDouble()) {
					QCERR_AND_THROW(run_fail, "Error: compensate_angle_conf error.");
				}
				cross_angle_vec.back().m_crosstalk_angle = iter->value.GetDouble();
			}

			m_compensate_angle.insert(std::make_pair(qubis_pair, cross_angle_vec));
		}

		return;
	}

	bool read_adjacent_matrix() {
		std::unique_ptr<ArchGraph> p_arch_graph = JsonBackendParser<ArchGraph>::Parse(m_config_reader.get_root_element());
		m_qubit_size = p_arch_graph->get_vertex_count();
		m_qubit_topo_matrix = p_arch_graph->get_adjacent_matrix();
		return true;
	}

	void read_high_frequency_qubit() {
		m_high_frequency_qubits.clear();
		/*auto& doc = m_config_reader.get_root_element();
		if (!(doc.HasMember(VIRTUAL_Z_CONFIG)))
		{
			QCERR_AND_THROW(init_fail, "Error: virtual_Z_config error.");
		}

		const auto& virtual_z_config = doc[VIRTUAL_Z_CONFIG];*/
		const auto& virtual_z_config = get_crosstalk_config();

		if (!(virtual_z_config.HasMember(HIGH_FREQUENCY_QUBIT)))
		{
			QCERR_AND_THROW(runtime_error, "Error: failed to read Crosstalk_config: no HighFrequencyQubit");
		}

		auto& high_frequency_qubit_conf = virtual_z_config[HIGH_FREQUENCY_QUBIT];
		for (int i = 0; i < high_frequency_qubit_conf.Size(); ++i)
		{
			m_high_frequency_qubits.push_back(high_frequency_qubit_conf[i].GetInt());
		}

		return;
	}

	const rapidjson::Value& get_crosstalk_config()
	{
		auto& doc = m_config_reader.get_root_element();
		if (!(doc.HasMember(VIRTUAL_Z_CONFIG)))
		{
			QCERR_AND_THROW(init_fail, "Error: virtual_Z_config error.");
		}

		return doc[VIRTUAL_Z_CONFIG];
	}

private:
	//QuantumChipConfig m_config;
	size_t m_qubit_size;
	std::vector<std::vector<int>> m_qubit_topo_matrix;
	std::vector<int> m_high_frequency_qubits;
	std::map<std::pair<int, int>, std::vector<QCrosstalkAngle>> m_compensate_angle;
	std::map<int, QGate> m_compensate_gates;
	TopologSequence<pOptimizerNodeInfo> m_layer_info;
	std::list<std::pair<NodeIter, QMeasure>> m_measure_buf;
	JsonConfigParam m_config_reader;
};

/*******************************************************************
*                 class DecomposeU3
********************************************************************/
class DecomposeU3 : protected TraverseByNodeIter
{
public:
	~DecomposeU3() {}
	static DecomposeU3& get_instance() {
		static DecomposeU3 _instance;
		return _instance;
	}

	void execute(QProg src_prog) {
		traverse_qprog(src_prog);
	}

	void execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam& cir_param, NodeIter& cur_node_iter) override {
		decompost_U3(cur_node, parent_node, cir_param, cur_node_iter);
	}

protected:
	DecomposeU3() {}
	void decompost_U3(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam& cir_param, NodeIter& cur_node_iter) {
		const GateType gt = (GateType)(cur_node->getQGate()->getGateType());
		if (gt == U3_GATE)
		{
			QGATE_SPACE::U3* u3_gate = dynamic_cast<QGATE_SPACE::U3*>(cur_node->getQGate());
			double theta = u3_gate->get_theta();
			double phi = u3_gate->get_phi();
			double lambda = u3_gate->get_lambda();
			QVec gate_qubits;
			cur_node->getQuBitVector(gate_qubits);

			QCircuit cir = build_cir_equal_to_U3(gate_qubits[0], theta, phi, lambda);
#if PRINT_TRACE
			{
				/* just for test */
				const auto _mat_tmp = getCircuitMatrix(cir);
				QStat _mat_u3;
				u3_gate->getMatrix(_mat_u3);
				if (0 == mat_compare(_mat_tmp, _mat_u3, 1e-13)) {
					cout << "decompose u3 okkkkkkkkkkkkkkkkkkkkk." << endl;
				}
				else {
					cout << "decompose u3 ffffffffffffffffailed." << endl;
					//return;
				}
		}
#endif // PRINT_TRACE

			//set-dagger
			cir.setDagger(cur_node->isDagger() ^ (cir_param.m_is_dagger));

			//set-control
			QVec ctrl_qubits;
			if ((0 < cur_node->getControlVector(ctrl_qubits)) || (cir_param.m_control_qubits.size() > 0))
			{
				//get control info
				auto increased_control = QCircuitParam::get_real_append_qubits(cir_param.m_control_qubits, ctrl_qubits);
				ctrl_qubits.insert(ctrl_qubits.end(), increased_control.begin(), increased_control.end());
				cir.setControl(ctrl_qubits);
			}

			//replace
			auto node_type = parent_node->getNodeType();
			switch (node_type)
			{
			case CIRCUIT_NODE:
			{
				auto cir_node = std::dynamic_pointer_cast<AbstractQuantumCircuit>(parent_node);
				cir_node->insertQNode(cur_node_iter, std::dynamic_pointer_cast<QNode>(cir.getImplementationPtr()));
				cir_node->deleteQNode(cur_node_iter);
			}
			break;

			case PROG_NODE:
			{
				auto prog_node = std::dynamic_pointer_cast<AbstractQuantumProgram>(parent_node);
				prog_node->insertQNode(cur_node_iter, std::dynamic_pointer_cast<QNode>(cir.getImplementationPtr()));
				prog_node->deleteQNode(cur_node_iter);
			}
			break;

			default:
				QCERR_AND_THROW_ERRSTR(run_fail, "Error: failed to delete target QNode, Node type error.");
				break;
			}
		}
	}

    double to_real_angle(const double& src_angle)
    {
        const double base = 2.0 * PI;
        const uint32_t loop = src_angle / base;
        return src_angle - ((double)loop * base);
    }

	QCircuit build_cir_equal_to_U3(Qubit* q, const double theta, const double phi, const double lamda) {
		QCircuit cir;
        const double real_theta = to_real_angle(theta);
        if ((abs(real_theta - PI) < 1e-14)) {
            cir << RPhi(q, PI, PI - lamda) << RZ(q, lamda - PI + phi);
        }
        else 
        {
            cir << RZ(q, lamda) << RX(q, PI * 0.5) << RZ(q, (PI + theta))
                << RX(q, (PI * 0.5)) << RZ(q, ((PI * 3) + phi));
            /*cir << RZ(q, lamda - (PI * 0.5) - (PI * 0.5)) << RX(q, PI * 0.5) << RZ(q, (PI - theta))
                << RX(q, (PI * 0.5)) << RZ(q, (phi - (PI * 0.5) + (PI * 0.5)));*/
        }

		return cir;
	}

private:
};

/*******************************************************************
*                 class TransferToU3Gate
********************************************************************/
class TransferToU3Gate
{
public:
	TransferToU3Gate(QuantumMachine* quantum_machine);
	~TransferToU3Gate() {}

	/* @brief  Transform Quantum program
	 * @param[in]  QProg&  quantum program
	 * @return     void
	 */
	virtual void transform(QProg&);

private:
	QuantumMachine* m_quantum_machine;
	std::map<int, std::string>  m_gatetype; /**< Quantum gatetype map   */
};

TransferToU3Gate::TransferToU3Gate(QuantumMachine* quantum_machine)
	:m_quantum_machine(quantum_machine)
{
	m_gatetype.insert(pair<int, string>(U3_GATE, "U3"));
	m_gatetype.insert(pair<int, string>(CZ_GATE, "CZ"));
}

void TransferToU3Gate::transform(QProg& prog)
{
	vector<vector<string>> ValidQGateMatrix(KMETADATA_GATE_TYPE_COUNT, vector<string>(0));
	vector<vector<string>> QGateMatrix(KMETADATA_GATE_TYPE_COUNT, vector<string>(0));

	QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype[U3_GATE]);
	QGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE].emplace_back(m_gatetype[CZ_GATE]);

	SingleGateTypeValidator::GateType(QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE],
		ValidQGateMatrix[MetadataGateType::METADATA_SINGLE_GATE]);  /* single gate data MetadataValidity */
	DoubleGateTypeValidator::GateType(QGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE],
		ValidQGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE]);  /* double gate data MetadataValidity */

	TransformDecomposition traversal_vector(ValidQGateMatrix, QGateMatrix, m_quantum_machine);

	traversal_vector.TraversalOptimizationMerge(prog);
	PTrace("after u3 optimizer:\n");
	PTraceCircuit(prog);
}

/*******************************************************************
*                 class VirtualZTransfer
********************************************************************/
class VirtualZTransfer : public ProcessOnTraversing
{
public:
	VirtualZTransfer(QProg src_prog, bool b_del_rz_gate = false);
	~VirtualZTransfer() {}

	void run();
	void process(const bool on_travel_end = false) override;

protected:
	void process_single_gate(const size_t& qubit_index, const bool& on_travel_end);
	void output_new_prog(bool b_on_traversal_end);
	double get_single_angle_parameter(const pOptimizerNodeInfo& node);

	void handle_RZ_gate(const pOptimizerNodeInfo& cur_node, pOptimizerNodeInfo& last_node,
		std::vector<pOptimizerNodeInfo>& new_node_vec, QCircuit& tmp_cir);
	void handle_RX_gate(const pOptimizerNodeInfo& cur_node, pOptimizerNodeInfo& last_node,
		std::vector<pOptimizerNodeInfo>& new_node_vec, QCircuit& tmp_cir);
	void handle_CZ_gate(const pOptimizerNodeInfo& cur_node, pOptimizerNodeInfo& last_node,
		std::vector<pOptimizerNodeInfo>& new_node_vec, QCircuit& tmp_cir);
	void handle_RPhi_gate(const pOptimizerNodeInfo& cur_node, pOptimizerNodeInfo& last_node,
		std::vector<pOptimizerNodeInfo>& new_node_vec, QCircuit& tmp_cir);
	void handle_ECHO_gate(const pOptimizerNodeInfo& cur_node, pOptimizerNodeInfo& last_node,
		std::vector<pOptimizerNodeInfo>& new_node_vec, QCircuit& tmp_cir);
	void handle_unknow_gate(const pOptimizerNodeInfo& cur_node, pOptimizerNodeInfo& last_node,
		std::vector<pOptimizerNodeInfo>& new_node_vec, QCircuit& tmp_cir);

public:
	QProg m_new_prog;

private:
	QProg m_src_prog;
	TopologSequence<pOptimizerNodeInfo> m_topolog_sequence;
	std::vector<NodeIter> m_z_gate_vec;
	threadPool m_thread_pool;
	std::atomic<size_t> m_finished_job_cnt;
	std::vector<QCircuit> m_tmp_cir;
	std::mutex m_queue_mutex;
	std::map<size_t, bool> m_b_output; /* true: normal output, false:pause */
	std::vector<pOptimizerNodeInfo> m_tmp_double_gates;
	const bool m_b_del_rz_gate;
};

VirtualZTransfer::VirtualZTransfer(QProg src_prog, bool b_del_rz_gate/* = false*/)
	:m_src_prog(src_prog), m_b_del_rz_gate(b_del_rz_gate)
{
	m_thread_pool.init_thread_pool(4);
}

void VirtualZTransfer::run()
{
	run_traversal(m_src_prog);
}

void VirtualZTransfer::process(const bool on_travel_end /*= false*/)
{
	m_finished_job_cnt = 0;

	for (const auto& item : m_cur_gates_buffer)
	{
		//create thread process
		//process_single_gate(item.first, on_travel_end);
		m_thread_pool.append(std::bind(&VirtualZTransfer::process_single_gate, this, item.first, on_travel_end));
	}

	while (m_finished_job_cnt != m_cur_gates_buffer.size()) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }

	output_new_prog(on_travel_end);
}

double VirtualZTransfer::get_single_angle_parameter(const pOptimizerNodeInfo& node)
{
	auto& node_iter = node->m_iter;
	auto p_gate_node = std::dynamic_pointer_cast<AbstractQGateNode>(*node_iter);
	auto p_single_angle_gate = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter*>(p_gate_node->getQGate());
	double angle = p_single_angle_gate->getParameter();

	if ((p_gate_node->isDagger()) ^ (node->m_is_dagger))
	{
		angle *= -1;
	}

	return angle;
}

void VirtualZTransfer::handle_RZ_gate(const pOptimizerNodeInfo& cur_node, pOptimizerNodeInfo& last_node,
	std::vector<pOptimizerNodeInfo>& new_node_vec, QCircuit& tmp_cir)
{
	const GateType last_gt_type = (GateType)(last_node->m_type);
	if (last_gt_type == RZ_GATE)
	{
		//merge
		double angle = get_single_angle_parameter((cur_node));
		angle += get_single_angle_parameter(last_node);

		tmp_cir << RZ((cur_node)->m_target_qubits[0], angle);
		last_node = std::make_shared<OptimizerNodeInfo>(tmp_cir.getLastNodeIter(), 0,
			(cur_node)->m_target_qubits, (cur_node)->m_control_qubits, RZ_GATE, (cur_node)->m_parent_node, false);
	}
	else
	{
		//update last_node
		new_node_vec.push_back(last_node);
		last_node = cur_node;
	}
}

void VirtualZTransfer::handle_RX_gate(const pOptimizerNodeInfo& cur_node, pOptimizerNodeInfo& last_node,
	std::vector<pOptimizerNodeInfo>& new_node_vec, QCircuit& tmp_cir)
{
	const GateType last_gt_type = (GateType)(last_node->m_type);
	if (last_gt_type == RZ_GATE)
	{
		//swap position by virtual z gate
		double rx_angle = get_single_angle_parameter(cur_node);
		double last_rz_angle = get_single_angle_parameter(last_node);
		double phi = -last_rz_angle;
		while (phi <= 0) 
		{
			phi += 2 * PI;
		}
		tmp_cir << RPhi((cur_node)->m_target_qubits[0], rx_angle, phi);
		pOptimizerNodeInfo tmp_node = std::make_shared<OptimizerNodeInfo>(tmp_cir.getLastNodeIter(), 0,
			(cur_node)->m_target_qubits, (cur_node)->m_control_qubits, RPHI_GATE, (cur_node)->m_parent_node, false);

		new_node_vec.push_back(tmp_node);
	}
	else if (last_gt_type == RX_GATE)
	{
		//merge
		double angle = get_single_angle_parameter(cur_node);
		angle += get_single_angle_parameter(last_node);

		tmp_cir << RX((cur_node)->m_target_qubits[0], angle);
		last_node = std::make_shared<OptimizerNodeInfo>(tmp_cir.getLastNodeIter(), 0,
			(cur_node)->m_target_qubits, (cur_node)->m_control_qubits, RX_GATE, (cur_node)->m_parent_node, false);
	}
	else
	{
		new_node_vec.push_back(last_node);
		last_node = (cur_node);
	}
}

void VirtualZTransfer::handle_CZ_gate(const pOptimizerNodeInfo& cur_node, pOptimizerNodeInfo& last_node,
	std::vector<pOptimizerNodeInfo>& new_node_vec, QCircuit& tmp_cir)
{
	const GateType last_gt_type = (GateType)(last_node->m_type);
	if (last_gt_type == RZ_GATE)
	{
		//swap
		new_node_vec.push_back(cur_node);
	}
	else
	{
		new_node_vec.push_back(last_node);
		last_node = cur_node;
	}
}

void VirtualZTransfer::handle_RPhi_gate(const pOptimizerNodeInfo& cur_node, pOptimizerNodeInfo& last_node,
	std::vector<pOptimizerNodeInfo>& new_node_vec, QCircuit& tmp_cir)
{
	const GateType last_gt_type = (GateType)(last_node->m_type);
	if (last_gt_type == RZ_GATE)
	{
		//RPHI
		auto p_phi_gate = std::dynamic_pointer_cast<AbstractQGateNode>(*((cur_node)->m_iter));
		QGATE_SPACE::RPhi* rphi_gate = dynamic_cast<QGATE_SPACE::RPhi*>(p_phi_gate->getQGate());
		double phi = rphi_gate->get_phi();
		double theta = rphi_gate->getBeta();

		auto p_rz_gate = std::dynamic_pointer_cast<AbstractQGateNode>(*(last_node->m_iter));
		auto gate_parameter = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter*>(p_rz_gate->getQGate());
		double gate_angle = gate_parameter->getParameter();
		//(phi - gate_angle) > 0 ? (phi - gate_angle) : (phi - gate_angle + 360)
		double phi_ = phi - gate_angle;
		while (phi_ <= 0)
		{
			phi_ += 2 * PI;
		}
		tmp_cir << RPhi((cur_node)->m_target_qubits[0], theta, phi_);
		pOptimizerNodeInfo tmp_node = std::make_shared<OptimizerNodeInfo>(tmp_cir.getLastNodeIter(), 0,
			(cur_node)->m_target_qubits, (cur_node)->m_control_qubits, RPHI_GATE, (cur_node)->m_parent_node, false);

		new_node_vec.push_back(tmp_node);
	}
	else
	{
		new_node_vec.push_back(last_node);
		last_node = cur_node;
	}
}

void VirtualZTransfer::handle_ECHO_gate(const pOptimizerNodeInfo& cur_node, pOptimizerNodeInfo& last_node,
	std::vector<pOptimizerNodeInfo>& new_node_vec, QCircuit& tmp_cir)
{
	//swap
	const GateType last_gt_type = (GateType)(last_node->m_type);
	if (last_gt_type == RZ_GATE)
	{
		//swap
		new_node_vec.push_back(cur_node);
	}
	else
	{
		new_node_vec.push_back(last_node);
		last_node = cur_node;
	}
}

void VirtualZTransfer::handle_unknow_gate(const pOptimizerNodeInfo& cur_node, pOptimizerNodeInfo& last_node,
	std::vector<pOptimizerNodeInfo>& new_node_vec, QCircuit& tmp_cir)
{
	new_node_vec.push_back(last_node);
	last_node = cur_node;
}

void VirtualZTransfer::process_single_gate(const size_t& qubit_index, const bool& on_travel_end)
{
    do
    {
        QCircuit tmp_cir;
        std::vector<pOptimizerNodeInfo>& node_vec = m_cur_gates_buffer.at(qubit_index);
        if (node_vec.size() == 0) {
            break;
        }

        std::vector<pOptimizerNodeInfo> new_node_vec;
        pOptimizerNodeInfo last_node = node_vec.front();
        auto itr = ++(node_vec.begin());
        const auto& gate_vec_size = m_cur_gates_buffer.get_target_qubit_sink_size(qubit_index);
        const uint32_t error_last_gate_type = 36;
        for (size_t _i = 1; (_i < gate_vec_size) && (itr != node_vec.end()); ++itr, ++_i)
        {
            const DAGNodeType cur_gt_type = (DAGNodeType)((*itr)->m_type);
            if ((cur_gt_type == NUKNOW_SEQ_NODE_TYPE) || (cur_gt_type > DAGNodeType::MAX_GATE_TYPE))
            {
                if (!(on_travel_end && m_b_del_rz_gate && (RZ_GATE == last_node->m_type))) {
                    new_node_vec.push_back(last_node);
                }

                if ((itr + 1) != node_vec.end() && (_i + 1) < gate_vec_size) {
                    last_node = *itr;
                    continue;
                }

                new_node_vec.push_back(*itr);
                last_node = nullptr;
                break;
            }

            switch (cur_gt_type)
            {
            case RZ_GATE:
                handle_RZ_gate(*itr, last_node, new_node_vec, tmp_cir);
                break;

            case RX_GATE:
                handle_RX_gate(*itr, last_node, new_node_vec, tmp_cir);
                break;

            case CZ_GATE:
                handle_CZ_gate(*itr, last_node, new_node_vec, tmp_cir);
                break;

            case RPHI_GATE:
                handle_RPhi_gate(*itr, last_node, new_node_vec, tmp_cir);
                break;

            case ECHO_GATE:
            case BARRIER_GATE:
                handle_ECHO_gate(*itr, last_node, new_node_vec, tmp_cir);
                break;

            default:
                handle_unknow_gate(*itr, last_node, new_node_vec, tmp_cir);
                break;
            }
        }

        if ((nullptr != last_node))
        {
            if (!(on_travel_end && m_b_del_rz_gate && (RZ_GATE == last_node->m_type))) {
                new_node_vec.push_back(last_node);
            }
        }

        m_cur_gates_buffer.get_target_qubit_sink_size(qubit_index) = new_node_vec.size();
        node_vec.swap(new_node_vec);

        m_queue_mutex.lock();
        m_tmp_cir.push_back(tmp_cir);
        m_queue_mutex.unlock();
    } while (false);

	++m_finished_job_cnt;
}

void VirtualZTransfer::output_new_prog(bool b_on_traversal_end)
{
	if (m_b_output.size() == 0)
	{
		for (auto& item : m_cur_gates_buffer)
		{
			m_b_output.insert(std::make_pair(item.first, true));
		}
	}

	while (true)
	{
		if (!b_on_traversal_end)
		{
			/* Keep at least 2 nodes */
			for (auto& item : m_cur_gates_buffer)
			{
				if (2 > m_cur_gates_buffer.get_target_qubit_sink_size(item.first))
				{
					return;
				}
			}
		}

		bool b_finished = true;
		for (auto& item : m_cur_gates_buffer)
		{
			if (0 != m_cur_gates_buffer.get_target_qubit_sink_size(item.first))
			{
				b_finished = false;
			}
			else
			{
				m_b_output.at(item.first) = false;
			}
		}
		if (b_finished)
		{
			break;
		}

		for (auto& item : m_cur_gates_buffer)
		{
			if (!(m_b_output.at(item.first)))
			{
				continue;
			}
			const auto& p_node = item.second.front();
			if ((p_node->m_target_qubits.size() == 1) && (p_node->m_control_qubits.size() == 0))
			{
				m_new_prog.pushBackNode(*(p_node->m_iter));
				//item.second.erase(item.second.begin());
				m_cur_gates_buffer.remove(item.first, item.second.begin());
				continue;
			}
			else
			{
				m_tmp_double_gates.push_back(p_node);
				m_b_output.at(item.first) = false;
			}
		}

		bool b_all_qubit_waiting = false;
		for (const auto& qubit_stat : m_b_output)
		{
			b_all_qubit_waiting = (b_all_qubit_waiting || qubit_stat.second);
		}

		if (b_all_qubit_waiting)
		{
			continue;
		}

		const size_t old_size = m_tmp_double_gates.size();
		for (auto itr1 = m_tmp_double_gates.begin(); itr1 != m_tmp_double_gates.end();)
		{
			bool b_found_same_node = false;
			int same_num = 1;
			const auto relation_qubit_num = (*itr1)->m_target_qubits.size() + (*itr1)->m_control_qubits.size();
			for (auto itr2 = (itr1 + 1); itr2 != m_tmp_double_gates.end(); ++itr2)
			{
				//bool b_same_qubits = true;
				if ((*itr1)->m_iter == (*itr2)->m_iter)
				{
					++same_num;
				}

				if (relation_qubit_num == same_num)
				{
					b_found_same_node = true;
				}
			}

			if (b_found_same_node)
			{
				auto found_same_node = (*itr1);
				m_new_prog.pushBackNode(*(found_same_node->m_iter));

				for (size_t h = 0; h < (found_same_node->m_target_qubits.size()); ++h)
				{
					const auto qubit_index = found_same_node->m_target_qubits.at(h)->get_phy_addr();
					m_cur_gates_buffer.remove(qubit_index, m_cur_gates_buffer.at(qubit_index).begin());
					/*m_cur_gates_buffer.at(qubit_index).erase(m_cur_gates_buffer.at(qubit_index).begin());
					--m_cur_buffer_pos.at(qubit_index);*/
					m_b_output.at(qubit_index) = true;
				}

				for (size_t h = 0; h < (found_same_node->m_control_qubits.size()); ++h)
				{
					if (BARRIER_GATE != found_same_node->m_type)
					{
						QCERR_AND_THROW_ERRSTR(run_fail, "Error: wrong multi-control-gate type.");
					}
					const auto qubit_index = found_same_node->m_control_qubits.at(h)->get_phy_addr();
					m_cur_gates_buffer.remove(qubit_index, m_cur_gates_buffer.at(qubit_index).begin());
					/*m_cur_gates_buffer.at(qubit_index).erase(m_cur_gates_buffer.at(qubit_index).begin());
					--m_cur_buffer_pos.at(qubit_index);*/
					m_b_output.at(qubit_index) = true;
				}

				for (auto itr2 = (itr1 + 1); itr2 != m_tmp_double_gates.end(); )
				{
					if ((*itr1)->m_iter == (*itr2)->m_iter)
					{
						itr2 = m_tmp_double_gates.erase(itr2);
					}
					else
					{
						++itr2;
					}
				}

				itr1 = m_tmp_double_gates.erase(itr1);
			}
			else
			{
				++itr1;
			}
		}

		if (old_size == m_tmp_double_gates.size())
		{
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: unknow error on VirtualZTransfer::output_new_prog.");
		}
	}
}

/*******************************************************************
*                      public interface
********************************************************************/
void QPanda::transfer_to_u3_gate(QProg& prog, QuantumMachine* quantum_machine)
{
	TransferToU3Gate transfer(quantum_machine);
	transfer.transform(prog);
}

void QPanda::transfer_to_u3_gate(QCircuit& circuit, QuantumMachine* quantum_machine)
{
	QProg tmp_prog(circuit);
	transfer_to_u3_gate(tmp_prog, quantum_machine);
	circuit = QProgFlattening::prog_flatten_to_cir(tmp_prog);
}

void QPanda::transfer_to_rotating_gate(QProg& prog, QuantumMachine* quantum_machine, const std::string& config_data /*= CONFIG_PATH*/)
{
	transfer_to_u3_gate(prog, quantum_machine);

	decompose_U3(prog, config_data);
}

void QPanda::cir_crosstalk_compensation(QProg& prog, const std::string& config_data/* = CONFIG_PATH*/)
{
	try
	{
		prog = QProgCrosstalkCompensation(config_data).do_crosstalk_compensation(prog);
	}
	catch (const std::exception& e)
	{
		cout << "error on src-prog:" << prog << endl;
		cout << "config_data:\n" << config_data << endl;
		QCERR_AND_THROW(run_fail, "Error: cir_crosstalk_compensation failed, catched an exception:" << e.what());
	}
}

void QPanda::virtual_z_transform(QCircuit& cir, QuantumMachine* quantum_machine, const bool b_del_rz_gate/* = false*/, const std::string& config_data/* = CONFIG_PATH*/)
{
	QProg tmp_prog(cir);

	virtual_z_transform(tmp_prog, quantum_machine, b_del_rz_gate, config_data);

	cir = QProgFlattening::prog_flatten_to_cir(tmp_prog);
}

void QPanda::virtual_z_transform(QProg& prog, QuantumMachine* quantum_machine, const bool b_del_rz_gate/* = false*/, const std::string& config_data /*= CONFIG_PATH*/)
{
	cir_optimizer_by_config(prog, config_data);

	transfer_to_rotating_gate(prog, quantum_machine, config_data);
	PTrace("after transfer_to_rotating_gate.\n");
	PTraceCircuit(prog);
	

	//cir_crosstalk_compensation(prog, config_data);
	//PTrace("after cir_crosstalk_compensation.\n");
	//PTraceCircuit(prog);

	move_rz_backward(prog, b_del_rz_gate);
	flatten(prog, true);
}

void QPanda::move_rz_backward(QProg& prog, const bool b_del_rz_gate)
{
	VirtualZTransfer virtual_z(prog, b_del_rz_gate);
	virtual_z.run();
	prog = virtual_z.m_new_prog;
}

void QPanda::decompose_U3(QProg& prog, const std::string& config_data /*= CONFIG_PATH*/)
{
	std::vector<std::pair<QCircuit, QCircuit>> optimitzer_cir;
	QCircuitOptimizerConfig u3_config_reader(config_data);
	u3_config_reader.get_u3_replace_cir(optimitzer_cir);
//#if PRINT_TRACE
//	for (size_t i = 0; i < optimitzer_cir.size(); ++i)
//	{
//		PTrace("Src:\n");
//		PTraceCircuit(optimitzer_cir.at(i).first);
//		PTrace("\nDst:\n");
//		PTraceCircuit(optimitzer_cir.at(i).second);
//		PTrace("\n-----------\n");
//	}
//#endif
	sub_cir_replace(prog, optimitzer_cir);
	PTrace("after u3 replace by config:\n");
	PTraceCircuit(prog);

	DecomposeU3::get_instance().execute(prog);
}
