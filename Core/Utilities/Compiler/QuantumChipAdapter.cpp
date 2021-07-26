#include "Core/Utilities/Compiler/QuantumChipAdapter.h"
#include "Core/Utilities/QProgInfo/MetadataValidity.h"
#include "Core/Utilities/Tools/QCircuitOptimize.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include "Core/Utilities/QProgInfo/QuantumMetadata.h"
#include "Core/Utilities/Tools/QProgFlattening.h"
#include "Core/Utilities/QProgTransform/TopologyMatch.h"


using namespace std;
using namespace QGATE_SPACE;

USING_QPANDA

#define PRINT_TRACE 0
#if PRINT_TRACE
#define PTrace printf
#define PTraceCircuit(cir) (std::cout << cir << endl)
#define PTraceCircuitMat(cir) { auto m = getCircuitMatrix(cir); std::cout << m << endl; }
#define PTraceMat(mat) (std::cout << (mat) << endl)
#else
#define PTrace
#define PTraceCircuit(cir)
#define PTraceCircuitMat(cir)
#define PTraceMat(mat)
#endif

QuantumChipAdapter::QuantumChipAdapter(QuantumMachine *quantum_machine, bool b_mapping/* = true*/, const std::string config_data/* = CONFIG_PATH*/)
	:m_quantum_machine(quantum_machine), m_b_enable_mapping(b_mapping), m_config_data(config_data)
{
	m_valid_gate.resize(KMETADATA_GATE_TYPE_COUNT, vector<string>(0));
	m_gates.resize(KMETADATA_GATE_TYPE_COUNT, vector<string>(0));

	init();
}

void QuantumChipAdapter::init()
{
	QuantumMetadata meta_data(m_config_data);
	std::vector<string> vec_single_gates;
	std::vector<string> vec_double_gates;
	meta_data.getQGate(vec_single_gates, vec_double_gates);

	for (auto& item : vec_single_gates)
	{
		m_gates[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(item);
	}
	for (auto& item : vec_double_gates)
	{
		m_gates[MetadataGateType::METADATA_DOUBLE_GATE].emplace_back(item);
	}
	SingleGateTypeValidator::GateType(m_gates[MetadataGateType::METADATA_SINGLE_GATE],
		m_valid_gate[MetadataGateType::METADATA_SINGLE_GATE]);  /* single gate data MetadataValidity */
	DoubleGateTypeValidator::GateType(m_gates[MetadataGateType::METADATA_DOUBLE_GATE],
		m_valid_gate[MetadataGateType::METADATA_DOUBLE_GATE]);  /* double gate data MetadataValidity */

	m_p_transf_decompos = std::make_shared<TransformDecomposition>(m_valid_gate, m_gates, m_quantum_machine);
}

void QuantumChipAdapter::adapter_conversion(QProg& prog, QVec &new_qvec)
{
	PTrace("decompose double qgate and multiple gate.\n");
	m_p_transf_decompos->decompose_double_qgate(prog);

	if (m_b_enable_mapping)
	{
		PTrace("Topological sequence mapping.\n");
		mapping(prog);
		new_qvec.clear();
		new_qvec = m_new_qvec;
	}
	
	PTrace("cir optimizer.\n");
	cir_optimizer_by_config(prog, m_config_data);

	PTrace("decompose double qgate.\n");
	m_p_transf_decompos->decompose_double_qgate(prog, false);

	PTrace("meta gate transform.\n");
	m_p_transf_decompos->meta_gate_transform(prog);
}

void QuantumChipAdapter::mapping(QProg &prog)
{
	QVec used_qubits;
	get_all_used_qubits(prog, used_qubits);

	JsonConfigParam config;
	config.load_config(m_config_data);
	std::vector<std::vector<double>> qubit_matrix;
	int qubit_num = 0;
	config.getMetadataConfig(qubit_num, qubit_matrix);
    prog = topology_match(prog, m_new_qvec, m_quantum_machine,
                          CNOT_GATE_METHOD, IBM_QX5_ARCH, m_config_data);

    return ;
}

/*******************************************************************
*                      public interface
********************************************************************/
void QPanda::quantum_chip_adapter(QProg& prog, QuantumMachine *quantum_machine, QVec &new_qvec, bool b_mapping/* = true*/, const std::string config_data/* = CONFIG_PATH*/)
{
	QuantumChipAdapter(quantum_machine, b_mapping, config_data).adapter_conversion(prog, new_qvec);
}

void QPanda::quantum_chip_adapter(QCircuit& cir, QuantumMachine *quantum_machine, QVec &new_qvec, bool b_mapping/* = true*/, const std::string config_data/*= CONFIG_PATH*/)
{
	QProg prog(cir);
	quantum_chip_adapter(prog, quantum_machine, new_qvec, b_mapping, config_data);
	cir = QProgFlattening::prog_flatten_to_cir(prog);
}
