#include <math.h>
#include <map>
#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/chrono.h"
#include "pybind11/eigen.h"
#include "pybind11/operators.h"
#include <pybind11/stl_bind.h>
#include "pybind11/eigen.h"
#include "pybind11/operators.h"
#include "QPandaConfig.h"
#include "QPanda.h"

USING_QPANDA
using namespace std;
using namespace pybind11::literals;
using std::map;
namespace py = pybind11;

//template<>
//struct py::detail::type_caster<QVec>
//    : py::detail::list_caster<QVec, Qubit*> { };

void init_quantum_machine(py::module &);
void init_variational(py::module &);
void init_qalg(py::module &);

#define BIND_CLASSICALCOND_OPERATOR_OVERLOAD(OP) .def(py::self OP py::self)\
                                                 .def(py::self OP cbit_size_t())\
                                                 .def(cbit_size_t() OP py::self)

PYBIND11_MODULE(pyQPanda, m)
{
    init_quantum_machine(m);
    init_variational(m);
    init_qalg(m);

    m.doc() = "";
    m.def("init",
        &init,
        "to init the environment. Use this at the beginning"
    );

    m.def("init_quantum_machine", &initQuantumMachine,
        "Create and initialize a quantum machine",
        py::return_value_policy::reference);

    m.def("get_qstate", &getQState,
        "get prog qstate",
        py::return_value_policy::automatic);

#define DEFINE_DESTROY(type)\
    m.def("destroy_quantum_machine", [] (type *machine){\
        destroyQuantumMachine(machine);\
    },\
        "destroy a quantum machine", py::return_value_policy::automatic)

    DEFINE_DESTROY(CPUQVM);
    DEFINE_DESTROY(CPUSingleThreadQVM);
    DEFINE_DESTROY(GPUQVM);
    DEFINE_DESTROY(NoiseQVM);

	m.def("finalize", []() { finalize(); },
		"to finalize the environment. Use this at the end",
		py::return_value_policy::reference
	);

    m.def("qAlloc", []() {return qAlloc(); },
        "Allocate a qubit",
        py::return_value_policy::reference
    );

    m.def("qAlloc", [](size_t size) {return qAlloc(size); },
        "Allocate a qubits",
        py::return_value_policy::reference
    );


	m.def("qAlloc_many", [](size_t size) {
		std::vector<Qubit *> temp = qAllocMany(size);
		return temp;
	},
        "Allocate several qubits",
        py::return_value_policy::reference
        );

    m.def("cAlloc", []() {return cAlloc(); },
        "Allocate a CBit",
        py::return_value_policy::reference
    );

    /* add */
    m.def("cAlloc", [](size_t size) {return cAlloc(size); },
        "Allocate a CBit",
        py::return_value_policy::reference
    );

    m.def("cAlloc_many", [](size_t size) {return cAllocMany(size); },
        "Allocate several CBits",
        py::return_value_policy::reference
    );

    m.def("cFree", &cFree, "Free a CBit");

    m.def("cFree_all", &cFreeAll, "Free several CBit");

    m.def("apply_QGate", &apply_QGate,
        "Apply QGate to qubits",
        py::return_value_policy::reference
    );

    m.def("getstat", &getstat,
        "get the status(ptr) of the quantum machine");

    /* will delete */
    m.def("getAllocateQubitNum", &getAllocateQubitNum,
        "getAllocateQubitNum");

    m.def("getAllocateCMem", &getAllocateCMem, "getAllocateCMem");

    m.def("CreateEmptyQProg", &CreateEmptyQProg,
        "Create an empty QProg Container",
        py::return_value_policy::automatic
    );

    m.def("CreateWhileProg", CreateWhileProg,
        "Classical_condition"_a, "true_node"_a,
        "Create a WhileProg",
        py::return_value_policy::automatic
    );

    m.def("CreateIfProg", [](ClassicalCondition m, QProg &qn)
    {return CreateIfProg(m, qn); },
        "Classical_condition"_a, "true_node"_a,
        "Create a IfProg",
        py::return_value_policy::automatic
    );

    m.def("CreateIfProg", [](ClassicalCondition&m, QProg &qn1, QProg &qn2)
    {return CreateIfProg(m, qn1, qn2); },
        "Classical_condition"_a, "true_node"_a, "false_node"_a,
        "Create a IfProg",
        py::return_value_policy::automatic
    );

    m.def("CreateEmptyCircuit", &CreateEmptyCircuit,
        "Create an empty QCircuit Container",
        py::return_value_policy::automatic
    );

    /* new interface */
    m.def("get_allocate_qubit_num", &getAllocateQubitNum,
        "get allocate qubit num",
         py::return_value_policy::automatic);
    m.def("get_allocate_cmem_num", &getAllocateCMem,
          "get allocate cmemnum",
          py::return_value_policy::automatic);

    m.def("create_empty_qprog", &createEmptyQProg,
        "Create an empty QProg Container",
        py::return_value_policy::automatic
    );

    m.def("create_while_prog", &createWhileProg,
        "Classical_condition"_a, "true_branch"_a,
        "Create a WhileProg",
        py::return_value_policy::automatic
    );

    m.def("create_if_prog", [](ClassicalCondition m, QProg &true_branch)
    {return createIfProg(m, true_branch); },
        "Classical_condition"_a, "true_branch"_a,
        "Create a IfProg",
        py::return_value_policy::automatic
    );

    m.def("create_if_prog", [](ClassicalCondition&m, QProg &true_branch, QProg &false_branch)
    {return createIfProg(m, true_branch, false_branch); },
        "Classical_condition"_a, "true_branch"_a, "false_branch"_a,
        "Create a IfProg",
        py::return_value_policy::automatic
    );

    m.def("create_empty_circuit", &createEmptyCircuit,
        "Create an empty QCircuit Container",
        py::return_value_policy::automatic
    );


    m.def("directly_run", &directlyRun, "directly run");

    m.def("quick_measure", &quickMeasure, "qubit_list"_a, "shots"_a, "quick measure");


    m.def("Measure", &Measure, "qubit"_a, "cbit"_a,
        "Create a Measure operation",
        py::return_value_policy::automatic
    );

    m.def("measure_all", &MeasureAll, "qubit_list"_a, "cbit_list"_a,
        "Create a Measure operation",
        py::return_value_policy::automatic
    );
    m.def("H", &H, "Create a H gate",
        py::return_value_policy::automatic
    );

	m.def("Reset", &Reset, "Create a Reset node",
		py::return_value_policy::automatic
	);

    m.def("T", &T, "Create a T gate",
        py::return_value_policy::automatic
    );

    m.def("S", &S, "qubit"_a, "Create a S gate",
        py::return_value_policy::automatic
    );

	m.def("I", &I, "qubit"_a, "Create an I gate",
		py::return_value_policy::automatic
	);

    m.def("X", &X, "qubit"_a, "Create an X gate",
        py::return_value_policy::automatic
    );

    m.def("Y", &Y, "qubit"_a, "Create a Y gate",
        py::return_value_policy::automatic
    );

    m.def("Z", &Z, "qubit"_a, "Create a Z gate",
        py::return_value_policy::automatic
    );

    m.def("X1", &X1, "qubit"_a, "Create an X1 gate",
        py::return_value_policy::automatic
    );

    m.def("Y1", &Y1, "qubit"_a, "Create a Y1 gate",
        py::return_value_policy::automatic
    );

    m.def("Z1", &Z1, "qubit"_a, "Create a Z1 gate",
        py::return_value_policy::automatic
    );

    m.def("RX", &RX, "qubit"_a, "angle"_a, "Create a RX gate",
        py::return_value_policy::automatic
    );

    m.def("RY", &RY, "qubit"_a, "angle"_a, "Create a RY gate",
        py::return_value_policy::automatic
    );

    m.def("RZ", &RZ, "qubit"_a, "angle"_a, "Create a RZ gate",
        py::return_value_policy::automatic
    );

    m.def("U1", &U1, "qubit"_a, "angle"_a, "Create a U1 gate",
        py::return_value_policy::automatic
    );

    m.def("U2", &U2, "qubit"_a, "phi"_a, "lambda"_a, "Create a U2 gate",
        py::return_value_policy::automatic
    );

    m.def("U3", &U3, "qubit"_a, "theta"_a, "phi"_a, "lambda"_a, "Create a U3 gate",
        py::return_value_policy::automatic
    );

    m.def("CNOT", &CNOT, "control_qubit"_a, "target_qubit"_a, "Create a CNOT gate",
        py::return_value_policy::automatic
    );

    m.def("CZ", &CZ, "control_qubit"_a, "target_qubit"_a, "Create a CZ gate",
        py::return_value_policy::automatic
    );

    m.def("SWAP", &SWAP, "control_qubit"_a, "target_qubit"_a, "Create a SWAP gate",
        py::return_value_policy::automatic
    );

    m.def("U4", [](QStat & matrix, Qubit *qubit)
    {return U4(matrix, qubit); }, "matrix"_a, "qubit"_a,
        "Create a U4 gate",
        py::return_value_policy::automatic
    );

    m.def("U4", [](double alpha, double beta, double gamma, double delta,
        Qubit * qubit)
    {return U4(alpha, beta, gamma, delta, qubit); }, "alpha"_a, "beta"_a, "delta"_a, "gamma"_a, "qubit"_a,
        "Create a U4 gate",
        py::return_value_policy::automatic
    );

    m.def("CU", [](double alpha, double beta, double gamma, double delta,
        Qubit * controlQBit, Qubit * targetQBit)
    {return CU(alpha, beta, gamma, delta, controlQBit, targetQBit); },
        "alpha"_a, "beta"_a, "delta"_a, "gamma"_a, "control_qubit"_a, "target_qubit"_a,
        "Create a CU gate",
        py::return_value_policy::automatic
    );

    m.def("CU", [](QStat & matrix, Qubit * controlQBit, Qubit * targetQBit)
    {return CU(matrix, controlQBit, targetQBit); },
        "matrix"_a, "control_qubit"_a, "target_qubit"_a,
        "Create a CU gate",
        py::return_value_policy::automatic
    );

    m.def("iSWAP",
        [](Qubit * controlQBit, Qubit * targetQBit)
    {return iSWAP(controlQBit, targetQBit); },
        "control_qubit"_a, "target_qubit"_a,
        "Create a iSWAP gate",
        py::return_value_policy::automatic
    );

    m.def("iSWAP",
        [](Qubit * controlQBit, Qubit * targetQBit, double theta)
    {return iSWAP(controlQBit, targetQBit, theta); },
        "control_qubit"_a, "target_qubit"_a, "angle"_a,
        "Create a iSWAP gate",
        py::return_value_policy::automatic
    );

    m.def("CR", &CR, "control_qubit"_a, "target_qubit"_a, "angle"_a, "Create a CR gate",
        py::return_value_policy::automatic
    );

    /* new interface */
    m.def("U4", [](Qubit *qubit, QStat & matrix)
    {return U4(qubit, matrix); }, "matrix"_a, "qubit"_a,
        "Create a U4 gate",
        py::return_value_policy::automatic
    );

    m.def("U4", [](Qubit * qubit, double alpha, double beta, double gamma, double delta)
    {return U4(qubit, alpha, beta, gamma, delta); }, "alpha"_a, "beta"_a, "delta"_a, "gamma"_a, "qubit"_a,
        "Create a U4 gate",
        py::return_value_policy::automatic
    );

    m.def("CU", [](Qubit * controlQBit, Qubit * targetQBit,
                double alpha, double beta, double gamma, double delta)
    {return CU(controlQBit, targetQBit, alpha, beta, gamma, delta); },
        "alpha"_a, "beta"_a, "delta"_a, "gamma"_a, "control_qubit"_a, "target_qubit"_a,
        "Create a CU gate",
        py::return_value_policy::automatic
    );

    m.def("CU", [](Qubit * controlQBit, Qubit * targetQBit, QStat & matrix)
    {return CU(controlQBit, targetQBit, matrix); },
        "matrix"_a, "control_qubit"_a, "target_qubit"_a,
        "Create a CU gate",
        py::return_value_policy::automatic
    );

    m.def("QDouble", [](Qubit * controlQBit, Qubit * targetQBit, QStat & matrix)
    {return QDouble(controlQBit, targetQBit, matrix); },
        "matrix"_a, "control_qubit"_a, "target_qubit"_a,
        "Create a CU gate",
        py::return_value_policy::automatic
    );



	m.def("print_matrix", [](QStat& mat) {
		auto mat_str = matrix_to_string(mat);
		std::cout << mat_str << endl;
		return mat_str;
	}, "mat"_a,
        "output matrix information to consol",
        py::return_value_policy::automatic
    );

    m.def("is_match_topology", &isMatchTopology, "gate"_a, "vecTopoSt"_a,
        "Whether the qgate matches the quantum topology",
        py::return_value_policy::automatic
    );

	py::class_<NodeInfo>(m, "NodeInfo")
		.def(py::init<>())
		.def_readwrite("m_itr", &NodeInfo::m_itr)
		.def_readwrite("m_node_type", &NodeInfo::m_node_type)
		.def_readwrite("m_gate_type", &NodeInfo::m_gate_type)
		.def_readwrite("m_is_dagger", &NodeInfo::m_is_dagger)
		.def_readwrite("m_qubits", &NodeInfo::m_qubits)
		.def_readwrite("m_control_qubits", &NodeInfo::m_control_qubits)
		.def("clear", &NodeInfo::clear);

    m.def("get_adjacent_qgate_type", [](QProg &prog, NodeIter &node_iter)
    {
        std::vector<NodeInfo> adjacent_nodes;
        getAdjacentQGateType(prog, node_iter, adjacent_nodes);
        return adjacent_nodes;
    }, "get the adjacent(the front one and the back one) nodes.",
        py::return_value_policy::automatic
        );

    m.def("is_swappable", &isSwappable, "prog"_a, "target_nodeItr_1"_a, "target_nodeItr_2"_a,
        "judge the specialed two NodeIters whether can be exchanged",
        py::return_value_policy::automatic
    );

    m.def("is_supported_qgate_type", &isSupportedGateType, "target_nodeItr"_a,
        "judge if the target node is a supported QGate type",
        py::return_value_policy::automatic
    );

    /* will delete */

    m.def("to_originir", [](QProg & qn, QuantumMachine *qvm)
        {return transformQProgToOriginIR(qn, qvm); },
        py::return_value_policy::automatic_reference
    );
    m.def("to_originir", [](QCircuit & qn, QuantumMachine *qvm)
        {return transformQProgToOriginIR(qn, qvm); },
        py::return_value_policy::automatic_reference
    );
    m.def("to_originir", [](QGate & qn, QuantumMachine *qvm)
        {return transformQProgToOriginIR(qn, qvm); },
        py::return_value_policy::automatic_reference
    );
    m.def("to_originir", [](QIfProg & qn, QuantumMachine *qvm)
        {return transformQProgToOriginIR(qn, qvm); },
        py::return_value_policy::automatic_reference
    );
    m.def("to_originir", [](QWhileProg & qn, QuantumMachine *qvm)
        {return transformQProgToOriginIR(qn, qvm); },
        py::return_value_policy::automatic_reference
    );
    m.def("to_originir", [](QMeasure & qn, QuantumMachine *qvm)
        {return transformQProgToOriginIR(qn, qvm); },
        py::return_value_policy::automatic_reference
    );

	m.def("originir_to_qprog", [](string file_path, QuantumMachine *qvm) {
		QVec qv;
		std::vector<ClassicalCondition> cv;
		return transformOriginIRToQProg(file_path, qvm, qv, cv);
	},
		py::return_value_policy::automatic_reference
		);

    m.def("to_QASM", [](QProg prog,QuantumMachine *qvm) {
        py::list retData;
        std::string qasmStr = convert_qprog_to_qasm(prog, qvm);
        retData.append(qasmStr);

        return retData;
    },"prog"_a,"quantum machine"_a,py::return_value_policy::automatic_reference
        );

    m.def("to_Quil",&transformQProgToQuil , "program"_a, "quantum machine"_a ,"QProg to Quil",
        py::return_value_policy::automatic_reference
        );

    m.def("count_gate", [](QProg & qn)
    {return getQGateNumber(qn); },
        "quantum_prog"_a,
        "Count quantum gate num under quantum program, quantum circuit",
        py::return_value_policy::automatic
    );

    m.def("count_gate", [](QCircuit & qn)
    {return getQGateNumber(qn); },
        "quantum_circuit"_a,
        "Count quantum gate num under quantum program, quantum circuit",
        py::return_value_policy::automatic
    );

    m.def("get_clock_cycle", [](QProg prog) {
        extern QuantumMachine* global_quantum_machine;
        return getQProgClockCycle(prog, global_quantum_machine);
    }
        , "program"_a, "Get Quantum Program Clock Cycle",
        py::return_value_policy::automatic_reference
        );

    m.def("get_bin_data", [](QProg prog) {    /* will delete */


        /* new interface */
        extern QuantumMachine* global_quantum_machine;
        return transformQProgToBinary(prog, global_quantum_machine);
    }
        , "program"_a, "Get quantum program binary data",
        py::return_value_policy::automatic_reference
        );

    m.def("bin_to_prog", [](const std::vector<uint8_t>& data, QVec & qubits,
        std::vector<ClassicalCondition>& cbits, QProg & prog) {
        extern QuantumMachine* global_quantum_machine;
        return binaryQProgDataParse(global_quantum_machine, data, qubits, cbits, prog);
    }
        , "data"_a, "qlist"_a, "clist"_a, "program"_a,
        "Parse quantum program interface for  binary data vector",
        py::return_value_policy::automatic_reference
        );
    m.def("get_bin_str", [](QProg prog, QuantumMachine *qvm) {
        auto data = transformQProgToBinary(prog, qvm);
        auto base64_data = Base64::encode(data.data(), data.size()); // 将得到的二进制数据以base64的方式编码
        std::string data_str(base64_data.begin(), base64_data.end());
        return data_str;
        }, "prog"_a, "quantum machine"_a, py::return_value_policy::automatic);

    /* new interface */

    m.def("transform_qprog_to_quil", &transformQProgToQuil
        , "program"_a,"quantum machine"_a, "QProg to Quil",
        py::return_value_policy::automatic_reference
        );

    m.def("get_qgate_num", [](QProg & qn)
    {return getQGateNumber(qn); },
        "quantum_prog"_a,
        "Count quantum gate num under quantum program, quantum circuit",
        py::return_value_policy::automatic
    );

    m.def("get_qgate_num", [](QCircuit & qn)
    {return getQGateNumber(qn); },
        "quantum_circuit"_a,
        "Count quantum gate num under quantum program, quantum circuit",
        py::return_value_policy::automatic
    );

    m.def("get_qprog_clock_cycle", &getQProgClockCycle,
        "program"_a,"quantum machine"_a "Get Quantum Program Clock Cycle",
        py::return_value_policy::automatic_reference
        );

    m.def("transform_qprog_to_binary", [](QProg prog,QuantumMachine * qvm) {
        return transformQProgToBinary(prog, qvm);
    }
        , "program"_a,"quantum machine"_a "Get quantum program binary data",
        py::return_value_policy::automatic_reference
        );

    m.def("transform_qprog_to_binary", [](QProg prog, QuantumMachine * qvm,string file_path) {
	
		return transformQProgToBinary(prog, qvm, file_path);
        }
        , "program"_a, "quantum machine"_a ,"file path"_a,"Get quantum program binary data",
            py::return_value_policy::automatic_reference
            );

    m.def("get_qprog_clock_cycle", &getQProgClockCycle,
        "program"_a, "QuantumMachine"_a, "Get Quantum Program Clock Cycle",
        py::return_value_policy::automatic_reference
        );

    m.def("transform_binary_data_to_qprog", [](QuantumMachine *qm, std::vector<uint8_t> data) {
        QVec qubits;
        std::vector<ClassicalCondition> cbits;
        QProg prog;
        transformBinaryDataToQProg(qm, data, qubits, cbits, prog);
        return prog;
    }
        , "QuantumMachine"_a, "data"_a,
        "Parse quantum program interface for binary data",
        py::return_value_policy::automatic_reference
        );

    m.def("transform_qprog_to_originir", [](QProg prog, QuantumMachine *qm) {
        return transformQProgToOriginIR(prog, qm);
    }
        , "program"_a, "QuantumMachine"_a, "QProg to originir",
        py::return_value_policy::automatic_reference
        );

	m.def("transform_originir_to_qprog", [](string file_path, QuantumMachine *qvm) {
		QVec qv;
		std::vector<ClassicalCondition> cv;
		return transformOriginIRToQProg(file_path, qvm, qv, cv);
	},
		py::return_value_policy::automatic_reference
		);

    py::enum_<SingleGateTransferType>(m, "SingleGateTransferType")
        .value("SINGLE_GATE_INVALID", SINGLE_GATE_INVALID)
        .value("ARBITRARY_ROTATION", ARBITRARY_ROTATION)
        .value("DOUBLE_CONTINUOUS", DOUBLE_CONTINUOUS)
        .value("SINGLE_CONTINUOUS_DISCRETE", SINGLE_CONTINUOUS_DISCRETE)
        .value("DOUBLE_DISCRETE", DOUBLE_DISCRETE)
        .export_values();

    py::enum_<DoubleGateTransferType>(m, "DoubleGateTransferType")
        .value("DOUBLE_GATE_INVALID", DOUBLE_GATE_INVALID)
        .value("DOUBLE_BIT_GATE", DOUBLE_BIT_GATE)
        .export_values();

    m.def("validate_single_qgate_type", [](std::vector<string> single_gates) {
        py::list ret_date;
        std::vector<string> valid_gates;
        auto type = validateSingleQGateType(single_gates, valid_gates);
        ret_date.append(static_cast<SingleGateTransferType>(type));
        ret_date.append(valid_gates);
        return ret_date;
    }
        , "Single QGates"_a, "get valid QGates and valid single QGate type",
        py::return_value_policy::automatic
        );

    m.def("validate_double_qgate_type", [](std::vector<string> double_gates) {
        py::list ret_data;
        std::vector<string> valid_gates;
        auto type = validateDoubleQGateType(double_gates, valid_gates);
        ret_data.append(static_cast<DoubleGateTransferType>(type));
        ret_data.append(valid_gates);
        return ret_data;
    }, "Double QGates"_a, "get valid QGates and valid double QGate type",
        py::return_value_policy::automatic_reference
        );

    m.def("get_unsupport_qgate_num", [](QProg prog, const vector<vector<string>> &gates) {
        return getUnsupportQGateNum(prog, gates);
    },
        "get unsupport QGate_num",
        py::return_value_policy::automatic
    );

    m.def("get_qgate_num", [](QProg prog) {
        return getQGateNum(prog);
    },
        "get QGate_num",
        py::return_value_policy::automatic
        );

	m.def("flatten", [](QProg &prog){
		flatten(prog);
	},
		"flatten quantum program",
		py::return_value_policy::automatic
		);

	m.def("flatten", [](QCircuit &circuit) {
		flatten(circuit);
	},
		"flatten quantum circuit",
		py::return_value_policy::automatic
		);

	m.def("convert_qprog_to_binary", [](QProg prog, QuantumMachine * qvm) {
		return convert_qprog_to_binary(prog, qvm);
	}
		, "program"_a, "quantum machine"_a "get quantum program binary data",
		py::return_value_policy::automatic_reference
		);

	m.def("convert_qprog_to_binary", [](QProg prog, QuantumMachine * qvm, string file_path) {
		convert_qprog_to_binary(prog, qvm, file_path);
	}
		, "program"_a, "quantum machine"_a, "file path"_a, "store quantum program in binary file ",
		py::return_value_policy::automatic_reference
		);

	m.def("convert_binary_data_to_qprog", [](QuantumMachine *qm, std::vector<uint8_t> data) {
		QVec qubits;
		std::vector<ClassicalCondition> cbits;
		QProg prog;
		convert_binary_data_to_qprog(qm, data, qubits, cbits, prog);
		return prog;
	}
		, "QuantumMachine"_a, "data"_a,
		"Parse quantum program interface for binary data",
		py::return_value_policy::automatic_reference
		);

	m.def("convert_originir_to_qprog", [](std::string file_path, QuantumMachine* qvm) {
		py::list ret_data;
		QVec qv;
		std::vector<ClassicalCondition> cv;
		QProg prog = convert_originir_to_qprog(file_path, qvm, qv, cv);
		py::list qubit_list;
		for (auto q : qv)
			qubit_list.append(q);

		ret_data.append(prog);
		ret_data.append(qubit_list);
		ret_data.append(cv);

		return ret_data;
	},
		"file_name"_a, "QuantumMachine"_a, "convert OriginIR to QProg",
		py::return_value_policy::automatic_reference
		);

	m.def("convert_qprog_to_originir", [](QProg prog, QuantumMachine *qm) {
		return convert_qprog_to_originir(prog, qm);
	}
		, "quantum program"_a, "quantum machine"_a, "convert QProg to OriginIR",
		py::return_value_policy::automatic_reference
		);

	m.def("convert_qprog_to_quil", &convert_qprog_to_quil,
		"quantum program"_a, "quantum machine"_a, "convert QProg to Quil",
		py::return_value_policy::automatic_reference
	);


	m.def("convert_qasm_to_qprog", [](std::string file_path, QuantumMachine* qvm) {
		py::list ret_data;
		QVec qv;
		std::vector<ClassicalCondition> cv;
		QProg prog = convert_qasm_to_qprog(file_path, qvm, qv, cv);
		py::list qubit_list;
		for (auto q : qv)
			qubit_list.append(q);

		ret_data.append(prog);
		ret_data.append(qubit_list);
		ret_data.append(cv);
		return ret_data;
	},
		"file_name"_a, "quantum machine"_a, "convert QASM to QProg",
		py::return_value_policy::automatic_reference
		);

	m.def("convert_qprog_to_qasm", [](QProg prog, QuantumMachine *qvm) {
		py::list retData;
		std::string qasmStr = convert_qprog_to_qasm(prog, qvm);
		retData.append(qasmStr);

		return retData;
	}, "prog"_a, "quantum machine"_a, py::return_value_policy::automatic_reference
	);

	m.def("cast_qprog_qgate", &cast_qprog_qgate,
		"quantum program"_a,  "cast QProg to QGate",
		py::return_value_policy::automatic_reference
	);

	m.def("cast_qprog_qmeasure", &cast_qprog_qmeasure,
		"quantum program"_a, "cast QProg to QMeasure",
		py::return_value_policy::automatic_reference
	);

	m.def("cast_qprog_qcircuit", [](QProg prog) {
		QCircuit cir;
		cast_qprog_qcircuit(prog, cir);
		return cir;
	}
		,"quantum program"_a, "cast QProg to QCircuit",
		py::return_value_policy::automatic_reference
		);

	m.def("topology_match", [](QProg prog, QVec qv, QuantumMachine *qvm, SwapQubitsMethod method, ArchType arch_type) {
		py::list ret_data;
		QProg out_prog = topology_match(prog, qv, qvm, method, arch_type);
		py::list qubit_list;
		for (auto q : qv)
			qubit_list.append(q);

		ret_data.append(out_prog);
		ret_data.append(qubit_list);
		return ret_data;
	},
		"prog"_a, "qubits"_a, "quantum machine"_a, "SwapQubitsMethod"_a = CNOT_GATE_METHOD, "ArchType"_a = IBM_QX5_ARCH,
		py::return_value_policy::automatic_reference
		);

	m.def("qcodar_match", [](QProg prog, QVec qv, QuantumMachine *qvm , QCodarGridDevice arch_type, 
		size_t m, size_t n , size_t run_times, const std::string config_data) {
		py::list ret_data;

		QProg out_prog;
		switch (arch_type)
		{
		case IBM_Q20_TOKYO:
		case IBM_Q53:
		case GOOGLE_Q54:
			out_prog = qcodar_match_by_target_meachine(prog, qv, qvm, arch_type, run_times);
			break;

		case SIMPLE_TYPE:
			out_prog = qcodar_match_by_simple_type(prog, qv, qvm, m, n, run_times);
			break;

		case ORIGIN_VIRTUAL:
			out_prog = qcodar_match_by_config(prog, qv, qvm, config_data, run_times);
			break;

		default:
			QCERR_AND_THROW_ERRSTR(runtime_error, "Error: QCodarGridDevice error on qcodar match.");
			break;
		}

		py::list qubit_list;

		for (auto q : qv)
			qubit_list.append(q);

		ret_data.append(out_prog);
		ret_data.append(qubit_list);
		return ret_data;
	}, 
		"prog"_a, "qubits"_a, "quantum machine"_a, "QCodarGridDevice"_a= SIMPLE_TYPE, "m"_a = 2, "n"_a=4, "run_times"_a=5, "config_data"_a = CONFIG_PATH,
		"/**\
		* @brief   A Contextual Duration - Aware Qubit Mapping for V arious NISQ Devices\
		* @ingroup Utilities\
		* @param[in]  QProg  quantum program\
		* @param[in, out]  QVec  qubit  vector\
		* @param[in]  QuantumMachine*  quantum machine\
        * @param[in]  QCodarGridDevice Device type, currently supported models include: \
		              IBM_Q20_TOKYO: IBM real phisical quantum chip\
	                  IBM_Q53: IBM real phisical quantum chip\
				      GOOGLE_Q54：Google real phisical quantum chip\
				      SIMPLE_TYPE：Simulator quantum chip\
					  ORIGIN_VIRTUAL：by config\
		* @param[in]  size_t   m : the length of the topology\
		* @param[in]  size_t   n : the  width of the topology\
		* @param[in]  size_t   run_times : the number of times  run the remapping, better parameters get better results\
		* @return    QProg   mapped  quantum program\
		* @note	 QCodarGridDevice : SIMPLE_TYPE  It's a simple undirected  topology graph, build a topology based on the values of m(rows) and n(cloumns)\
		* / ",
		py::return_value_policy::automatic_reference
	);



    /* will delete */
    m.def("PMeasure", &PMeasure,
        "Get the probability distribution over qubits",
        py::return_value_policy::automatic
    );

    m.def("PMeasure_no_index", &PMeasure_no_index,
        "Get the probability distribution over qubits",
        py::return_value_policy::automatic
    );

    m.def("accumulateProbability", &accumulateProbability,
        "Accumulate the probability from a prob list",
        py::return_value_policy::automatic
    );
    m.def("accumulate_probabilities", &accumulateProbability, "probability_list"_a,
        "Accumulate the probability from a prob list",
        py::return_value_policy::automatic
    );

    /* new interface */

    m.def("pmeasure", &pMeasure,
        "Get the probability distribution over qubits",
        py::return_value_policy::automatic
    );

    m.def("pmeasure_no_index", &pMeasureNoIndex,
        "Get the probability distribution over qubits",
        py::return_value_policy::automatic
    );

    m.def("accumulate_probability", &accumulateProbability, "probability_list"_a,
        "Accumulate the probability from a prob list",
        py::return_value_policy::automatic
    );


    m.def("run_with_configuration", &runWithConfiguration, "program"_a,
        "cbit_list"_a,
        "shots"_a,
        "run with configuration",
        py::return_value_policy::automatic
    );

    m.def("prob_run_tuple_list", probRunTupleList, "program"_a, "qubit_list"_a, "select_max"_a = -1,
        py::return_value_policy::reference);
    m.def("prob_run_list", probRunList, "program"_a, "qubit_list"_a, "select_max"_a = -1,
        py::return_value_policy::reference);
    m.def("prob_run_dict", probRunDict, "program"_a, "qubit_list"_a, "select_max"_a = -1,
        py::return_value_policy::reference);

    /* add */
    m.def("get_tuple_list", getProbTupleList, "qubit_list"_a, "select_max"_a = -1,
        py::return_value_policy::reference);
    m.def("get_prob_list", getProbList, "qubit_list"_a, "select_max"_a = -1,
        py::return_value_policy::reference);
    m.def("get_prob_dict", getProbDict, "qubit_list"_a, "select_max"_a = -1,
        py::return_value_policy::reference);

    py::class_<ClassicalProg>(m, "ClassicalProg")
        .def(py::init<ClassicalCondition &>());

    py::class_<QProg>(m, "QProg")
        .def(py::init<>())
        .def(py::init<QProg&>())
        .def(py::init<QCircuit &>())
        .def(py::init<QIfProg &>())
        .def(py::init<QWhileProg &>())
        .def(py::init<QGate &>())
        .def(py::init<QMeasure &>())
		.def(py::init<QReset &>())
        .def(py::init<ClassicalCondition &>())
        .def(py::init([](NodeIter & iter) {
        if (!(*iter))
        {
            QCERR("iter is null");
            throw runtime_error("iter is null");
        }

        if (PROG_NODE == (*iter)->getNodeType())
        {
            auto gate_node = std::dynamic_pointer_cast<AbstractQuantumProgram>(*iter);
            return QProg(gate_node);
        }
        else
        {
            QCERR("node type error");
            throw runtime_error("node type error");
        }
    }
))
        .def("insert", &QProg::operator<<<QProg >,
            py::return_value_policy::reference)
        .def("insert", &QProg::operator<<<QGate>,
            py::return_value_policy::reference)
        .def("insert", &QProg::operator<<<QCircuit>,
            py::return_value_policy::reference)
        .def("insert", &QProg::operator<<<QIfProg>,
            py::return_value_policy::reference)
        .def("insert", &QProg::operator<<<QWhileProg>,
            py::return_value_policy::reference)
        .def("insert", &QProg::operator<<<QMeasure>,
            py::return_value_policy::reference)
		.def("insert", &QProg::operator<<<QReset>,
			py::return_value_policy::reference)
        .def("insert", &QProg::operator<<<ClassicalCondition>,
            py::return_value_policy::reference)
        .def("begin",&QProg::getFirstNodeIter,
        py::return_value_policy::reference)
        .def("end",&QProg::getEndNodeIter,
        py::return_value_policy::reference)
        .def("last",&QProg::getLastNodeIter,
        py::return_value_policy::reference)
		.def("__repr__", [](QProg& p) {return draw_qprog(p); },
		py::return_value_policy::reference)
        .def("head",&QProg::getHeadNodeIter,
        py::return_value_policy::reference);


    py::implicitly_convertible<QGate, QProg>();
    py::implicitly_convertible<QCircuit, QProg>();
    py::implicitly_convertible<QIfProg, QProg>();
    py::implicitly_convertible<QWhileProg, QProg>();
    py::implicitly_convertible<QMeasure, QProg>();
	py::implicitly_convertible<QReset, QProg>();
    py::implicitly_convertible<ClassicalCondition, QProg>();


    py::class_<QCircuit>(m, "QCircuit")
        .def(py::init<>())
        .def(py::init([](NodeIter & iter) {
        if (!(*iter))
        {
            QCERR("iter is null");
            throw runtime_error("iter is null");
        }

        if (CIRCUIT_NODE == (*iter)->getNodeType())
        {
            auto gate_node = std::dynamic_pointer_cast<AbstractQuantumCircuit>(*iter);
            return QCircuit(gate_node);
        }
        else
        {
            QCERR("node type error");
            throw runtime_error("node type error");
        }
    }
        ))
        .def("insert", &QCircuit::operator<< <QCircuit>,
            py::return_value_policy::reference)
        .def("insert", &QCircuit::operator<< <QGate>,
            py::return_value_policy::reference)
        .def("dagger", &QCircuit::dagger,
            py::return_value_policy::automatic)
        .def("control", &QCircuit::control,
            py::return_value_policy::automatic)
        .def("set_dagger", &QCircuit::setDagger)
        .def("set_control", &QCircuit::setControl)
        .def("begin",&QCircuit::getFirstNodeIter,
            py::return_value_policy::reference)
        .def("end",&QCircuit::getEndNodeIter,
            py::return_value_policy::reference)
        .def("last",&QCircuit::getLastNodeIter,
            py::return_value_policy::reference)
        .def("head",&QCircuit::getHeadNodeIter,
            py::return_value_policy::reference)
		.def("__repr__", [](QCircuit& p) {return draw_qprog(p); },
			py::return_value_policy::reference);


    /* hide */
    py::class_<HadamardQCircuit, QCircuit>(m, "hadamard_circuit")
        .def(py::init<QVec&>());

    py::class_<QGate>(m, "QGate")
        .def(py::init([](NodeIter & iter) {
        if (!(*iter))
        {
            QCERR("iter is null");
            throw runtime_error("iter is null");
        }

        if (GATE_NODE == (*iter)->getNodeType())
        {
            auto gate_node = std::dynamic_pointer_cast<AbstractQGateNode>(*iter);
            return QGate(gate_node);
        }
        else
        {
            QCERR("node type error");
            throw runtime_error("node type error");
        }
    }
        ))
        .def("dagger", &QGate::dagger)
        .def("control", &QGate::control)
        .def("is_dagger", &QGate::isDagger)
        .def("set_dagger", &QGate::setDagger)
        .def("set_control", &QGate::setControl)
        .def("get_target_qubit_num", &QGate::getTargetQubitNum)
        .def("get_control_qubit_num", &QGate::getControlQubitNum)
        .def("get_qubits",&QGate::getQuBitVector,py::return_value_policy::automatic)
        .def("get_control_qubits", &QGate::getControlVector,py::return_value_policy::automatic)
        .def("gate_type", [](QGate & qgate) {
        return qgate.getQGate()->getGateType();
    })
        .def("gate_matrix", [](QGate & qgate) {
        QStat matrix;
        qgate.getQGate()->getMatrix(matrix);
        return matrix;
    }, py::return_value_policy::automatic);


    py::class_<QIfProg>(m, "QIfProg")
        .def(py::init([](NodeIter & iter) {
            if (!(*iter))
            {
                QCERR("iter is null");
                throw runtime_error("iter is null");
            }

            if (QIF_START_NODE == (*iter)->getNodeType())
            {
                auto gate_node = std::dynamic_pointer_cast<AbstractControlFlowNode>(*iter);
                return QIfProg(gate_node);
            }
            else
            {
                QCERR("node type error");
                throw runtime_error("node type error");
            }
    }
        ))
        .def(py::init<ClassicalCondition &, QProg>())
        .def(py::init<ClassicalCondition &, QProg, QProg>())
        .def("get_true_branch", [](QIfProg & self) {
        auto true_branch = self.getTrueBranch();
        if (!true_branch)
        {
            QCERR("true branch is null");
            throw runtime_error("true branch is null");
        }

        auto type = true_branch->getNodeType();
        if (PROG_NODE != type)
        {
            QCERR("true branch node type error");
            throw runtime_error("true branch node type error");
        }

        return QProg(true_branch);
    },
            py::return_value_policy::automatic)
        .def("get_false_branch", [](QIfProg & self) {
        auto false_branch = self.getFalseBranch();
        if (!false_branch)
        {
            return QProg();
        }

        auto type = false_branch->getNodeType();
        if (PROG_NODE != type)
        {
            QCERR("false branch node type error");
            throw runtime_error("true branch node type error");
        }

        return QProg(false_branch);
    },
            py::return_value_policy::automatic)
        .def("get_classical_condition", &QIfProg::getClassicalCondition,
             py::return_value_policy::automatic);


    py::class_<QWhileProg>(m, "QWhileProg")
        .def(py::init([](NodeIter & iter) {
        if (!(*iter))
        {
            QCERR("iter is null");
            throw runtime_error("iter is null");
        }

        if (WHILE_START_NODE == (*iter)->getNodeType())
        {
            auto gate_node = std::dynamic_pointer_cast<AbstractControlFlowNode>(*iter);
            return QWhileProg(gate_node);
        }
        else
        {
            QCERR("node type error");
            throw runtime_error("node type error");
        }
    }
        ))
        .def(py::init<ClassicalCondition, QProg>())
        .def("get_true_branch", [](QWhileProg & self) {
        auto true_branch = self.getTrueBranch();
        if (!true_branch)
        {
            QCERR("true branch is null");
            throw runtime_error("true branch is null");
        }

        auto type = true_branch->getNodeType();
        if (PROG_NODE != type)
        {
            QCERR("true branch node type error");
            throw runtime_error("true branch node type error");
        }

        return QProg(true_branch);
    },
             py::return_value_policy::automatic)
        .def("get_classical_condition", &QWhileProg::getClassicalCondition,
             py::return_value_policy::automatic);;

    py::class_<QMeasure>(m, "QMeasure")
        .def(py::init([](NodeIter & iter) {
        if (!(*iter))
        {
            QCERR("iter is null");
            throw runtime_error("iter is null");
        }

        if (MEASURE_GATE == (*iter)->getNodeType())
        {
            auto gate_node = std::dynamic_pointer_cast<AbstractQuantumMeasure>(*iter);
            return QMeasure(gate_node);
        }
        else
        {
            QCERR("node type error");
            throw runtime_error("node type error");
        }
    }));

	py::class_<QReset>(m, "QReset")
		.def(py::init([](NodeIter & iter) {
		if (!(*iter))
		{
			QCERR("iter is null");
			throw runtime_error("iter is null");
		}

		if (RESET_NODE == (*iter)->getNodeType())
		{
			auto gate_node = std::dynamic_pointer_cast<AbstractQuantumReset>(*iter);
			return QReset(gate_node);
		}
		else
		{
			QCERR("node type error");
			throw runtime_error("node type error");
		}
	}));

    py::class_<Qubit>(m, "Qubit")
        .def("getPhysicalQubitPtr", &Qubit::getPhysicalQubitPtr, py::return_value_policy::reference)
		.def("get_phy_addr", &Qubit::get_phy_addr, py::return_value_policy::reference)
        ;

    py::class_<PhysicalQubit>(m, "PhysicalQubit")
        .def("getQubitAddr", &PhysicalQubit::getQubitAddr, py::return_value_policy::reference)
        ;

    py::class_<CBit>(m, "CBit")
        .def("getName", &CBit::getName);

    py::class_<ClassicalCondition>(m, "ClassicalCondition")
        .def("get_val", &ClassicalCondition::get_val,"get value")
        .def("set_val",&ClassicalCondition::set_val,"set value")
        BIND_CLASSICALCOND_OPERATOR_OVERLOAD(<)
        BIND_CLASSICALCOND_OPERATOR_OVERLOAD(<=)
        BIND_CLASSICALCOND_OPERATOR_OVERLOAD(>)
        BIND_CLASSICALCOND_OPERATOR_OVERLOAD(>=)
        BIND_CLASSICALCOND_OPERATOR_OVERLOAD(+)
        BIND_CLASSICALCOND_OPERATOR_OVERLOAD(-)
        BIND_CLASSICALCOND_OPERATOR_OVERLOAD(*)
        BIND_CLASSICALCOND_OPERATOR_OVERLOAD(/ )
        BIND_CLASSICALCOND_OPERATOR_OVERLOAD(== )
        ;

    m.def("add", [](ClassicalCondition a, ClassicalCondition b)
    {return a + b; });

    m.def("add", [](ClassicalCondition a, cbit_size_t b)
    {return a + b; });

    m.def("add", [](cbit_size_t a, ClassicalCondition b)
    {return a + b; });

    m.def("sub", [](ClassicalCondition a, ClassicalCondition b)
    {return a - b; });

    m.def("sub", [](ClassicalCondition a, cbit_size_t b)
    {return a - b; });

    m.def("sub", [](cbit_size_t a, ClassicalCondition b)
    {return a - b; });

    m.def("mul", [](ClassicalCondition a, ClassicalCondition b)
    {return a * b; });

    m.def("mul", [](ClassicalCondition a, cbit_size_t b)
    {return a * b; });

    m.def("mul", [](cbit_size_t a, ClassicalCondition b)
    {return a * b; });

    m.def("div", [](ClassicalCondition a, ClassicalCondition b)
    {return a / b; });

    m.def("div", [](ClassicalCondition a, cbit_size_t b)
    {return a / b; });

    m.def("div", [](cbit_size_t a, ClassicalCondition b)
    {return a / b; });

    m.def("equal", [](ClassicalCondition a, ClassicalCondition b)
    {return a == b; });

    m.def("equal", [](ClassicalCondition a, cbit_size_t b)
    {return a == b; });

    m.def("equal", [](cbit_size_t a, ClassicalCondition b)
    {return a == b; });

    m.def("assign", [](ClassicalCondition a, ClassicalCondition b)
    {return a = b; });

    m.def("assign", [](ClassicalCondition a, cbit_size_t b)
    {return a = b; });


    py::enum_<QError>(m, "QError")
        .value("UndefineError", QError::undefineError)
        .value("qErrorNone", QError::qErrorNone)
        .value("qParameterError", QError::qParameterError)
        .value("qubitError", QError::qubitError)
        .value("loadFileError", QError::loadFileError)
        .value("initStateError", QError::initStateError)
        .value("destroyStateError", QError::destroyStateError)
        .value("setComputeUnitError", QError::setComputeUnitError)
        .value("runProgramError", QError::runProgramError)
        .value("getResultError", QError::getResultError)
        .value("getQStateError", QError::getQStateError)
        ;

    py::class_<OriginCollection>(m, "OriginCollection")
        .def(py::init<>())
        .def(py::init<std::string>())
        .def(py::init<OriginCollection>())
        .def("setNames", [](OriginCollection& c, py::args args) {
            std::vector<std::string> all_key;
            for (auto arg : args) { all_key.push_back(std::string(py::str(arg))); }
            c = all_key;
        })
        .def("insertValue", [](OriginCollection& c, std::string key, py::args args) {
            int i = 1;
            auto vector = c.getKeyVector();
            c.addValue(vector[0], key);
            for (auto arg : args) {
                c.addValue(vector[i],std::string(py::str (arg)));
                i++;
                }
        })
        .def("insertValue", [](OriginCollection& c, int key, py::args args) {
        int i = 1;
        auto vector = c.getKeyVector();
        c.addValue(vector[0], key);
        for (auto arg : args) {
            c.addValue(vector[i], std::string(py::str(arg)));
            i++;
        }
    })
        .def("getValue",&OriginCollection::getValue,"Get value by Key name")
        .def("getValueByKey", [](OriginCollection & c, std::string key_value) {
        return c.getValueByKey(key_value);
    }, "Get Value by key value")
        .def("getValueByKey", [](OriginCollection & c, int key_value) {
        return c.getValueByKey(key_value);
    }, "Get Value by key value")
        .def("open", &OriginCollection::open, "Open json file")
        .def("write", &OriginCollection::write, "write json file")
        .def("getJsonString", &OriginCollection::getJsonString, "Get Json String")
        .def("getFilePath", &OriginCollection::getFilePath, "Get file path")
        .def("getKeyVector", &OriginCollection::getKeyVector, "Get key vector");

    py::class_<QResult>(m, "QResult")
        .def("getResultMap", &QResult::getResultMap, py::return_value_policy::reference);

    m.def("vector_dot", &vector_dot,"x"_a,"y"_a, "Inner product of vector x and y");
    m.def("all_cut_of_graph", &all_cut_of_graph, "generate graph of maxcut problem");

    m.def("vector_dot", &vector_dot, "Inner product of vector x and y");
    m.def("all_cut_of_graph", &all_cut_of_graph, "generate graph of maxcut problem");
    //combine pyQPandaVariational and pyQPanda



    py::enum_<QPanda::OptimizerType>(m, "OptimizerType", py::arithmetic())
        .value("NELDER_MEAD", QPanda::OptimizerType::NELDER_MEAD)
        .value("POWELL", QPanda::OptimizerType::POWELL)
        .value("GRADIENT", QPanda::OptimizerType::GRADIENT)
        .export_values();

    py::class_<QPanda::AbstractOptimizer>(m, "AbstractOptimizer")
        .def("registerFunc", &QPanda::AbstractOptimizer::registerFunc)
        .def("setXatol", &QPanda::AbstractOptimizer::setXatol)
        .def("exec", &QPanda::AbstractOptimizer::exec)
        .def("getResult", &QPanda::AbstractOptimizer::getResult)
        .def("setAdaptive", &QPanda::AbstractOptimizer::setAdaptive)
        .def("setDisp", &QPanda::AbstractOptimizer::setDisp)
        .def("setFatol", &QPanda::AbstractOptimizer::setFatol)
        .def("setMaxFCalls", &QPanda::AbstractOptimizer::setMaxFCalls)
        .def("setMaxIter", &QPanda::AbstractOptimizer::setMaxIter);

    py::class_<QPanda::OptimizerFactory>(m, "OptimizerFactory")
        .def(py::init<>())
        .def("makeOptimizer", py::overload_cast<const QPanda::OptimizerType &>
            (&QPanda::OptimizerFactory::makeOptimizer), "Please input OptimizerType ")
        .def("makeOptimizer", py::overload_cast<const std::string &>
            (&QPanda::OptimizerFactory::makeOptimizer), "Please input the Optimizer's name(string)")
        ;

    py::class_<QPanda::QOptimizationResult>(m, "QOptimizationResult")
        .def(py::init<std::string &, size_t &, size_t &, std::string &, double &, vector_d &>())
        .def_readwrite("message",&QPanda::QOptimizationResult::message)
        .def_readwrite("fcalls", &QPanda::QOptimizationResult::fcalls)
        .def_readwrite("fun_val", &QPanda::QOptimizationResult::fun_val)
        .def_readwrite("iters", &QPanda::QOptimizationResult::iters)
        .def_readwrite("key", &QPanda::QOptimizationResult::key)
        .def_readwrite("para", &QPanda::QOptimizationResult::para);

    py::enum_<NodeType>(m, "NodeType")
        .value("NODE_UNDEFINED", NodeType::NODE_UNDEFINED)
        .value("GATE_NODE", NodeType::GATE_NODE)
        .value("CIRCUIT_NODE", NodeType::CIRCUIT_NODE)
        .value("PROG_NODE", NodeType::PROG_NODE)
        .value("MEASURE_GATE", NodeType::MEASURE_GATE)
        .value("WHILE_START_NODE", NodeType::WHILE_START_NODE)
        .value("QIF_START_NODE", NodeType::QIF_START_NODE)
        .value("CLASS_COND_NODE", NodeType::CLASS_COND_NODE)
		.value("RESET_NODE", NodeType::RESET_NODE);

    py::class_<NodeIter>(m, "NodeIter")
        .def(py::init<>())
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("get_next", &NodeIter::getNextIter, py::return_value_policy::automatic)
        .def("get_pre", [](NodeIter & iter) {
        return --iter;
        }, py::return_value_policy::reference)
        .def("get_node_type", [](NodeIter & iter) {
            auto node_type = (*iter)->getNodeType();
            return node_type;
        },
                py::return_value_policy::automatic);

    py::implicitly_convertible<ClassicalCondition, ClassicalProg>();
    py::implicitly_convertible<cbit_size_t, ClassicalCondition>();


    m.def("get_matrix", [](QProg prog, const NodeIter nodeItrStart, const NodeIter nodeItrEnd) {
        return getCircuitMatrix(prog, nodeItrStart, nodeItrEnd);
    }, py::arg("prog"), py::arg("nodeItrStart") = NodeIter(), py::arg("nodeItrEnd") = NodeIter()
        , "get the target prog  matrix",
        py::return_value_policy::automatic
        );

	py::class_<LayerNodeInfo>(m, "LayerNodeInfo")
		.def(py::init<>([](const NodeIter iter, QVec target_qubits, QVec control_qubits,
			GateType type, const bool dagger){
		return LayerNodeInfo(iter, target_qubits, control_qubits, type, dagger);}))
		.def_readwrite("m_iter", &LayerNodeInfo::m_iter)
		.def_readwrite("m_target_qubits", &LayerNodeInfo::m_target_qubits)
		.def_readwrite("m_ctrl_qubits", &LayerNodeInfo::m_ctrl_qubits)
		.def_readwrite("m_cbits", &LayerNodeInfo::m_cbits)
		.def_readwrite("m_params", &LayerNodeInfo::m_params)
		.def_readwrite("m_name", &LayerNodeInfo::m_name)
		.def_readwrite("m_type", &LayerNodeInfo::m_type)
		.def_readwrite("m_dagger", &LayerNodeInfo::m_dagger);

	m.def("circuit_layer", [](QProg prg) {
		py::list ret_data;
		auto layer_info = prog_layer(prg);
		std::vector<std::vector<LayerNodeInfo>> tmp_layer(layer_info.size());
		size_t layer_index = 0;
		for (auto& cur_layer : layer_info)
		{
			for (auto& node_item : cur_layer)
			{
				//single gate first
				if ((node_item.first->m_ctrl_qubits.size() == 0) && (node_item.first->m_target_qubits.size() == 1))
				{
					tmp_layer[layer_index].insert(tmp_layer[layer_index].begin(), LayerNodeInfo(node_item.first));
				}
				else
				{
					tmp_layer[layer_index].push_back(LayerNodeInfo(node_item.first));
				}
			}

			++layer_index;
		}
		ret_data.append(tmp_layer);

		std::vector<int> vec_qubits_in_use;
		get_all_used_qubits(prg, vec_qubits_in_use);
		ret_data.append(vec_qubits_in_use);

		std::vector<int> vec_cbits_in_use;
		get_all_used_class_bits(prg, vec_cbits_in_use);
		ret_data.append(vec_cbits_in_use);

		return ret_data;
	}, py::arg("prog"),
		"quantum circuit layering",
		py::return_value_policy::automatic
		);

    m.def("draw_qprog", [](QProg prg, const NodeIter itr_start, const NodeIter itr_end) {
		auto text_pic_str = draw_qprog(prg, itr_start, itr_end);
		std::cout << text_pic_str << endl;
		//return text_pic_str;
    }, py::arg("prog"), py::arg("itr_start") = NodeIter(), py::arg("itr_end") = NodeIter(),
        "output a quantum prog/circuit to console by text-pic(UTF-8 code), \
        and will save the text-pic in file named QCircuitTextPic.txt in the same time in current path",
        py::return_value_policy::automatic
        );

	m.def("fit_to_gbk", &fit_to_gbk, py::arg("utf8_str"),
		"Adapting utf8 characters to GBK",
		py::return_value_policy::automatic
		);

	m.def("draw_qprog_with_clock", [](QProg prg, const NodeIter itr_start, const NodeIter itr_end) {
		auto text_pic_str = draw_qprog_with_clock(prg, itr_start, itr_end);
		std::cout << text_pic_str << endl;
		//return text_pic_str;
	}, py::arg("prog"), py::arg("itr_start") = NodeIter(), py::arg("itr_end") = NodeIter(),
		"output a quantum prog/circuit to console by text-pic(UTF-8 code) with time sequence, \
        and will save the text-pic in file named QCircuitTextPic.txt in the same time in current path",
		py::return_value_policy::automatic
		);

	m.def("fill_qprog_by_I", [](QProg &prg) {
		return fill_qprog_by_I(prg);
	}, py::arg("prog"),
		"Fill the input QProg by I gate, get a new quantum program",
		py::return_value_policy::automatic
		);

	m.def("matrix_decompose", [](QVec& qubits, QStat& src_mat) {
		return matrix_decompose(qubits, src_mat);
	}, "qubits"_a, "matrix"_a,
		"Decomposition of quantum gates by Chi Kwong Li and Diane Christine Pelejo",
		py::return_value_policy::automatic
		);

#define QUERY_REPLACE(GRAPH_NODE,QUERY_NODE,REPLACE_NODE) \
    m.def("graph_query_replace", [](GRAPH_NODE &graph_node, QUERY_NODE &query_node,\
                                       REPLACE_NODE &replace_node, QuantumMachine *qvm)\
    {\
        QProg prog;\
        graph_query_replace(graph_node, query_node, replace_node, prog, qvm); \
        return prog;\
    },py::return_value_policy::automatic);

	m.def("quantum_chip_adapter", [](QProg prog, QuantumMachine *quantum_machine, bool b_mapping = true, 
		const std::string config_data = CONFIG_PATH) {
		py::list ret_data;

		QVec new_qvec;
		quantum_chip_adapter(prog, quantum_machine, new_qvec, b_mapping, config_data);
		if (!b_mapping)
		{
			get_all_used_qubits(prog, new_qvec);
		}

		ret_data.append(prog);
		ret_data.append(new_qvec);
		return ret_data;
	}, "prog"_a, "quantum machine"_a, "b_mapping"_a=true, "config data"_a = CONFIG_PATH,
		"/**\
		* @brief  Quantum chip adaptive conversion\
		* @ingroup Utilities\
		* @param[in]  QProg Quantum Program\
		* @param[in]  QuantumMachine*  quantum machine pointer\
        * @param[out] QVec& Quantum bits after mapping.\
                      Note: if b_mapping is false, the input QVec will be misoperated.\
		* @param[in]  bool whether or not perform the mapping operation.\
		* @param[in] const std::string It can be configuration file or configuration data, which can be distinguished by file suffix,\
		            so the configuration file must be end with \".json\", default is CONFIG_PATH\
		* @return The new quantum program and the mapped qubit vector\
		* / ",
		py::return_value_policy::automatic
		);

	m.def("decompose_multiple_control_qgate", [](QProg prog, QuantumMachine *quantum_machine,
		const std::string config_data = CONFIG_PATH) {
		decompose_multiple_control_qgate(prog, quantum_machine, config_data);
		return prog;
	}, "prog"_a, "quantum machine"_a, "config data"_a = CONFIG_PATH,
		"/**\
		* @brief Decompose multiple control QGate\
		* @ingroup Utilities\
		* @param[in]  QProg&   Quantum Program\
		* @param[in]  QuantumMachine*  quantum machine pointer\
		* @param[in] const std::string It can be configuration file or configuration data, which can be distinguished by file suffix,\
		             so the configuration file must be end with \".json\", default is CONFIG_PATH\
		* @return new Qprog\
		* / ",
		py::return_value_policy::automatic
		);

	m.def("get_all_used_qubits", [](QProg prog) {
		QVec vec_qubits_in_use;
		get_all_used_qubits(prog, vec_qubits_in_use);
		return vec_qubits_in_use;
	}, "qprog"_a,
		"Get all the used  quantum bits in the input prog",
		py::return_value_policy::automatic
		);

	m.def("get_all_used_qubits_to_int", [](QProg prog) {
		std::vector<int> vec_qubits_in_use;
		get_all_used_qubits(prog, vec_qubits_in_use);
		return vec_qubits_in_use;
	}, "qprog"_a,
		"Get all the used  quantum bits in the input prog, return all the index of qubits",
		py::return_value_policy::automatic
		);
    QUERY_REPLACE(QProg, QCircuit, QCircuit)
    QUERY_REPLACE(QProg, QCircuit, QGate)
    QUERY_REPLACE(QProg, QGate, QCircuit)
    QUERY_REPLACE(QProg, QGate, QGate)

    QUERY_REPLACE(QCircuit, QCircuit, QCircuit)
    QUERY_REPLACE(QCircuit, QCircuit, QGate)
    QUERY_REPLACE(QCircuit, QGate, QCircuit)
    QUERY_REPLACE(QCircuit, QGate, QGate)

    QUERY_REPLACE(QGate, QCircuit, QCircuit)
    QUERY_REPLACE(QGate, QCircuit, QGate)
    QUERY_REPLACE(QGate, QGate, QCircuit)
    QUERY_REPLACE(QGate, QGate, QGate);
}

