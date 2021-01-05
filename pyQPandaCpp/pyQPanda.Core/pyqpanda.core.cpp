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
void init_components(py::module&);
void init_core_class(py::module &);

PYBIND11_MODULE(pyQPanda, m)
{
    init_quantum_machine(m);
    init_core_class(m);
    init_variational(m);
    init_qalg(m);
    init_components(m);

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

    m.def("BARRIER", [](Qubit* qubit)
    {return BARRIER(qubit); },
        "qubit"_a,
        "Create an BARRIER gate",
        py::return_value_policy::automatic
    );

    m.def("BARRIER", [](QVec qubits)
    {return BARRIER(qubits); },
        "qubit list"_a,
        "Create an BARRIER gate",
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
        [](Qubit* first_qubit, Qubit* second_qubit)
    {return iSWAP(first_qubit, second_qubit); },
        "qubit"_a, "qubit"_a,
        "Create a iSWAP gate",
        py::return_value_policy::automatic
    );

    m.def("iSWAP",
        [](Qubit* first_qubit, Qubit* second_qubit, double theta)
    {return iSWAP(first_qubit, second_qubit, theta); },
        "qubit"_a, "qubit"_a, "angle"_a,
        "Create a iSWAP gate",
        py::return_value_policy::automatic
    );

    m.def("SqiSWAP",
        [](Qubit* first_qubit, Qubit* second_qubit)
    {return SqiSWAP(first_qubit, second_qubit); },
        "qubit"_a, "qubit"_a,
        "Create a SqiSWAP gate",
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



    m.def("print_matrix", [](QStat& mat, const int precision) {
        auto mat_str = matrix_to_string(mat, precision);
        std::cout << mat_str << endl;
        return mat_str;
    }, "mat"_a, "precision"_a = 8,
        "/**\
          * @brief  output matrix information to consol\
          * @ingroup Utilities\
          * @param[in] the target matrix\
          * @param[in] const int: precision, default is 8\
          */",
        py::return_value_policy::automatic
        );

    m.def("is_match_topology", &isMatchTopology, "gate"_a, "vecTopoSt"_a,
        "Whether the qgate matches the quantum topology",
        py::return_value_policy::automatic
    );

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

    m.def("to_Quil", &transformQProgToQuil, "program"_a, "quantum machine"_a, "QProg to Quil",
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
        , "program"_a, "quantum machine"_a, "QProg to Quil",
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

    m.def("get_qprog_clock_cycle", &get_qprog_clock_cycle,
        py::arg("prog"), py::arg("qm"), py::arg("optimize") = false, "Get Quantum Program Clock Cycle",
        py::return_value_policy::automatic_reference
    );

    m.def("transform_qprog_to_binary", [](QProg prog, QuantumMachine * qvm) {
        return transformQProgToBinary(prog, qvm);
    }
        , "program"_a, "quantum machine"_a "Get quantum program binary data",
        py::return_value_policy::automatic_reference
        );

    m.def("transform_qprog_to_binary", [](QProg prog, QuantumMachine * qvm, string file_path) {

        return transformQProgToBinary(prog, qvm, file_path);
    }
        , "program"_a, "quantum machine"_a, "file path"_a, "Get quantum program binary data",
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

    m.def("flatten", [](QProg &prog) {
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

	m.def("convert_originir_str_to_qprog", [](std::string originir_str, QuantumMachine* qvm) {
		py::list ret_data;
		QVec qv;
		std::vector<ClassicalCondition> cv;
		QProg prog = convert_originir_string_to_qprog(originir_str, qvm, qv, cv);
		py::list qubit_list;
		for (auto q : qv)
			qubit_list.append(q);

		ret_data.append(prog);
		ret_data.append(qubit_list);
		ret_data.append(cv);

		return ret_data;
	},
		"originir_str"_a, "QuantumMachine"_a, "convert OriginIR to QProg",
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


    m.def("convert_qasm_string_to_qprog", [](std::string qasm_str, QuantumMachine* qvm) {
        py::list ret_data;
        QVec qv;
        std::vector<ClassicalCondition> cv;
        QProg prog = convert_qasm_string_to_qprog(qasm_str, qvm, qv, cv);
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
        return convert_qprog_to_qasm(prog, qvm);
    }, "prog"_a, "quantum machine"_a, py::return_value_policy::automatic_reference
    );

    m.def("cast_qprog_qgate", &cast_qprog_qgate,
        "quantum program"_a, "cast QProg to QGate",
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
        , "quantum program"_a, "cast QProg to QCircuit",
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

    m.def("qcodar_match", [](QProg prog, QVec qv, QuantumMachine *qvm, QCodarGridDevice arch_type,
        size_t m, size_t n, size_t run_times, const std::string config_data) {
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
        "prog"_a, "qubits"_a, "quantum machine"_a, "QCodarGridDevice"_a = SIMPLE_TYPE, "m"_a = 2, "n"_a = 4, "run_times"_a = 5, "config_data"_a = CONFIG_PATH,
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


    m.def("vector_dot", &vector_dot, "x"_a, "y"_a, "Inner product of vector x and y");
    m.def("all_cut_of_graph", &all_cut_of_graph, "generate graph of maxcut problem");

    m.def("vector_dot", &vector_dot, "Inner product of vector x and y");
    m.def("all_cut_of_graph", &all_cut_of_graph, "generate graph of maxcut problem");

    m.def("get_matrix", [](QProg prog, const bool b_bid_endian, const NodeIter nodeItrStart, const NodeIter nodeItrEnd) {
        return getCircuitMatrix(prog, b_bid_endian, nodeItrStart, nodeItrEnd);
    }, py::arg("prog"), py::arg("b_bid_endian") = false, py::arg("nodeItrStart") = NodeIter(), py::arg("nodeItrEnd") = NodeIter()
        , "/**\
		* @brief  get the target matrix between the input two Nodeiters\
		* @ingroup Utilities\
		* @param[in] const bool Qubit order mark of output matrix,\
		true for positive sequence(Bid Endian), false for inverted order(Little Endian), default is false\
		* @param[in] nodeItrStart the start NodeIter\
		* @param[in] nodeItrEnd the end NodeIter\
		* @return the target matrix include all the QGate's matrix (multiply).\
		* @see\
		* / ",
        py::return_value_policy::automatic
        );

    m.def("circuit_layer", [](QProg prg) {
        py::list ret_data;
        auto layer_info = prog_layer(prg);
        std::vector<std::vector<NodeInfo>> tmp_layer(layer_info.size());
        size_t layer_index = 0;
        for (auto& cur_layer : layer_info)
        {
            for (auto& node_item : cur_layer)
            {
                const pOptimizerNodeInfo& n = node_item.first;
                //single gate first
                if ((node_item.first->m_control_qubits.size() == 0) && (node_item.first->m_target_qubits.size() == 1))
                {
                    tmp_layer[layer_index].insert(tmp_layer[layer_index].begin(),
                        NodeInfo(n->m_iter, n->m_target_qubits,
                            n->m_control_qubits, n->m_type,
                            n->m_is_dagger));
                }
                else
                {
                    tmp_layer[layer_index].push_back(NodeInfo(n->m_iter, n->m_target_qubits,
                        n->m_control_qubits, n->m_type,
                        n->m_is_dagger));
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

    m.def("draw_qprog_text", [](QProg prg, const NodeIter itr_start, const NodeIter itr_end) {
        return draw_qprog(prg, itr_start, itr_end);
    }, py::arg("prog"), py::arg("itr_start") = NodeIter(), py::arg("itr_end") = NodeIter(),
        "Convert a quantum prog/circuit to text-pic(UTF-8 code), \
        and will save the text-pic in file named QCircuitTextPic.txt in the same time in current path",
        py::return_value_policy::automatic
        );

    m.def("fit_to_gbk", &fit_to_gbk, py::arg("utf8_str"),
        "Adapting utf8 characters to GBK",
        py::return_value_policy::automatic
    );

    m.def("draw_qprog_text_with_clock", [](QProg prog, const std::string config_data, const NodeIter itr_start, const NodeIter itr_end) {
        return draw_qprog_with_clock(prog, config_data, itr_start, itr_end);
    }, py::arg("prog"), "config_data"_a = CONFIG_PATH, py::arg("itr_start") = NodeIter(), py::arg("itr_end") = NodeIter(),
        "Convert a quantum prog/circuit to text-pic(UTF-8 code) with time sequence, \
        and will save the text-pic in file named QCircuitTextPic.txt in the same time in current path",
        py::return_value_policy::automatic
        );

    m.def("fill_qprog_by_I", [](QProg &prg) {
        return fill_qprog_by_I(prg);
    }, py::arg("prog"),
        "Fill the input QProg by I gate, get a new quantum program",
        py::return_value_policy::automatic
        );

    m.def("matrix_decompose", [](QVec& qubits, QStat& src_mat, const DecompositionMode mode = DecompositionMode::QR) {
        return matrix_decompose(qubits, src_mat, mode);
    }, "qubits"_a, "matrix"_a, "mode"_a = DecompositionMode::QR,
        "/**\
		* @brief  matrix decomposition\
		* @ingroup Utilities\
		* @param[in]  QVec& the used qubits\
		* @param[in]  QStat& The target matrix\
		* @param[in]  DecompositionMode decomposition mode, default is QR\
		* @return    QCircuit The quantum circuit for target matrix\
		* @see DecompositionMode\
		* / ",
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
    }, "prog"_a, "quantum_machine"_a, "b_mapping"_a = true, "config_data"_a = CONFIG_PATH,
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
    }, "prog"_a, "quantum_machine"_a, "config_data"_a = CONFIG_PATH,
        "/**\
		* @brief Decompose multiple control QGate\
		* @param[in]  QProg&   Quantum Program\
		* @param[in]  QuantumMachine*  quantum machine pointer\
		* @param[in] const std::string It can be configuration file or configuration data, which can be distinguished by file suffix,\
		             so the configuration file must be end with \".json\", default is CONFIG_PATH\
		* @return new Qprog\
		* / ",
        py::return_value_policy::automatic
        );

    m.def("transform_to_base_qgate", [](QProg prog, QuantumMachine *quantum_machine,
        const std::string config_data = CONFIG_PATH) {
        transform_to_base_qgate(prog, quantum_machine, config_data);
        return prog;
    }, "prog"_a, "quantum_machine"_a, "config_data"_a = CONFIG_PATH,
        "/**\
		* @brief Basic quantum - gate conversion\
		* @param[in]  QProg&   Quantum Program\
		* @param[in]  QuantumMachine*  quantum machine pointer\
		* @param[in] const std::string& It can be configuration file or configuration data, which can be distinguished by file suffix,\
		             so the configuration file must be end with \".json\", default is CONFIG_PATH\
		* @return\
		* @note Quantum circuits or programs cannot contain multiple-control gates.\
		* / ",
        py::return_value_policy::automatic
        );

    m.def("circuit_optimizer", [](QProg prog, const std::vector<std::pair<QCircuit, QCircuit>>& optimizer_cir_vec,
        const std::vector<QCircuitOPtimizerMode>& mode_list = std::vector<QCircuitOPtimizerMode>(0)) {
		int mode = 0;
		for (const auto& m : mode_list){
			mode |= m;
		}
        sub_cir_optimizer(prog, optimizer_cir_vec, mode);
        return prog;
    }, "prog"_a, "optimizer_cir_vec"_a = std::vector<std::pair<QCircuit, QCircuit>>(), "mode_list"_a = std::vector<QCircuitOPtimizerMode>(0),
        "/**\
		* @brief QCircuit optimizer\
		* @ingroup Utilities\
		* @param[in, out]  QProg(or QCircuit) the source prog(or circuit)\
		* @param[in] std::vector<std::pair<QCircuit, QCircuit>>\
		* @param[in] const std::vector<QCircuitOPtimizerMode> Optimise mode(see QCircuitOPtimizerMode), Support several models :\
	                 Merge_H_X: Optimizing continuous H or X gate\
		             Merge_U3 : merge continues single gates to a U3 gate\
		             Merge_RX : merge continues RX gates\
		             Merge_RY : merge continues RY gates\
		             Merge_RZ : merge continues RZ gates\
		* @return  new prog\
		* / ",
        py::return_value_policy::automatic
        );

    m.def("circuit_optimizer_by_config", [](QProg prog, const std::string config_data,
		const std::vector<QCircuitOPtimizerMode>& mode_list) {
		int mode = 0;
		for (const auto& m : mode_list) {
			mode |= m;
		}
        cir_optimizer_by_config(prog, config_data, mode);
        return prog;
    }, "prog"_a, "config_data"_a = CONFIG_PATH, "mode"_a = std::vector<QCircuitOPtimizerMode>(0),
        "/**\
		* @brief QCircuit optimizer\
		* @ingroup Utilities\
		* @param[in, out]  QProg(or QCircuit) the source prog(or circuit)\
		* @param[in] const std::string It can be configuration file or configuration data, which can be distinguished by file suffix,\
		             so the configuration file must be end with \".json\", default is CONFIG_PATH\
		* @param[in] const std::vector<QCircuitOPtimizerMode> Optimise mode(see QCircuitOPtimizerMode), Support several models :\
	                 Merge_H_X: Optimizing continuous H or X gate\
		             Merge_U3 : merge continues single gates to a U3 gate\
		             Merge_RX : merge continues RX gates\
		             Merge_RY : merge continues RY gates\
		             Merge_RZ : merge continues RZ gates\
		* @return  new prog\
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

    m.def("state_fidelity", [](const QStat &state1, const QStat &state2)
    {
        return state_fidelity(state1, state2);
    }, "state1"_a, "state2"_a,
        "Get state fidelity",
        py::return_value_policy::automatic
        );

    m.def("state_fidelity", [](const std::vector<QStat> &matrix1, const std::vector<QStat> &matrix2)
    {
        return state_fidelity(matrix1, matrix2);
    }, "state1"_a, "state2"_a,
        "Get state fidelity",
        py::return_value_policy::automatic
        );

    m.def("state_fidelity", [](const QStat &state, const vector<QStat> &matrix)
    {
        return state_fidelity(state, matrix);
    }, "state1"_a, "state2"_a,
        "Get state fidelity",
        py::return_value_policy::automatic
        );

    m.def("state_fidelity", [](const vector<QStat> &matrix, const QStat &state)
    {
        return state_fidelity(matrix, state);
    }, "state1"_a, "state2"_a,
        "Get state fidelity",
        py::return_value_policy::automatic
        );

	m.def("get_circuit_optimal_topology", [](QProg& prog, QuantumMachine* quantum_machine, 
		const size_t max_connect_degree, const std::string config_data) {
		return get_circuit_optimal_topology(prog, quantum_machine, max_connect_degree, config_data);
	}, "prog"_a, "quantum_machine"_a, "max_connect_degree"_a, "config_data"_a = CONFIG_PATH,
		"Get the optimal topology of the input circuit",
		py::return_value_policy::automatic
		);

    m.def("get_double_gate_block_topology", [](QProg& prog) {
        return get_double_gate_block_topology(prog);
    }, "prog"_a,
        "get double gate block topology",
        py::return_value_policy::automatic
        );

    m.def("del_weak_edge", [](TopologyData& topo_data) {
        del_weak_edge(topo_data);
        return topo_data;
    }, "matrix"_a,
        "Delete weakly connected edges",
        py::return_value_policy::automatic
        );

    m.def("del_weak_edge2", [](TopologyData& topo_data, const size_t max_connect_degree, std::vector<int>& sub_graph_set) {
        py::list ret_data;

        std::vector<weight_edge> candidate_edges;
        std::vector<int> intermediary_points = del_weak_edge(topo_data, max_connect_degree, sub_graph_set, candidate_edges);

        ret_data.append(topo_data);
        ret_data.append(intermediary_points);
        ret_data.append(candidate_edges);
        return ret_data;
    }, "matrix"_a, "int"_a, "list"_a,
        "Delete weakly connected edges",
        py::return_value_policy::automatic
        );

    m.def("del_weak_edge3", [](TopologyData& topo_data, std::vector<int>& sub_graph_set, const size_t max_connect_degree,
        const double lamda1, const double lamda2, const double lamda3) {
        py::list ret_data;

        std::vector<int> intermediary_points = del_weak_edge(topo_data, sub_graph_set, max_connect_degree, lamda1, lamda2, lamda3);

        ret_data.append(topo_data);
        ret_data.append(intermediary_points);
        return ret_data;
    }, "matrix"_a, "list"_a, "int"_a, "data"_a, "data"_a, "data"_a,
        "Delete weakly connected edges",
        py::return_value_policy::automatic
        );

    m.def("recover_edges", [](TopologyData& topo_data, const size_t max_connect_degree, std::vector<weight_edge>& candidate_edges) {
        recover_edges(topo_data, max_connect_degree, candidate_edges);
        return topo_data;
    }, "matrix"_a, "int"_a, "list"_a,
        "/**\
		* @brief Recover edges from the candidate edges\
		* @ingroup Utilities\
		* @param[in] const TopologyData& the target graph\
		* @param[in] const size_t The max connect - degree\
		* @param[in] std::vector<weight_edge>& Thecandidate edges\
		* @return\
		* / ",
        py::return_value_policy::automatic
        );

    m.def("get_complex_points", [](TopologyData& topo_data, const size_t max_connect_degree) {
        return get_complex_points(topo_data, max_connect_degree);
    }, "matrix"_a, "int"_a,
        "Get complex points",
        py::return_value_policy::automatic
        );

    m.def("split_complex_points", [](std::vector<int>& complex_points, const size_t max_connect_degree,
        const TopologyData& topo_data, int split_method = 0) {
        return split_complex_points(complex_points, max_connect_degree, topo_data, static_cast<ComplexVertexSplitMethod>(split_method));
    }, "data"_a, "int"_a, "matrix"_a, "int"_a = 0,
        "Splitting complex points into multiple points",
        py::return_value_policy::automatic
        );

    m.def("replace_complex_points", [](TopologyData& src_topo_data, const size_t max_connect_degree,
        const std::vector<std::pair<int, TopologyData>>& sub_topo_vec) {
        replace_complex_points(src_topo_data, max_connect_degree, sub_topo_vec);
        return src_topo_data;
    }, "matrix"_a, "int"_a, "data"_a,
        "Replacing complex points with subgraphs",
        py::return_value_policy::automatic
        );

    m.def("get_sub_graph", [](TopologyData& topo_data) {
        return get_sub_graph(topo_data);
    }, "matrix"_a,
        "Get sub graph",
        py::return_value_policy::automatic
        );

    m.def("estimate_topology", [](const TopologyData& topo_data) {
        return estimate_topology(topo_data);
    }, "matrix"_a,
        "Evaluate topology performance",
        py::return_value_policy::automatic
        );

    m.def("planarity_testing", [](const TopologyData& topo_data) {
        return planarity_testing(topo_data);
    }, "matrix"_a,
        "/**\
		* @brief  planarity testing\
		* @ingroup Utilities\
		* @param[in] const TopologyData& the target graph\
		* @return bool If the input graph is planarity, return true, otherwise retuen false.\
		* / ",
        py::return_value_policy::automatic
        );
    /* =============================test end =============================*/

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

