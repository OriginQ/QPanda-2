#include <math.h>
#include <map>
#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/chrono.h"
#include "Core/QPanda.h"
#include "Components/MaxCutProblemGenerator/MaxCutProblemGenerator.h"
#include "pybind11/eigen.h"
#include "pybind11/operators.h"
#include <pybind11/stl_bind.h>
#include "Core/Utilities/OriginCollection.h"
#include "Core/Utilities/Utilities.h"
#include "Components/DataStruct.h"
#include "Components/Optimizer/AbstractOptimizer.h"
#include "Components/Optimizer/OptimizerFactory.h"
#include "Components/Optimizer/OriginNelderMead.h"
#include "Core/QuantumMachine/QCloudMachine.h"
#include "Core/Utilities/Transform/QProgClockCycle.h"
#include "Components/Operator/FermionOperator.h"
#include "QPandaConfig.h"
#include "Core/Utilities/QCircuitInfo.h"
#include "Core/Utilities/QProgToDAG/GraphMatch.h"
#include "Core/Utilities/Visualization/QVisualization.h"
#include "Core/QuantumMachine/SingleAmplitudeQVM.h"
#include "Core/QuantumMachine/PartialAmplitudeQVM.h"
#include "pybind11/eigen.h"
#include "pybind11/operators.h"
#include "Core/Utilities/base64.hpp"

USING_QPANDA
using namespace std;
using namespace pybind11::literals;
using std::map;
namespace py = pybind11;

template<>
struct py::detail::type_caster<QVec>
    : py::detail::list_caster<QVec, Qubit*> { };
  
void init_quantum_machine(py::module &);
void init_variational(py::module &);

#define BIND_CLASSICALCOND_OPERATOR_OVERLOAD(OP) .def(py::self OP py::self)\
                                                 .def(py::self OP cbit_size_t())\
                                                 .def(cbit_size_t() OP py::self)

PYBIND11_MODULE(pyQPanda, m)
{
    init_quantum_machine(m);
    init_variational(m);

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

    m.def("finalize", &finalize,
        "to finalize the environment. Use this at the end"
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

    m.def("T", &T, "Create a T gate",
        py::return_value_policy::automatic
    );

    m.def("T", &T, "qubit"_a, "Create a T gate",
        py::return_value_policy::automatic
    );

    m.def("S", &S, "qubit"_a, "Create a S gate",
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

    m.def("CNOT", &CNOT, "control_qubit"_a, "target_qubit"_a, "Create a CNOT gate",
        py::return_value_policy::automatic
    );

    m.def("CZ", &CZ, "control_qubit"_a, "target_qubit"_a, "Create a CZ gate",
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

	m.def("print_mat", &printMat, "mat"_a,
		"output matrix information to consol",
		py::return_value_policy::automatic
	);

    m.def("is_match_topology", &isMatchTopology, "gate"_a, "vecTopoSt"_a,
        "Whether the qgate matches the quantum topology",
        py::return_value_policy::automatic
    );

    m.def("get_adjacent_qgate_type", [](QProg &prog, NodeIter &node_iter)
    {
        std::vector<NodeIter> front_and_back_iter;
        getAdjacentQGateType(prog, node_iter, front_and_back_iter);
        return front_and_back_iter;
    }, "get the adjacent qgates's(the front one and the back one) type",
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

	m.def("originir_to_qprog", [](string file_path, QuantumMachine *qvm)
		{return transformOriginIRToQProg(file_path, qvm); },
		py::return_value_policy::automatic_reference
	);

    m.def("to_QASM", [](QProg prog,QuantumMachine *qvm, IBMQBackends ibmBackend) {
        py::list retData;
        std::string qasmStr = transformQProgToQASM(prog, qvm, (IBMQBackends)ibmBackend);
        retData.append(qasmStr);

        std::string IBMBackendName = QProgToQASM::getIBMQBackendName((IBMQBackends)ibmBackend);
        retData.append(IBMBackendName);

        return retData;
    },"prog"_a,"quantum machine"_a,"IBMQBackends"_a= IBMQ_QASM_SIMULATOR,py::return_value_policy::automatic_reference
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

#ifdef USE_CURL
    m.def("get_bin_data", [](QProg prog) {
        extern QuantumMachine* global_quantum_machine;
        return qProgToBinary(prog, global_quantum_machine);
    }
        , "program"_a, "Get quantum program binary data",
        py::return_value_policy::automatic_reference
        );
#endif // USE_CURL

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

	m.def("transform_qprog_to_qasm", [](QProg prog, QuantumMachine *qvm, IBMQBackends ibmBackend) {
		py::list retData;
		std::string qasmStr = transformQProgToQASM(prog, qvm, (IBMQBackends)ibmBackend);
		retData.append(qasmStr);

		std::string IBMBackendName = QProgToQASM::getIBMQBackendName((IBMQBackends)ibmBackend);
		retData.append(IBMBackendName);

		return retData;
		}, "prog"_a, "quantum machine"_a, "IBMQBackends"_a = IBMQ_QASM_SIMULATOR, py::return_value_policy::automatic_reference
	);


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


	m.def("transform_qprog_to_originir", [](QProg prog, QuantumMachine *qvm) {
		return transformQProgToOriginIR(prog, qvm);
		},
		"program"_a,"quantum machine"_a, "QProg to originir",
        py::return_value_policy::automatic_reference
        );

#ifdef USE_CURL
    m.def("qprog_to_binary", [](QProg prog) {
        extern QuantumMachine* global_quantum_machine;
        return qProgToBinary(prog, global_quantum_machine);
    }
        , "program"_a, "Get quantum program binary data",
        py::return_value_policy::automatic_reference
        );
#endif // USE_CURL

    m.def("get_qprog_clock_cycle", &getQProgClockCycle, 
		"program"_a, "QuantumMachine"_a, "Get Quantum Program Clock Cycle",
        py::return_value_policy::automatic_reference
        );

    m.def("transform_binary_data_to_qprog", [](QuantumMachine *qm, const std::vector<uint8_t>& data, QVec & qubits,
        std::vector<ClassicalCondition>& cbits, QProg & prog) {
        return binaryQProgDataParse(qm, data, qubits, cbits, prog);
    }
        , "QuantumMachine"_a, "data"_a, "qlist"_a, "clist"_a, "program"_a,
        "Parse quantum program interface for  binary data vector",
        py::return_value_policy::automatic_reference
        );

    m.def("transform_qprog_to_originir", [](QProg prog, QuantumMachine *qm) {
        return transformQProgToOriginIR(prog, qm);
    }
        , "program"_a, "QuantumMachine"_a, "QProg to originir",
        py::return_value_policy::automatic_reference
        );

    m.def("transform_originir_to_qprog", &transformOriginIRToQProg,
		"file_name"_a, "QuantumMachine"_a, "originir to QProg",
        py::return_value_policy::automatic_reference
        );

    m.def("validate_single_qgate_type", &validateSingleQGateType,
        "validate single QGate type",
        py::return_value_policy::automatic
    );

    m.def("validate_double_qgate_type", &validateDoubleQGateType,
        "validate double QGate type",
        py::return_value_policy::automatic
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
        .def("insert", &QProg::operator<<<ClassicalCondition>,
            py::return_value_policy::reference)
        .def("begin",&QProg::getFirstNodeIter, 
        py::return_value_policy::reference)
        .def("end",&QProg::getEndNodeIter, 
        py::return_value_policy::reference)
        .def("last",&QProg::getLastNodeIter, 
        py::return_value_policy::reference)
        .def("head",&QProg::getHeadNodeIter, 
        py::return_value_policy::reference);


    py::implicitly_convertible<QGate, QProg>();
    py::implicitly_convertible<QCircuit, QProg>();
    py::implicitly_convertible<QIfProg, QProg>();
    py::implicitly_convertible<QWhileProg, QProg>();
    py::implicitly_convertible<QMeasure, QProg>();
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

    py::class_<Qubit>(m, "Qubit")
        .def("getPhysicalQubitPtr", &Qubit::getPhysicalQubitPtr, py::return_value_policy::reference)
        ;

    py::class_<PhysicalQubit>(m, "PhysicalQubit")
        .def("getQubitAddr", &PhysicalQubit::getQubitAddr, py::return_value_policy::reference)
        ;

    py::class_<CBit>(m, "CBit")
        .def("getName", &CBit::getName);
    
    py::class_<ClassicalCondition>(m, "ClassicalCondition")
        .def("eval", &ClassicalCondition::eval,"get value")
        .def("setValue",&ClassicalCondition::setValue,"set value")
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
        .value("CLASS_COND_NODE", NodeType::CLASS_COND_NODE);

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


	m.def("get_matrix", [](QProg & prog, const NodeIter nodeItrStart, const NodeIter nodeItrEnd) {
		return getMatrix(prog, nodeItrStart, nodeItrEnd);
	}, py::arg("prog"), py::arg("nodeItrStart") = NodeIter(), py::arg("nodeItrEnd") = NodeIter()
		, "get the target prog  matrix",
		py::return_value_policy::automatic
		);

	m.def("get_matrix", [](QCircuit & cir, const NodeIter nodeItrStart, const NodeIter nodeItrEnd) {
		return getMatrix(cir, nodeItrStart, nodeItrEnd);
	}, py::arg("circuit"), py::arg("nodeItrStart") = NodeIter(), py::arg("nodeItrEnd") = NodeIter()
		, "get the target circuit  matrix",
		py::return_value_policy::automatic
		);

	m.def("print_qprog", [](QProg &prg, const NodeIter nodeItrStart, const NodeIter nodeItrEnd) {
		return printProg(prg, nodeItrStart, nodeItrEnd);
	}, py::arg("prog"), py::arg("nodeItrStart") = NodeIter(), py::arg("nodeItrEnd") = NodeIter(),
		"output a quantum prog/circuit to console by text-pic(UTF-8 code), \
		and will save the text-pic in file named QCircuitTextPic.txt in the same time in current path",
		py::return_value_policy::automatic
		);

	m.def("print_qprog", [](QCircuit &cir, const NodeIter nodeItrStart, const NodeIter nodeItrEnd) {
		return printProg(cir, nodeItrStart, nodeItrEnd);
	}, py::arg("circuit"), py::arg("nodeItrStart") = NodeIter(), py::arg("nodeItrEnd") = NodeIter(),
		"output a quantum prog/circuit to console by text-pic(UTF-8 code), \
		and will save the text-pic in file named QCircuitTextPic.txt in the same time in current path",
		py::return_value_policy::automatic
		);

#define QUERY_REPLACE(GRAPH_NODE,QUERY_NODE,REPLACE_NODE) \
m.def("graph_query_replace", [](GRAPH_NODE &graph_node, QUERY_NODE &query_node,\
                                   REPLACE_NODE &replace_node, QProg &prog, QuantumMachine *qvm)\
{\
    GraphMatch qm;\
    return qm.graphQueryReplace(graph_node, query_node, replace_node, prog, qvm);\
});

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

