#include <math.h>
#include <map>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/chrono.h"
#include "Core/QPanda.h"
#include "MaxCutProblemGenerator/MaxCutProblemGenerator.h"
#include "pybind11/eigen.h"
#include "pybind11/operators.h"
#include <pybind11/stl_bind.h>
#include "Core/Utilities/OriginCollection.h"
#include "QAlg/DataStruct.h"
#include "Optimizer/AbstractOptimizer.h"
#include "Optimizer/OptimizerFactory.h"
#include "Optimizer/OriginNelderMead.h"
#include "Core/QuantumMachine/QCloudMachine.h"
#include "Core/Utilities/Transform/QProgClockCycle.h"
#include "Operator/FermionOperator.h"
#include "QPandaConfig.h"
#include "QAlg/Utils/Utilities.h"

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
        "Allocate several qubits",
        py::return_value_policy::reference
    );


    m.def("qAlloc_many", [](size_t size) {
         std::vector<Qubit *> temp =  qAllocMany(size);
         return temp;
    },
        "Allocate several qubits",
        py::return_value_policy::reference
    );

    m.def("cAlloc", []() {return cAlloc(); },
        "Allocate a CBit",
        py::return_value_policy::reference
    );

    m.def("cAlloc_many", [](size_t size) {return cAllocMany(size); },
        "Allocate several CBits",
        py::return_value_policy::reference
    );

    m.def("cFree", &cFree, "Free a CBit");

    m.def("apply_QGate", &apply_QGate,
        "Apply QGate to qubits",
        py::return_value_policy::reference
    );

    m.def("getstat", &getstat,
        "get the status(ptr) of the quantum machine");

    m.def("getAllocateQubitNum", &getAllocateQubitNum,
        "getAllocateQubitNum");

    m.def("getAllocateCMem", &getAllocateCMem, "getAllocateCMem");

    m.def("directly_run", &directlyRun, "directly run");

    m.def("quick_measure", &quickMeasure, "qubit_list"_a, "shots"_a, "quick measure");

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

    m.def("measure", &Measure,"qubit"_a,"cbit"_a,
        "Create a Measure operation",
        py::return_value_policy::automatic
    );

    m.def("measure_all", &MeasureAll,"qubit_list"_a, "cbit_list"_a,
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
        "control_qubit"_a, "target_qubit"_a,"angle"_a,
        "Create a iSWAP gate",
        py::return_value_policy::automatic
    );

    m.def("CR", &CR, "control_qubit"_a, "target_qubit"_a,"angle"_a, "Create a CR gate",
        py::return_value_policy::automatic
    );

    m.def("to_QRunes", [](QProg & qn,QuantumMachine *qvm)
    {return transformQProgToQRunes(qn,qvm); },
        py::return_value_policy::automatic_reference
    );
    m.def("to_QRunes", [](QCircuit & qn, QuantumMachine *qvm)
    {return transformQProgToQRunes(qn, qvm); },
        py::return_value_policy::automatic_reference
    );
    m.def("to_QRunes", [](QGate & qn, QuantumMachine *qvm)
    {return transformQProgToQRunes(qn, qvm); },
        py::return_value_policy::automatic_reference
    );
    m.def("to_QRunes", [](QIfProg & qn, QuantumMachine *qvm)
    {return transformQProgToQRunes(qn, qvm); },
        py::return_value_policy::automatic_reference
    );
    m.def("to_QRunes", [](QWhileProg & qn, QuantumMachine *qvm)
    {return transformQProgToQRunes(qn, qvm); },
        py::return_value_policy::automatic_reference
    );
    m.def("to_QRunes", [](QMeasure & qn, QuantumMachine *qvm)
    {return transformQProgToQRunes(qn, qvm); },
        py::return_value_policy::automatic_reference
    );

	m.def("to_QASM", [](QProg prog) {
		extern QuantumMachine* global_quantum_machine;
		return transformQProgToQASM(prog, global_quantum_machine);
		}
		,"program"_a, "QProg to QASM",
        py::return_value_policy::automatic_reference
    );

    m.def("to_Quil", [](QProg prog) {
		extern QuantumMachine* global_quantum_machine;
		return transformQProgToQuil(prog, global_quantum_machine);
		}
		, "program"_a, "QProg to Quil",
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

    m.def("get_bin_data", [](QProg prog) {
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
    m.def("accumulate_probabilities", &accumulateProbability,"probability_list"_a,
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

    py::class_<ClassicalProg>(m, "ClassicalProg")
        .def(py::init<ClassicalCondition &>());

    py::class_<QProg>(m, "QProg")
        .def(py::init<>())
        .def(py::init<QCircuit &>())
        .def(py::init<QIfProg &>())
        .def(py::init<QWhileProg &>())
        .def(py::init<QGate &>())
        .def(py::init<QMeasure &>())
        .def(py::init<ClassicalCondition &>())
        .def("insert", &QProg::operator<<<QProg>,
            py::return_value_policy::reference);

        py::implicitly_convertible<QCircuit, QProg>();
        py::implicitly_convertible<QIfProg, QProg>();
        py::implicitly_convertible<QWhileProg, QProg>();
        py::implicitly_convertible<QGate, QProg>();
        py::implicitly_convertible<QMeasure, QProg>();
        py::implicitly_convertible<ClassicalCondition, QProg>();

    py::class_<QCircuit>(m, "QCircuit")
        .def(py::init<>())
        .def("insert", &QCircuit::operator<< <QCircuit>, 
            py::return_value_policy::reference)
        .def("insert", &QCircuit::operator<< <QGate>,
            py::return_value_policy::reference)
        .def("dagger", &QCircuit::dagger)
        .def("control", &QCircuit::control);

    py::class_<HadamardQCircuit, QCircuit>(m, "hadamard_circuit")
        .def(py::init<QVec&>());

    py::class_<QGate>(m, "QGate")
        .def("dagger", &QGate::dagger)
        .def("control", &QGate::control);


    py::class_<QIfProg>(m, "QIfProg")
            .def(py::init<ClassicalCondition &, QProg>())
            .def(py::init<ClassicalCondition &, QProg, QProg>());

    py::class_<QWhileProg>(m, "QWhileProg")
            .def(py::init<ClassicalCondition , QProg>());

    py::class_<QMeasure>(m, "QMeasure");

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

	py::class_<FermionOperator>(m, "FermionOperator")
		.def(py::init<>())
		.def(py::init<>([](const complex_d &val)
			{ return FermionOperator(val); }))
		.def(py::init<>([](const std::string &key, const complex_d &val)
			{ return FermionOperator(key, val); }))
		.def(py::init<>([](const FermionOperator::FermionMap &map)
			{ return FermionOperator(map); }))
		.def("normal_ordered", &FermionOperator::normal_ordered)
		.def("isEmpty", &FermionOperator::isEmpty)
		.def("setAction", &FermionOperator::setAction)
		.def("setErrorThreshold", &FermionOperator::setErrorThreshold)
		.def("error_threshold", &FermionOperator::error_threshold)
		.def("data", &FermionOperator::data)
		.def(py::self + py::self)
		.def(py::self - py::self)
		.def(py::self * py::self)
		.def(py::self += py::self)
		.def(py::self -= py::self)
		.def(py::self *= py::self)
		.def(py::self + QPanda::complex_d())
		.def(py::self * QPanda::complex_d())
		.def(py::self - QPanda::complex_d())
		.def(QPanda::complex_d() + py::self)
		.def(QPanda::complex_d() * py::self)
		.def(QPanda::complex_d() - py::self)
		.def("toString", &FermionOperator::toString)
		.def("__str__", &FermionOperator::toString);

    py::implicitly_convertible<ClassicalCondition, ClassicalProg>();
    py::implicitly_convertible<cbit_size_t, ClassicalCondition>();
}
