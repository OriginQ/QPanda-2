#include <math.h>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/chrono.h"
#include "Core/QPanda.h"
#include "MaxCutProblemGenerator/MaxCutProblemGenerator.h"
#include "Variational/utils.h"
#include "pybind11/eigen.h"
#include "pybind11/operators.h"
#include "Core/Utilities/OriginCollection.h"
#include "Variational/Optimizer.h"
#include "Core/VirtualQuantumProcessor/NoiseQPU/NoiseModel.h"
#include <map>
#include "QAlg/DataStruct.h"
#include "Optimizer/AbstractOptimizer.h"
#include "Optimizer/OptimizerFactory.h"
#include "Optimizer/OriginNelderMead.h"
#include "Core/QuantumMachine/QCloudMachine.h"
#include "Core/Utilities/Transform/QProgClockCycle.h"
#include "QPandaConfig.h"

USING_QPANDA
using namespace std;
using namespace pybind11::literals;
using std::map;
namespace py = pybind11;
namespace Var = QPanda::Variational;

#define GET_FEED_PTR_NO_OFFSET(ptr_name, classname) \
    QPanda::QGate(classname::*ptr_name)() const \
    = &classname::feed

#define GET_FEED_PTR_WITH_OFFSET(ptr_name, classname) \
    QPanda::QGate(classname::*ptr_name)( \
        std::map<size_t, double>) const \
    = &classname::feed

template<>
struct py::detail::type_caster<QVec>
    : py::detail::list_caster<QVec, Qubit*> { };

namespace QPanda {
namespace Variational {
const var py_stack(int axis, std::vector<var>& args)
{
    std::vector<std::shared_ptr<impl>> vimpl;
    for (auto &arg : args)
        vimpl.push_back(arg.pimpl);
        Var::var res(make_shared<impl_stack>(axis, args));
        for (const std::shared_ptr<impl>& _impl : vimpl) {
            _impl->parents.push_back(res.pimpl);
        }
        return res;
    }
} // namespace QPanda
} // namespace Variational

#define BIND_VAR_OPERATOR_OVERLOAD(OP) .def(py::self OP py::self)\
                                       .def(py::self OP double())\
                                       .def(double() OP py::self)

#define BIND_CLASSICALCOND_OPERATOR_OVERLOAD(OP) .def(py::self OP py::self)\
                                                 .def(py::self OP cbit_size_t())\
                                                 .def(cbit_size_t() OP py::self)

PYBIND11_MODULE(pyQPanda, m)
{   
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

    m.def("qAlloc_many", [](size_t size) {return qAllocMany(size); },
        "Allocate several qubits",
        py::return_value_policy::reference
    );


    m.def("measure_all", &MeasureAll, "directly run");

    m.def("cAlloc", []() {return cAlloc(); },
        "Allocate a CBit",
        py::return_value_policy::reference
    );

    m.def("cAlloc_many", [](size_t size) {return cAllocMany(size); },
        "Allocate several CBits",
        py::return_value_policy::reference
    );

    m.def("cFree", &cFree, "Free a CBit");

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


    m.def("CreateWhileProg", [](ClassicalCondition& m, QProg & qn)
    {QNode * node = (QNode *)&qn;
    return CreateWhileProg(m, node); },
        "Create a WhileProg",
        py::return_value_policy::automatic
        );

    m.def("CreateWhileProg", [](ClassicalCondition& m, QCircuit & qn)
    {QNode * node = (QNode *)&qn;
    return CreateWhileProg(m, node); },
        "Create a WhileProg",
        py::return_value_policy::automatic
        );

    m.def("CreateWhileProg", [](ClassicalCondition& m, QGate & qn)
    {QNode * node = (QNode *)&qn;
    return CreateWhileProg(m, node); },
        "Classical_condition"_a, "Qnode"_a,
        "Create a WhileProg",
        py::return_value_policy::automatic
        );

    m.def("CreateIfProg", [](ClassicalCondition& m, QProg & qn)
    {QNode * node = (QNode *)&qn;
    return CreateIfProg(m, node); },
        "Classical_condition"_a, "true_node"_a,
        "Create a IfProg",
        py::return_value_policy::automatic
        );

    m.def("CreateIfProg", [](ClassicalCondition& m, QCircuit & qn)
    {QNode * node = (QNode *)&qn;
    return CreateIfProg(m, node); },
        "Classical_condition"_a, "true_node"_a,
        "Create a IfProg",
        py::return_value_policy::automatic
        );

    m.def("CreateIfProg", [](ClassicalCondition& m, QGate & qn)
    {QNode * node = (QNode *)&qn;
    return CreateIfProg(m, node); },
        "Classical_condition"_a, "true_node"_a,
        "Create a IfProg",
        py::return_value_policy::automatic
        );

    m.def("CreateIfProg", [](ClassicalCondition&m, QGate & qn1, QProg & qn2)
    {QNode * node1 = (QNode *)&qn1;
    QNode * node2 = (QNode *)&qn2;
    return CreateIfProg(m, node1, node2); },
        "Classical_condition"_a, "true_node"_a, "false_node"_a,
        "Create a IfProg",
        py::return_value_policy::automatic
        );

    m.def("CreateIfProg", [](ClassicalCondition&m, QGate & qn1, QCircuit & qn2)
    {QNode * node1 = (QNode *)&qn1;
    QNode * node2 = (QNode *)&qn2;
    return CreateIfProg(m, node1, node2); },
        "Classical_condition"_a, "true_node"_a, "false_node"_a,
        "Create a IfProg",
        py::return_value_policy::automatic
        );

    m.def("CreateIfProg", [](ClassicalCondition&m, QGate & qn1, QGate & qn2)
    {QNode * node1 = (QNode *)&qn1;
    QNode * node2 = (QNode *)&qn2;
    return CreateIfProg(m, node1, node2); },
        "Classical_condition"_a, "true_node"_a, "false_node"_a,
        "Create a IfProg",
        py::return_value_policy::automatic
        );

    m.def("CreateIfProg", [](ClassicalCondition&m, QCircuit & qn1, QGate & qn2)
    {QNode * node1 = (QNode *)&qn1;
    QNode * node2 = (QNode *)&qn2;
    return CreateIfProg(m, node1, node2); },
        "Classical_condition"_a, "true_node"_a, "false_node"_a,
        "Create a IfProg",
        py::return_value_policy::automatic
        );

    m.def("CreateIfProg", [](ClassicalCondition&m, QCircuit & qn1, QCircuit & qn2)
    {QNode * node1 = (QNode *)&qn1;
    QNode * node2 = (QNode *)&qn2;
    return CreateIfProg(m, node1, node2); },
        "Classical_condition"_a, "true_node"_a, "false_node"_a,
        "Create a IfProg",
        py::return_value_policy::automatic
        );

    m.def("CreateIfProg", [](ClassicalCondition&m, QCircuit & qn1, QProg & qn2)
    {QNode * node1 = (QNode *)&qn1;
    QNode * node2 = (QNode *)&qn2;
    return CreateIfProg(m, node1, node2); },
        "Classical_condition"_a, "true_node"_a, "false_node"_a,
        "Create a IfProg",
        py::return_value_policy::automatic
        );

    m.def("CreateIfProg", [](ClassicalCondition&m, QProg & qn1, QGate & qn2)
    {QNode * node1 = (QNode *)&qn1;
    QNode * node2 = (QNode *)&qn2;
    return CreateIfProg(m, node1, node2); },
        "Classical_condition"_a, "true_node"_a, "false_node"_a,
        "Create a IfProg",
        py::return_value_policy::automatic
        );


    m.def("CreateIfProg", [](ClassicalCondition&m, QProg & qn1, QCircuit & qn2)
    {QNode * node1 = (QNode *)&qn1;
    QNode * node2 = (QNode *)&qn2;
    return CreateIfProg(m, node1, node2); },
        "Classical_condition"_a, "true_node"_a, "false_node"_a,
        "Create a IfProg",
        py::return_value_policy::automatic
        );

    m.def("CreateIfProg", [](ClassicalCondition&m, QProg & qn1, QProg & qn2)
    {QNode * node1 = (QNode *)&qn1;
    QNode * node2 = (QNode *)&qn2;
    return CreateIfProg(m, node1, node2); },
        "Classical_condition"_a, "true_node"_a, "false_node"_a,
        "Create a IfProg",
        py::return_value_policy::automatic
        );

    m.def("CreateEmptyCircuit", &CreateEmptyCircuit,
        "Create an empty QCircuit Container",
        py::return_value_policy::automatic
    );

    m.def("Measure", &Measure,"qubit"_a,"cbit"_a,
        "Create a Measure operation",
        py::return_value_policy::automatic
    );

    m.def("meas_all", &MeasureAll,"qubit_list"_a, "cbit_list"_a,
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

    m.def("to_QRunes", &transformQProgToQRunes,"program"_a, "qvm"_a, "QProg to QRunes",
        py::return_value_policy::automatic_reference
    );

    m.def("to_QASM", &transformQProgToQASM, "program"_a, "qvm"_a, "QProg to QASM",
        py::return_value_policy::automatic_reference
    );

    m.def("to_Quil", &transformQProgToQuil, "program"_a, "qvm"_a, "QProg to Quil",
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

    m.def("get_clock_cycle", &getQProgClockCycle, "qvm"_a,"program"_a, "Get Quantum Program Clock Cycle",
        py::return_value_policy::automatic_reference
    );

    m.def("get_bin_data", &transformQProgToBinary, "program"_a ,"qvm"_a, "Get quantum program binary data",
        py::return_value_policy::automatic_reference
    );

#ifdef USE_CURL
    m.def("get_bin_str", &QProgToBinary, "program"_a, "qvm"_a, "Get quantum program binary data string",
        py::return_value_policy::automatic_reference
    );
#endif // USE_CURL

    m.def("bin_to_prog", &binaryQProgDataParse, "qvm"_a, "data"_a, "qlist"_a, "clist"_a, "program"_a,
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
        .def("insert", &QProg::operator<<<QGate>,
            py::return_value_policy::reference)
        .def("insert", &QProg::operator<<<QProg>,
            py::return_value_policy::reference)
        .def("insert", &QProg::operator<<<QCircuit>,
            py::return_value_policy::reference)
        .def("insert", &QProg::operator<<<QMeasure>,
            py::return_value_policy::reference)
        .def("insert", &QProg::operator<<<QIfProg>,
            py::return_value_policy::reference)
        .def("insert", &QProg::operator<<<QWhileProg>,
            py::return_value_policy::reference)
        .def("insert", &QProg::operator<<<ClassicalProg>,
            py::return_value_policy::reference);

    py::class_<QCircuit>(m, "QCircuit")
        .def(py::init<>())
        .def("insert", &QCircuit::operator<< <QCircuit>, 
            py::return_value_policy::reference)
        .def("insert", &QCircuit::operator<< <QGate>,
            py::return_value_policy::reference)
        .def("dagger", &QCircuit::dagger)
        .def("control", &QCircuit::control);

    py::class_<HadamardQCircuit, QCircuit>(m, "Hadamard_Circuit")
        .def(py::init<QVec&>());


    py::class_<QGate>(m, "QGate")
        .def("dagger", &QGate::dagger)
        .def("control", &QGate::control);

    py::class_<QIfProg>(m, "QIfProg");
    py::class_<QWhileProg>(m, "QWhileProg");
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
        BIND_CLASSICALCOND_OPERATOR_OVERLOAD(>=);

        
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

    py::enum_<QMachineType>(m, "QMachineType")
        .value("CPU", QMachineType::CPU)
        .value("GPU", QMachineType::GPU)
        .value("CPU_SINGLE_THREAD", QMachineType::CPU_SINGLE_THREAD)
        .value("NOISE", QMachineType::NOISE)
        .export_values();

    py::enum_<NOISE_MODEL>(m, "NoiseModel")
        .value("DAMPING_KRAUS_OPERATOR", NOISE_MODEL::DAMPING_KRAUS_OPERATOR)
        .value("DECOHERENCE_KRAUS_OPERATOR", NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR)
        .value("DEPHASING_KRAUS_OPERATOR", NOISE_MODEL::DEPHASING_KRAUS_OPERATOR)
        .value("PAULI_KRAUS_MAP", NOISE_MODEL::PAULI_KRAUS_MAP)
        .value("DOUBLE_DAMPING_KRAUS_OPERATOR", NOISE_MODEL::DOUBLE_DAMPING_KRAUS_OPERATOR)
        .value("DOUBLE_DECOHERENCE_KRAUS_OPERATOR", NOISE_MODEL::DOUBLE_DECOHERENCE_KRAUS_OPERATOR)
        .value("DOUBLE_DEPHASING_KRAUS_OPERATOR", NOISE_MODEL::DOUBLE_DEPHASING_KRAUS_OPERATOR)
        ;

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

    Qubit*(QuantumMachine::*qalloc)() = &QuantumMachine::allocateQubit;
    QVec (QuantumMachine::*qallocMany)(size_t) = &QuantumMachine::allocateQubits;
    vector<ClassicalCondition>(QuantumMachine::*callocMany)(size_t) = &QuantumMachine::allocateCBits;
    void (QuantumMachine::*free_qubit)(Qubit *) = &QuantumMachine::Free_Qubit;
    void (QuantumMachine::*free_qubits)(QVec&) = &QuantumMachine::Free_Qubits;
    void (QuantumMachine::*free_cbit)(ClassicalCondition & ) = &QuantumMachine::Free_CBit;
    void (QuantumMachine::*free_cbits)(vector<ClassicalCondition> &) = &QuantumMachine::Free_CBits;
    QMachineStatus * (QuantumMachine::*get_status)() const = &QuantumMachine::getStatus;
    size_t (QuantumMachine::*get_allocate_qubit)() = &QuantumMachine::getAllocateQubit;
    size_t (QuantumMachine::*get_allocate_CMem)() = &QuantumMachine::getAllocateCMem;
    void (QuantumMachine::*_finalize)() = &QuantumMachine::finalize;


    Qubit*(QVec::*qvec_subscript_cbit_size_t)(cbit_size_t) = &QVec::operator[];
    Qubit*(QVec::*qvec_subscript_cc)(ClassicalCondition&) = &QVec::operator[];

    py::class_<QVec>(m, "QVec")
        .def(py::init<>())
        .def(py::init<const QVec &>())
        .def("__getitem__", qvec_subscript_cbit_size_t, py::return_value_policy::reference)
        .def("__getitem__", qvec_subscript_cc, py::return_value_policy::reference)
        .def("__len__", &QVec::size);


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


    py::class_<QuantumMachine>(m, "QuantumMachine")
        .def("finalize", &QuantumMachine::finalize, "finalize")
        .def("get_qstate", &QuantumMachine::getQState, "getState", py::return_value_policy::automatic)
        .def("qAlloc", qalloc, "Allocate a qubit", py::return_value_policy::reference)
        .def("qAlloc_many", qallocMany, "Allocate a list of qubits", "n_qubit"_a,
            py::return_value_policy::reference)
        .def("cAlloc", calloc, "Allocate a cbit", py::return_value_policy::reference)
        .def("cAlloc_many", callocMany, "Allocate a list of cbits", "n_cbit"_a,
            py::return_value_policy::reference)
        .def("qFree", free_qubit, "Free a qubit")
        .def("qFree_all", free_qubits, "qubit_list"_a,
            "Free a list of qubits")
        .def("cFree", free_cbit, "Free a cbit")
        .def("cFree_all", free_cbits, "cbit_list"_a,
            "Free a list of cbits")
        .def("getStatus", get_status, "get the status(ptr) of the quantum machine",
            py::return_value_policy::reference)
        .def("getAllocateQubitNum", get_allocate_qubit, "getAllocateQubitNum", py::return_value_policy::reference)
        .def("getAllocateCMem", get_allocate_CMem, "getAllocateCMem", py::return_value_policy::reference)
        .def("directly_run", &QuantumMachine::directlyRun, "program"_a,
            py::return_value_policy::reference)
        .def("run_with_configuration", [](QuantumMachine &qvm, QProg & prog, vector<ClassicalCondition> & cc_vector, py::dict param) {
        py::object json = py::module::import("json");
        py::object dumps = json.attr("dumps");
        auto json_string = std::string(py::str(dumps(param)));
        rapidjson::Document doc;
        auto & alloc = doc.GetAllocator();
        doc.Parse(json_string.c_str());
        return qvm.runWithConfiguration(prog, cc_vector, doc);
    },
            "program"_a, "cbit_list"_a, "data"_a,
        py::return_value_policy::automatic);


#define DEFINE_IDEAL_QVM(class_name)\
    py::class_<class_name,QuantumMachine>(m, #class_name)\
        .def(py::init<>())\
        .def("initQVM", &class_name::init, "init quantum virtual machine")\
        .def("finalize", &class_name::finalize, "finalize")\
        .def("get_qstate", &class_name::getQState, "getState",py::return_value_policy::automatic)\
        .def("qAlloc", qalloc, "Allocate a qubit", py::return_value_policy::reference)\
        .def("qAlloc_many", qallocMany, "Allocate a list of qubits", "n_qubit"_a,\
            py::return_value_policy::reference)\
        .def("cAlloc", calloc, "Allocate a cbit", py::return_value_policy::reference)\
        .def("cAlloc_many", callocMany, "Allocate a list of cbits", "n_cbit"_a,\
            py::return_value_policy::reference)\
        .def("qFree", free_qubit, "Free a qubit")\
        .def("qFree_all", free_qubits, "qubit_list"_a,\
            "Free a list of qubits")\
        .def("cFree", free_cbit, "Free a cbit")\
        .def("cFree_all", free_cbits, "cbit_list"_a,\
            "Free a list of cbits")\
        .def("getStatus", get_status, "get the status(ptr) of the quantum machine",\
            py::return_value_policy::reference)\
        .def("getAllocateQubitNum", get_allocate_qubit, "getAllocateQubitNum", py::return_value_policy::reference)\
        .def("getAllocateCMem", get_allocate_CMem, "getAllocateCMem", py::return_value_policy::reference)\
        .def("directly_run", &class_name::directlyRun, "program"_a,\
            py::return_value_policy::reference)\
        .def("run_with_configuration", [](class_name &qvm, QProg &prog, vector<ClassicalCondition> & cc_vector, py::dict param){\
            py::object json = py::module::import("json");\
            py::object dumps = json.attr("dumps");\
            auto json_string = std::string(py::str(dumps(param)));\
            rapidjson::Document doc;\
            auto & alloc = doc.GetAllocator();\
            doc.Parse(json_string.c_str());\
            return qvm.runWithConfiguration(prog, cc_vector, doc);\
        }, \
            "program"_a, "cbit_list"_a, "data"_a,\
            py::return_value_policy::automatic)\
        .def("pmeasure", &class_name::PMeasure, "qubit_list"_a, "select_max"_a = -1,\
            "Get the probability distribution over qubits", py::return_value_policy::reference)\
        .def("pmeasure_no_index", &class_name::PMeasure_no_index, "qubit_list"_a,\
            "Get the probability distribution over qubits", py::return_value_policy::reference)\
        .def("get_prob_tuple_list", &class_name::getProbTupleList, "qubit_list"_a, "select_max"_a = -1,\
            py::return_value_policy::reference)\
        .def("get_prob_list", &class_name::getProbList, "qubit_list"_a, "select_max"_a = -1,\
            py::return_value_policy::reference)\
        .def("get_prob_dict", &class_name::getProbDict, "qubit_list"_a, "select_max"_a = -1,\
            py::return_value_policy::reference)\
        .def("prob_run_tuple_list", &class_name::probRunDict, "program"_a, "qubit_list"_a, "select_max"_a = -1,\
            py::return_value_policy::reference)\
        .def("prob_run_list", &class_name::probRunList, "program"_a, "qubit_list"_a, "select_max"_a = -1,\
            py::return_value_policy::reference)\
        .def("prob_run_dict", &class_name::probRunDict, "program"_a, "qubit_list"_a, "select_max"_a = -1,\
            py::return_value_policy::reference)\
        .def("quick_measure", &class_name::quickMeasure, "qubit_list"_a, "shots"_a,\
            py::return_value_policy::reference)\

    DEFINE_IDEAL_QVM(CPUQVM);
    DEFINE_IDEAL_QVM(CPUSingleThreadQVM);
    DEFINE_IDEAL_QVM(GPUQVM);

    py::class_<NoiseQVM, QuantumMachine>(m, "NoiseQVM")
        .def(py::init<>())
        .def("initQVM", [](NoiseQVM& qvm, py::dict param) {
            py::object json = py::module::import("json");
            py::object dumps = json.attr("dumps");
            auto json_string = std::string(py::str(dumps(param)));
            rapidjson::Document doc(rapidjson::kObjectType); 
            doc.Parse(json_string.c_str()); 
            qvm.init(doc);
        }, "init quantum virtual machine")
        .def("finalize", &NoiseQVM::finalize, "finalize")
        .def("get_qstate", &NoiseQVM::getQState, "getState", py::return_value_policy::automatic)
        .def("qAlloc", qalloc, "Allocate a qubit", py::return_value_policy::reference)
        .def("qAlloc_many", qallocMany, "Allocate a list of qubits", "n_qubit"_a,
            py::return_value_policy::reference)
        .def("cAlloc", calloc, "Allocate a cbit", py::return_value_policy::reference)
        .def("cAlloc_many", callocMany, "Allocate a list of cbits", "n_cbit"_a,
            py::return_value_policy::reference)
        .def("qFree", free_qubit, "Free a qubit")
        .def("qFree_all", free_qubits, "qubit_list"_a,
            "Free a list of qubits")
        .def("cFree", free_cbit, "Free a cbit")
        .def("cFree_all", free_cbits, "cbit_list"_a,
            "Free a list of cbits")
        .def("getStatus", get_status, "get the status(ptr) of the quantum machine",
            py::return_value_policy::reference)
        .def("getAllocateQubitNum", get_allocate_qubit, "getAllocateQubitNum", py::return_value_policy::reference)
        .def("getAllocateCMem", get_allocate_CMem, "getAllocateCMem", py::return_value_policy::reference)
        .def("directly_run", &NoiseQVM::directlyRun, "program"_a,
            py::return_value_policy::reference)
        .def("run_with_configuration", [](NoiseQVM &qvm, QProg & prog, vector<ClassicalCondition> & cc_vector, py::dict param) {
            py::object json = py::module::import("json");
            py::object dumps = json.attr("dumps");
            auto json_string = std::string(py::str(dumps(param)));
            rapidjson::Document doc;
            auto & alloc = doc.GetAllocator();
            doc.Parse(json_string.c_str());
            return qvm.runWithConfiguration(prog, cc_vector, doc);
        },
            "program"_a, "cbit_list"_a, "data"_a,
        py::return_value_policy::automatic);



    py::class_<QResult>(m, "QResult")
        .def("getResultMap", &QResult::getResultMap, py::return_value_policy::reference);
    
    m.def("vector_dot", &vector_dot,"x"_a,"y"_a, "Inner product of vector x and y");
    m.def("all_cut_of_graph", &all_cut_of_graph, "generate graph of maxcut problem");

    m.def("vector_dot", &vector_dot, "Inner product of vector x and y");
    m.def("all_cut_of_graph", &all_cut_of_graph, "generate graph of maxcut problem");
    //combine pyQPandaVariational and pyQPanda
    py::class_<Var::var>(m, "var")
        .def(py::init<double>())
        .def(py::init<py::EigenDRef<Eigen::MatrixXd>>())
        .def(py::init<double,bool>())
        .def(py::init<py::EigenDRef<Eigen::MatrixXd>,bool>())
        .def("get_value", &Var::var::getValue)
        .def("set_value", &Var::var::setValue)
        .def("clone", &Var::var::clone)
        BIND_VAR_OPERATOR_OVERLOAD(+)
        BIND_VAR_OPERATOR_OVERLOAD(-)
        BIND_VAR_OPERATOR_OVERLOAD(*)
        BIND_VAR_OPERATOR_OVERLOAD(/ )
        .def("__getitem__", [](Var::var& v, int idx) {return v[idx]; }, py::is_operator())
        .def(py::self == py::self);

    py::class_<Var::VariationalQuantumGate>
        (m, "VariationalQuantumGate")
        .def("get_constants",&Var::VariationalQuantumGate::get_constants, py::return_value_policy::reference);


    GET_FEED_PTR_NO_OFFSET(feed_vqg_h_no_ptr, Var::VariationalQuantumGate_H);

    py::class_<Var::VariationalQuantumGate_H, Var::VariationalQuantumGate>
        (m, "VariationalQuantumGate_H")
        .def(py::init<QPanda::Qubit*>())
        .def("feed", feed_vqg_h_no_ptr);
    GET_FEED_PTR_NO_OFFSET(feed_vqg_x_no_ptr, Var::VariationalQuantumGate_X);

    py::class_<Var::VariationalQuantumGate_X, Var::VariationalQuantumGate>
        (m, "VariationalQuantumGate_X")
        .def(py::init<QPanda::Qubit*>())
        .def("feed", feed_vqg_x_no_ptr);

    GET_FEED_PTR_NO_OFFSET(feed_vqg_rx_no_ptr, Var::VariationalQuantumGate_RX);
    GET_FEED_PTR_WITH_OFFSET(feed_vqg_rx_with_ptr, Var::VariationalQuantumGate_RX);

    py::class_<Var::VariationalQuantumGate_RX, Var::VariationalQuantumGate>
        (m, "VariationalQuantumGate_RX")
        .def(py::init<QPanda::Qubit*, Var::var>())
        .def(py::init<QPanda::Qubit*, double>())
        .def("feed", feed_vqg_rx_no_ptr)
        .def("feed", feed_vqg_rx_with_ptr);

    GET_FEED_PTR_NO_OFFSET(feed_vqg_ry_no_ptr, Var::VariationalQuantumGate_RY);
    GET_FEED_PTR_WITH_OFFSET(feed_vqg_ry_with_ptr, Var::VariationalQuantumGate_RY);

    py::class_<Var::VariationalQuantumGate_RY, Var::VariationalQuantumGate>
        (m, "VariationalQuantumGate_RY")
        .def(py::init<QPanda::Qubit*, Var::var>())
        .def(py::init<QPanda::Qubit*, double>())
        .def("feed", feed_vqg_ry_no_ptr)
        .def("feed", feed_vqg_ry_with_ptr);

    GET_FEED_PTR_NO_OFFSET(feed_vqg_rz_no_ptr, Var::VariationalQuantumGate_RZ);
    GET_FEED_PTR_WITH_OFFSET(feed_vqg_rz_with_ptr, Var::VariationalQuantumGate_RZ);

    py::class_<Var::VariationalQuantumGate_RZ, Var::VariationalQuantumGate>
        (m, "VariationalQuantumGate_RZ")
        .def(py::init<QPanda::Qubit*, Var::var>())
        .def(py::init<QPanda::Qubit*, double>())
        .def("feed", feed_vqg_rz_no_ptr)
        .def("feed", feed_vqg_rz_with_ptr);

    GET_FEED_PTR_NO_OFFSET(feed_vqg_crx_no_ptr, Var::VariationalQuantumGate_CRX);

    py::class_<Var::VariationalQuantumGate_CRX, Var::VariationalQuantumGate>
        (m, "VariationalQuantumGate_CRX")
        .def(py::init<QPanda::Qubit*, QVec &, double>())
        .def(py::init<Var::VariationalQuantumGate_CRX &>())
        .def("feed", feed_vqg_crx_no_ptr);

    GET_FEED_PTR_NO_OFFSET(feed_vqg_cry_no_ptr, Var::VariationalQuantumGate_CRY);

    py::class_<Var::VariationalQuantumGate_CRY, Var::VariationalQuantumGate>
        (m, "VariationalQuantumGate_CRY")
        .def(py::init<QPanda::Qubit*, QVec &, double>())
        .def(py::init<Var::VariationalQuantumGate_CRY &>())
        .def("feed", feed_vqg_cry_no_ptr);

    GET_FEED_PTR_NO_OFFSET(feed_vqg_crz_no_ptr, Var::VariationalQuantumGate_CRZ);

    py::class_<Var::VariationalQuantumGate_CRZ, Var::VariationalQuantumGate>
        (m, "VariationalQuantumGate_CRZ")
        .def(py::init<QPanda::Qubit*, QVec &, double>())
        .def(py::init<Var::VariationalQuantumGate_CRZ &>())
        .def("feed", feed_vqg_crz_no_ptr);



    GET_FEED_PTR_NO_OFFSET(feed_vqg_cnot_no_ptr, Var::VariationalQuantumGate_CNOT);

    py::class_<Var::VariationalQuantumGate_CNOT, Var::VariationalQuantumGate>
        (m, "VariationalQuantumGate_CNOT")
        .def(py::init<QPanda::Qubit*, QPanda::Qubit*>())
        .def("feed", feed_vqg_cnot_no_ptr);

    GET_FEED_PTR_NO_OFFSET(feed_vqg_cz_no_ptr, Var::VariationalQuantumGate_CZ);

    py::class_<Var::VariationalQuantumGate_CZ, Var::VariationalQuantumGate>
        (m, "VariationalQuantumGate_CZ")
        .def(py::init<QPanda::Qubit*, QPanda::Qubit*>())
        .def("feed", feed_vqg_cz_no_ptr);

    QCircuit(Var::VariationalQuantumCircuit::*feed_vqc_with_ptr)
        (const std::vector<std::tuple<weak_ptr<Var::VariationalQuantumGate>,
            size_t, double>>) const
        = &Var::VariationalQuantumCircuit::feed;

    QCircuit(Var::VariationalQuantumCircuit::*feed_vqc_no_ptr)() const
        = &Var::VariationalQuantumCircuit::feed;

    Var::VariationalQuantumCircuit& (Var::VariationalQuantumCircuit::*insert_vqc_vqc)
        (Var::VariationalQuantumCircuit)
        = &Var::VariationalQuantumCircuit::insert;

    Var::VariationalQuantumCircuit& (Var::VariationalQuantumCircuit::*insert_vqc_qc)
        (QCircuit) = &Var::VariationalQuantumCircuit::insert;

    Var::VariationalQuantumCircuit& (Var::VariationalQuantumCircuit::*insert_vqc_qg)
        (QGate&) = &Var::VariationalQuantumCircuit::insert;



    py::class_<Var::VariationalQuantumCircuit>
        (m, "VariationalQuantumCircuit")
        .def(py::init<>())
        .def(py::init<QCircuit>())
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_H>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_X>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_RX>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_RY>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_RZ>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_CNOT>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_CZ>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_CRX>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_CRY>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_CRZ>, py::return_value_policy::reference)
        .def("insert", insert_vqc_vqc, py::return_value_policy::reference)
        .def("insert", insert_vqc_qc, py::return_value_policy::reference)
        .def("insert", insert_vqc_qg, py::return_value_policy::reference)
        .def("feed", feed_vqc_no_ptr)
        .def("feed", feed_vqc_with_ptr);

    py::class_<Var::expression>(m, "expression")
        .def(py::init<Var::var>())
        .def("find_leaves", &Var::expression::findLeaves)
        .def("find_non_consts", (std::unordered_set<Var::var>(Var::expression::*)(const std::vector<Var::var>&)) &Var::expression::findNonConsts)
        .def("propagate", (MatrixXd(Var::expression::*)()) &Var::expression::propagate)
        .def("propagate", (MatrixXd(Var::expression::*)(const std::vector<Var::var>&))
            &Var::expression::propagate)
        .def("backprop", (void(Var::expression::*)(std::unordered_map<Var::var, MatrixXd>&))
            &Var::expression::backpropagate)
        .def("backprop", (void(Var::expression::*)(std::unordered_map<Var::var, MatrixXd>&,
            const std::unordered_set<Var::var>&))&Var::expression::backpropagate)
        .def("get_root", &Var::expression::getRoot);
    m.def("eval", Var::eval);
    m.def("eval", [](Var::var v) {return eval(v, true); });

    m.def("_back", [](Var::expression& exp,
        std::unordered_map<Var::var, MatrixXd>& derivatives,
        const std::unordered_set<Var::var>& leaves) {
        Var::back(exp, derivatives, leaves);
        return derivatives;
    });
    m.def("_back", [](Var::expression& exp,
        std::unordered_map<Var::var, MatrixXd>& derivatives) {
        Var::back(exp, derivatives);
        return derivatives;
    });
    m.def("_back", [](const Var::var& v,
        std::unordered_map<Var::var, MatrixXd>& derivatives) {
        Var::back(v, derivatives);
        return derivatives;
    });
    m.def("_back", [](const Var::var& v,
        std::unordered_map<Var::var, MatrixXd>& derivatives,
        const std::unordered_set<Var::var>& leaves) {
        Var::back(v, derivatives, leaves);
        return derivatives;
    });

    m.def("exp", Var::exp);
    m.def("log", Var::log);
    m.def("poly", Var::poly);
    m.def("dot", Var::dot);
    m.def("inverse", Var::inverse);
    m.def("transpose", Var::transpose);
    m.def("sum", Var::sum);
    m.def("stack", [](int axis, py::args args) {
        std::vector<Var::var> vars;
        for (auto arg : args)
        {
            vars.push_back(py::cast<Var::var>(arg));
        }
        return py_stack(axis, vars);
    });
    const Var::var(*qop_plain)(Var::VariationalQuantumCircuit&,
        QPanda::PauliOperator,
        QPanda::QuantumMachine*,
        std::vector<Qubit*>) = Var::qop;

    const Var::var(*qop_map)(Var::VariationalQuantumCircuit&,
        QPanda::PauliOperator,
        QPanda::QuantumMachine*,
        std::map<size_t, Qubit*>) = Var::qop;
    m.def("qop", qop_plain,"VariationalQuantumCircuit"_a,"Hamiltonian"_a,"QuantumMachine"_a,"qubitList"_a);
    m.def("qop", qop_map, "VariationalQuantumCircuit"_a, "Hamiltonian"_a, "QuantumMachine"_a, "qubitList"_a);
	m.def("qop_pmeasure", Var::qop_pmeasure);
    py::implicitly_convertible<double, Var::var>();
    py::implicitly_convertible<ClassicalCondition, ClassicalProg>();
    py::implicitly_convertible<cbit_size_t, ClassicalCondition>();

	py::class_<Var::Optimizer>(m, "Optimizer")
		.def("get_variables", &Var::Optimizer::get_variables)
		.def("get_loss", &Var::Optimizer::get_loss)
		.def("run", &Var::Optimizer::run);
	

	py::enum_<Var::OptimizerMode>(m, "OptimizerMode");

	py::class_<Var::VanillaGradientDescentOptimizer, std::shared_ptr<Var::VanillaGradientDescentOptimizer>>
		(m, "VanillaGradientDescentOptimizer")
		.def(py::init<>([](
			Var::var lost_function,
			double learning_rate = 0.01,
			double stop_condition = 1.e-6,
			Var::OptimizerMode mode = Var::OptimizerMode::MINIMIZE) {
		return Var::VanillaGradientDescentOptimizer(lost_function,
			learning_rate, stop_condition, mode);
	}))
		.def("minimize", &Var::VanillaGradientDescentOptimizer::minimize)
		.def("get_variables", &Var::VanillaGradientDescentOptimizer::get_variables)
		.def("get_loss", &Var::VanillaGradientDescentOptimizer::get_loss)
		.def("run", &Var::VanillaGradientDescentOptimizer::run);
		

	py::class_<Var::MomentumOptimizer, std::shared_ptr<Var::MomentumOptimizer>>
		(m, "MomentumOptimizer")
		.def(py::init<>([](
			Var::var lost,
			double learning_rate = 0.01,
			double momentum = 0.9) {
		return Var::MomentumOptimizer(lost,
			learning_rate, momentum);
			}))
		.def("minimize", &Var::MomentumOptimizer::minimize)
		.def("get_variables", &Var::MomentumOptimizer::get_variables)
		.def("get_loss", &Var::MomentumOptimizer::get_loss)
		.def("run", &Var::MomentumOptimizer::run);

	py::class_<Var::AdaGradOptimizer, std::shared_ptr<Var::AdaGradOptimizer>>
		(m, "AdaGradOptimizer")
		.def(py::init<>([](
			Var::var lost,
			double learning_rate = 0.01,
			double initial_accumulator_value = 0.0,
			double epsilon = 1e-10) {
		return Var::AdaGradOptimizer(lost,
			learning_rate, initial_accumulator_value, epsilon);
		}))
		.def("minimize", &Var::AdaGradOptimizer::minimize)
		.def("get_variables", &Var::AdaGradOptimizer::get_variables)
		.def("get_loss", &Var::AdaGradOptimizer::get_loss)
		.def("run", &Var::AdaGradOptimizer::run);

	py::class_<Var::RMSPropOptimizer, std::shared_ptr<Var::RMSPropOptimizer>>
		(m, "RMSPropOptimizer")
		.def(py::init<>([](
			Var::var lost,
			double learning_rate = 0.001,
			double decay = 0.9,
			double epsilon = 1e-10) {
		return Var::RMSPropOptimizer(lost,
			learning_rate, decay, epsilon);
			}))
		.def("minimize", &Var::RMSPropOptimizer::minimize)
		.def("get_variables", &Var::RMSPropOptimizer::get_variables)
		.def("get_loss", &Var::RMSPropOptimizer::get_loss)
		.def("run", &Var::RMSPropOptimizer::run);


	py::class_<Var::AdamOptimizer, std::shared_ptr<Var::AdamOptimizer>>
		(m, "AdamOptimizer")
		.def(py::init<>([](
			Var::var lost,
			double learning_rate = 0.001,
			double beta1 = 0.9,
			double beta2 = 0.999,
			double epsilon = 1e-8) {
		return Var::AdamOptimizer(lost,
			learning_rate, beta1, beta2, epsilon);
			}))
		.def("minimize", &Var::AdamOptimizer::minimize)
		.def("get_variables", &Var::AdamOptimizer::get_variables)
		.def("get_loss", &Var::AdamOptimizer::get_loss)
		.def("run", &Var::AdamOptimizer::run);

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

#ifdef USE_CURL
    py::class_<QCloudMachine, QuantumMachine>(m, "QCloud")
        .def(py::init<>())
        .def("initQVM", &QCloudMachine::init, "init quantum virtual machine")
        .def("finalize", &QCloudMachine::finalize, "finalize")
        .def("qAlloc", qalloc, "Allocate a qubit", py::return_value_policy::reference)
        .def("qAlloc_many", qallocMany, "Allocate a list of qubits", "n_qubit"_a,
            py::return_value_policy::reference)
        .def("cAlloc", calloc, "Allocate a cbit", py::return_value_policy::reference)
        .def("cAlloc_many", callocMany, "Allocate a list of cbits", "n_cbit"_a,
            py::return_value_policy::reference)
        .def("qFree", free_qubit, "Free a qubit")
        .def("qFree_all", free_qubits, "qubit_list"_a,
            "Free a list of qubits")
        .def("cFree", free_cbit, "Free a cbit")
        .def("cFree_all", free_cbits, "cbit_list"_a,
            "Free a list of cbits")
        .def("getAllocateQubitNum", get_allocate_qubit, "getAllocateQubitNum", py::return_value_policy::reference)
        .def("getAllocateCMem", get_allocate_CMem, "getAllocateCMem", py::return_value_policy::reference)
        .def("run_with_configuration", [](QCloudMachine &qcm, QProg & prog, py::dict param)
            {
                py::object json = py::module::import("json");
                py::object dumps = json.attr("dumps");
                auto json_string = std::string(py::str(dumps(param)));
                rapidjson::Document doc;
                auto &alloc = doc.GetAllocator();
                doc.Parse(json_string.c_str());
                return qcm.runWithConfiguration(prog, doc);
            })
        .def("prob_run_dict", [](QCloudMachine &qcm, QProg & prog, QVec qvec, py::dict param)
            {
                py::object json = py::module::import("json");
                py::object dumps = json.attr("dumps");
                auto json_string = std::string(py::str(dumps(param)));
                rapidjson::Document doc;
                auto &alloc = doc.GetAllocator();
                doc.Parse(json_string.c_str());
                return qcm.probRunDict(prog, qvec, doc);
            })
        .def("get_result", [](QCloudMachine &qcm, std::string tasdid)
        {
            return qcm.getResult(tasdid);
        });
#endif // USE_CURL
}
