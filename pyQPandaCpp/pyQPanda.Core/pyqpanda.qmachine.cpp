#include <math.h>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/chrono.h"
#include "Core/Core.h"
#include "Core/QuantumMachine/SingleAmplitudeQVM.h"
#include "Core/QuantumMachine/PartialAmplitudeQVM.h"
#include "Core/QuantumMachine/QCloudMachine.h"
#include "Core/VirtualQuantumProcessor/NoiseQPU/NoiseModel.h"
USING_QPANDA
using namespace std;
using namespace pybind11::literals;
using std::map;
namespace py = pybind11;

void init_quantum_machine(py::module &m)
{
    py::enum_<QMachineType>(m, "QMachineType")
        .value("CPU", QMachineType::CPU)
        .value("GPU", QMachineType::GPU)
        .value("CPU_SINGLE_THREAD", QMachineType::CPU_SINGLE_THREAD)
        .value("NOISE", QMachineType::NOISE)
        .export_values();

	py::enum_<IBMQBackends>(m, "IBMQBackends")
		.value("IBMQ_QASM_SIMULATOR", IBMQBackends::IBMQ_QASM_SIMULATOR)
		.value("IBMQ_16_MELBOURNE", IBMQBackends::IBMQ_16_MELBOURNE)
		.value("IBMQX2", IBMQBackends::IBMQX2)
		.value("IBMQX4", IBMQBackends::IBMQX4)
		.export_values();

    py::enum_<NOISE_MODEL>(m, "NoiseModel")
        .value("DAMPING_KRAUS_OPERATOR", NOISE_MODEL::DAMPING_KRAUS_OPERATOR)
        .value("DECOHERENCE_KRAUS_OPERATOR", NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR)
        .value("DEPHASING_KRAUS_OPERATOR", NOISE_MODEL::DEPHASING_KRAUS_OPERATOR)
        .value("PAULI_KRAUS_MAP", NOISE_MODEL::PAULI_KRAUS_MAP)
        .value("DECOHERENCE_KRAUS_OPERATOR_P1_P2", NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR_P1_P2)
        .value("BITFLIP_KRAUS_OPERATOR", NOISE_MODEL::BITFLIP_KRAUS_OPERATOR)
        .value("DEPOLARIZING_KRAUS_OPERATOR", NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR)
        .value("BIT_PHASE_FLIP_OPRATOR", NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR)
        .value("PHASE_DAMPING_OPRATOR", NOISE_MODEL::PHASE_DAMPING_OPRATOR)
        ;

    Qubit*(QVec::*qvec_subscript_cbit_size_t)(size_t) = &QVec::operator[];
    Qubit*(QVec::*qvec_subscript_cc)(ClassicalCondition&) = &QVec::operator[];

    py::class_<QVec>(m, "QVec")
        .def(py::init<>())
        .def(py::init<std::vector<Qubit *> &>())
        .def(py::init<const QVec &>())
        .def("__getitem__", [](QVec & self,int num) {
        return self[num];
    }, py::return_value_policy::reference)
        .def("__getitem__", qvec_subscript_cc, py::return_value_policy::reference)
        .def("__len__", &QVec::size)
        .def("append", [](QVec & self, Qubit * qubit) {
        self.push_back(qubit);
    })
        .def("pop", [](QVec & self) {
        self.pop_back();
    });

    py::implicitly_convertible<std::vector<Qubit *>, QVec>();

    Qubit*(QuantumMachine::*qalloc)() = &QuantumMachine::allocateQubit;
    ClassicalCondition(QuantumMachine::*cAlloc)() = &QuantumMachine::allocateCBit;
    ClassicalCondition(QuantumMachine::*calloc_by_address)(size_t) = &QuantumMachine::allocateCBit;
    QVec(QuantumMachine::*qallocMany)(size_t) = &QuantumMachine::allocateQubits;
    vector<ClassicalCondition>(QuantumMachine::*callocMany)(size_t) = &QuantumMachine::allocateCBits;
    void (QuantumMachine::*free_qubit)(Qubit *) = &QuantumMachine::Free_Qubit;
    void (QuantumMachine::*free_qubits)(QVec&) = &QuantumMachine::Free_Qubits;
    void (QuantumMachine::*free_cbit)(ClassicalCondition &) = &QuantumMachine::Free_CBit;
    void (QuantumMachine::*free_cbits)(vector<ClassicalCondition> &) = &QuantumMachine::Free_CBits;
    QMachineStatus * (QuantumMachine::*get_status)() const = &QuantumMachine::getStatus;
    size_t(QuantumMachine::*get_allocate_qubit)() = &QuantumMachine::getAllocateQubit;
    size_t(QuantumMachine::*get_allocate_CMem)() = &QuantumMachine::getAllocateCMem;
    void (QuantumMachine::*_finalize)() = &QuantumMachine::finalize;
    Qubit* (QuantumMachine::*allocate_qubit_through_phy_address)(size_t) = &QuantumMachine::allocateQubitThroughPhyAddress;
    Qubit* (QuantumMachine::*allocate_qubit_through_vir_address)(size_t) = &QuantumMachine::allocateQubitThroughVirAddress;
    std::map<GateType, size_t>(QuantumMachine::*get_gate_time_map)() const = &QuantumMachine::getGateTimeMap;
    bool(QuantumMachine::*swap_qubit_physical_address)(Qubit*, Qubit*) = &QuantumMachine::swapQubitPhysicalAddress;

    
    py::class_<QuantumMachine>(m, "QuantumMachine")
		.def("set_configure", [](QuantumMachine &qvm, size_t max_qubit, size_t max_cbit) 
            {
		        Configuration config = { max_qubit, max_cbit };
		        qvm.setConfig(config);
	        }, "set QVM max qubit and max cbit", "max_qubit"_a, "max_cbit"_a)
        .def("finalize", &QuantumMachine::finalize, "finalize")
        .def("get_qstate", &QuantumMachine::getQState, "getState", py::return_value_policy::automatic)
        .def("qAlloc", qalloc, "Allocate a qubit", py::return_value_policy::reference)
        .def("qAlloc_many", [](QuantumMachine & self,size_t num) 
            {
                vector<Qubit *> temp = self.allocateQubits(num);
                return temp;
            }, "Allocate a list of qubits", "n_qubit"_a,
            py::return_value_policy::reference)
        .def("cAlloc", cAlloc, "Allocate a cbit", py::return_value_policy::reference)
        .def("cAlloc_many", callocMany, "Allocate a list of cbits", "n_cbit"_a,
            py::return_value_policy::reference)
        .def("qFree", free_qubit, "Free a qubit")
        .def("qFree_all", free_qubits, "qubit_list"_a,
            "Free a list of qubits")
        .def("cFree", free_cbit, "Free a cbit")
        .def("cFree_all", free_cbits, "cbit_list"_a,
            "Free a list of cbits")

        /*will delete*/
        .def("getAllocateQubitNum", get_allocate_qubit, "getAllocateQubitNum", py::return_value_policy::reference)
        .def("getAllocateCMem", get_allocate_CMem, "getAllocateCMem", py::return_value_policy::reference)
        .def("getStatus", get_status, "get the status(ptr) of the quantum machine",
            py::return_value_policy::reference)

        /* new interface */
        .def("cAlloc", calloc_by_address, "Allocate a cbit", "cbit"_a, py::return_value_policy::reference)
        .def("get_status", get_status, "get the status(ptr) of the quantum machine",py::return_value_policy::reference)
        .def("get_allocate_qubit_num", get_allocate_qubit, "getAllocateQubitNum", py::return_value_policy::reference)\
        .def("get_allocate_cmem_num", get_allocate_CMem, "getAllocateCMem", py::return_value_policy::reference)\
        .def("allocate_qubit_through_phy_address", allocate_qubit_through_phy_address, "address"_a, py::return_value_policy::reference)
        .def("allocate_qubit_through_vir_address", allocate_qubit_through_vir_address, "address"_a, py::return_value_policy::reference)
        .def("get_gate_time_map", get_gate_time_map, py::return_value_policy::reference)
        .def("swap_qubit_physical_address", swap_qubit_physical_address, "qubit"_a, "qubit"_a, py::return_value_policy::reference)

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
		.def("set_configure", [](class_name &qvm, size_t max_qubit, size_t max_cbit) { \
	    Configuration config = { max_qubit, max_cbit }; \
	    qvm.setConfig(config); \
	}, "set QVM max qubit and max cbit", "max_qubit"_a, "max_cbit"_a) \
        .def(py::init<>())\
        .def("finalize", &class_name::finalize, "finalize")\
        .def("get_qstate", &class_name::getQState, "getState",py::return_value_policy::automatic)\
        .def("qAlloc", qalloc, "Allocate a qubit", py::return_value_policy::reference)\
        .def("qAlloc_many", [](QuantumMachine & self,size_t num) { \
        vector<Qubit *> temp = self.allocateQubits(num);\
        return temp;\
        }, "Allocate a list of qubits", "n_qubit"_a,\
            py::return_value_policy::reference)\
        .def("cAlloc", cAlloc, "Allocate a cbit", py::return_value_policy::reference)\
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
        /*will delete*/\
        .def("initQVM", &class_name::init, "init quantum virtual machine")\
        .def("getAllocateQubitNum", get_allocate_qubit, "getAllocateQubitNum", py::return_value_policy::reference)\
        .def("getAllocateCMem", get_allocate_CMem, "getAllocateCMem", py::return_value_policy::reference)\
         /* new interface */\
        .def("init_qvm", &class_name::init, "init quantum virtual machine")\
        .def("get_allocate_qubit_num", get_allocate_qubit, "getAllocateQubitNum", py::return_value_policy::reference)\
        .def("get_allocate_cmem_num", get_allocate_CMem, "getAllocateCMem", py::return_value_policy::reference)\
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

#ifdef USE_CUDA
    DEFINE_IDEAL_QVM(GPUQVM);
#endif // USE_CUDA

    py::class_<NoiseQVM, QuantumMachine>(m, "NoiseQVM")
		.def("set_configure", [](NoiseQVM &qvm, size_t max_qubit, size_t max_cbit) {
		Configuration config = { max_qubit, max_cbit };
		qvm.setConfig(config);
	}, "set QVM max qubit and max cbit", "max_qubit"_a, "max_cbit"_a)
        .def(py::init<>())
        .def("finalize", &NoiseQVM::finalize, "finalize")
        .def("get_qstate", &NoiseQVM::getQState, "getState", py::return_value_policy::automatic)
        .def("qAlloc", qalloc, "Allocate a qubit", py::return_value_policy::reference)
        .def("qAlloc_many", [](QuantumMachine & self, size_t num) {
                    vector<Qubit *> temp = self.allocateQubits(num);
                    return temp;
                }, "Allocate a list of qubits", "n_qubit"_a,
            py::return_value_policy::reference)
        .def("cAlloc", cAlloc, "Allocate a cbit", py::return_value_policy::reference)
        .def("cAlloc_many", callocMany, "Allocate a list of cbits", "n_cbit"_a,
            py::return_value_policy::reference)
        .def("cAlloc", calloc_by_address, "Allocate a cbit", "cbit"_a, py::return_value_policy::reference)
        .def("allocate_qubit_through_phy_address", allocate_qubit_through_phy_address, "address"_a, py::return_value_policy::reference)
        .def("allocate_qubit_through_vir_address", allocate_qubit_through_vir_address, "address"_a, py::return_value_policy::reference)
        .def("qFree", free_qubit, "Free a qubit")
        .def("qFree_all", free_qubits, "qubit_list"_a,
            "Free a list of qubits")
        .def("cFree", free_cbit, "Free a cbit")
        .def("cFree_all", free_cbits, "cbit_list"_a,
            "Free a list of cbits")
        .def("getStatus", get_status, "get the status(ptr) of the quantum machine",
            py::return_value_policy::reference)

        /*will delete*/
        .def("initQVM", [](NoiseQVM& qvm, py::dict param) {
        py::object json = py::module::import("json");
        py::object dumps = json.attr("dumps");
        auto json_string = std::string(py::str(dumps(param)));
        rapidjson::Document doc(rapidjson::kObjectType);
        doc.Parse(json_string.c_str());
        qvm.init(doc);
        }, "init quantum virtual machine")
        .def("getAllocateQubitNum", get_allocate_qubit, "getAllocateQubitNum", py::return_value_policy::reference)
        .def("getAllocateCMem", get_allocate_CMem, "getAllocateCMem", py::return_value_policy::reference)

        /* new interface */
            .def("init_qvm", [](NoiseQVM& qvm, py::dict param) {
            py::object json = py::module::import("json");
            py::object dumps = json.attr("dumps");
            auto json_string = std::string(py::str(dumps(param)));
            rapidjson::Document doc(rapidjson::kObjectType);
            doc.Parse(json_string.c_str());
            qvm.init(doc);
        }, "init quantum virtual machine")
        .def("get_allocate_qubit_num", get_allocate_qubit, "getAllocateQubitNum", py::return_value_policy::reference)
        .def("get_allocate_cmem_num", get_allocate_CMem, "getAllocateCMem", py::return_value_policy::reference)

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


#define PMEASURE_BIN_INDEX(QVM_CLASS,NODE)\
.def("pmeasure_bin_index", [](QVM_CLASS &qvm,NODE node,string bin_index)\
{return qvm.PMeasure_bin_index(node, bin_index); })

#define PMEASURE_DEC_INDEX(QVM_CLASS,NODE)\
.def("pmeasure_dec_index", [](QVM_CLASS &qvm,NODE node,string bin_index)\
{return qvm.PMeasure_dec_index(node, bin_index); })

#define QVM_RUN(QVM_CLASS,NODE)\
.def("run", [](QVM_CLASS &qvm, NODE prog) {return qvm.run(prog); }, "load the quantum program")

    py::class_<SingleAmplitudeQVM, QuantumMachine>(m, "SingleAmpQVM")
		.def("set_configure", [](SingleAmplitudeQVM &qvm, size_t max_qubit, size_t max_cbit) {
		Configuration config = { max_qubit, max_cbit };
		qvm.setConfig(config);
	}, "set QVM max qubit and max cbit", "max_qubit"_a, "max_cbit"_a)
        .def(py::init<>())
        .def("finalize", &SingleAmplitudeQVM::finalize, "finalize")
        .def("qAlloc", qalloc, "Allocate a qubit", py::return_value_policy::reference)
        .def("qAlloc_many", [](QuantumMachine & self, size_t num) {
        vector<Qubit *> temp = self.allocateQubits(num);
        return temp;
    }, "Allocate a list of qubits", "n_qubit"_a,
            py::return_value_policy::reference)
        .def("cAlloc", cAlloc, "Allocate a cbit", py::return_value_policy::reference)
        .def("cAlloc_many", callocMany, "Allocate a list of cbits", "n_cbit"_a,
            py::return_value_policy::reference)
        .def("qFree", free_qubit, "Free a qubit")
        .def("qFree_all", free_qubits, "qubit_list"_a,
            "Free a list of qubits")
        .def("cFree", free_cbit, "Free a cbit")
        .def("cFree_all", free_cbits, "cbit_list"_a,
            "Free a list of cbits")

        /*will delete*/
        .def("initQVM", &SingleAmplitudeQVM::init, "init quantum virtual machine")

        /* new interface */
        .def("init_qvm", &SingleAmplitudeQVM::init, "init quantum virtual machine")

        QVM_RUN(SingleAmplitudeQVM, QProg)
        QVM_RUN(SingleAmplitudeQVM, QCircuit) 
        QVM_RUN(SingleAmplitudeQVM, QGate)
        .def("run", [](SingleAmplitudeQVM &qvm, std::string QRunes_file) {return qvm.run(QRunes_file); }, "load and parser the quantum program")

        .def("get_qstate", &SingleAmplitudeQVM::getQState, "Get the quantum state of quantum program",
            py::return_value_policy::automatic_reference)

        PMEASURE_BIN_INDEX(SingleAmplitudeQVM, QProg)
        PMEASURE_BIN_INDEX(SingleAmplitudeQVM, QCircuit)
        PMEASURE_BIN_INDEX(SingleAmplitudeQVM, QGate)

        PMEASURE_DEC_INDEX(SingleAmplitudeQVM, QProg)
        PMEASURE_DEC_INDEX(SingleAmplitudeQVM, QCircuit)
        PMEASURE_DEC_INDEX(SingleAmplitudeQVM, QGate)

        .def("pmeasure", [](SingleAmplitudeQVM &qvm, QVec qvec, std::string select_max)
            {return qvm.PMeasure(qvec, select_max); })
        .def("pmeasure", [](SingleAmplitudeQVM &qvm, std::string select_max)
            {return qvm.PMeasure(select_max); })
        .def("get_prob_dict", [](SingleAmplitudeQVM &qvm, QVec qvec, std::string select_max)
            {return qvm.getProbDict(qvec, select_max); })
        .def("prob_run_dict", [](SingleAmplitudeQVM &qvm, QProg prog, QVec qvec, std::string select_max)
            {return qvm.probRunDict(prog, qvec, select_max); });

py::class_<PartialAmplitudeQVM, QuantumMachine>(m, "PartialAmpQVM")
		.def("set_configure", [](PartialAmplitudeQVM &qvm, size_t max_qubit, size_t max_cbit) {
		Configuration config = { max_qubit, max_cbit };
		qvm.setConfig(config);
	}, "set QVM max qubit and max cbit", "max_qubit"_a, "max_cbit"_a)
        .def(py::init<>())
        .def("finalize", &PartialAmplitudeQVM::finalize, "finalize")
        .def("qAlloc", qalloc, "Allocate a qubit", py::return_value_policy::reference)
        .def("qAlloc_many", [](QuantumMachine & self, size_t num) {
        vector<Qubit *> temp = self.allocateQubits(num);
        return temp;
    }, "Allocate a list of qubits", "n_qubit"_a,
            py::return_value_policy::reference)
        .def("cAlloc", cAlloc, "Allocate a cbit", py::return_value_policy::reference)
        .def("cAlloc_many", callocMany, "Allocate a list of cbits", "n_cbit"_a,
            py::return_value_policy::reference)
        .def("qFree", free_qubit, "Free a qubit")
        .def("qFree_all", free_qubits, "qubit_list"_a,
            "Free a list of qubits")
        .def("cFree", free_cbit, "Free a cbit")
        .def("cFree_all", free_cbits, "cbit_list"_a,
            "Free a list of cbits")

        /*will delete*/
        .def("initQVM", &PartialAmplitudeQVM::init, "init quantum virtual machine")

        /* new interface */
        .def("init_qvm", &PartialAmplitudeQVM::init, "init quantum virtual machine")

        QVM_RUN(PartialAmplitudeQVM, QProg)
        QVM_RUN(PartialAmplitudeQVM, QCircuit)
        QVM_RUN(PartialAmplitudeQVM, QGate)

        .def("run", [](PartialAmplitudeQVM &qvm, std::string QRunes_file) {return qvm.run(QRunes_file); }, "load and parser the quantum program")

        .def("get_qstate", &PartialAmplitudeQVM::getQState, "Get the quantum state of quantum program",
            py::return_value_policy::automatic_reference)

        .def("pmeasure", [](PartialAmplitudeQVM &qvm, QVec qvec, std::string select_max)
    {return qvm.PMeasure(qvec, select_max); })
        .def("pmeasure", [](PartialAmplitudeQVM &qvm, std::string select_max)
    {return qvm.PMeasure(select_max); })

        .def("pmeasure_bin_index", &PartialAmplitudeQVM::PMeasure_bin_index, "bin_index"_a,
            "PMeasure_bin_index", py::return_value_policy::automatic_reference)
        .def("pmeasure_dec_index", &PartialAmplitudeQVM::PMeasure_dec_index, "dec_index"_a,
            "PMeasure_dec_index", py::return_value_policy::automatic_reference)
        .def("get_prob_dict", [](PartialAmplitudeQVM &qvm, QVec qvec, std::string select_max)
            {return qvm.getProbDict(qvec, select_max); })
        .def("prob_run_dict", [](PartialAmplitudeQVM &qvm, QProg prog, QVec qvec, std::string select_max)
            {return qvm.probRunDict(prog, qvec, select_max); })
        .def("pmeasure_subset", [](PartialAmplitudeQVM &qvm, QProg &prog, std::vector<std::string> index_vec)
            {return qvm.pMeasureSubset(prog, index_vec); });

#ifdef USE_CURL
    py::class_<QCloudMachine, QuantumMachine>(m, "QCloud")
        .def(py::init<>())
        .def("finalize", &QCloudMachine::finalize, "finalize")
        .def("qAlloc", qalloc, "Allocate a qubit", py::return_value_policy::reference)
        .def("qAlloc_many", [](QuantumMachine & self, size_t num) 
            {
                vector<Qubit *> temp = self.allocateQubits(num);
                return temp;
            }, "Allocate a list of qubits", "n_qubit"_a,
            py::return_value_policy::reference)
        .def("cAlloc", cAlloc, "Allocate a cbit", py::return_value_policy::reference)
        .def("cAlloc_many", callocMany, "Allocate a list of cbits", "n_cbit"_a,
            py::return_value_policy::reference)
        .def("qFree", free_qubit, "Free a qubit")
        .def("qFree_all", free_qubits, "qubit_list"_a,
            "Free a list of qubits")
        .def("cFree", free_cbit, "Free a cbit")
        .def("cFree_all", free_cbits, "cbit_list"_a,
            "Free a list of cbits")
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
             })
        /*will delete*/
        .def("initQVM", &QCloudMachine::init, "init quantum virtual machine")
        /* new interface */
        .def("init_qvm", &QCloudMachine::init, "init quantum virtual machine");
#endif // USE_CURL

		py::implicitly_convertible<CPUQVM, QuantumMachine>();
		py::implicitly_convertible<GPUQVM, QuantumMachine>();
		py::implicitly_convertible<CPUSingleThreadQVM, QuantumMachine>();
		py::implicitly_convertible<NoiseQVM, QuantumMachine>();
		py::implicitly_convertible<SingleAmplitudeQVM, QuantumMachine>();
		py::implicitly_convertible<PartialAmplitudeQVM, QuantumMachine>();

}
