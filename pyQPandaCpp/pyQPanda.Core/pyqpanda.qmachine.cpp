#include <map>
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
#include "template_generator.h"
USING_QPANDA
namespace py = pybind11;
using namespace pybind11::literals;

// template<>
// struct py::detail::type_caster<QVec>  : py::detail::list_caster<QVec, Qubit*> {};

void export_quantum_machine(py::module &m)
{
     py::class_<QuantumMachine>(m, "QuantumMachine")
         .def(
             "set_configure",
             [](QuantumMachine &qvm, size_t max_qubit, size_t max_cbit)
             {
                  Configuration config = {max_qubit, max_cbit};
                  qvm.setConfig(config);
             },
             py::arg("max_qubit"),
             py::arg("max_cbit"),
             "set QVM max qubit and max cbit")
         .def("finalize", &QuantumMachine::finalize, "finalize")
         .def("get_qstate", &QuantumMachine::getQState, "getState", py::return_value_policy::automatic)
         .def("qAlloc", &QuantumMachine::allocateQubit, "Allocate a qubit", py::return_value_policy::reference)
         .def("qAlloc_many",
              &QuantumMachine::allocateQubits,
              py::arg("qubit_num"),
              "Allocate a list of qubits",
              py::return_value_policy::reference)
         .def("cAlloc",
              py::overload_cast<>(&QuantumMachine::allocateCBit),
              "Allocate a cbit",
              py::return_value_policy::reference)
         .def("cAlloc_many",
              &QuantumMachine::allocateCBits,
              py::arg("cbit_num"),
              "Allocate a list of cbits",
              py::return_value_policy::reference)
         .def("qFree",
              &QuantumMachine::Free_Qubit,
              py::arg("qubit"),
              "Free a qubit")
         .def("qFree_all",
              &QuantumMachine::Free_Qubits,
              py::arg("qubit_list"),
              "Free a list of qubits")
         .def("qFree_all", py::overload_cast<QVec &>(&QuantumMachine::qFreeAll), "Free all of qubits")

         .def("cFree", &QuantumMachine::Free_CBit, "Free a cbit")
         .def("cFree_all",
              &QuantumMachine::Free_CBits,
              py::arg("cbit_list"),
              "Free a list of cbits")
         .def("cFree_all", py::overload_cast<>(&QuantumMachine::cFreeAll), "Free all of cbits")
         .def("getStatus", &QuantumMachine::getStatus, "get the status(ptr) of the quantum machine", py::return_value_policy::reference_internal)

         /*will delete*/
         .def("initQVM", &QuantumMachine::init, "init quantum virtual machine")
         .def("getAllocateQubitNum", &QuantumMachine::getAllocateQubit, "getAllocateQubitNum", py::return_value_policy::reference)
         .def("getAllocateCMem", &QuantumMachine::getAllocateCMem, "getAllocateCMem", py::return_value_policy::reference)

         /* new interface */
         .def("init_qvm", &QuantumMachine::init, "init quantum virtual machine")
         .def("init_state",
              &QuantumMachine::initState,
              py::arg_v("state", QStat(), "QStat()"),
              py::arg_v("qlist", QVec(), "QVec()"),
              py::return_value_policy::reference)
         .def("cAlloc",
              py::overload_cast<size_t>(&QuantumMachine::allocateCBit),
              py::arg("cbit"),
              "Allocate a cbit",
              py::return_value_policy::reference)
         .def("get_status", &QuantumMachine::getStatus, "get the status(ptr) of the quantum machine", py::return_value_policy::reference_internal)
         .def("get_allocate_qubit_num", &QuantumMachine::getAllocateQubit, "getAllocateQubitNum", py::return_value_policy::reference)
         .def("get_allocate_cmem_num", &QuantumMachine::getAllocateCMem, "getAllocateCMem", py::return_value_policy::reference)
         .def("allocate_qubit_through_phy_address",
              &QuantumMachine::allocateQubitThroughPhyAddress,
              py::arg("address"),
              py::return_value_policy::reference)
         .def("allocate_qubit_through_vir_address",
              &QuantumMachine::allocateQubitThroughVirAddress,
              py::arg("address"),
              py::return_value_policy::reference)
         .def("get_gate_time_map", &QuantumMachine::getGateTimeMap, py::return_value_policy::reference)
         //.def("get_allocate_qubits", get_allocate_qubits, "qubit vector"_a,  py::return_value_policy::reference)
         //.def("get_allocate_cbits", get_allocate_cbits, "cbit vector"_a,  py::return_value_policy::reference)

         .def(
             "get_allocate_qubits",
             [](QuantumMachine &self)
             {
                  QVec qv;
                  self.get_allocate_qubits(qv);
                  return qv;
             },
             "get allocate qubits of QuantumMachine",
             py::return_value_policy::reference)

         .def(
             "get_allocate_cbits",
             [](QuantumMachine &self)
             {
                  std::vector<ClassicalCondition> cv;
                  self.get_allocate_cbits(cv);
                  return cv;
             },
             "get allocate cbits of QuantumMachine",
             py::return_value_policy::reference)

         .def("get_expectation",
              py::overload_cast<QProg, const QHamiltonian &, const QVec &>(&QuantumMachine::get_expectation),
              py::arg("qprog"),
              py::arg("hamiltonian"),
              py::arg("qubit_list"),
              py::return_value_policy::reference)
         .def("get_expectation",
              py::overload_cast<QProg, const QHamiltonian &, const QVec &, int>(&QuantumMachine::get_expectation),
              py::arg("qprog"),
              py::arg("hamiltonian"),
              py::arg("qubit_list"),
              py::arg("shots"),
              py::return_value_policy::reference)
         .def("get_processed_qgate_num", &QuantumMachine::get_processed_qgate_num, py::return_value_policy::reference)
         .def("async_run",
              &QuantumMachine::async_run,
              py::arg("qprog"),
              py::arg_v("noise_model", NoiseModel(), "NoiseModel()"),
              "Run quantum prog asynchronously at background\n"
              "Use get_processed_qgate_num() to get check the asynchronous process progress\n"
              "Use is_async_finished() check whether asynchronous process finished\n"
              "Use get_async_result() block current code and get asynchronous process result unitll it finished",
              py::return_value_policy::reference)
         .def("is_async_finished", &QuantumMachine::is_async_finished, py::return_value_policy::reference)
         .def("get_async_result", &QuantumMachine::get_async_result, py::return_value_policy::reference)
         .def("directly_run",
              &QuantumMachine::directlyRun,
              py::arg("qprog"),
              py::arg_v("noise_model", NoiseModel(), "NoiseModel()"),
              py::return_value_policy::reference)
         .def(
             "run_with_configuration",
             [](QuantumMachine &qvm, QProg &prog, vector<ClassicalCondition> &cc_vector, py::dict param, NoiseModel noise_model = NoiseModel())
             {
                  py::object json = py::module::import("json");
                  py::object dumps = json.attr("dumps");
                  auto json_string = std::string(py::str(dumps(param)));
                  rapidjson::Document doc;
                  auto &alloc = doc.GetAllocator();
                  doc.Parse(json_string.c_str());
                  return qvm.runWithConfiguration(prog, cc_vector, doc, noise_model);
             },
             py::arg("qprog"),
             py::arg("cbit_list"),
             py::arg("data"),
             py::arg_v("noise_model", NoiseModel(), "NoiseModel()"),
             py::return_value_policy::automatic)

         .def("run_with_configuration",
              py::overload_cast<QProg &, vector<ClassicalCondition> &, int, const NoiseModel &>(&QuantumMachine::runWithConfiguration),
              py::arg("qprog"),
              py::arg("cbit_list"),
              py::arg("shot"),
              py::arg_v("noise_model", NoiseModel(), "NoiseModel()"),
              py::return_value_policy::automatic)
         .def("run_with_configuration",
              py::overload_cast<QProg &, vector<int> &, int, const NoiseModel &>(&QuantumMachine::runWithConfiguration),
              py::arg("qprog"),
              py::arg("cbit_list"),
              py::arg("shot"),
              py::arg_v("noise_model", NoiseModel(), "NoiseModel()"),
              py::return_value_policy::automatic);

     /*
       just inherit abstract base class wrappered by trampoline class, will implement C++ like polymorphism in python
       no neeed use another trampoline class wrapper child, unless children have derived children class
     */
     py::class_<CPUQVM, QuantumMachine> cpu_qvm(m, "CPUQVM");
     /*
       we should declare these function in py::class_<IdealQVM>, then CPUQVM inherit form it,
       but as we won't want to export IdealQVM to user, this may the only way
     */
     export_idealqvm_func<CPUQVM>::export_func(cpu_qvm);
     py::class_<CPUSingleThreadQVM, QuantumMachine> cpu_single_thread_qvm(m, "CPUSingleThreadQVM");
     export_idealqvm_func<CPUSingleThreadQVM>::export_func(cpu_single_thread_qvm);

#ifdef USE_CUDA
     py::class_<GPUQVM, QuantumMachine> gpu_qvm(m, "GPUQVM");
     export_idealqvm_func<GPUQVM>::export_func(gpu_qvm);
#endif // USE_CUDA

     py::class_<NoiseQVM, QuantumMachine>(m, "NoiseQVM")
         .def(py::init<>())
         .def("set_max_threads",
              &NoiseQVM::set_parallel_threads,
              py::arg("size"),
              "set NoiseQVM max thread size",
              py::return_value_policy::automatic)
         .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, double>(&NoiseQVM::set_noise_model))
         .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &, double>(&NoiseQVM::set_noise_model))
         .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, double, const QVec &>(&NoiseQVM::set_noise_model))
         .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &, double, const QVec &>(&NoiseQVM::set_noise_model))
         .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, double, const std::vector<QVec> &>(&NoiseQVM::set_noise_model))
         .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, double, double, double>(&NoiseQVM::set_noise_model))
         .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &, double, double, double>(&NoiseQVM::set_noise_model))
         .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, double, double, double, const QVec &>(&NoiseQVM::set_noise_model))
         .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &, double, double, double, const QVec &>(&NoiseQVM::set_noise_model))
         .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, double, double, double, const std::vector<QVec> &>(&NoiseQVM::set_noise_model))
         .def("set_measure_error",
              py::overload_cast<const NOISE_MODEL &, double, const QVec &>(&NoiseQVM::set_measure_error),
              py::arg("model"),
              py::arg("prob"),
              py::arg("qubits") = QVec())
         .def("set_measure_error",
              py::overload_cast<const NOISE_MODEL &, double, double, double, const QVec &>(&NoiseQVM::set_measure_error),
              py::arg("model"),
              py::arg("T1"),
              py::arg("T2"),
              py::arg("t_gate"),
              py::arg("qubits") = QVec())
         .def("set_mixed_unitary_error", py::overload_cast<const GateType &, const std::vector<QStat> &, const std::vector<double> &>(&NoiseQVM::set_mixed_unitary_error))
         .def("set_mixed_unitary_error", py::overload_cast<const GateType &, const std::vector<QStat> &, const std::vector<double> &, const QVec &>(&NoiseQVM::set_mixed_unitary_error))
         .def("set_mixed_unitary_error", py::overload_cast<const GateType &, const std::vector<QStat> &, const std::vector<double> &, const std::vector<QVec> &>(&NoiseQVM::set_mixed_unitary_error))
         .def("set_reset_error",
              &NoiseQVM::set_reset_error,
              py::arg("p0"),
              py::arg("p1"),
              py::arg("qubits") = QVec())

         .def("set_readout_error",
              &NoiseQVM::set_readout_error,
              py::arg("probs_list"),
              py::arg("qubits") = QVec())

         .def("set_rotation_error", &NoiseQVM::set_rotation_error)

         /*will delete*/
         .def(
             "initQVM",
             [](NoiseQVM &qvm, py::dict param)
             {
                  py::object json = py::module::import("json");
                  py::object dumps = json.attr("dumps");
                  auto json_string = std::string(py::str(dumps(param)));
                  rapidjson::Document doc(rapidjson::kObjectType);
                  doc.Parse(json_string.c_str());
                  qvm.init(doc);
             },
             "init quantum virtual machine")

         /* new interface */
         .def(
             "init_qvm",
             [](NoiseQVM &qvm, py::dict param)
             {
                  py::object json = py::module::import("json");
                  py::object dumps = json.attr("dumps");
                  auto json_string = std::string(py::str(dumps(param)));
                  rapidjson::Document doc(rapidjson::kObjectType);
                  doc.Parse(json_string.c_str());
                  qvm.init(doc);
             },
             py::arg("json_config"),
             "init quantum virtual machine")
         .def("init_qvm", py::overload_cast<>(&NoiseQVM::init), "init quantum virtual machine");
     // .def(
     //     "directly_run",
     //     [](NoiseQVM &qvm, QProg &prog)
     //     { qvm.directlyRun(prog); },
     //     "directly_run a prog", "program"_a, py::return_value_policy::reference)
     //     .def(
     //         "init_state",
     //         [](NoiseQVM &self, const QStat &state, const QVec &qlist)
     //         { self.initState(state, qlist); },
     //         py::arg("state") = QStat(),
     //         py::arg("qlist") = QVec(),
     //         py::return_value_policy::reference);

     // .def(
     //     "run_with_configuration",
     //      [](NoiseQVM &qvm, QProg &prog, vector<ClassicalCondition> &cc_vector, py::dict param)
     //     {
     // py::object json = py::module::import("json");
     // py::object dumps = json.attr("dumps");
     // auto json_string = std::string(py::str(dumps(param)));
     // rapidjson::Document doc;
     // auto &alloc = doc.GetAllocator();
     // doc.Parse(json_string.c_str());
     // return qvm.runWithConfiguration(prog, cc_vector, doc); },
     //     "program"_a, "cbit_list"_a, "data"_a, py::return_value_policy::automatic)
     // .def(
     //     "run_with_configuration", [](NoiseQVM &qvm, QProg &prog, vector<ClassicalCondition> &cc_vector, int shots)
     //     { return qvm.runWithConfiguration(prog, cc_vector, shots); },
     //     "program"_a, "cbit_list"_a, "data"_a, py::return_value_policy::automatic)
     // .def(
     //     "run_with_configuration", [](QuantumMachine &qvm, QProg &prog, vector<int> &cbit_addrs, int shots)
     //     { return qvm.runWithConfiguration(prog, cbit_addrs, shots); },
     //     "program"_a, "cbit_addr_list"_a, "data"_a, py::return_value_policy::automatic);

     py::class_<SingleAmplitudeQVM, QuantumMachine>(m, "SingleAmpQVM")
         .def(py::init<>())

         //     .def("init_qvm", &SingleAmplitudeQVM::init, "init quantum virtual machine")
         .def("run",
              py::overload_cast<QProg &, QVec &, size_t, size_t>(&SingleAmplitudeQVM::run),
              py::arg("prog"),
              py::arg("qv"),
              py::arg("max_rank") = 30,
              py::arg("alloted_time") = 5,
              "run the quantum program",
              py::return_value_policy::automatic_reference)
         .def("run",
              py::overload_cast<QProg &, QVec &, size_t, const std::vector<qprog_sequence_t> &>(&SingleAmplitudeQVM::run),
              "run the quantum program")
         .def("get_sequence", &SingleAmplitudeQVM::getSequence, "get prog sequence", py::return_value_policy::automatic_reference)
         .def("get_quick_map_vertice", &SingleAmplitudeQVM::getQuickMapVertice, "get quick map vertice", py::return_value_policy::automatic_reference)

         .def("pmeasure_bin_index", &SingleAmplitudeQVM::pMeasureBinindex, "PMeasure by binary index", py::return_value_policy::automatic_reference)
         .def("pmeasure_dec_index", &SingleAmplitudeQVM::pMeasureDecindex, "PMeasure by decimal  index", py::return_value_policy::automatic_reference)

         .def("get_prob_dict", py::overload_cast<QVec>(&SingleAmplitudeQVM::getProbDict))
         .def("get_prob_dict", py::overload_cast<const std::vector<int> &>(&SingleAmplitudeQVM::getProbDict))
         .def("prob_run_dict", py::overload_cast<QProg &, QVec>(&SingleAmplitudeQVM::probRunDict))
         .def("prob_run_dict", py::overload_cast<QProg &, const std::vector<int> &>(&SingleAmplitudeQVM::probRunDict));

     py::class_<PartialAmplitudeQVM, QuantumMachine>(m, "PartialAmpQVM")
         .def(py::init<>())

         .def("init_qvm",
              &PartialAmplitudeQVM::init,
              py::arg_v("type", BackendType::CPU, "BackendType.CPU"),
              "init quantum virtual machine")

         .def("run",
              py::overload_cast<QProg &, const NoiseModel &>(&PartialAmplitudeQVM::run<QProg>),
              py::arg("qprog"),
              py::arg_v("noise_model", NoiseModel(), "NoiseModel()"),
              "load the quantum program")
         .def("run",
              py::overload_cast<QCircuit &, const NoiseModel &>(&PartialAmplitudeQVM::run<QCircuit>),
              py::arg("qprog"),
              py::arg_v("noise_model", NoiseModel(), "NoiseModel()"),
              "load the quantum program")

         .def("pmeasure_bin_index", &PartialAmplitudeQVM::pmeasure_bin_index, "bin_index"_a, "PMeasure_bin_index", py::return_value_policy::automatic_reference)
         .def("pmeasure_dec_index", &PartialAmplitudeQVM::pmeasure_dec_index, "dec_index"_a, "PMeasure_dec_index", py::return_value_policy::automatic_reference)
         .def("pmeasure_subset", &PartialAmplitudeQVM::pmeasure_subset, "index_list"_a, "pmeasure_subset", py::return_value_policy::automatic_reference)
         .def("get_prob_dict", &PartialAmplitudeQVM::getProbDict)
         .def("prob_run_dict", &PartialAmplitudeQVM::probRunDict);

     py::class_<MPSQVM, QuantumMachine> mps_qvm(m, "MPSQVM");
     mps_qvm.def(py::init<>())

         //     .def("get_qstate", &MPSQVM::getQState, "getState", py::return_value_policy::automatic)
         //     .def("init_qvm", &MPSQVM::init, "init quantum virtual machine")

         //     .def("init_state", &MPSQVM::initState,
         //          py::arg("state") = QStat(),
         //          py::arg("qlist") = QVec(),
         //          py::return_value_policy::reference)

         // .def(
         //     "directly_run", [](MPSQVM &qvm, QProg &prog)
         //     { qvm.directlyRun(prog); },
         //     "directly_run a prog", "program"_a, py::return_value_policy::reference)
         // .def(
         //     "run_with_configuration", [](MPSQVM &qvm, QProg &prog, std::vector<ClassicalCondition> &cc_vector, py::dict param)
         //     {
         //             py::object json = py::module::import("json");
         //             py::object dumps = json.attr("dumps");
         //             auto json_string = std::string(py::str(dumps(param)));
         //             rapidjson::Document doc;
         //             auto & alloc = doc.GetAllocator();
         //             doc.Parse(json_string.c_str());
         //             return qvm.runWithConfiguration(prog, cc_vector, doc); },
         //     "program"_a, "cbit_list"_a, "data"_a, py::return_value_policy::automatic)
         // .def(
         //     "run_with_configuration", [](MPSQVM &qvm, QProg &prog, std::vector<ClassicalCondition> &cc_vector, int shots)
         //     { return qvm.runWithConfiguration(prog, cc_vector, shots); },
         //     "program"_a, "cbit_list"_a, "data"_a, py::return_value_policy::automatic)
         /* we should export idealQVM first then MPSQVM inherit, otherwise we have these ugly code */
         .def("pmeasure", &MPSQVM::PMeasure, "qubit_list"_a, "select_max"_a = -1, "Get the probability distribution over qubits", py::return_value_policy::reference)
         .def("pmeasure_no_index", &MPSQVM::PMeasure_no_index, "qubit_list"_a, "Get the probability distribution over qubits", py::return_value_policy::reference)
         .def("get_prob_tuple_list", &MPSQVM::getProbTupleList, "qubit_list"_a, "select_max"_a = -1, py::return_value_policy::reference)
         .def("get_prob_list", &MPSQVM::getProbList, "qubit_list"_a, "select_max"_a = -1, py::return_value_policy::reference)
         .def("get_prob_dict", &MPSQVM::getProbDict, "qubit_list"_a, "select_max"_a = -1, py::return_value_policy::reference)
         .def("prob_run_tuple_list", &MPSQVM::probRunTupleList, "program"_a, "qubit_list"_a, "select_max"_a = -1, py::return_value_policy::reference)
         .def("prob_run_list", &MPSQVM::probRunList, "program"_a, "qubit_list"_a, "select_max"_a = -1, py::return_value_policy::reference)
         .def("prob_run_dict", &MPSQVM::probRunDict, "program"_a, "qubit_list"_a, "select_max"_a = -1, py::return_value_policy::reference)
         .def("quick_measure", &MPSQVM::quickMeasure, "qubit_list"_a, "shots"_a, py::return_value_policy::reference)

         .def("pmeasure_bin_index", &MPSQVM::pmeasure_bin_index, "program"_a, "string"_a, py::return_value_policy::reference)
         .def("pmeasure_dec_index", &MPSQVM::pmeasure_dec_index, "program"_a, "string"_a, py::return_value_policy::reference)
         .def("pmeasure_bin_subset", &MPSQVM::pmeasure_bin_subset, "program"_a, "string_list"_a, py::return_value_policy::reference)
         .def("pmeasure_dec_subset", &MPSQVM::pmeasure_dec_subset, "program"_a, "string_list"_a, py::return_value_policy::reference)

         // The all next MPSQVM functions are only for noise simulation

         /* bit-flip,phase-flip,bit-phase-flip,phase-damping,amplitude-damping,depolarizing*/
         .def("set_noise_model", py::overload_cast<NOISE_MODEL, GateType, double>(&MPSQVM::set_noise_model))
         .def("set_noise_model", py::overload_cast<NOISE_MODEL, GateType, double, const std::vector<QVec> &>(&MPSQVM::set_noise_model))

         /*decoherence error*/
         .def("set_noise_model", py::overload_cast<NOISE_MODEL, GateType, double, double, double>(&MPSQVM::set_noise_model))
         .def("set_noise_model", py::overload_cast<NOISE_MODEL, GateType, double, double, double, const std::vector<QVec> &>(&MPSQVM::set_noise_model))

         /*mixed unitary error*/
         .def("set_mixed_unitary_error", py::overload_cast<GateType, const std::vector<QStat> &, const std::vector<QVec> &>(&MPSQVM::set_mixed_unitary_error))
         .def("set_mixed_unitary_error", py::overload_cast<GateType, const std::vector<QStat> &, const prob_vec &, const std::vector<QVec> &>(&MPSQVM::set_mixed_unitary_error))
         .def("set_mixed_unitary_error", py::overload_cast<GateType, const std::vector<QStat> &>(&MPSQVM::set_mixed_unitary_error))
         .def("set_mixed_unitary_error", py::overload_cast<GateType, const std::vector<QStat> &, const prob_vec &>(&MPSQVM::set_mixed_unitary_error))

         /*readout error*/
         .def("set_readout_error", &MPSQVM::set_readout_error, "readout_params"_a, "qubits"_a, py::return_value_policy::reference)

         /*measurement error*/
         .def("set_measure_error", py::overload_cast<NOISE_MODEL, double>(&MPSQVM::set_measure_error))
         .def("set_measure_error", py::overload_cast<NOISE_MODEL, double, double, double>(&MPSQVM::set_measure_error))

         /*rotation error*/
         .def("set_rotation_error", &MPSQVM::set_rotation_error, "param"_a, py::return_value_policy::reference)

         /*reset error*/
         .def("set_reset_error", &MPSQVM::set_reset_error, "reset_0_param"_a, "reset_1_param"_a, py::return_value_policy::reference)

         /* bit-flip,phase-flip,bit-phase-flip,phase-damping,amplitude-damping,depolarizing*/
         .def("add_single_noise_model", py::overload_cast<NOISE_MODEL, GateType, double>(&MPSQVM::add_single_noise_model))
         /*decoherence error*/
         .def("add_single_noise_model", py::overload_cast<NOISE_MODEL, GateType, double, double, double>(&MPSQVM::add_single_noise_model));
     // export_idealqvm_func<MPSQVM>::export_func(mps_qvm);

     /*combine error*/

#ifdef USE_CURL

     py::enum_<RealChipType>(m, "real_chip_type")
         .value("origin_wuyuan_d3", RealChipType::ORIGIN_WUYUAN_D3)
         .value("origin_wuyuan_d4", RealChipType::ORIGIN_WUYUAN_D4)
         .value("origin_wuyuan_d5", RealChipType::ORIGIN_WUYUAN_D5)
         .export_values();

     py::enum_<TaskStatus>(m, "task_status")
         .value("waiting", TaskStatus::WAITING)
         .value("computing", TaskStatus::COMPUTING)
         .value("finished", TaskStatus::FINISHED)
         .value("failed", TaskStatus::FAILED)
         .value("queuing", TaskStatus::QUEUING)
         .export_values();

     // py::enum_<RealChipType>(m, "RealChipType")
     //     .value("origin_wuyuan_d3", RealChipType::ORIGIN_WUYUAN_D3)
     //     .value("origin_wuyuan_d4", RealChipType::ORIGIN_WUYUAN_D4)
     //     .value("origin_wuyuan_d5", RealChipType::ORIGIN_WUYUAN_D5)
     //     .export_values();

     // py::enum_<TaskStatus>(m, "TaskStatus")
     //     .value("waiting", TaskStatus::WAITING)
     //     .value("computing", TaskStatus::COMPUTING)
     //     .value("finished", TaskStatus::FINISHED)
     //     .value("failed", TaskStatus::FAILED)
     //     .value("queuing", TaskStatus::QUEUING)
     //     .export_values();

     py::class_<QCloudMachine, QuantumMachine>(m, "QCloud")
         .def(py::init<>())
         .def("init_qvm",
              &QCloudMachine::init,
              py::arg("token"),
              py::arg("is_logged") = false,
              "init quantum virtual machine")

         // url setting
         .def("set_qcloud_api", &QCloudMachine::set_qcloud_api)

         .def("set_inquire_url", &QCloudMachine::set_inquire_url)
         .def("set_compute_url", &QCloudMachine::set_compute_url)
         .def("set_batch_inquire_url", &QCloudMachine::set_batch_inquire_url)
         .def("set_batch_compute_url", &QCloudMachine::set_batch_compute_url)

         // noise
         .def("set_noise_model", &QCloudMachine::set_noise_model)
         .def("noise_measure",
              &QCloudMachine::noise_measure,
              py::arg("prog"),
              py::arg("shot"),
              py::arg("task_name") = "Qurator Experiment")

         // full_amplitude
         .def("full_amplitude_measure",
              &QCloudMachine::full_amplitude_measure,
              py::arg("prog"),
              py::arg("shot"),
              py::arg("task_name") = "Qurator Experiment")

         .def("full_amplitude_pmeasure",
              &QCloudMachine::full_amplitude_pmeasure,
              py::arg("prog"),
              py::arg("qvec"),
              py::arg("task_name") = "Qurator Experiment")

         // partial_amplitude
         .def("partial_amplitude_pmeasure",
              &QCloudMachine::partial_amplitude_pmeasure,
              py::arg("prog"),
              py::arg("amp_vec"),
              py::arg("task_name") = "Qurator Experiment")

         // single_amplitude
         .def("single_amplitude_pmeasure",
              &QCloudMachine::single_amplitude_pmeasure,
              py::arg("prog"),
              py::arg("amplitude"),
              py::arg("task_name") = "Qurator Experiment")

         // real chip measure
         .def("real_chip_measure",
              &QCloudMachine::real_chip_measure,
              py::arg("prog"),
              py::arg("shot"),
              py::arg_v("chip_id", RealChipType::ORIGIN_WUYUAN_D5, "real_chip_type.origin_wuyuan_d5"),
              py::arg("is_amend") = true,
              py::arg("is_mapping") = true,
              py::arg("is_optimization") = true,
              py::arg("task_name") = "Qurator Experiment")

         // real chip get_state_fidelity
         .def("get_state_fidelity",
              &QCloudMachine::get_state_fidelity,
              py::arg("prog"),
              py::arg("shot"),
              py::arg_v("chip_id", RealChipType::ORIGIN_WUYUAN_D5, "real_chip_type.origin_wuyuan_d5"),
              py::arg("is_amend") = true,
              py::arg("is_mapping") = true,
              py::arg("is_optimization") = true,
              py::arg("task_name") = "Qurator Experiment")

         // real chip get_state_tomography_density
         .def("get_state_tomography_density",
              &QCloudMachine::get_state_tomography_density,
              py::arg("prog"),
              py::arg("shot"),
              py::arg_v("chip_id", RealChipType::ORIGIN_WUYUAN_D5, "real_chip_type.origin_wuyuan_d5"),
              py::arg("is_amend") = true,
              py::arg("is_mapping") = true,
              py::arg("is_optimization") = true,
              py::arg("task_name") = "Qurator Experiment")

         // get_expectation
         .def("get_expectation",
              &QCloudMachine::get_expectation,
              py::arg("prog"),
              py::arg("hamiltonian"),
              py::arg("qvec"),
              py::arg("status"),
              py::arg("task_name") = "Qurator Experiment")

         // get_expectation commit
         .def("get_expectation_commit",
              &QCloudMachine::get_expectation,
              py::arg("prog"),
              py::arg("hamiltonian"),
              py::arg("qvec"),
              py::arg("status"),
              py::arg("task_name") = "QPanda Experiment")

         // get_expectation_exec
         .def("get_expectation_exec",
              &QCloudMachine::get_expectation_exec,
              py::arg("taskid"),
              py::arg("status"))

         // get_expectation_query
         .def("get_expectation_query",
              &QCloudMachine::get_expectation_query,
              py::arg("taskid"),
              py::arg("status"))

         // full_amplitude
         .def("full_amplitude_measure_batch",
              &QCloudMachine::full_amplitude_measure_batch,
              py::arg("prog_array"),
              py::arg("shot"),
              py::arg("task_name") = "QPanda Experiment")

         .def("full_amplitude_pmeasure_batch",
              &QCloudMachine::full_amplitude_pmeasure_batch,
              py::arg("prog_array"),
              py::arg("qvec"),
              py::arg("task_name") = "QPanda Experiment")

         // partial_amplitude
         .def("partial_amplitude_pmeasure_batch",
              &QCloudMachine::partial_amplitude_pmeasure_batch,
              py::arg("prog_array"),
              py::arg("amp_vec"),
              py::arg("task_name") = "QPanda Experiment")

         // single_amplitude
         .def("single_amplitude_pmeasure_batch",
              &QCloudMachine::single_amplitude_pmeasure_batch,
              py::arg("prog"),
              py::arg("amplitude"),
              py::arg("task_name") = "QPanda Experiment")

         // noise
         .def("noise_measure_batch",
              &QCloudMachine::noise_measure_batch,
              py::arg("prog_array"),
              py::arg("shot"),
              py::arg("task_name") = "QPanda Experiment")

         // real chip measure
         .def("real_chip_measure_batch",
              &QCloudMachine::real_chip_measure_batch,
              py::arg("prog_array"),
              py::arg("shot"),
              py::arg_v("chip_id", RealChipType::ORIGIN_WUYUAN_D3, "real_chip_type.origin_wuyuan_d3"),
              py::arg("is_amend") = true,
              py::arg("is_mapping") = true,
              py::arg("is_optimization") = true,
              py::arg("task_name") = "QPanda Experiment")

         .def("full_amplitude_measure_batch_commit",
              &QCloudMachine::full_amplitude_measure_batch_commit,
              py::arg("prog_array"),
              py::arg("shot"),
              py::arg("status"),
              py::arg("task_name") = "QPanda Experiment")

         .def("full_amplitude_pmeasure_batch_commit",
              &QCloudMachine::full_amplitude_pmeasure_batch_commit,
              py::arg("prog_array"),
              py::arg("qvec"),
              py::arg("status"),
              py::arg("task_name") = "QPanda Experiment")

         .def("real_chip_measure_batch_commit",
              &QCloudMachine::real_chip_measure_batch_commit,
              py::arg("prog_array"),
              py::arg("shot"),
              py::arg("status"),
              py::arg_v("chip_id", RealChipType::ORIGIN_WUYUAN_D3, "real_chip_type.origin_wuyuan_d3"),
              py::arg("is_amend") = true,
              py::arg("is_mapping") = true,
              py::arg("is_optimization") = true,
              py::arg("task_name") = "QPanda Experiment")

         .def("full_amplitude_measure_batch_query",
              &QCloudMachine::full_amplitude_measure_batch_query,
              py::arg("taskid_map"))

         .def("full_amplitude_pmeasure_batch_query",
              &QCloudMachine::full_amplitude_pmeasure_batch_query,
              py::arg("taskid_map"))

         .def("real_chip_measure_batch_query",
              &QCloudMachine::real_chip_measure_batch_query,
              py::arg("taskid_map"));

#endif // USE_CURL

     py::implicitly_convertible<CPUQVM, QuantumMachine>();
     py::implicitly_convertible<GPUQVM, QuantumMachine>();
     py::implicitly_convertible<CPUSingleThreadQVM, QuantumMachine>();
     py::implicitly_convertible<NoiseQVM, QuantumMachine>();
     py::implicitly_convertible<SingleAmplitudeQVM, QuantumMachine>();
     py::implicitly_convertible<PartialAmplitudeQVM, QuantumMachine>();
}
