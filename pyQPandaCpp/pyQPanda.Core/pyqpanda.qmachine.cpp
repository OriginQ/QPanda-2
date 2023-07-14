#include "Core/Core.h"
#include "Core/QuantumMachine/SingleAmplitudeQVM.h"
#include "Core/QuantumMachine/PartialAmplitudeQVM.h"
#include "Core/VirtualQuantumProcessor/NoiseQPU/NoiseModel.h"
#include "Core/VirtualQuantumProcessor/SparseQVM/SparseQVM.h"
#include "Core/VirtualQuantumProcessor/Stabilizer/Stabilizer.h"
#include "Core/VirtualQuantumProcessor/DensityMatrix/DensityMatrixSimulator.h"
#include <map>
#include <math.h>
#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/chrono.h"
#include "pybind11/numpy.h"
#include "pybind11/operators.h"
#include <pybind11/stl_bind.h>
#include "pybind11/eigen.h"
#include "template_generator.h"

#if defined(USE_OPENSSL) && defined(USE_CURL)
#include "Core/QuantumCloud/QCloudMachine.h"
#endif

USING_QPANDA
namespace py = pybind11;
using namespace pybind11::literals;

// template<>
// struct py::detail::type_caster<QVec>  : py::detail::list_caster<QVec, Qubit*> {};
void export_quantum_machine(py::module &m)
{
    py::class_<QuantumMachine>(m, "QuantumMachine", "quantum machine base class")
        .def(
            "set_configure",
            [](QuantumMachine &qvm, size_t max_qubit, size_t max_cbit)
            {
                Configuration config = { max_qubit, max_cbit };
                qvm.setConfig(config);
            },
            py::arg("max_qubit"),
        py::arg("max_cbit"),
        "set QVM max qubit and max cbit\n"
        "\n"
        "Args:\n"
        "    max_qubit: quantum machine max qubit num \n"
        "    max_cbit: quantum machine max cbit num \n"
        "\n"
        "Returns:\n"
        "    none\n"
        "Raises:\n"
        "    run_fail: An error occurred in set_configure\n")
        .def("finalize", &QuantumMachine::finalize,
            "finalize quantum machine\n"
            "\n"
            "Args:\n"
            "    none\n"
            "\n"
            "Returns:\n"
            "    none\n"
            "Raises:\n"
            "    run_fail: An error occurred in finalize\n"
        )
        .def("get_qstate", &QuantumMachine::getQState,
            "Get the status of the Quantum machine\n"
            "\n"
            "Args:\n"
            "    none\n"
            "\n"
            "Returns:\n"
            "    the status of the Quantum machine, see QMachineStatus\n"
            "\n"
            "Raises:\n"
            "    init_fail: An error occurred\n",
            py::return_value_policy::automatic)
        .def("qAlloc", &QuantumMachine::allocateQubit,
            "Allocate a qubits\n"
            "After init()\n"
            "\n"
            "Args:\n"
            "    qubit_addr: qubit physic address, should in [0,29)\n"
            "\n"
            "Returns:\n"
            "    pyQPanda.Qubit: None, if qubit_addr error, or reached max number of allowed qubit",
            py::return_value_policy::reference)
        /*.def("qAlloc_many",
             &QuantumMachine::allocateQubits,
             py::arg("qubit_num"),
             "Allocate a list of qubits",
            [](QuantumMachine &self, int qubit_num)
            {
                 std::vector<Qubit*> qv = self.qAllocMany(qubit_num);
                return qv;
            },
            py::return_value_policy::reference)*/
        .def("qAlloc_many",
            [](QuantumMachine &self, size_t qubit_num)
    {
        auto qv = static_cast<std::vector<Qubit*>>(self.qAllocMany(qubit_num));
        return qv;
    },
            py::arg("qubit_num"),
        "Allocate several qubits\n"
        "After init()\n"
        "\n"
        "Args:\n"
        "    qubit_num: numbers of qubit want to be created\n"
        "\n"
        "Returns:\n"
        "    list[pyQPanda.Qubit]: list of qubit",
        py::return_value_policy::reference)
        .def("cAlloc",
            py::overload_cast<>(&QuantumMachine::allocateCBit),
            "Allocate a CBit\n"
            "After init()\n"
            "\n"
            "Args:\n"
            "    cbit_addr: cbit address, should in [0,29)"
            "\n"
            "Returns:\n"
            "    classic result cbit",
            py::return_value_policy::reference)
        .def("cAlloc_many",
            &QuantumMachine::allocateCBits,
            py::arg("cbit_num"),
            "Allocate several CBits\n"
            "After init()\n"
            "\n"
            "Args:\n"
            "    cbit_num: numbers of cbit want to be created\n"
            "\n"
            "Returns:\n"
            "    list of cbit",
            py::return_value_policy::reference)
        .def("qFree",
            &QuantumMachine::Free_Qubit,
            py::arg("qubit"),
            "Free a CBit\n"
            "\n"
            "Args:\n"
            "    CBit: a CBit\n"
            "\n"
            "Returns:\n"
            "    none\n")
        .def("qFree_all",
            &QuantumMachine::Free_Qubits,
            py::arg("qubit_list"),
            "Free all cbits\n"
            "\n"
            "Args:\n"
            "    none\n"
            "\n"
            "Returns:\n"
            "    none\n")
        .def("qFree_all", py::overload_cast<QVec &>(&QuantumMachine::qFreeAll),
            "Free all qubits\n"
            "\n"
            "Args:\n"
            "    none\n"
            "\n"
            "Returns:\n"
            "    none\n")

        .def("cFree", &QuantumMachine::Free_CBit,
            "Free a CBit\n"
            "\n"
            "Args:\n"
            "    CBit: a CBit\n"
            "\n"
            "Returns:\n"
            "    none\n")
        .def("cFree_all",
            &QuantumMachine::Free_CBits,
            py::arg("cbit_list"),
            "Free all cbits\n"
            "\n"
            "Args:\n"
            "    none\n"
            "\n"
            "Returns:\n"
            "    none\n")
        .def("cFree_all", py::overload_cast<>(&QuantumMachine::cFreeAll),
            "Free all cbits\n"
            "\n"
            "Args:\n"
            "    none\n"
            "\n"
            "Returns:\n"
            "    none\n")
        .def("getStatus", &QuantumMachine::getStatus,
            "Get the status of the Quantum machine\n"
            "\n"
            "Args:\n"
            "    none\n"
            "\n"
            "Returns:\n"
            "    the status of the Quantum machine, see QMachineStatus\n"
            "\n"
            "Raises:\n"
            "    init_fail: An error occurred\n",
            py::return_value_policy::reference_internal)

        /*will delete*/
        .def("initQVM", &QuantumMachine::init,
            "Init the global unique quantum machine at background.\n"
            "\n"
            "Args:\n"
            "    machine_type: quantum machine type, see pyQPanda.QMachineType\n"
            "\n"
            "Returns:\n"
            "    bool: ture if initialization success")
        .def("getAllocateQubitNum", &QuantumMachine::getAllocateQubit,
            "Get allocated qubits of QuantumMachine\n"
            "\n"
            "Args:\n"
            "    none\n"
            "\n"
            "Returns:\n"
            "    qubit list\n"
            "\n"
            "Raises:\n"
            "    run_fail: An error occurred in allocated qubits of QuantumMachine\n",
            py::return_value_policy::reference)
        .def("getAllocateCMem", &QuantumMachine::getAllocateCMem,
            "Get allocated cbits of QuantumMachine\n"
            "\n"
            "Args:\n"
            "    none\n"
            "\n"
            "Returns:\n"
            "    cbit list\n"
            "\n"
            "Raises:\n"
            "    run_fail: An error occurred in allocated cbits of QuantumMachine\n",
            py::return_value_policy::reference)

        /* new interface */
        .def("init_qvm", &QuantumMachine::init,
            "Init the global unique quantum machine at background.\n"
            "\n"
            "Args:\n"
            "    machine_type: quantum machine type, see pyQPanda.QMachineType\n"
            "\n"
            "Returns:\n"
            "    bool: ture if initialization success")
        .def("init_state",
            &QuantumMachine::initState,
            py::arg_v("state", QStat(), "QStat()"),
            py::arg_v("qlist", QVec(), "QVec()"),
            "Get allocated cbits of QuantumMachine\n"
            "\n"
            "Args:\n"
            "    none\n"
            "\n"
            "Returns:\n"
            "    cbit list\n"
            "\n"
            "Raises:\n"
            "    run_fail: An error occurred in allocated cbits of QuantumMachine\n",
            py::return_value_policy::reference)
        .def("cAlloc",
            py::overload_cast<size_t>(&QuantumMachine::allocateCBit),
            py::arg("cbit"),
            "Allocate a CBit\n"
            "After init()\n"
            "\n"
            "Args:\n"
            "    cbit_addr: cbit address, should in [0,29)"
            "\n"
            "Returns:\n"
            "    classic result cbit",
            py::return_value_policy::reference)
        .def("get_status", &QuantumMachine::getStatus,
            "Get the status of the Quantum machine\n"
            "\n"
            "Args:\n"
            "    none\n"
            "\n"
            "Returns:\n"
            "    the status of the Quantum machine, see QMachineStatus\n"
            "\n"
            "Raises:\n"
            "    init_fail: An error occurred\n", py::return_value_policy::reference_internal)
        .def("get_allocate_qubit_num", &QuantumMachine::getAllocateQubit,
            "Get allocated qubits of QuantumMachine\n"
            "\n"
            "Args:\n"
            "    none\n"
            "\n"
            "Returns:\n"
            "    qubit list\n"
            "\n"
            "Raises:\n"
            "    run_fail: An error occurred in allocated qubits of QuantumMachine\n",
            py::return_value_policy::reference)
        .def("get_allocate_cmem_num", &QuantumMachine::getAllocateCMem,
            "Get allocated cbits of QuantumMachine\n"
            "\n"
            "Args:\n"
            "    none\n"
            "\n"
            "Returns:\n"
            "    cbit list\n"
            "\n"
            "Raises:\n"
            "    run_fail: An error occurred in allocated cbits of QuantumMachine\n", py::return_value_policy::reference)
        .def("allocate_qubit_through_phy_address",
            &QuantumMachine::allocateQubitThroughPhyAddress,
            py::arg("address"),
            "allocate qubits through phy address\n"
            "\n"
            "Args:\n"
            "    address: qubit phy address\n"
            "\n"
            "Returns:\n"
            "    Qubit\n"
            "\n"
            "Raises:\n"
            "    run_fail: An error occurred in allocated qubits of QuantumMachine\n",
            py::return_value_policy::reference)
        .def("allocate_qubit_through_vir_address",
            &QuantumMachine::allocateQubitThroughVirAddress,
            py::arg("address"),
            "allocate qubits through vir address\n"
            "\n"
            "Args:\n"
            "    address: qubit vir address\n"
            "\n"
            "Returns:\n"
            "    Qubit\n"
            "\n"
            "Raises:\n"
            "    run_fail: An error occurred in allocated qubits of QuantumMachine\n",
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
            "Get allocated qubits of QuantumMachine\n"
        "\n"
        "Args:\n"
        "    none\n"
        "\n"
        "Returns:\n"
        "    qubit list\n"
        "\n"
        "Raises:\n"
        "    run_fail: An error occurred in allocated qubits of QuantumMachine\n",
        py::return_value_policy::reference)

        .def(
            "get_allocate_cbits",
            [](QuantumMachine &self)
    {
        std::vector<ClassicalCondition> cv;
        self.get_allocate_cbits(cv);
        return cv;
    },
            "Get allocated cbits of QuantumMachine\n"
        "\n"
        "Args:\n"
        "    none\n"
        "\n"
        "Returns:\n"
        "    cbit list\n"
        "\n"
        "Raises:\n"
        "    run_fail: An error occurred in allocated cbits of QuantumMachine\n",
        py::return_value_policy::reference)

        .def("get_expectation",
            py::overload_cast<QProg, const QHamiltonian &, const QVec &>(&QuantumMachine::get_expectation),
            py::arg("qprog"),
            py::arg("hamiltonian"),
            py::arg("qubit_list"),
            "get expectation of current hamiltonian\n"
            "\n"
            "Args:\n"
            "    qprog : quantum prog \n"
            "    hamiltonian: selected hamiltonian \n"
            "    qubit_list : measure qubit list \n"
            "\n"
            "Returns:\n"
            "    double : expectation of current hamiltonian\n"
            "\n"
            "Raises:\n"
            "    run_fail: An error occurred in get_expectation\n",
            py::return_value_policy::reference)
        .def("get_expectation",
            py::overload_cast<QProg, const QHamiltonian &, const QVec &, int>(&QuantumMachine::get_expectation),
            py::arg("qprog"),
            py::arg("hamiltonian"),
            py::arg("qubit_list"),
            py::arg("shots"),
            "get expectation of current hamiltonian\n"
            "\n"
            "Args:\n"
            "    qprog : quantum prog \n"
            "    hamiltonian: selected hamiltonian \n"
            "    qubit_list : measure qubit list \n"
            "    shots : measure shots \n"
            "\n"
            "Returns:\n"
            "    double : expectation of current hamiltonian\n"
            "\n"
            "Raises:\n"
            "    run_fail: An error occurred in get_expectation\n",
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
            py::call_guard<py::gil_scoped_release>(),
            "Directly run quantum prog\n"
            "After init()\n"
            "\n"
            "Args:\n"
            "    qprog: quantum program\n"
            "    noise_model: noise model, default is no noise. noise model only work on CPUQVM now\n"
            "\n"
            "Returns:\n"
            "    Dict[str, bool]: result of quantum program execution one shot.\n"
            "                     first is the final qubit register state, second is it's measure probability",
            py::return_value_policy::reference)
        .def("run_with_configuration",
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
        py::call_guard<py::gil_scoped_release>(),
        "Run quantum program with configuration\n"
        "\n"
        "Args:\n"
        "    program: quantum program\n"
        "    cbit_list: classic cbits list\n"
        "    shots: repeate run quantum program times\n"
        "    noise_model: noise model, default is no noise. noise model only work on CPUQVM now\n"
        "\n"
        "Returns:\n"
        "    result of quantum program execution in shots.\n"
        "    first is the final qubit register state, second is it's hit shot"
        "Raises:\n"
        "    run_fail: An error occurred in measure quantum program\n",
        py::return_value_policy::automatic)

        .def("run_with_configuration",
            py::overload_cast<QProg &, vector<ClassicalCondition> &, int, const NoiseModel &>(&QuantumMachine::runWithConfiguration),
            py::arg("qprog"),
            py::arg("cbit_list"),
            py::arg("shot"),
            py::arg_v("noise_model", NoiseModel(), "NoiseModel()"),
            py::call_guard<py::gil_scoped_release>(),
            "Run quantum program with configuration\n"
            "\n"
            "Args:\n"
            "    program: quantum program\n"
            "    cbit_list: classic cbits list\n"
            "    shots: repeate run quantum program times\n"
            "    noise_model: noise model, default is no noise. noise model only work on CPUQVM now\n"
            "\n"
            "Returns:\n"
            "    result of quantum program execution in shots.\n"
            "    first is the final qubit register state, second is it's hit shot"
            "Raises:\n"
            "    run_fail: An error occurred in measure quantum program\n",
            py::return_value_policy::automatic)
        .def("run_with_configuration",
            py::overload_cast<QProg &, int, const NoiseModel &>(&QuantumMachine::runWithConfiguration),
            py::arg("qprog"),
            py::arg("shot"),
            py::arg_v("noise_model", NoiseModel(), "NoiseModel()"),
            py::call_guard<py::gil_scoped_release>(),
            "Run quantum program with configuration\n"
            "\n"
            "Args:\n"
            "    program: quantum program\n"
            "    shots: repeate run quantum program times\n"
            "    noise_model: noise model, default is no noise. noise model only work on CPUQVM now\n"
            "\n"
            "Returns:\n"
            "    result of quantum program execution in shots.\n"
            "    first is the final qubit register state, second is it's hit shot"
            "Raises:\n"
            "    run_fail: An error occurred in measure quantum program\n",
            py::return_value_policy::automatic)
        .def("run_with_configuration",
            py::overload_cast<QProg &, vector<int> &, int, const NoiseModel &>(&QuantumMachine::runWithConfiguration),
            py::arg("qprog"),
            py::arg("cbit_list"),
            py::arg("shot"),
            py::arg_v("noise_model", NoiseModel(), "NoiseModel()"),
            py::call_guard<py::gil_scoped_release>(),
            "Run quantum program with configuration\n"
            "\n"
            "Args:\n"
            "    program: quantum program\n"
            "    cbit_list: classic cbits list\n"
            "    shots: repeate run quantum program times\n"
            "    noise_model: noise model, default is no noise. noise model only work on CPUQVM now\n"
            "\n"
            "Returns:\n"
            "    result of quantum program execution in shots.\n"
            "    first is the final qubit register state, second is it's hit shot"
            "Raises:\n"
            "    run_fail: An error occurred in measure quantum program\n",
            py::return_value_policy::automatic);

    /*
      just inherit abstract base class wrappered by trampoline class, will implement C++ like polymorphism in python
      no neeed use another trampoline class wrapper child, unless children have derived children class
    */
    py::class_<CPUQVM, QuantumMachine> cpu_qvm(m, "CPUQVM", "quantum machine cpu");
    cpu_qvm.def("init_qvm", py::overload_cast<bool>(&CPUQVM::init));
    cpu_qvm.def("init_qvm", py::overload_cast<>(&CPUQVM::init));
    cpu_qvm.def("set_max_threads",
        &CPUQVM::set_parallel_threads,
        py::arg("size"),
        "set CPUQVM max thread size",
        py::return_value_policy::automatic);
    /*
      we should declare these function in py::class_<IdealQVM>, then CPUQVM inherit form it,
      but as we won't want to export IdealQVM to user, this may the only way
    */
    export_idealqvm_func<CPUQVM>::export_func(cpu_qvm);
    py::class_<CPUSingleThreadQVM, QuantumMachine> cpu_single_thread_qvm(m,
        "CPUSingleThreadQVM", "quantum machine class for cpu single thread");
    export_idealqvm_func<CPUSingleThreadQVM>::export_func(cpu_single_thread_qvm);

#ifdef USE_CUDA
    py::class_<GPUQVM, QuantumMachine> gpu_qvm(m, "GPUQVM");
    export_idealqvm_func<GPUQVM>::export_func(gpu_qvm);
#endif // USE_CUDA

    py::class_<NoiseQVM, QuantumMachine>(m, "NoiseQVM", "quantum machine class for simulate noise prog")
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

    py::class_<SingleAmplitudeQVM, QuantumMachine>(m, "SingleAmpQVM", "quantum single amplitude machine class")
        .def(py::init<>())

        //     .def("init_qvm", &SingleAmplitudeQVM::init, "init quantum virtual machine")
        .def("run",
            py::overload_cast<QProg &, QVec &, size_t, size_t>(&SingleAmplitudeQVM::run),
            py::arg("prog"),
            py::arg("qv"),
            py::arg("max_rank") = 30,
            py::arg("alloted_time") = 5,
            "run the quantum program\n"
            "\n"
            "Args:\n"
            "    QProg: quantum prog \n"
            "    QVec: qubits list\n"
            "    size_t: max_rank\n"
            "    size_t: alloted_time\n"
            "\n"
            "Returns:\n"
            "    none\n"
            "Raises:\n"
            "    run_fail: An error occurred in run\n",
            py::return_value_policy::automatic_reference)
        .def("run",
            py::overload_cast<QProg &, QVec &, size_t, const std::vector<qprog_sequence_t> &>(&SingleAmplitudeQVM::run),
            "run the quantum program\n"
            "\n"
            "Args:\n"
            "    QProg: quantum prog \n"
            "    QVec: qubits list\n"
            "    size_t: max_rank\n"
            "    list: sequences\n"
            "\n"
            "Returns:\n"
            "    none\n"
            "Raises:\n"
            "    run_fail: An error occurred in run\n")
        .def("get_sequence", &SingleAmplitudeQVM::getSequence, "get prog sequence", py::return_value_policy::automatic_reference)
        .def("get_quick_map_vertice", &SingleAmplitudeQVM::getQuickMapVertice, "get quick map vertice", py::return_value_policy::automatic_reference)

        .def("pmeasure_bin_index", &SingleAmplitudeQVM::pMeasureBinindex,
            "pmeasure bin index quantum state amplitude\n"
            "\n"
            "Args:\n"
            "    string : bin string\n"
            "\n"
            "Returns:\n"
            "    double : bin amplitude prob\n"
            "Raises:\n"
            "    run_fail: An error occurred in pmeasure_bin_index\n", py::return_value_policy::automatic_reference)
        .def("pmeasure_dec_index", &SingleAmplitudeQVM::pMeasureDecindex,
            "pmeasure dec index quantum state amplitude\n"
            "\n"
            "Args:\n"
            "    string : dec string\n"
            "\n"
            "Returns:\n"
            "    double : dec amplitude prob\n"
            "Raises:\n"
            "    run_fail: An error occurred in pmeasure_dec_index\n", py::return_value_policy::automatic_reference)

        .def("pmeasure_bin_amplitude", &SingleAmplitudeQVM::pmeasure_bin_index,
            "pmeasure bin index quantum state amplitude\n"
            "\n"
            "Args:\n"
            "    string : bin string\n"
            "\n"
            "Returns:\n"
            "    complex : bin amplitude\n"
            "Raises:\n"
            "    run_fail: An error occurred in pmeasure_bin_index\n", py::return_value_policy::automatic_reference)
        
        .def("pmeasure_dec_amplitude", &SingleAmplitudeQVM::pmeasure_dec_index,
            "pmeasure dec index quantum state amplitude\n"
            "\n"
            "Args:\n"
            "    string : dec string\n"
            "\n"
            "Returns:\n"
            "    complex : dec amplitude amplitude\n"
            "Raises:\n"
            "    run_fail: An error occurred in pmeasure_dec_index\n", py::return_value_policy::automatic_reference)

        .def("get_prob_dict", py::overload_cast<QVec>(&SingleAmplitudeQVM::getProbDict),
            "Get pmeasure result as dict\n"
            "\n"
            "Args:\n"
            "    qubit_list: pmeasure qubits list\n"
            "\n"
            "Returns:\n"
            "    measure result of quantum machine\n"
            "Raises:\n"
            "    run_fail: An error occurred in get_prob_dict\n")
        .def("get_prob_dict", py::overload_cast<const std::vector<int> &>(&SingleAmplitudeQVM::getProbDict),
            "Get pmeasure result as dict\n"
            "\n"
            "Args:\n"
            "    qubit_list: pmeasure qubits list\n"
            "\n"
            "Returns:\n"
            "    measure result of quantum machine"
            "Raises:\n"
            "    run_fail: An error occurred in get_prob_dict\n")
        .def("prob_run_dict", py::overload_cast<QProg &, QVec>(&SingleAmplitudeQVM::probRunDict),
            "Run quantum program and get pmeasure result as dict\n"
            "\n"
            "Args:\n"
            "    qprog: quantum program\n"
            "    qubit_list: pmeasure qubits list\n"
            "\n"
            "Returns:\n"
            "    measure result of quantum machine\n"
            "Raises:\n"
            "    run_fail: An error occurred in measure quantum program\n")
        .def("prob_run_dict", py::overload_cast<QProg &, const std::vector<int> &>(&SingleAmplitudeQVM::probRunDict),
            "Run quantum program and get pmeasure result as dict\n"
            "\n"
            "Args:\n"
            "    qprog: quantum program\n"
            "    qubit_list: pmeasure qubits list\n"
            "\n"
            "Returns:\n"
            "    measure result of quantum machine\n"
            "Raises:\n"
            "    run_fail: An error occurred in measure quantum program\n");

    py::class_<PartialAmplitudeQVM, QuantumMachine>(m, "PartialAmpQVM", "quantum partial amplitude machine class")
        .def(py::init<>())

        .def("init_qvm",
            &PartialAmplitudeQVM::init,
            py::arg_v("type", BackendType::CPU, "BackendType.CPU"),
            "init quantum virtual machine")

        .def("run",
            py::overload_cast<QProg &, const NoiseModel &>(&PartialAmplitudeQVM::run<QProg>),
            py::arg("qprog"),
            py::arg_v("noise_model", NoiseModel(), "NoiseModel()"),
            "run the quantum program\n"
            "\n"
            "Args:\n"
            "    QProg: quantum prog \n"
            "    size_t : NoiseModel\n"
            "\n"
            "Returns:\n"
            "    none\n"
            "Raises:\n"
            "    run_fail: An error occurred in run\n")
        .def("run",
            py::overload_cast<QCircuit &, const NoiseModel &>(&PartialAmplitudeQVM::run<QCircuit>),
            py::arg("qprog"),
            py::arg_v("noise_model", NoiseModel(), "NoiseModel()"),
            "run the quantum program\n"
            "\n"
            "Args:\n"
            "    QProg: quantum prog \n"
            "    size_t : NoiseModel\n"
            "\n"
            "Returns:\n"
            "    none\n"
            "Raises:\n"
            "    run_fail: An error occurred in run\n")

        .def("pmeasure_bin_index", &PartialAmplitudeQVM::pmeasure_bin_index, "bin_index"_a,
            "pmeasure bin index quantum state amplitude\n"
            "\n"
            "Args:\n"
            "    string : bin string\n"
            "\n"
            "Returns:\n"
            "    complex : bin amplitude\n"
            "Raises:\n"
            "    run_fail: An error occurred in pmeasure_bin_index\n", py::return_value_policy::automatic_reference)
        .def("pmeasure_dec_index", &PartialAmplitudeQVM::pmeasure_dec_index, "dec_index"_a,
            "pmeasure dec index quantum state amplitude\n"
            "\n"
            "Args:\n"
            "    string : dec string\n"
            "\n"
            "Returns:\n"
            "    complex : dec amplitude\n"
            "Raises:\n"
            "    run_fail: An error occurred in pmeasure_dec_index\n", py::return_value_policy::automatic_reference)
        .def("pmeasure_subset", &PartialAmplitudeQVM::pmeasure_subset, "index_list"_a,
            "pmeasure quantum state amplitude subset\n"
            "\n"
            "Args:\n"
            "    list : dec state string list\n"
            "\n"
            "Returns:\n"
            "    list : dec amplitude result list\n"
            "Raises:\n"
            "    run_fail: An error occurred in pmeasure_dec_index\n", py::return_value_policy::automatic_reference)
        .def("get_prob_dict", &PartialAmplitudeQVM::getProbDict,
            "Get pmeasure result as dict\n"
            "\n"
            "Args:\n"
            "    qubit_list: pmeasure qubits list\n"
            "\n"
            "Returns:\n"
            "    measure result of quantum machine\n"
            "Raises:\n"
            "    run_fail: An error occurred in get_prob_dict\n")
        .def("prob_run_dict", &PartialAmplitudeQVM::probRunDict,
            "Run quantum program and get pmeasure result as dict\n"
            "\n"
            "Args:\n"
            "    qprog: quantum program\n"
            "    qubit_list: pmeasure qubits list\n"
            "\n"
            "Returns:\n"
            "    measure result of quantum machine\n"
            "Raises:\n"
            "    run_fail: An error occurred in measure quantum program\n");

    py::class_<SparseSimulator, QuantumMachine>(m, "SparseQVM", "quantum sparse machine class")
        .def(py::init<>())

        .def("init_qvm",
            &SparseSimulator::init,
            "init quantum virtual machine")

        .def("prob_run_dict", &SparseSimulator::probRunDict,
            "Run quantum program and get pmeasure result as dict\n"
            "\n"
            "Args:\n"
            "    qprog: quantum program\n"
            "\n"
            "Returns:\n"
            "    measure result of quantum machine\n"
            "Raises:\n"
            "    run_fail: An error occurred in measure quantum program\n")
        .def("directlyRun", &SparseSimulator::directlyRun,
            "Run quantum program and get pmeasure result as dict\n"
            "\n"
            "Args:\n"
            "    qprog: quantum program\n"
            "\n"
            "Returns:\n"
            "     Dict[str, bool]: result of quantum program execution one shot.\n"
            "Raises:\n"
            "    run_fail: An error occurred in measure quantum program\n")

        .def("directly_run", &SparseSimulator::directlyRun,
            "Run quantum program and get pmeasure result as dict\n"
            "\n"
            "Args:\n"
            "    qprog: quantum program\n"
            "\n"
            "Returns:\n"
            "    measure result of quantum machine\n"
            "Raises:\n"
            "    run_fail: An error occurred in measure quantum program\n")

        .def("run_with_configuration", &SparseSimulator::runWithConfiguration,
            "Run quantum program and get pmeasure result as dict\n"
            "\n"
            "Args:\n"
            "    qprog: quantum program\n"
            "\n"
            "Args:\n"
            "    cbits: quantum cbits\n"
            "\n"
            "Args:\n"
            "    shots: samble shots\n"
            "\n"
            "Returns:\n"
            "    measure result of quantum machine\n"
            "Raises:\n"
            "    run_fail: An error occurred in measure quantum program\n");

    py::class_<DensityMatrixSimulator, QuantumMachine>(m, "DensityMatrixSimulator", "simulator for density matrix")
        .def(py::init<>())
        .def("init_qvm",
            &DensityMatrixSimulator::init,
            py::arg("is_double_precision") = true,
            "init quantum virtual machine")

        .def("get_probability",
            py::overload_cast<QProg&, size_t>(&DensityMatrixSimulator::get_probability),
            py::arg("prog"),
            py::arg("index"),
            "Run quantum program and get index probability\n"
            "\n"
            "Args:\n"
            "    prog: quantum program \n"
            "    index: measure index in [0,2^N - 1] \n"
            "\n"
            "Returns:\n"
            "    probability result of quantum program \n"
            "Raises:\n"
            "    run_fail: An error occurred in get_probability\n")

        .def("get_probability",
            py::overload_cast<QProg&, std::string>(&DensityMatrixSimulator::get_probability),
            py::arg("prog"),
            py::arg("index"),
            "Run quantum program and get index probability\n"
            "\n"
            "Args:\n"
            "    prog: quantum program \n"
            "    index: measure index in [0,2^N - 1] \n"
            "\n"
            "Returns:\n"
            "    probability result of quantum program \n"
            "Raises:\n"
            "    run_fail: An error occurred in get_probability\n")

        .def("get_probabilities",
            py::overload_cast<QProg&>(&DensityMatrixSimulator::get_probabilities),
            py::arg("prog"),
            "Run quantum program and get all indices probabilities\n"
            "\n"
            "Args:\n"
            "    prog: quantum program \n"
            "\n"
            "Returns:\n"
            "    probabilities result of quantum program \n"
            "Raises:\n"
            "    run_fail: An error occurred in get_probabilities\n")

        .def("get_probabilities",
            py::overload_cast<QProg&, QVec>(&DensityMatrixSimulator::get_probabilities),
            py::arg("prog"),
            py::arg("qubits"),
            "Run quantum program and get all indices probabilities for current qubits\n"
            "\n"
            "Args:\n"
            "    prog: quantum program \n"
            "    qubits: select qubits \n"
            "\n"
            "Returns:\n"
            "    probabilities result of quantum program \n"
            "Raises:\n"
            "    run_fail: An error occurred in get_probabilities\n")

        .def("get_probabilities",
            py::overload_cast<QProg&, Qnum>(&DensityMatrixSimulator::get_probabilities),
            py::arg("prog"),
            py::arg("qubits"),
            "Run quantum program and get all indices probabilities for current qubits\n"
            "\n"
            "Args:\n"
            "    prog: quantum program \n"
            "    qubits: select qubits \n"
            "\n"
            "Returns:\n"
            "    probabilities result of quantum program \n"
            "Raises:\n"
            "    run_fail: An error occurred in get_probabilities\n")

        .def("get_probabilities",
            py::overload_cast<QProg&, std::vector<string>>(&DensityMatrixSimulator::get_probabilities),
            py::arg("prog"),
            py::arg("indices"),
            "Run quantum program and get all indices probabilities for current binary indices\n"
            "\n"
            "Args:\n"
            "    prog: quantum program \n"
            "    indices: select binary indices \n"
            "\n"
            "Returns:\n"
            "    probabilities result of quantum program \n"
            "Raises:\n"
            "    run_fail: An error occurred in get_probabilities\n")

        .def("get_expectation",
            py::overload_cast<QProg&, const QHamiltonian&, const QVec&>(&DensityMatrixSimulator::get_expectation),
            py::arg("prog"),
            py::arg("hamiltonian"),
            py::arg("qubits"),
            "Run quantum program and hamiltonian expection for current qubits\n"
            "\n"
            "Args:\n"
            "    prog: quantum program \n"
            "    hamiltonian: QHamiltonian \n"
            "    qubits: select qubits \n"
            "\n"
            "Returns:\n"
            "    hamiltonian expection for current qubits\n"
            "Raises:\n"
            "    run_fail: An error occurred in get_expectation\n")

        .def("get_expectation",
            py::overload_cast<QProg&, const QHamiltonian&, const Qnum&>(&DensityMatrixSimulator::get_expectation),
            py::arg("prog"),
            py::arg("hamiltonian"),
            py::arg("qubits"),
            "Run quantum program and hamiltonian expection for current qubits\n"
            "\n"
            "Args:\n"
            "    prog: quantum program \n"
            "    hamiltonian: QHamiltonian \n"
            "    qubits: select qubits \n"
            "\n"
            "Returns:\n"
            "    hamiltonian expection for current qubits\n"
            "Raises:\n"
            "    run_fail: An error occurred in get_expectation\n")

        .def("get_density_matrix",
            py::overload_cast<QProg&>(&DensityMatrixSimulator::get_density_matrix),
            py::arg("prog"),
            "Run quantum program and get full density matrix\n"
            "\n"
            "Args:\n"
            "    prog: quantum program \n"
            "\n"
            "Returns:\n"
            "    full density matrix \n"
            "Raises:\n"
            "    run_fail: An error occurred in get_density_matrix\n")

        .def("get_reduced_density_matrix",
            py::overload_cast<QProg&, const QVec&>(&DensityMatrixSimulator::get_reduced_density_matrix),
            py::arg("prog"),
            py::arg("qubits"),
            "Run quantum program and get density matrix for current qubits\n"
            "\n"
            "Args:\n"
            "    prog: quantum program \n"
            "    qubits: quantum program select qubits\n"
            "\n"
            "Returns:\n"
            "    density matrix \n"
            "Raises:\n"
            "    run_fail: An error occurred in get_reduced_density_matrix\n")

        .def("get_reduced_density_matrix",
            py::overload_cast<QProg&, const Qnum&>(&DensityMatrixSimulator::get_reduced_density_matrix),
            py::arg("prog"),
            py::arg("qubits"),
            "Run quantum program and get density matrix for current qubits\n"
            "\n"
            "Args:\n"
            "    prog: quantum program \n"
            "    qubits: quantum program select qubits\n"
            "\n"
            "Returns:\n"
            "    density matrix \n"
            "Raises:\n"
            "    run_fail: An error occurred in get_reduced_density_matrix\n")

        /* bit-flip, phase-flip, bit-phase-flip, phase-damping, amplitude-damping, depolarizing*/
        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, double>(&DensityMatrixSimulator::set_noise_model))
        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &, double>(&DensityMatrixSimulator::set_noise_model))
        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, double, const QVec &>(&DensityMatrixSimulator::set_noise_model))
        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &, double, const QVec &>(&DensityMatrixSimulator::set_noise_model))
        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, double, const std::vector<QVec> &>(&DensityMatrixSimulator::set_noise_model))

        /*decoherence error*/
        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, double, double, double>(&DensityMatrixSimulator::set_noise_model))
        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &, double, double, double>(&DensityMatrixSimulator::set_noise_model))
        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, double, double, double, const QVec &>(&DensityMatrixSimulator::set_noise_model))
        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &, double, double, double, const QVec &>(&DensityMatrixSimulator::set_noise_model))
        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, double, double, double, const std::vector<QVec> &>(&DensityMatrixSimulator::set_noise_model));

        py::class_<Stabilizer, QuantumMachine>(m, "Stabilizer", "simulator for basic clifford simulator")
            .def(py::init<>())
            .def("init_qvm",
                &Stabilizer::init,
                "init quantum virtual machine")
            .def("run_with_configuration",
                &Stabilizer::runWithConfiguration,
                py::arg("qprog"),
                py::arg("shot"),
                py::arg_v("noise_model", NoiseModel(), "NoiseModel()"),

                "Run quantum program and get shots result \n"
                "\n"
                "Args:\n"
                "    prog: quantum program \n"
                "    int: measure shots\n"
                "\n"
                "Returns:\n"
                "    shots result of quantum program \n"
                "Raises:\n"
                "    run_fail: An error occurred in run_with_configuration\n")

            .def("prob_run_dict",
                &Stabilizer::probRunDict,
                py::arg("qprog"),
                py::arg("qubits"),
                py::arg_v("select_max", -1, "-1"),
                "Run quantum program and get probabilities\n"
                "\n"
                "Args:\n"
                "    prog: quantum program \n"
                "    qubits: pmeasure qubits\n"
                "\n"
                "Returns:\n"
                "    probabilities result of quantum program \n"
                "Raises:\n"
                "    run_fail: An error occurred in prob_run_dict\n");

    py::class_<MPSQVM, QuantumMachine> mps_qvm(m, "MPSQVM", "quantum matrix product state machine class");
    mps_qvm.def(py::init<>())
        .def("pmeasure", &MPSQVM::PMeasure, "qubit_list"_a, "select_max"_a = -1,
            "Get the probability distribution over qubits\n"
            "\n"
            "Args:\n"
            "    qubit_list: qubit list to measure\n"
            "    select_max: max returned element num in returnd tuple, should in [-1, 1<<len(qubit_list)]\n"
            "                default is -1, means no limit\n"
            "\n"
            "Returns:\n"
            "    measure result of quantum machine in tuple form", py::return_value_policy::reference)
        .def("pmeasure_no_index", &MPSQVM::PMeasure_no_index, "qubit_list"_a,
            "Get the probability distribution over qubits\n"
            "\n"
            "Args:\n"
            "    qubit_list: qubit list to measure\n"
            "\n"
            "Returns:\n"
            "    measure result of quantum machine in list form", py::return_value_policy::reference)
        .def("get_prob_tuple_list", &MPSQVM::getProbTupleList, "qubit_list"_a, "select_max"_a = -1,
            "Get pmeasure result as list\n"
            "\n"
            "Args:\n"
            "    qubit_list: pmeasure qubits list\n"
            "    select_max: max returned element num in returnd tuple, should in [-1, 1<<len(qubit_list)]\n"
            "                default is -1, means no limit"
            "\n"
            "Returns:\n"
            "    measure result of quantum machine\n"
            "Raises:\n"
            "    run_fail: An error occurred in get_prob_tuple_list\n",
            py::return_value_policy::reference)
        .def("get_prob_list", &MPSQVM::getProbList, "qubit_list"_a, "select_max"_a = -1,
            "Get pmeasure result as list\n"
            "\n"
            "Args:\n"
            "    qubit_list: pmeasure qubits list\n"
            "    select_max: max returned element num in returnd tuple, should in [-1, 1<<len(qubit_list)]\n"
            "                default is -1, means no limit"
            "\n"
            "Returns:\n"
            "    measure result of quantum machine\n"
            "Raises:\n"
            "    run_fail: An error occurred in get_prob_list\n",
            py::return_value_policy::reference)
        .def("get_prob_dict", &MPSQVM::getProbDict, "qubit_list"_a, "select_max"_a = -1,
            "Get pmeasure result as dict\n"
            "\n"
            "Args:\n"
            "    qubit_list: pmeasure qubits list\n"
            "    select_max: max returned element num in returnd tuple, should in [-1, 1<<len(qubit_list)]\n"
            "                default is -1, means no limit"
            "\n"
            "Returns:\n"
            "    measure result of quantum machine\n"
            "Raises:\n"
            "    run_fail: An error occurred in get_prob_dict\n",
            py::return_value_policy::reference)
        .def("prob_run_tuple_list", &MPSQVM::probRunTupleList, "program"_a, "qubit_list"_a, "select_max"_a = -1,
            "Run quantum program and get pmeasure result as tuple list\n"
            "\n"
            "Args:\n"
            "    qprog: quantum program\n"
            "    qubit_list: pmeasure qubits list\n"
            "    select_max: max returned element num in returnd tuple, should in [-1, 1<<len(qubit_list)]\n"
            "              default is -1, means no limit"
            "\n"
            "Returns:\n"
            "  measure result of quantum machine\n"
            "Raises:\n"
            "    run_fail: An error occurred in prob_run_tuple_list\n",
            py::return_value_policy::reference)
        .def("prob_run_list", &MPSQVM::probRunList, "program"_a, "qubit_list"_a, "select_max"_a = -1,
            "Run quantum program and get pmeasure result as list\n"
            "\n"
            "Args:\n"
            "    qprog: quantum program\n"
            "    qubit_list: pmeasure qubits list\n"
            "    select_max: max returned element num in returnd tuple, should in [-1, 1<<len(qubit_list)]\n"
            "                default is -1, means no limit"
            "\n"
            "Returns:\n"
            "    measure result of quantum machine\n"
            "Raises:\n"
            "    run_fail: An error occurred in measure quantum program\n",
            py::return_value_policy::reference)
        .def("prob_run_dict", &MPSQVM::probRunDict, "program"_a, "qubit_list"_a, "select_max"_a = -1,
            "Run quantum program and get pmeasure result as dict\n"
            "\n"
            "Args:\n"
            "    qprog: quantum program\n"
            "    qubit_list: pmeasure qubits list\n"
            "    select_max: max returned element num in returnd tuple, should in [-1, 1<<len(qubit_list)]\n"
            "                default is -1, means no limit"
            "\n"
            "Returns:\n"
            "    measure result of quantum machine\n"
            "Raises:\n"
            "    run_fail: An error occurred in measure quantum program\n",
            py::return_value_policy::reference)
        .def("quick_measure", &MPSQVM::quickMeasure, "qubit_list"_a, "shots"_a,
            "Quick measure\n"
            "\n"
            "Args:\n"
            "    qubit_list: qubit list to measure\n"
            "    shots: the repeat num  of measure operate\n"
            "\n"
            "Returns:\n"
            "    result of quantum program\n"
            "Raises:\n"
            "    run_fail: An error occurred in measure quantum program\n",
            py::return_value_policy::reference)

        .def("pmeasure_bin_index", &MPSQVM::pmeasure_bin_index, "program"_a, "string"_a,
            "pmeasure bin index quantum state amplitude\n"
            "\n"
            "Args:\n"
            "    string : bin string\n"
            "\n"
            "Returns:\n"
            "    complex : bin amplitude\n"
            "Raises:\n"
            "    run_fail: An error occurred in pmeasure_bin_index\n",
            py::return_value_policy::reference)
        .def("pmeasure_dec_index", &MPSQVM::pmeasure_dec_index, "program"_a, "string"_a,
            "pmeasure dec index quantum state amplitude\n"
            "\n"
            "Args:\n"
            "    string : dec string\n"
            "\n"
            "Returns:\n"
            "    complex : dec amplitude\n"
            "Raises:\n"
            "    run_fail: An error occurred in pmeasure_dec_index\n",
            py::return_value_policy::reference)
        .def("pmeasure_bin_subset", &MPSQVM::pmeasure_bin_subset, "program"_a, "string_list"_a,
            "pmeasure quantum state amplitude subset\n"
            "\n"
            "Args:\n"
            "    list : bin state string list\n"
            "\n"
            "Returns:\n"
            "    list : bin amplitude result list\n"
            "Raises:\n"
            "    run_fail: An error occurred in pmeasure_bin_subset\n",
            py::return_value_policy::reference)
        .def("pmeasure_dec_subset", &MPSQVM::pmeasure_dec_subset, "program"_a, "string_list"_a,
            "pmeasure quantum state amplitude subset\n"
            "\n"
            "Args:\n"
            "    list : dec state string list\n"
            "\n"
            "Returns:\n"
            "    list : dec amplitude result list\n"
            "Raises:\n"
            "    run_fail: An error occurred in pmeasure_dec_subset\n",
            py::return_value_policy::reference)

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

#if defined(USE_OPENSSL) && defined(USE_CURL)

    py::enum_<RealChipType>(m, "real_chip_type", "origin quantum real chip type enum")
        .value("origin_wuyuan_d3", RealChipType::ORIGIN_WUYUAN_D3)
        .value("origin_wuyuan_d4", RealChipType::ORIGIN_WUYUAN_D4)
        .value("origin_wuyuan_d5", RealChipType::ORIGIN_WUYUAN_D5)
        .export_values();

    py::class_<QCloudMachine, QuantumMachine>(m, "QCloud", "origin quantum cloud machine")
        .def(py::init<>())
        .def("init_qvm",
            &QCloudMachine::init,
            py::arg("token"),
            py::arg("is_logged") = false,
            "init quantum virtual machine")

        // url setting
        .def("set_qcloud_api", &QCloudMachine::set_qcloud_api)

        // noise
        .def("set_noise_model", &QCloudMachine::set_noise_model)
        .def("noise_measure",
            &QCloudMachine::noise_measure,
            py::arg("prog"),
            py::arg("shot"),
            py::arg("task_name") = "QPanda Experiment")

        // full_amplitude
        .def("full_amplitude_measure",
            &QCloudMachine::full_amplitude_measure,
            py::arg("prog"),
            py::arg("shot"),
            py::arg("task_name") = "QPanda Experiment")

        .def("full_amplitude_pmeasure",
            &QCloudMachine::full_amplitude_pmeasure,
            py::arg("prog"),
            py::arg("qvec"),
            py::arg("task_name") = "QPanda Experiment")

        // partial_amplitude
        .def("partial_amplitude_pmeasure",
            &QCloudMachine::partial_amplitude_pmeasure,
            py::arg("prog"),
            py::arg("amp_vec"),
            py::arg("task_name") = "QPanda Experiment")

        // single_amplitude
        .def("single_amplitude_pmeasure",
            &QCloudMachine::single_amplitude_pmeasure,
            py::arg("prog"),
            py::arg("amplitude"),
            py::arg("task_name") = "QPanda Experiment")
/*
        // real chip expectation!
        .def("real_chip_expectation",
            [](QCloudMachine& machine,
               QProg& prog,
               const std::string &hamiltonian,
               const std::vector<uint32_t> qubits,
               int shot,
               int chip_id = (int)RealChipType::ORIGIN_WUYUAN_D5,
               bool is_amend = true,
               bool is_mapping = true,
               bool is_optimization = true,
               std::string task_name = "QPanda Experiment")
            {
                auto real_chip_type = static_cast<RealChipType>(chip_id);
                return machine.real_chip_expectation(prog, hamiltonian, qubits, shot, real_chip_type, is_amend, is_mapping, is_optimization, task_name);
            },
            py::arg("prog"),
            py::arg("hamiltonian"),
            py::arg("qubits"),
            py::arg("shot"),
            py::arg("chip_id") = (int)RealChipType::ORIGIN_WUYUAN_D5,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("task_name") = "QPanda Experiment")
*/
        // real chip measure
        .def("real_chip_measure",
            [](QCloudMachine& machine,
               QProg& prog, 
               int shot, 
               int chip_id = (int)RealChipType::ORIGIN_WUYUAN_D5,
               bool is_amend = true,
               bool is_mapping = true,
               bool is_optimization = true,
               std::string task_name = "QPanda Experiment")
            {
                auto real_chip_type = static_cast<RealChipType>(chip_id);
                return machine.real_chip_measure(prog, shot, real_chip_type, is_amend, is_mapping, is_optimization, task_name);
            },
            py::arg("prog"),
            py::arg("shot"),
            py::arg("chip_id") = (int)RealChipType::ORIGIN_WUYUAN_D5,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("task_name") = "QPanda Experiment")

        // get state fidelity
        .def("get_state_fidelity",
            [](QCloudMachine& machine,
               QProg& prog,
               int shot,
               int chip_id = (int)RealChipType::ORIGIN_WUYUAN_D5,
               bool is_amend = true,
               bool is_mapping = true,
               bool is_optimization = true,
               std::string task_name = "QPanda Experiment")
            {
                auto real_chip_type = static_cast<RealChipType>(chip_id);
                return machine.get_state_fidelity(prog, shot, real_chip_type, is_amend, is_mapping, is_optimization, task_name);
            },
            py::arg("prog"),
            py::arg("shot"),
            py::arg("chip_id") = (int)RealChipType::ORIGIN_WUYUAN_D5,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("task_name") = "QPanda Experiment")


        // real chip get_state_tomography_density
        .def("get_state_tomography_density",
            [](QCloudMachine& machine,
               QProg& prog,
               int shot,
               int chip_id = (int)RealChipType::ORIGIN_WUYUAN_D5,
               bool is_amend = true,
               bool is_mapping = true,
               bool is_optimization = true,
               std::string task_name = "QPanda Experiment")
            {
                auto real_chip_type = static_cast<RealChipType>(chip_id);
                return machine.get_state_tomography_density(prog, shot, real_chip_type, is_amend, is_mapping, is_optimization, task_name);
            },
            py::arg("prog"),
            py::arg("shot"),
            py::arg("chip_id") = (int)RealChipType::ORIGIN_WUYUAN_D5,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("task_name") = "QPanda Experiment")

#if 0

        // real chip measure
        .def("real_chip_measure",
            &QCloudMachine::real_chip_measure,
            py::arg("prog"),
            py::arg("shot"),
            py::arg_v("chip_id", (size_t)RealChipType::ORIGIN_WUYUAN_D5, "real_chip_type.origin_wuyuan_d5"),
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("task_name") = "QPanda Experiment")

        // real chip get_state_fidelity
        .def("get_state_fidelity",
            &QCloudMachine::get_state_fidelity,
            py::arg("prog"),
            py::arg("shot"),
            py::arg_v("chip_id", (size_t)RealChipType::ORIGIN_WUYUAN_D5, "real_chip_type.origin_wuyuan_d5"),
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("task_name") = "QPanda Experiment")

        // real chip get_state_tomography_density
        .def("get_state_tomography_density",
            &QCloudMachine::get_state_tomography_density,
            py::arg("prog"),
            py::arg("shot"),
            py::arg_v("chip_id", (size_t)RealChipType::ORIGIN_WUYUAN_D5, "real_chip_type.origin_wuyuan_d5"),
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("task_name") = "QPanda Experiment")
#endif

        // get_expectation
        .def("get_expectation",
            &QCloudMachine::get_expectation,
            py::arg("prog"),
            py::arg("hamiltonian"),
            py::arg("qvec"),
            py::arg("task_name") = "QPanda Experiment")

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
            py::arg_v("chip_id", (size_t)RealChipType::ORIGIN_WUYUAN_D3, "real_chip_type.origin_wuyuan_d3"),
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("task_name") = "QPanda Experiment");

   
#endif // USE_CURL

    py::implicitly_convertible<CPUQVM, QuantumMachine>();
    py::implicitly_convertible<GPUQVM, QuantumMachine>();
    py::implicitly_convertible<CPUSingleThreadQVM, QuantumMachine>();
    py::implicitly_convertible<NoiseQVM, QuantumMachine>();
    py::implicitly_convertible<SingleAmplitudeQVM, QuantumMachine>();
    py::implicitly_convertible<PartialAmplitudeQVM, QuantumMachine>();
    py::implicitly_convertible<Stabilizer, QuantumMachine>();
    py::implicitly_convertible<DensityMatrixSimulator, QuantumMachine>();
}
