#include "Core/Core.h"
#include "Core/QuantumMachine/SingleAmplitudeQVM.h"
#include "Core/QuantumMachine/PartialAmplitudeQVM.h"
#include "Core/VirtualQuantumProcessor/NoiseQPU/NoiseModel.h"
#include "Core/VirtualQuantumProcessor/SparseQVM/SparseQVM.h"
#include "Core/VirtualQuantumProcessor/Stabilizer/Stabilizer.h"
#include "Core/QuantumCloud/QCloudService.h"
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

USING_QPANDA
namespace py = pybind11;
using namespace pybind11::literals;

// template<>
// struct py::detail::type_caster<QVec>  : py::detail::list_caster<QVec, Qubit*> {};
void export_quantum_machine(py::module &m)
{

    py::class_<QuantumMachine>(m, "QuantumMachine", "quantum machine base class")
        .def("set_configure",
            [](QuantumMachine &qvm, size_t max_qubit, size_t max_cbit){
                Configuration config = { max_qubit, max_cbit };
                qvm.setConfig(config);
            },
            py::arg("max_qubit"),
            py::arg("max_cbit"),
            "Set the maximum qubit and cbit numbers for the QVM.\n"
            "\n"
            "Args:\n"
            "     max_qubit: Maximum number of qubits in the quantum machine.\n"
            "\n"
            "     max_cbit: Maximum number of cbits in the quantum machine.\n"
            "\n"
            "Returns:\n"
            "     None\n")

        .def("finalize", &QuantumMachine::finalize,
            "Finalize the quantum machine.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     None\n")

        .def("get_qstate", &QuantumMachine::getQState,
            "Get the status of the quantum machine.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     QMachineStatus: The current status of the quantum machine.\n",
            py::return_value_policy::automatic)

        .def("qAlloc", &QuantumMachine::allocateQubit,
            "Allocate a qubit.\n"
            "\n"
            "This function must be called after init().\n"
            "\n"
            "Args:\n"
            "     qubit_addr: The physical address of the qubit, should be in the range [0, 29).\n",
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
            "Allocate multiple qubits.\n"
            "\n"
            "This function must be called after init().\n"
            "\n"
            "Args:\n"
            "     qubit_num: The number of qubits to allocate.\n"
            "\n"
            "Returns:\n"
            "     list[Qubit]: A list of allocated qubits.\n",
            py::return_value_policy::reference)

        .def("cAlloc",
            py::overload_cast<>(&QuantumMachine::allocateCBit),
            "Allocate a classical bit (CBit).\n"
            "\n"
            "This function must be called after init().\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     CBit: A reference to the allocated classical bit.\n",
            py::return_value_policy::reference
        )

        .def("cAlloc_many", &QuantumMachine::allocateCBits,
            py::arg("cbit_num"),
            "Allocate multiple classical bits (CBits).\n"
            "\n"
            "This function must be called after init().\n"
            "\n"
            "Args:\n"
            "     cbit_num: The number of classical bits to allocate.\n"
            "\n"
            "Returns:\n"
            "     list[CBit]: A list of allocated classical bits.\n",
            py::return_value_policy::reference
        )

        .def("qFree",
            &QuantumMachine::Free_Qubit,
            py::arg("qubit"),
            "Free a qubit.\n"
            "\n"
            "This function deallocates a previously allocated qubit.\n"
            "\n"
            "Args:\n"
            "     qubit: The Qubit to be freed.\n"
            "\n"
            "Returns:\n"
            "     None: This function does not return a value.\n"
        )

        .def("qFree_all",
            &QuantumMachine::Free_Qubits,
            py::arg("qubit_list"),
            "Free all allocated qubits.\n"
            "\n"
            "This function deallocates all qubits that have been previously allocated.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     None: This function does not return a value.\n"
        )

        .def("qFree_all", py::overload_cast<QVec &>(&QuantumMachine::qFreeAll),
            "Free all qubits.\n"
            "\n"
            "This function deallocates all qubits provided in the input vector.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     None: This function does not return a value.\n"
        )

        .def("cFree", &QuantumMachine::Free_CBit,
            "Free a classical bit (CBit).\n"
            "\n"
            "This function deallocates a previously allocated classical bit.\n"
            "\n"
            "Args:\n"
            "     CBit: The classical bit to be freed.\n"
            "\n"
            "Returns:\n"
            "     None: This function does not return a value.\n"
        )

        .def("cFree_all",
            &QuantumMachine::Free_CBits,
            py::arg("cbit_list"),
                "Free all allocated classical bits (CBits).\n"
                "\n"
                "This function deallocates all classical bits provided in the input list.\n"
                "\n"
                "Args:\n"
                "     None\n"
                "\n"
                "Returns:\n"
                "     None: This function does not return a value.\n"
            )

        .def("cFree_all", py::overload_cast<>(&QuantumMachine::cFreeAll),
            "Free all classical bits (CBits).\n"
            "\n"
            "This function deallocates all classical bits that have been previously allocated.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     None: This function does not return a value.\n"
        )

        .def("getStatus", &QuantumMachine::getStatus,
            "Get the status of the Quantum machine.\n"
            "\n"
            "This function retrieves the current status of the Quantum machine.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     QMachineStatus: The status of the Quantum machine.\n",
            py::return_value_policy::reference_internal
        )

        /*will delete*/
        .def("initQVM", &QuantumMachine::init,
            "Initialize the global unique quantum machine in the background.\n"
            "\n"
            "This function sets up the quantum machine based on the specified type.\n"
            "\n"
            "Args:\n"
            "     machine_type: The type of quantum machine to initialize, as defined in pyQPanda.QMachineType.\n"
            "\n"
            "Returns:\n"
            "     bool: True if the initialization is successful, otherwise false.\n"
        )

        .def("getAllocateQubitNum", &QuantumMachine::getAllocateQubit,
            "Get the list of allocated qubits in the QuantumMachine.\n"
            "\n"
            "This function retrieves the qubits that have been allocated for use in the quantum machine.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     List of allocated qubits.\n",
            py::return_value_policy::reference
        )

        .def("getAllocateCMem", &QuantumMachine::getAllocateCMem,
            "Get the list of allocated classical bits (cbits) in the QuantumMachine.\n"
            "\n"
            "This function retrieves the cbits that have been allocated for use in the quantum machine.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     List of allocated cbits.\n",
            py::return_value_policy::reference
        )

        /* new interface */
        .def("init_qvm", &QuantumMachine::init,
            "Initialize the global unique quantum machine in the background.\n"
            "\n"
            "This function sets up the quantum machine based on the specified type.\n"
            "\n"
            "Args:\n"
            "     machine_type: The type of quantum machine to initialize, as defined in pyQPanda.QMachineType.\n"
            "\n"
            "Returns:\n"
            "     bool: True if the initialization is successful, otherwise false.\n"
        )

        .def("init_state",
            &QuantumMachine::initState,
            py::arg_v("state", QStat(), "QStat()"),
            py::arg_v("qlist", QVec(), "QVec()"),
            py::return_value_policy::reference,
            "Initialize the quantum state of the QuantumMachine.\n"
            "\n"
            "This function sets the initial state of the quantum machine.\n"
            "\n"
            "Args:\n"
            "     state: The initial quantum state, represented as a QStat object. Defaults to QStat().\n"
            "\n"
            "     qlist: The list of qubits to which the state will be applied, represented as a QVec object. Defaults to QVec().\n"
            "\n"
            "Returns:\n"
            "     Reference to the updated quantum machine.\n"
        )

        .def("init_sparse_state",
            &QuantumMachine::initSparseState,
            py::arg_v("state", std::map<std::string, qcomplex_t>(), "std::map<std::string, qcomplex_t>()"),
            py::arg_v("qlist", QVec(), "QVec()"),
            py::return_value_policy::reference,
            "Initialize a sparse quantum state for the QuantumMachine.\n"
            "\n"
            "This function sets the initial sparse state of the quantum machine.\n"
            "\n"
            "Args:\n"
            "     state: A map representing the sparse state, where keys are state identifiers and values are qcomplex_t. Defaults to an empty map.\n"
            "\n"
            "     qlist: The list of qubits to which the sparse state will be applied, represented as a QVec object. Defaults to QVec().\n"
            "\n"
            "Returns:\n"
            "     Reference to the updated quantum machine.\n"
        )

        .def("cAlloc", py::overload_cast<size_t>(&QuantumMachine::allocateCBit),
            py::arg("cbit"),
            "Allocate a classical bit (CBit) in the QuantumMachine.\n"
            "\n"
            "This function allocates a CBit after the quantum machine has been initialized.\n"
            "\n"
            "Args:\n"
            "     cbit_addr: The address of the CBit to allocate, which should be in the range [0, 29).\n"
            "\n"
            "Returns:\n"
            "     Reference to the allocated CBit.\n",
            py::return_value_policy::reference
        )

        .def("get_status", &QuantumMachine::getStatus,
            "Retrieve the status of the QuantumMachine.\n"
            "\n"
            "This function returns the current status of the quantum machine.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     The status of the Quantum machine, represented as a QMachineStatus.\n",
            py::return_value_policy::reference_internal
        )

        .def("get_allocate_qubit_num", &QuantumMachine::getAllocateQubit,
            "Retrieve the list of allocated qubits in the QuantumMachine.\n"
            "\n"
            "This function returns the currently allocated qubits.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A list of allocated qubits.\n",
            py::return_value_policy::reference
        )

        .def("get_allocate_cmem_num", &QuantumMachine::getAllocateCMem,
            "Retrieve the list of allocated cbits in the QuantumMachine.\n"
            "\n"
            "This function returns the currently allocated cbits.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A list of allocated cbits.\n",
            py::return_value_policy::reference
        )

        .def("allocate_qubit_through_phy_address", &QuantumMachine::allocateQubitThroughPhyAddress,
            py::arg("address"),
            "Allocate qubits through physical address.\n"
            "\n"
            "This function allocates qubits using the specified physical address.\n"
            "\n"
            "Args:\n"
            "     address: The physical address of the qubit.\n"
            "\n"
            "Returns:\n"
            "     The allocated qubit.\n",
            py::return_value_policy::reference
        )

        .def("allocate_qubit_through_vir_address", &QuantumMachine::allocateQubitThroughVirAddress,
            py::arg("address"),
            "Allocate a qubit using its physical address.\n"
            "\n"
            "This function allocates a qubit based on the specified physical address.\n"
            "\n"
            "Args:\n"
            "     address: The physical address of the qubit to allocate.\n"
            "\n"
            "Returns:\n"
            "     A reference to the allocated Qubit.\n",
            py::return_value_policy::reference
        )

        .def("get_gate_time_map", &QuantumMachine::getGateTimeMap,
            "Retrieve the gate time mapping for the QuantumMachine.\n"
            "\n"
            "This function returns a map of gates to their corresponding execution times.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A reference to the gate time map.\n",
            py::return_value_policy::reference
        )

//.def("get_allocate_qubits", get_allocate_qubits, "qubit vector"_a,  py::return_value_policy::reference)
//.def("get_allocate_cbits", get_allocate_cbits, "cbit vector"_a,  py::return_value_policy::reference)

        .def(
            "get_allocate_qubits",
            [](QuantumMachine &self)
            {
                QVec qv;
                self.get_allocate_qubits(qv);
                return static_cast<std::vector<Qubit*>>(qv);
            },
            "Retrieve the list of allocated qubits in the QuantumMachine.\n"
            "\n"
            "This function returns a list of currently allocated qubits.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A list of allocated qubits.\n",
            py::return_value_policy::reference
        )

        .def(
        "get_allocate_cbits",
            [](QuantumMachine &self)
            {
                std::vector<ClassicalCondition> cv;
                self.get_allocate_cbits(cv);
                return cv;
            },
            "Retrieve the list of allocated cbits in the QuantumMachine.\n"
            "\n"
            "This function returns a list of currently allocated cbits.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A list of allocated cbits.\n",
            py::return_value_policy::reference
        )

        .def("get_expectation",
            py::overload_cast<QProg, const QHamiltonian &, const QVec &>(&QuantumMachine::get_expectation),
            py::arg("qprog"),
            py::arg("hamiltonian"),
            py::arg("qubit_list"),
            "Calculate the expectation value of the given Hamiltonian.\n"
            "\n"
            "This function computes the expectation value based on the provided quantum program,\n"
            "\n"
            "Hamiltonian, and list of qubits to measure.\n"
            "\n"
            "Args:\n"
            "     qprog: The quantum program to execute.\n"
            "\n"
            "     hamiltonian: The Hamiltonian for which the expectation is calculated.\n"
            "\n"
            "     qubit_list: A list of qubits to measure.\n"
            "\n"
            "Returns:\n"
            "     A double representing the expectation value of the current Hamiltonian.\n",
            py::return_value_policy::reference
        )

        .def("get_expectation",
            py::overload_cast<QProg, const QHamiltonian &, const QVec &, int>(&QuantumMachine::get_expectation),
            py::arg("qprog"),
            py::arg("hamiltonian"),
            py::arg("qubit_list"),
            py::arg("shots"),
            "Calculate the expectation value of the given Hamiltonian with specified measurement shots.\n"
            "\n"
            "This function computes the expectation value based on the provided quantum program,\n"
            "\n"
            "Hamiltonian, list of qubits to measure, and the number of measurement shots.\n"
            "\n"
            "Args:\n"
            "     qprog: The quantum program to execute.\n"
            "\n"
            "     hamiltonian: The Hamiltonian for which the expectation is calculated.\n"
            "\n"
            "     qubit_list: A list of qubits to measure.\n"
            "\n"
            "     shots: The number of measurement shots to perform.\n"
            "\n"
            "Returns:\n"
            "     A double representing the expectation value of the current Hamiltonian.\n",
            py::return_value_policy::reference
        )

        .def("get_processed_qgate_num", &QuantumMachine::get_processed_qgate_num, 
            py::return_value_policy::reference,
            "Retrieve the number of processed quantum gates.\n"
            "\n"
            "This function returns the total count of quantum gates that have been processed\n"
            "\n"
            "by the QuantumMachine.\n"
            "\n"
            "Returns:\n"
            "     An integer representing the number of processed quantum gates.\n"
        )

        .def("async_run",
            &QuantumMachine::async_run,
            py::arg("qprog"),
            py::arg_v("noise_model", NoiseModel(), "NoiseModel()"),
            "Execute the quantum program asynchronously in the background.\n"
            "\n"
            "This function runs the specified quantum program without blocking the main thread.\n"
            "\n"
            "You can check the progress using get_processed_qgate_num(), determine if the process\n"
            "\n"
            "is finished with is_async_finished(), and retrieve results with get_async_result().\n"
            "\n"
            "Args:\n"
            "     qprog: The quantum program to run.\n"
            "\n"
            "     noise_model: (optional) The noise model to apply (default is NoiseModel()).\n"
            "\n"
            "Returns:\n"
            "     A reference indicating the status of the asynchronous operation.\n",
            py::return_value_policy::reference
        )

        .def("is_async_finished", &QuantumMachine::is_async_finished, 
            py::return_value_policy::reference,
            "Check if the asynchronous quantum program execution is complete.\n"
            "\n"
            "This function returns a boolean indicating whether the asynchronous process\n"
            "\n"
            "initiated by async_run() has finished.\n"
            "\n"
            "Returns:\n"
            "     True if the process is complete, False otherwise.\n"
        )

        .def("get_async_result", &QuantumMachine::get_async_result, 
            py::return_value_policy::reference,
            "Retrieve the result of the asynchronous quantum program execution.\n"
            "\n"
            "This function blocks the current code until the asynchronous process initiated\n"
            "\n"
            "by async_run() is complete, then returns the results.\n"
            "\n"
            "Returns:\n"
            "     The result of the asynchronous execution.\n"
        )

        .def("directly_run",
            &QuantumMachine::directlyRun,
            py::arg("qprog"),
            py::arg_v("noise_model", NoiseModel(), "NoiseModel()"),
            py::call_guard<py::gil_scoped_release>(),
            py::return_value_policy::reference,
            "Directly execute the quantum program.\n"
            "\n"
            "This function runs the specified quantum program immediately after the\n"
            "\n"
            "initialization (init()). It supports an optional noise model, which is\n"
            "\n"
            "currently only applicable to CPUQVM.\n"
            "\n"
            "Args:\n"
            "     qprog: The quantum program to run.\n"
            "\n"
            "     noise_model: (optional) The noise model to apply (default is no noise).\n"
            "\n"
            "Returns:\n"
            "     A dictionary with the execution results:\n"
            "         The final qubit register state.\n"
            "         The measurement probabilities.\n"
        )

        .def("run_with_configuration",
            [](QuantumMachine &qvm, QProg &prog, vector<ClassicalCondition> &cc_vector, 
                py::dict param, NoiseModel noise_model)
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
            "Execute the quantum program with a specified configuration.\n"
            "\n"
            "This function runs the quantum program using the provided classical bits,\n"
            "\n"
            "parameters, and an optional noise model. It supports multiple shots for\n"
            "\n"
            "repeated execution.\n"
            "\n"
            "Args:\n"
            "     qprog: The quantum program to execute.\n"
            "\n"
            "     cbit_list: The list of classical bits.\n"
            "\n"
            "     data: Parameters for the execution in dictionary form.\n"
            "\n"
            "     noise_model: (optional) The noise model to apply (default is no noise).\n"
            "\n"
            "Returns:\n"
            "     The execution results over the specified shots, including:\n"
            "         The final qubit register state.\n"
            "         The count of hits for each outcome.\n",
            py::return_value_policy::automatic
        )

        .def("run_with_configuration",
            py::overload_cast<QProg &, vector<ClassicalCondition> &, int, const NoiseModel &>(&QuantumMachine::runWithConfiguration),
            py::arg("qprog"),
            py::arg("cbit_list"),
            py::arg("shot"),
            py::arg_v("noise_model", NoiseModel(), "NoiseModel()"),
            py::call_guard<py::gil_scoped_release>(),
            "Execute the quantum program with a specified configuration.\n"
            "\n"
            "This function runs the quantum program using the provided classical bits,\n"
            "\n"
            "parameters, and an optional noise model. It supports multiple shots for\n"
            "\n"
            "repeated execution.\n"
            "\n"
            "Args:\n"
            "     qprog: The quantum program to execute.\n"
            "\n"
            "     cbit_list: The list of classical bits.\n"
            "\n"
            "     shot: The number of times to repeat the execution.\n"
            "\n"
            "     noise_model: (optional) The noise model to apply (default is no noise).\n"
            "\n"
            "Note: Noise models currently only work on CPUQVM.\n"
            "\n"
            "Returns:\n"
            "     A tuple containing the results of the quantum program execution:\n"
            "         The final qubit register state.\n"
            "         The count of hits for each outcome.\n",
            py::return_value_policy::automatic
        )

        .def("run_with_configuration",
            py::overload_cast<QProg &, int, const NoiseModel &>(&QuantumMachine::runWithConfiguration),
            py::arg("qprog"),
            py::arg("shot"),
            py::arg_v("noise_model", NoiseModel(), "NoiseModel()"),
            py::call_guard<py::gil_scoped_release>(),
            "Execute the quantum program with a specified configuration.\n"
            "\n"
            "This function runs the quantum program using the provided quantum program,\n"
            "\n"
            "number of shots, and an optional noise model. It supports multiple shots\n"
            "\n"
            "for repeated execution.\n"
            "\n"
            "Args:\n"
            "     qprog: The quantum program to execute.\n"
            "\n"
            "     shot: The number of times to repeat the execution.\n"
            "\n"
            "     noise_model: (optional) The noise model to apply (default is no noise).\n"
            "\n"
            "Returns:\n"
            "     The execution results over the specified shots, including:\n"
            "         The final qubit register state.\n"
            "         The count of hits for each outcome.\n",
            py::return_value_policy::automatic
        )

        .def("run_with_configuration",
            py::overload_cast<QProg &, vector<int> &, int, const NoiseModel &>(&QuantumMachine::runWithConfiguration),
            py::arg("qprog"),
            py::arg("cbit_list"),
            py::arg("shot"),
            py::arg_v("noise_model", NoiseModel(), "NoiseModel()"),
            py::call_guard<py::gil_scoped_release>(),
            "Execute the quantum program with a specified configuration.\n"
            "\n"
            "This function runs the quantum program using the provided classical bits,\n"
            "\n"
            "the number of shots for repeated execution, and an optional noise model.\n"
            "\n"
            "Args:\n"
            "     qprog: The quantum program to execute.\n"
            "\n"
            "     cbit_list: The list of classical bits.\n"
            "\n"
            "     shot: The number of times to repeat the execution.\n"
            "\n"
            "     noise_model: (optional) The noise model to apply (default is no noise). Note: Noise models currently work only on CPUQVM.\n"
            "\n"
            "Returns:\n"
            "     A tuple containing the execution results over the specified shots:\n"
            "         The final qubit register state.\n"
            "         The count of hits for each outcome.\n",
            py::return_value_policy::automatic
        );

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
        "Set the maximum number of threads for the CPU quantum virtual machine (QVM).\n"
        "\n"
        "Args:\n"
        "     size: The maximum number of threads to use.\n"
        "\n"
        "Returns:\n"
        "     None: This method does not return a value.\n",
        py::return_value_policy::automatic
    );


    /*
      we should declare these function in py::class_<IdealQVM>, then CPUQVM inherit form it,
      but as we won't want to export IdealQVM to user, this may the only way
    */
    export_idealqvm_func<CPUQVM>::export_func(cpu_qvm);
    py::class_<CPUSingleThreadQVM, QuantumMachine> cpu_single_thread_qvm(m,
        "CPUSingleThreadQVM", "quantum machine class for cpu single thread");
    export_idealqvm_func<CPUSingleThreadQVM>::export_func(cpu_single_thread_qvm);

#ifdef USE_CUDA


    py::class_<FullAmplitudeQVM, QuantumMachine> full_qvm(m, "FullAmplitudeQVM");
    export_idealqvm_func<FullAmplitudeQVM>::export_func(full_qvm);
    full_qvm.def("init_qvm", [](FullAmplitudeQVM& qvm, std::string backend)
        {
            if ("CPU" == backend || "cpu" == backend)
            {
                return qvm.init(BackendType::CPU);
            }
            else if ("GPU" == backend || "gpu" == backend)
            {
                return qvm.init(BackendType::GPU);

            }
            else
            {
                QCERR_AND_THROW(std::runtime_error, "FullAmplitudeQVM init only support 'CPU' or 'GPU'.");
            }
        });

    full_qvm.def("init_qvm", [](FullAmplitudeQVM& qvm)
        {
            return qvm.init(BackendType::CPU);
        });


    py::class_<GPUQVM, QuantumMachine> gpu_qvm(m, "GPUQVM");
    export_idealqvm_func<GPUQVM>::export_func(gpu_qvm);
#endif // USE_CUDA

    py::class_<NoiseQVM, QuantumMachine>(m, "NoiseQVM", "quantum machine class for simulate noise prog")
        .def(py::init<>())

        .def("set_max_threads",
            &NoiseQVM::set_parallel_threads,
            py::arg("size"),
            "Set the maximum number of threads for the noise quantum virtual machine (NoiseQVM).\n"
            "\n"
            "Args:\n"
            "     size: The maximum number of threads to utilize.\n"
            "\n"
            "Returns:\n"
            "     None: This method does not return a value.\n",
            py::return_value_policy::automatic)

        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, double>(&NoiseQVM::set_noise_model),
            "Set the noise model for the quantum virtual machine.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model to be applied.\n"
            "\n"
            "     gate_type: The type of gate for which the noise model is relevant.\n"
            "\n"
            "     noise_level: A double representing the level of noise to apply.\n"
            "\n"
            "Returns:\n"
            "     None, as the function configures the noise model in place.\n")

        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &, double>
            (&NoiseQVM::set_noise_model),
            "Set the noise model for multiple gate types in the quantum virtual machine.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model to be applied.\n"
            "\n"
            "     gate_types: A vector of gate types for which the noise model is relevant.\n"
            "\n"
            "     noise_level: A double representing the level of noise to apply.\n"
            "\n"
            "Returns:\n"
            "     None, as the function configures the noise model in place for the specified gate types.\n")

        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, double, const QVec &>
            (&NoiseQVM::set_noise_model),
            "Set the noise model for a specific gate type and qubit vector in the quantum virtual machine.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model to be applied.\n"
            "\n"
            "     gate_type: The type of gate for which the noise model is relevant.\n"
            "\n"
            "     noise_level: A double representing the level of noise to apply.\n"
            "\n"
            "     qubits: A vector of qubits (QVec) affected by the noise model.\n"
            "\n"
            "Returns:\n"
            "     None, as the function configures the noise model in place for the specified gate type and qubits.\n")

        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &, double, const QVec &>
            (&NoiseQVM::set_noise_model),
            "Set the noise model for multiple gate types and a specific qubit vector in the quantum virtual machine.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model to be applied.\n"
            "\n"
            "     gate_types: A vector of gate types for which the noise model is relevant.\n"
            "\n"
            "     noise_level: A double representing the level of noise to apply.\n"
            "\n"
            "     qubits: A vector of qubits (QVec) affected by the noise model.\n"
            "\n"
            "Returns:\n"
            "     None, as the function configures the noise model in place for the specified gate types and qubits.\n")

        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, double, const std::vector<QVec> &>
            (&NoiseQVM::set_noise_model),
            "Set the noise model for a specific gate type and multiple qubit vectors in the quantum virtual machine.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model to be applied.\n"
            "\n"
            "     gate_type: The type of gate for which the noise model is relevant.\n"
            "\n"
            "     noise_level: A double representing the level of noise to apply.\n"
            "\n"
            "     qubits: A vector of qubit vectors (std::vector<QVec>) affected by the noise model.\n"
            "\n"
            "Returns:\n"
            "     None, as the function configures the noise model in place for the specified gate type and qubit vectors.\n"
            )

        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, double, double, double>
            (&NoiseQVM::set_noise_model),
            "Set the noise model for a specific gate type with multiple noise parameters in the quantum virtual machine.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model to be applied.\n"
            "\n"
            "     gate_type: The type of gate for which the noise model is relevant.\n"
            "\n"
            "     noise_level1: A double representing the first level of noise to apply.\n"
            "\n"
            "     noise_level2: A double representing the second level of noise to apply.\n"
            "\n"
            "     noise_level3: A double representing the third level of noise to apply.\n"
            "\n"
            "Returns:\n"
            "     None, as the function configures the noise model in place for the specified gate type with the given noise parameters.\n"
            )

        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &, double, double, double>
            (&NoiseQVM::set_noise_model),
            "Set the noise model for multiple gate types with various noise parameters in the quantum virtual machine.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model to be applied.\n"
            "\n"
            "     gate_types: A vector of gate types for which the noise model is relevant.\n"
            "\n"
            "     noise_level1: A double representing the first level of noise to apply.\n"
            "\n"
            "     noise_level2: A double representing the second level of noise to apply.\n"
            "\n"
            "     noise_level3: A double representing the third level of noise to apply.\n"
            "\n"
            "Returns:\n"
            "     None, as the function configures the noise model in place for the specified gate types with the given noise parameters.\n"
            )

        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, double, double, double, const QVec &>
            (&NoiseQVM::set_noise_model),
            "Set the noise model for a specific gate type with multiple noise parameters affecting a specific qubit vector in the quantum virtual machine.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model to be applied.\n"
            "\n"
            "     gate_type: The type of gate for which the noise model is relevant.\n"
            "\n"
            "     noise_level1: A double representing the first level of noise to apply.\n"
            "\n"
            "     noise_level2: A double representing the second level of noise to apply.\n"
            "\n"
            "     noise_level3: A double representing the third level of noise to apply.\n"
            "\n"
            "     qubits: A specific qubit vector (QVec) affected by the noise model.\n"
            "\n"
            "Returns:\n"
            "     None, as the function configures the noise model in place for the specified gate type and qubit vector.\n"
            )

        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &, double, double, double, const QVec &>
            (&NoiseQVM::set_noise_model),
            "Set the noise model for multiple gate types with various noise parameters affecting a specific qubit vector in the quantum virtual machine.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model to be applied.\n"
            "\n"
            "     gate_types: A vector of gate types for which the noise model is relevant.\n"
            "\n"
            "     noise_level1: A double representing the first level of noise to apply.\n"
            "\n"
            "     noise_level2: A double representing the second level of noise to apply.\n"
            "\n"
            "     noise_level3: A double representing the third level of noise to apply.\n"
            "\n"
            "     qubits: A specific qubit vector (QVec) affected by the noise model.\n"
            "\n"
            "Returns:\n"
            "     None, as the function configures the noise model in place for the specified gate types and qubit vector.\n"
            )

        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, double, double, double, const std::vector<QVec> &>
            (&NoiseQVM::set_noise_model),
            "Set the noise model for a specific gate type with multiple noise parameters affecting a vector of qubit vectors in the quantum virtual machine.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model to be applied.\n"
            "\n"
            "     gate_type: The type of gate for which the noise model is relevant.\n"
            "\n"
            "     noise_level1: A double representing the first level of noise to apply.\n"
            "\n"
            "     noise_level2: A double representing the second level of noise to apply.\n"
            "\n"
            "     noise_level3: A double representing the third level of noise to apply.\n"
            "\n"
            "     qubits_list: A vector of qubit vectors (QVec) affected by the noise model.\n"
            "\n"
            "Returns:\n"
            "     None, as the function configures the noise model in place for the specified gate type and qubit vectors.\n"
            )

        .def("set_measure_error",
            py::overload_cast<const NOISE_MODEL &, double, const QVec &>(&NoiseQVM::set_measure_error),
            py::arg("model"),
            py::arg("prob"),
            py::arg("qubits") = QVec(),
            "Set the measurement error model in the quantum virtual machine.\n"
            "\n"
            "Args:\n"
            "     model: The noise model to be applied for measurement errors.\n"
            "\n"
            "     prob: A double representing the probability of measurement error.\n"
            "\n"
            "     qubits: A specific qubit vector (QVec) for which the measurement error applies (default is an empty QVec).\n"
            "\n"
            "Returns:\n"
            "     None, as the function configures the measurement error model in place for the specified qubit vector.\n"
            )

        .def("set_measure_error",
            py::overload_cast<const NOISE_MODEL &, double, double, double, const QVec &>(&NoiseQVM::set_measure_error),
            py::arg("model"),
            py::arg("T1"),
            py::arg("T2"),
            py::arg("t_gate"),
            py::arg("qubits") = QVec(),
            "Set the measurement error model in the quantum virtual machine with specific error parameters.\n"
            "\n"
            "Args:\n"
            "     model: The noise model to be applied for measurement errors.\n"
            "\n"
            "     T1: A double representing the relaxation time constant for the qubits.\n"
            "\n"
            "     T2: A double representing the dephasing time constant for the qubits.\n"
            "\n"
            "     t_gate: A double representing the time duration of the gate operation.\n"
            "\n"
            "     qubits: A specific qubit vector (QVec) for which the measurement error applies (default is an empty QVec).\n"
            "\n"
            "Returns:\n"
            "     None, as the function configures the measurement error model in place for the specified qubit vector.\n"
            )

        .def("set_mixed_unitary_error", py::overload_cast<const GateType &, const std::vector<QStat> &, const std::vector<double> &>
            (&NoiseQVM::set_mixed_unitary_error),
            "Set a mixed unitary error model for a specific gate type in the quantum virtual machine.\n"
            "\n"
            "Args:\n"
            "     gate_type: The type of gate for which the mixed unitary error model applies.\n"
            "\n"
            "     unitary_ops: A vector of unitary operations (QStat) representing the error model.\n"
            "\n"
            "     probabilities: A vector of doubles representing the probabilities associated with each unitary operation.\n"
            "\n"
            "Returns:\n"
            "     None, as the function configures the mixed unitary error model in place for the specified gate type.\n"
            )

        .def("set_mixed_unitary_error", py::overload_cast<const GateType &, const std::vector<QStat> &,
            const std::vector<double> &, const QVec &>
            (&NoiseQVM::set_mixed_unitary_error),
            "Set a mixed unitary error model for a specific gate type in the quantum virtual machine with specific qubits.\n"
            "\n"
            "Args:\n"
            "     gate_type: The type of gate for which the mixed unitary error model applies.\n"
            "\n"
            "     unitary_ops: A vector of unitary operations (QStat) representing the error model.\n"
            "\n"
            "     probabilities: A vector of doubles representing the probabilities associated with each unitary operation.\n"
            "\n"
            "     qubits: A specific qubit vector (QVec) for which the mixed unitary error applies.\n"
            "\n"
            "Returns:\n"
            "     None, as the function configures the mixed unitary error model in place for the specified gate type and qubits.\n"
            )

        .def("set_mixed_unitary_error", py::overload_cast<const GateType &, const std::vector<QStat> &, 
            const std::vector<double> &, const std::vector<QVec> &>
            (&NoiseQVM::set_mixed_unitary_error),
            "Set a mixed unitary error model for a specific gate type in the quantum virtual machine, targeting multiple qubits.\n"
            "\n"
            "Args:\n"
            "     gate_type: The type of gate for which the mixed unitary error model applies.\n"
            "\n"
            "     unitary_ops: A vector of unitary operations (QStat) representing the error model.\n"
            "\n"
            "     probabilities: A vector of doubles representing the probabilities associated with each unitary operation.\n"
            "\n"
            "     qubit_groups: A vector of qubit vectors (QVec) specifying the qubits affected by the error model.\n"
            "\n"
            "Returns:\n"
            "     None, as the function configures the mixed unitary error model in place for the specified gate type and qubit groups.\n"
            )
                

        .def("set_reset_error", (&NoiseQVM::set_reset_error),
            py::arg("p0"),
            py::arg("p1"),
            py::arg("qubits") = QVec(),
            "Set a reset error model for the quantum virtual machine.\n"
            "\n"
            "Args:\n"
            "     p0: Probability of the qubit resetting to state 0.\n"
            "\n"
            "     p1: Probability of the qubit resetting to state 1.\n"
            "\n"
            "     qubits: A vector of qubits (QVec) for which the reset error model applies. Defaults to all qubits if not specified.\n"
            "\n"
            "Returns:\n"
            "     None, as this function configures the reset error model in place for the specified qubits.\n"
            )

        .def("set_readout_error",
            &NoiseQVM::set_readout_error,
            py::arg("probs_list"),
            py::arg("qubits") = QVec(),
            "Set a readout error model for the quantum virtual machine.\n"
            "\n"
            "Args:\n"
            "     probs_list: A list of probabilities for readout errors associated with each qubit.\n"
            "\n"
            "     qubits: A vector of qubits (QVec) for which the readout error model applies. Defaults to all qubits if not specified.\n"
            "\n"
            "Returns:\n"
            "     None, as this function configures the readout error model in place for the specified qubits.\n"
            )

        .def("set_rotation_error", &NoiseQVM::set_rotation_error,
            "Set a rotation error model for the quantum virtual machine.\n"
            "\n"
            "Args:\n"
            "     None specified in the function signature, but typically would include error parameters for the rotation.\n"
            "\n"
            "Returns:\n"
            "     None, as this function configures the rotation error model in place for the quantum operations.\n"
            )

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
        .def("init_qvm",
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

        .def("init_qvm", py::overload_cast<>(&NoiseQVM::init),
            "init quantum virtual machine");

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
            py::overload_cast<QProg&, QVec&, size_t, size_t>(&SingleAmplitudeQVM::run),
            py::arg("prog"),
            py::arg("qv"),
            py::arg("max_rank") = 30,
            py::arg("alloted_time") = 5,
            "Run the quantum program.\n"
            "\n"
            "Args:\n"
            "     prog: A quantum program (QProg) to be executed.\n"
            "\n"
            "     qv: A list of qubits (QVec) involved in the program.\n"
            "\n"
            "     max_rank: The maximum rank to consider during execution (default is 30).\n"
            "\n"
            "     alloted_time: The time allocated for execution (default is 5 seconds).\n"
            "\n"
            "Returns:\n"
            "     None, as the function executes the program in place.\n",
            py::return_value_policy::automatic_reference)

        .def("run",
            py::overload_cast<QProg&, QVec&, size_t, const std::vector<qprog_sequence_t>&>
            (&SingleAmplitudeQVM::run),
            "Run the quantum program.\n"
            "\n"
            "Args:\n"
            "     prog: A quantum program (QProg) to be executed.\n"
            "\n"
            "     qv: A list of qubits (QVec) involved in the program.\n"
            "\n"
            "     max_rank: The maximum rank to consider during execution.\n"
            "\n"
            "     sequences: A list of sequences (std::vector<qprog_sequence_t>).\n"
            "\n"
            "Returns:\n"
            "     None, as the function executes the program in place.\n"
            )

        .def("get_sequence", &SingleAmplitudeQVM::getSequence,
            "Get the program sequence.\n"
            "\n"
            "Returns:\n"
            "     A reference to the current program sequence.\n",
            py::return_value_policy::automatic_reference)

        .def("get_quick_map_vertice", &SingleAmplitudeQVM::getQuickMapVertice,
            "Get the quick map vertices.\n"
            "\n"
            "Returns:\n"
            "     A reference to the quick map vertices.\n",
            py::return_value_policy::automatic_reference)

        .def("pmeasure_bin_index", &SingleAmplitudeQVM::pMeasureBinindex,
            "Measure the bin index of the quantum state amplitude.\n"
            "\n"
            "Args:\n"
            "     bin_string: A string representing the bin.\n"
            "\n"
            "Returns:\n"
            "     A double representing the amplitude probability of the bin.\n",
            py::return_value_policy::automatic_reference)


        .def("pmeasure_dec_index", &SingleAmplitudeQVM::pMeasureDecindex,
            "Measure the dec index of the quantum state amplitude.\n"
            "\n"
            "Args:\n"
            "     dec_string: A string representing the dec.\n"
            "\n"
            "Returns:\n"
            "     A double representing the amplitude probability of the dec.\n",
            py::return_value_policy::automatic_reference)

        .def("pmeasure_bin_amplitude", &SingleAmplitudeQVM::pmeasure_bin_index,
            "Measure the bin amplitude of the quantum state.\n"
            "\n"
            "Args:\n"
            "     bin_string: A string representing the bin.\n"
            "\n"
            "Returns:\n"
            "     A complex number representing the bin amplitude.\n",
            py::return_value_policy::automatic_reference)

        .def("pmeasure_dec_amplitude", &SingleAmplitudeQVM::pmeasure_dec_index,
            "Measure the dec amplitude of the quantum state.\n"
            "\n"
            "Args:\n"
            "     dec_string: A string representing the dec.\n"
            "\n"
            "Returns:\n"
            "     A complex number representing the dec amplitude.\n",
            py::return_value_policy::automatic_reference)

        .def("get_prob_dict", py::overload_cast<QVec>(&SingleAmplitudeQVM::getProbDict),
            "Get the pmeasure result as a dictionary.\n"
            "\n"
            "Args:\n"
            "     qubit_list: A list of qubits for pmeasure.\n"
            "\n"
            "Returns:\n"
            "     A dictionary containing the measurement results of the quantum machine.\n")

        .def("get_prob_dict", py::overload_cast<const std::vector<int>&>(&SingleAmplitudeQVM::getProbDict),
            "Get the pmeasure result as a dictionary.\n"
            "\n"
            "Args:\n"
            "     qubit_list: A list of qubits for pmeasure.\n"
            "\n"
            "Returns:\n"
            "     A dictionary containing the measurement results of the quantum machine.\n")

        .def("prob_run_dict", py::overload_cast<QProg&, QVec>(&SingleAmplitudeQVM::probRunDict),
            "Run the quantum program and get the pmeasure result as a dictionary.\n"
            "\n"
            "Args:\n"
            "     qprog: The quantum program to run.\n"
            "\n"
            "     qubit_list: A list of qubits for pmeasure.\n"
            "\n"
            "Returns:\n"
            "     A dictionary containing the measurement results of the quantum machine.\n")

        .def("prob_run_dict", py::overload_cast<QProg&, const std::vector<int>&>(&SingleAmplitudeQVM::probRunDict),
            "Run the quantum program and get the pmeasure result as a dictionary.\n"
            "\n"
            "Args:\n"
            "     qprog: The quantum program to run.\n"
            "\n"
            "     qubit_list: A list of qubits for pmeasure.\n"
            "\n"
            "Returns:\n"
            "     A dictionary containing the measurement results of the quantum machine.\n");


    py::class_<PartialAmplitudeQVM, QuantumMachine>(m, "PartialAmpQVM", "quantum partial amplitude machine class\n")
        .def(py::init<>())

        .def("init_qvm",
            [](PartialAmplitudeQVM& machine, int type)
            {
                auto backend_type = static_cast<BackendType>(type);
                return machine.init(backend_type);
            },
            py::arg("backend_type") = (int)BackendType::CPU)

        .def("run",
            py::overload_cast<QProg &, const NoiseModel &>(&PartialAmplitudeQVM::run<QProg>),
            py::arg("qprog"),
            py::arg_v("noise_model", NoiseModel(), "NoiseModel()"),
            "Run the quantum program.\n"
            "\n"
            "Args:\n"
            "     qprog: The quantum program to execute.\n"
            "\n"
            "     noise_model: An optional noise model (default is NoiseModel()).\n"
            "\n"
            "Returns:\n"
            "     None.\n")

        .def("run",
            py::overload_cast<QCircuit &, const NoiseModel &>(&PartialAmplitudeQVM::run<QCircuit>),
            py::arg("qprog"),
            py::arg_v("noise_model", NoiseModel(), "NoiseModel()"),
            "Run the quantum program.\n"
            "\n"
            "Args:\n"
            "     qprog: The quantum circuit to execute.\n"
            "\n"
            "     noise_model: An optional noise model (default is NoiseModel()).\n"
            "\n"
            "Returns:\n"
            "     None.\n")

        .def("pmeasure_bin_index", &PartialAmplitudeQVM::pmeasure_bin_index, "bin_index"_a,
            "Get the amplitude of the quantum state for the specified bin index.\n"
            "\n"
            "Args:\n"
            "     bin_index: A string representing the bin.\n"
            "\n"
            "Returns:\n"
            "     A complex number representing the amplitude of the bin.\n",
            py::return_value_policy::automatic_reference)

        .def("pmeasure_dec_index", &PartialAmplitudeQVM::pmeasure_dec_index, "dec_index"_a,
            "Get the amplitude of the quantum state for the specified decimal index.\n"
            "\n"
            "Args:\n"
            "     dec_index: A string representing the decimal.\n"
            "\n"
            "Returns:\n"
            "     A complex number representing the amplitude of the decimal.\n",
            py::return_value_policy::automatic_reference)

        .def("pmeasure_subset", &PartialAmplitudeQVM::pmeasure_subset, "index_list"_a,
            "Get the amplitudes of the quantum state for a subset of indices.\n"
            "\n"
            "Args:\n"
            "     index_list: A list of strings representing decimal states.\n"
            "\n"
            "Returns:\n"
            "     A list of complex numbers representing the amplitude results.\n",
            py::return_value_policy::automatic_reference)

        .def("get_prob_dict", &PartialAmplitudeQVM::getProbDict,
            "Get the measurement results as a dictionary.\n"
            "\n"
            "Args:\n"
            "     qubit_list: A list of qubits to measure.\n"
            "\n"
            "Returns:\n"
            "     A dictionary containing the measurement results of the quantum machine.\n")

        .def("prob_run_dict", &PartialAmplitudeQVM::probRunDict,
            "Run the quantum program and get the measurement results as a dictionary.\n"
            "\n"
            "Args:\n"
            "     qprog: The quantum program to execute.\n"
            "\n"
            "     qubit_list: A list of qubits to measure.\n"
            "\n"
            "Returns:\n"
            "     A dictionary containing the measurement results of the quantum machine.\n"
        );

    py::class_<SparseSimulator, QuantumMachine>(m, "SparseQVM", "quantum sparse machine class\n")
        .def(py::init<>())

        .def("init_qvm",
            &SparseSimulator::init,
            "init quantum virtual machine")

        .def("prob_run_dict", &SparseSimulator::probRunDict,
            "Run the quantum program and get the measurement results as a dictionary.\n"
            "\n"
            "Args:\n"
            "     qprog: The quantum program to execute.\n"
            "\n"
            "Returns:\n"
            "     A dictionary containing the measurement results of the quantum machine.\n")

        .def("directlyRun", &SparseSimulator::directlyRun,
            "Run the quantum program and get the measurement results as a dictionary.\n"
            "\n"
            "Args:\n"
            "     qprog: The quantum program to execute.\n"
            "\n"
            "Returns:\n"
            "     Dict[str, bool]: The result of the quantum program execution in one shot.\n")

        .def("directly_run", &SparseSimulator::directlyRun,
            "Run the quantum program and get the measurement results as a dictionary.\n"
            "\n"
            "Args:\n"
            "     qprog: The quantum program to execute.\n"
            "\n"
            "Returns:\n"
            "     The measurement results of the quantum machine.\n")

        .def("run_with_configuration", &SparseSimulator::runWithConfiguration,
            "Run the quantum program with the specified configuration and get the measurement results as a dictionary.\n"
            "\n"
            "Args:\n"
            "     qprog: The quantum program to execute.\n"
            "\n"
            "     cbits: The quantum classical bits.\n"
            "\n"
            "     shots: The number of sample shots.\n"
            "\n"
            "Returns:\n"
            "     The measurement results of the quantum machine.\n"
        );

    py::class_<DensityMatrixSimulator, QuantumMachine>(m, "DensityMatrixSimulator", "simulator for density matrix")
        .def(py::init<>())

        .def("init_qvm",
            [](DensityMatrixSimulator& machine, bool is_double_precision)
            {
                return machine.init(is_double_precision);
            },
            py::arg("is_double_precision") = true)

        .def("get_probability",
            py::overload_cast<QProg&, size_t>(&DensityMatrixSimulator::get_probability),
            py::arg("prog"),
            py::arg("index"),
            "Run the quantum program and get the probability for the specified index.\n"
            "\n"
            "Args:\n"
            "     prog: The quantum program to execute.\n"
            "\n"
            "     index: The measurement index in [0, 2^N      1].\n"
            "\n"
            "Returns:\n"
            "     The probability result of the quantum program.\n")

        .def("get_probability",
            py::overload_cast<QProg&, std::string>(&DensityMatrixSimulator::get_probability),
            py::arg("prog"),
            py::arg("index"),
            "Run the quantum program and get the probability for the specified index.\n"
            "\n"
            "Args:\n"
            "     prog: The quantum program to execute.\n"
            "\n"
            "     index: The measurement index in [0, 2^N      1].\n"
            "\n"
            "Returns:\n"
            "     The probability result of the quantum program.")

        .def("get_probabilities",
            py::overload_cast<QProg&>(&DensityMatrixSimulator::get_probabilities),
            py::arg("prog"),
            "Run the quantum program and get the probabilities for all indices.\n"
            "\n"
            "Args:\n"
            "     prog: The quantum program to execute.\n"
            "\n"
            "Returns:\n"
            "     The probabilities result of the quantum program.\n")

        .def("get_probabilities",
            py::overload_cast<QProg&, QVec>(&DensityMatrixSimulator::get_probabilities),
            py::arg("prog"),
            py::arg("qubits"),
            "Run the quantum program and get the probabilities for all indices for the specified qubits.\n"
            "\n"
            "Args:\n"
            "     prog: The quantum program to execute.\n"
            "\n"
            "     qubits: The selected qubits for measurement.\n"
            "\n"
            "Returns:\n"
            "     The probabilities result of the quantum program.\n")

        .def("get_probabilities",
            py::overload_cast<QProg&, Qnum>(&DensityMatrixSimulator::get_probabilities),
            py::arg("prog"),
            py::arg("qubits"),
            "Run the quantum program and get the probabilities for all indices for the specified qubits.\n"
            "\n"
            "Args:\n"
            "     prog: The quantum program to execute.\n"
            "\n"
            "     qubits: The selected qubits for measurement.\n"
            "\n"
            "Returns:\n"
            "     The probabilities result of the quantum program.\n")

        .def("get_probabilities",
            py::overload_cast<QProg&, std::vector<string>>(&DensityMatrixSimulator::get_probabilities),
            py::arg("prog"),
            py::arg("indices"),
            "Run the quantum program and get the probabilities for the specified binary indices.\n"
            "\n"
            "Args:\n"
            "     prog: The quantum program to execute.\n"
            "\n"
            "     indices: The selected binary indices for measurement.\n"
            "\n"
            "Returns:\n"
            "     The probabilities result of the quantum program.\n")

        .def("get_expectation",
            py::overload_cast<QProg&, const QHamiltonian&, const QVec&>(&DensityMatrixSimulator::get_expectation),
            py::arg("prog"),
            py::arg("hamiltonian"),
            py::arg("qubits"),
            "Run the quantum program and calculate the Hamiltonian expectation for the specified qubits.\n"
            "\n"
            "Args:\n"
            "     prog: The quantum program to execute.\n"
            "\n"
            "     hamiltonian: The QHamiltonian to use for the expectation value.\n"
            "\n"
            "     qubits: The selected qubits for measurement.\n"
            "\n"
            "Returns:\n"
            "     The Hamiltonian expectation for the specified qubits.\n")

        .def("get_expectation",
            py::overload_cast<QProg&, const QHamiltonian&, const Qnum&>(&DensityMatrixSimulator::get_expectation),
            py::arg("prog"),
            py::arg("hamiltonian"),
            py::arg("qubits"),
            "Run the quantum program and calculate the Hamiltonian expectation for the specified qubits.\n"
            "\n"
            "Args:\n"
            "     prog: The quantum program to execute.\n"
            "\n"
            "     hamiltonian: The QHamiltonian to use for the expectation value.\n"
            "\n"
            "     qubits: The selected qubits for measurement.\n"
            "\n"
            "Returns:\n"
            "     The Hamiltonian expectation for the specified qubits.\n")

        .def("get_density_matrix",
            py::overload_cast<QProg&>(&DensityMatrixSimulator::get_density_matrix),
            py::arg("prog"),
            "Run quantum program and get the full density matrix.\n"
            "\n"
            "Args:\n"
            "     prog: The quantum program to execute.\n"
            "\n"
            "Returns:\n"
            "     The full density matrix.\n")

        .def("get_reduced_density_matrix",
            py::overload_cast<QProg&, const QVec&>(&DensityMatrixSimulator::get_reduced_density_matrix),
            py::arg("prog"),
            py::arg("qubits"),
            "Run quantum program and get the density matrix for current qubits.\n"
            "\n"
            "Args:\n"
            "     prog: The quantum program to execute.\n"
            "\n"
            "     qubits: The selected qubits from the quantum program.\n"
            "\n"
            "Returns:\n"
            "     The density matrix for the specified qubits.\n")

        .def("get_reduced_density_matrix",
            py::overload_cast<QProg&, const Qnum&>(&DensityMatrixSimulator::get_reduced_density_matrix),
            py::arg("prog"),
            py::arg("qubits"),
            "Run quantum program and get the density matrix for current qubits.\n"
            "\n"
            "Args:\n"
            "     prog: The quantum program to execute.\n"
            "\n"
            "     qubits: The selected qubits from the quantum program.\n"
            "\n"
            "Returns:\n"
            "     The density matrix for the specified qubits.\n")

        /* bit-flip, phase-flip, bit-phase-flip, phase-damping, amplitude-damping, depolarizing*/
        .def("set_noise_model", py::overload_cast<const cmatrix_t&>(&DensityMatrixSimulator::set_noise_model),
            "Set the noise model for the density matrix simulator.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model represented as a complex matrix.\n"
            "\n"
            "Returns:\n"
            "     None.\n")

        .def("set_noise_model", py::overload_cast<const cmatrix_t&, const std::vector<GateType>&>
            (&DensityMatrixSimulator::set_noise_model),
            "Set the noise model for the density matrix simulator with specific gate types.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model represented as a complex matrix.\n"
            "\n"
            "     gate_types: A vector of gate types to which the noise model applies.\n"
            "\n"
            "Returns:\n"
            "     None.\n")

        .def("set_noise_model", py::overload_cast<const std::vector<cmatrix_t>&>(&DensityMatrixSimulator::set_noise_model),
            "Set multiple noise models for the density matrix simulator.\n"
            "\n"
            "Args:\n"
            "     noise_models: A vector of noise models, each represented as a complex matrix.\n"
            "\n"
            "Returns:\n"
            "     None.\n")

        .def("set_noise_model", py::overload_cast<const std::vector<cmatrix_t>&, const std::vector<GateType>&>
            (&DensityMatrixSimulator::set_noise_model),
            "Set multiple noise models for the density matrix simulator with specific gate types.\n"
            "\n"
            "Args:\n"
            "     noise_models: A vector of noise models, each represented as a complex matrix.\n"
            "\n"
            "     gate_types: A vector of gate types to which the noise models apply.\n"
            "\n"
            "Returns:\n"
            "     None.\n")


        /* bit-flip, phase-flip, bit-phase-flip, phase-damping, amplitude-damping, depolarizing*/
        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, double>
            (&DensityMatrixSimulator::set_noise_model),
            "Set a specific noise model for the density matrix simulator with a given gate type and probability.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model to apply.\n"
            "\n"
            "     gate_type: The specific gate type associated with the noise model.\n"
            "\n"
            "     probability: The probability of the noise occurring.\n"
            "\n"
            "Returns:\n"
            "     None.\n")

        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &, double>
            (&DensityMatrixSimulator::set_noise_model),
            "Set a specific noise model for the density matrix simulator with multiple gate types and a given probability.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model to apply.\n"
            "\n"
            "     gate_types: A vector of gate types associated with the noise model.\n"
            "\n"
            "     probability: The probability of the noise occurring.\n"
            "\n"
            "Returns:\n"
            "     None.\n")

        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, double, const QVec &>
            (&DensityMatrixSimulator::set_noise_model),
            "Set a specific noise model for the density matrix simulator with a given gate type, probability, and target qubits.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model to apply.\n"
            "\n"
            "     gate_type: The specific gate type associated with the noise model.\n"
            "\n"
            "     probability: The probability of the noise occurring.\n"
            "\n"
            "     qubits: The target qubits affected by the noise model.\n"
            "\n"
            "Returns:\n"
            "     None.\n")

        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &, 
            double, const QVec &>(&DensityMatrixSimulator::set_noise_model),
            "Set a specific noise model for the density matrix simulator with multiple gate types, a given probability, and target qubits.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model to apply.\n"
            "\n"
            "     gate_types: A vector of gate types associated with the noise model.\n"
            "\n"
            "     probability: The probability of the noise occurring.\n"
            "\n"
            "     qubits: The target qubits affected by the noise model.\n"
            "\n"
            "Returns:\n"
            "     None.\n")

        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, double, 
            const std::vector<QVec> &>(&DensityMatrixSimulator::set_noise_model),
            "Set a specific noise model for the density matrix simulator with a given gate type, probability, and groups of target qubits.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model to apply.\n"
            "\n"
            "     gate_type: The specific gate type associated with the noise model.\n"
            "\n"
            "     probability: The probability of the noise occurring.\n"
            "\n"
            "     qubit_groups: A vector of QVecs representing groups of target qubits affected by the noise model.\n"
            "\n"
            "Returns:\n"
            "     None.\n")


            /*decoherence error*/
        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, 
            double, double, double>(&DensityMatrixSimulator::set_noise_model),
            "Set a specific noise model for the density matrix simulator with a given gate type, probability, duration, and temperature.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model to apply.\n"
            "\n"
            "     gate_type: The specific gate type associated with the noise model.\n"
            "\n"
            "     probability: The probability of the noise occurring.\n"
            "\n"
            "     duration: The duration for which the noise model is applied.\n"
            "\n"
            "     temperature: The temperature affecting the noise characteristics.\n"
            "\n"
            "Returns:\n"
            "     None.\n")

        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &,
            double, double, double>(&DensityMatrixSimulator::set_noise_model),
            "Set a specific noise model for the density matrix simulator with multiple gate types, a given probability, duration, and temperature.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model to apply.\n"
            "\n"
            "     gate_types: A vector of gate types associated with the noise model.\n"
            "\n"
            "     probability: The probability of the noise occurring.\n"
            "\n"
            "     duration: The duration for which the noise model is applied.\n"
            "\n"
            "     temperature: The temperature affecting the noise characteristics.\n"
            "\n"
            "Returns:\n"
            "     None.\n")

        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, 
            double, double, double, const QVec &>(&DensityMatrixSimulator::set_noise_model),
            "Set a specific noise model for the density matrix simulator with a given gate type, probability, duration, temperature, and a target qubit.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model to apply.\n"
            "\n"
            "     gate_type: The specific gate type associated with the noise model.\n"
            "\n"
            "     probability: The probability of the noise occurring.\n"
            "\n"
            "     duration: The duration for which the noise model is applied.\n"
            "\n"
            "     temperature: The temperature affecting the noise characteristics.\n"
            "\n"
            "     target_qubit: The specific qubit targeted by the noise model.\n"
            "\n"
            "Returns:\n"
            "     None.\n")

        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &, 
            double, double, double, const QVec &>(&DensityMatrixSimulator::set_noise_model),
            "Set a specific noise model for the density matrix simulator with multiple gate types, probability, duration, temperature, and a target qubit.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model to apply.\n"
            "\n"
            "     gate_types: A vector of gate types associated with the noise model.\n"
            "\n"
            "     probability: The probability of the noise occurring.\n"
            "\n"
            "     duration: The duration for which the noise model is applied.\n"
            "\n"
            "     temperature: The temperature affecting the noise characteristics.\n"
            "\n"
            "     target_qubit: The specific qubit targeted by the noise model.\n"
            "\n"
            "Returns:\n"
            "     None.\n")

        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, 
            double, double, double, const std::vector<QVec> &>(&DensityMatrixSimulator::set_noise_model),
            "Set a specific noise model for the density matrix simulator with a given gate type, probability, duration, temperature, and multiple target qubits.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model to apply.\n"
            "\n"
            "     gate_type: The specific gate type associated with the noise model.\n"
            "\n"
            "     probability: The probability of the noise occurring.\n"
            "\n"
            "     duration: The duration for which the noise model is applied.\n"
            "\n"
            "     temperature: The temperature affecting the noise characteristics.\n"
            "\n"
            "     target_qubits: A vector of qubits targeted by the noise model.\n"
            "\n"
            "Returns:\n"
            "     None.\n");

    py::class_<Stabilizer, QuantumMachine>(m, "Stabilizer", "simulator for basic clifford simulator")
        .def(py::init<>())
        .def("init_qvm",
            &Stabilizer::init,
            "init quantum virtual machine")

        /* bit-flip, phase-flip, bit-phase-flip, phase-damping, depolarizing*/
        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, double>(&Stabilizer::set_noise_model),
            "Set a noise model for the Stabilizer simulator with a specific gate type and probability.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model to apply (e.g., bit-flip, phase-flip, etc.).\n"
            "\n"
            "     gate_type: The specific gate type associated with the noise model.\n"
            "\n"
            "     probability: The probability of the noise occurring.\n"
            "\n"
            "Returns:\n"
            "     None.\n")

        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &, 
            double>(&Stabilizer::set_noise_model),
            "Set a noise model for the Stabilizer simulator with multiple gate types and a specified probability.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model to apply (e.g., bit-flip, phase-flip, etc.).\n"
            "\n"
            "     gate_types: A vector of gate types associated with the noise model.\n"
            "\n"
            "     probability: The probability of the noise occurring.\n"
            "\n"
            "Returns:\n"
            "     None.\n")

        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, double,
            const QVec &>(&Stabilizer::set_noise_model),
            "Set a noise model for the Stabilizer simulator with a specific gate type, probability, and targeted qubits.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model to apply (e.g., bit-flip, phase-flip, etc.).\n"
            "\n"
            "     gate_type: The specific gate type associated with the noise model.\n"
            "\n"
            "     probability: The probability of the noise occurring.\n"
            "\n"
            "     target_qubits: The qubits targeted by the noise model.\n"
            "\n"
            "Returns:\n"
            "     None.\n")

        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const std::vector<GateType> &,
            double, const QVec &>(&Stabilizer::set_noise_model),
            "Set a noise model for the Stabilizer simulator with multiple gate types, a specified probability, and targeted qubits.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model to apply (e.g., bit-flip, phase-flip, etc.).\n"
            "\n"
            "     gate_types: A vector of gate types associated with the noise model.\n"
            "\n"
            "     probability: The probability of the noise occurring.\n"
            "\n"
            "     target_qubits: The qubits targeted by the noise model.\n"
            "\n"
            "Returns:\n"
            "     None.\n")

        .def("set_noise_model", py::overload_cast<const NOISE_MODEL &, const GateType &, double,
            const std::vector<QVec> &>(&Stabilizer::set_noise_model),
            "Set a noise model for the Stabilizer simulator with a specific gate type, probability, and multiple targeted qubits.\n"
            "\n"
            "Args:\n"
            "     noise_model: The noise model to apply (e.g., bit-flip, phase-flip, etc.).\n"
            "\n"
            "     gate_type: The specific gate type associated with the noise model.\n"
            "\n"
            "     probability: The probability of the noise occurring.\n"
            "\n"
            "     target_qubits: A vector of qubit vectors targeted by the noise model.\n"
            "\n"
            "Returns:\n"
            "     None.\n")

        .def("run_with_configuration",
            &Stabilizer::runWithConfiguration,
            py::arg("qprog"),
            py::arg("shot"),
            "Run quantum program and get shots result.\n"
            "\n"
            "Args:\n"
            "     qprog: Quantum program to execute.\n"
            "\n"
            "     shot: Number of measurement shots.\n"
            "\n"
            "Returns:\n"
            "     Shots result of the quantum program.\n")

        .def("prob_run_dict",
            &Stabilizer::probRunDict,
            py::arg("qprog"),
            py::arg("qubits"),
            py::arg_v("select_max", -1, "-1"),
            "Run quantum program and get probabilities.\n"
            "\n"
            "Args:\n"
            "     qprog: Quantum program to execute.\n"
            "\n"
            "     qubits: Qubits to be measured for probabilities.\n"
            "\n"
            "     select_max: Optional, selects the maximum number of probabilities to return.\n"
            "\n"
            "Returns:\n"
            "     Probabilities result of the quantum program.\n"
        );

    py::class_<MPSQVM, QuantumMachine> mps_qvm(m, "MPSQVM", "quantum matrix product state machine class");
        mps_qvm.def(py::init<>())

        .def("pmeasure", &MPSQVM::PMeasure, 
            "qubit_list"_a, "select_max"_a = -1,
            "Get the probability distribution over qubits.\n"
            "\n"
            "Args:\n"
            "     qubit_list: List of qubits to measure.\n"
            "\n"
            "     select_max: Maximum number of returned elements in the result tuple; should be in [-1, 1<<len(qubit_list)]. Default is -1, which means no limit.\n"
            "\n"
            "Returns:\n"
            "     Measure result of the quantum machine in tuple form.\n",
            py::return_value_policy::reference)

        .def("pmeasure_no_index", &MPSQVM::PMeasure_no_index, 
            "qubit_list"_a,
            "Get the probability distribution over qubits.\n"
            "\n"
            "Args:\n"
            "     qubit_list: List of qubits to measure.\n"
            "\n"
            "Returns:\n"
            "     Measure result of the quantum machine in list form.\n",
            py::return_value_policy::reference)

        .def("get_prob_tuple_list", &MPSQVM::getProbTupleList, 
            "qubit_list"_a, "select_max"_a = -1,
            "Get pmeasure result as list.\n"
            "\n"
            "Args:\n"
            "     qubit_list: List of qubits for pmeasure.\n"
            "\n"
            "     select_max: Maximum number of returned elements in the result tuple; should be in [-1, 1<<len(qubit_list)]. Default is -1, meaning no limit.\n"
            "\n"
            "Returns:\n"
            "     Measure result of the quantum machine.\n",
            py::return_value_policy::reference)

        .def("get_prob_list", &MPSQVM::getProbList, 
            "qubit_list"_a, "select_max"_a = -1,
            "Get pmeasure result as list.\n"
            "\n"
            "Args:\n"
            "     qubit_list: List of qubits for pmeasure.\n"
            "\n"
            "     select_max: Maximum number of returned elements in the result tuple; should be in [-1, 1<<len(qubit_list)]. Default is -1, meaning no limit.\n"
            "\n"
            "Returns:\n"
            "     Measure result of the quantum machine.\n",
            py::return_value_policy::reference)

        .def("get_prob_dict", &MPSQVM::getProbDict,
            "qubit_list"_a, "select_max"_a = -1,
            "Get pmeasure result as dict.\n"
            "\n"
            "Args:\n"
            "     qubit_list: List of qubits for pmeasure.\n"
            "\n"
            "     select_max: Maximum number of returned elements in the result tuple; should be in [-1, 1<<len(qubit_list)]. Default is -1, meaning no limit.\n"
            "\n"
            "Returns:\n"
            "     Measure result of the quantum machine.\n",
            py::return_value_policy::reference)

        .def("prob_run_tuple_list", &MPSQVM::probRunTupleList, 
            "program"_a, "qubit_list"_a, "select_max"_a = -1,
            "Run quantum program and get pmeasure result as tuple list.\n"
            "\n"
            "Args:\n"
            "     program: Quantum program to run.\n"
            "\n"
            "     qubit_list: List of qubits for pmeasure.\n"
            "\n"
            "     select_max: Maximum number of returned elements in the result tuple; should be in [-1, 1<<len(qubit_list)]. Default is -1, meaning no limit.\n"
            "\n"
            "Returns:\n"
            "     Measure result of the quantum machine.\n",
            py::return_value_policy::reference)

        .def("prob_run_list", &MPSQVM::probRunList, 
            "program"_a, "qubit_list"_a, "select_max"_a = -1,
            "Run quantum program and get pmeasure result as list.\n"
            "\n"
            "Args:\n"
            "     program: Quantum program to run.\n"
            "\n"
            "     qubit_list: List of qubits for pmeasure.\n"
            "\n"
            "     select_max: Maximum number of returned elements in the result; should be in [-1, 1<<len(qubit_list)]. Default is -1, meaning no limit.\n"
            "\n"
            "Returns:\n"
            "     Measure result of the quantum machine.\n",
            py::return_value_policy::reference)

        .def("prob_run_dict", &MPSQVM::probRunDict, 
            "program"_a, "qubit_list"_a, "select_max"_a = -1,
            "Run quantum program and get pmeasure result as dict.\n"
            "Args:\n"
            "     program: Quantum program to run.\n"
            "\n"
            "     qubit_list: List of qubits for pmeasure.\n"
            "\n"
            "     select_max: Maximum number of returned elements in the result; should be in [-1, 1<<len(qubit_list)]. Default is -1, meaning no limit.\n"
            "Returns:\n"
            "     Measure result of the quantum machine.\n",
            py::return_value_policy::reference)

        .def("quick_measure", &MPSQVM::quickMeasure, 
            "qubit_list"_a, "shots"_a,
            "Quick measure.\n"
            "\n"
            "Args:\n"
            "     qubit_list: List of qubits to measure.\n"
            "\n"
            "     shots: The number of repetitions for the measurement operation.\n"
            "\n"
            "Returns:\n"
            "     Result of the quantum program.\n",
            py::return_value_policy::reference)

        .def("pmeasure_bin_index", &MPSQVM::pmeasure_bin_index, 
            "program"_a, "string"_a,
            "Get pmeasure bin index quantum state amplitude.\n"
            "\n"
            "Args:\n"
            "     string: Bin string.\n"
            "\n"
            "Returns:\n"
            "     Complex: Bin amplitude.\n",
            py::return_value_policy::reference)

        .def("pmeasure_dec_index", &MPSQVM::pmeasure_dec_index, 
            "program"_a, "string"_a,
            "Get pmeasure decimal index quantum state amplitude.\n"
            "\n"
            "Args:\n"
            "     string: Decimal string.\n"
            "\n"
            "Returns:\n"
            "     Complex: Decimal amplitude.\n",
            py::return_value_policy::reference)

        .def("pmeasure_bin_subset", &MPSQVM::pmeasure_bin_subset, 
            "program"_a, "string_list"_a,
            "Get pmeasure quantum state amplitude subset.\n"
            "\n"
            "Args:\n"
            "     list: List of bin state strings.\n"
            "\n"
            "Returns:\n"
            "     List: Bin amplitude result list.\n",
            py::return_value_policy::reference)

        .def("pmeasure_dec_subset", &MPSQVM::pmeasure_dec_subset,
            "program"_a, "string_list"_a,
            "Get pmeasure quantum state amplitude subset.\n"
            "\n"
            "Args:\n"
            "     list: List of decimal state strings.\n"
            "\n"
            "Returns:\n"
            "     List: Decimal amplitude result list.\n",
            py::return_value_policy::reference)

        // The all next MPSQVM functions are only for noise simulation

        /* bit-flip,phase-flip,bit-phase-flip,phase-damping,amplitude-damping,depolarizing*/
        .def("set_noise_model", py::overload_cast<NOISE_MODEL, GateType, double>(&MPSQVM::set_noise_model),
            "Set the noise model for the quantum simulation.\n"
            "\n"
            "Args:\n"
            "     noise_model: Type of noise model (bit-flip, phase-flip, etc.).\n"
            "\n"
            "     gate_type: Type of gate affected by the noise.\n"
            "\n"
            "     noise_level: Level of noise to apply.\n"
            )

        .def("set_noise_model", py::overload_cast<NOISE_MODEL, GateType, 
            double, const std::vector<QVec> &>(&MPSQVM::set_noise_model),
            "Set the noise model for the quantum simulation with specific qubits.\n"
            "\n"
            "Args:\n"
            "     noise_model: Type of noise model (bit-flip, phase-flip, etc.).\n"
            "\n"
            "     gate_type: Type of gate affected by the noise.\n"
            "\n"
            "     noise_level: Level of noise to apply.\n"
            "\n"
            "     qubits: List of qubits to which the noise model will be applied.\n"
            )

        /*decoherence error*/
        .def("set_noise_model", py::overload_cast<NOISE_MODEL, GateType, 
            double, double, double>(&MPSQVM::set_noise_model),
            "Set the noise model for the quantum simulation with multiple noise levels.\n"
            "\n"
            "Args:\n"
            "     noise_model: Type of noise model (bit-flip, phase-flip, etc.).\n"
            "\n"
            "     gate_type: Type of gate affected by the noise.\n"
            "\n"
            "     noise_level_1: First noise level to apply.\n"
            "\n"
            "     noise_level_2: Second noise level to apply.\n"
            "\n"
            "     noise_level_3: Third noise level to apply.\n"
            )

        .def("set_noise_model", py::overload_cast<NOISE_MODEL, GateType,
            double, double, double, const std::vector<QVec> &>(&MPSQVM::set_noise_model),
            "Set the noise model for the quantum simulation with specific noise levels and qubits.\n"
            "\n"
            "Args:\n"
            "     noise_model: Type of noise model (bit-flip, phase-flip, etc.).\n"
            "\n"
            "     gate_type: Type of gate affected by the noise.\n"
            "\n"
            "     noise_level_1: First noise level to apply.\n"
            "\n"
            "     noise_level_2: Second noise level to apply.\n"
            "\n"
            "     noise_level_3: Third noise level to apply.\n"
            "\n"
            "     qubits: List of qubits to which the noise model will be applied.\n")

        /*mixed unitary error*/
        .def("set_mixed_unitary_error", py::overload_cast<GateType, const std::vector<QStat> &, 
            const std::vector<QVec> &>(&MPSQVM::set_mixed_unitary_error),
            "Set mixed unitary errors for the specified gate type.\n"
            "\n"
            "Args:\n"
            "     gate_type: Type of gate affected by the error.\n"
            "\n"
            "     unitary_errors: List of unitary error matrices.\n"
            "\n"
            "     qubits: List of qubits where the errors will be applied.\n")

        .def("set_mixed_unitary_error", py::overload_cast<GateType, const std::vector<QStat> &,
            const prob_vec &, const std::vector<QVec> &>(&MPSQVM::set_mixed_unitary_error),
            "Set mixed unitary errors with associated probabilities for the specified gate type.\n"
            "\n"
            "Args:\n"
            "     gate_type: Type of gate affected by the error.\n"
            "\n"
            "     unitary_errors: List of unitary error matrices.\n"
            "\n"
            "     probabilities: Probabilities associated with each unitary error.\n"
            "\n"
            "     qubits: List of qubits where the errors will be applied.\n")

        .def("set_mixed_unitary_error", py::overload_cast<GateType, 
            const std::vector<QStat> &>(&MPSQVM::set_mixed_unitary_error),
            "Set mixed unitary errors for the specified gate type.\n"
            "\n"
            "Args:\n"
            "     gate_type: Type of gate affected by the error.\n"
            "\n"
            "     unitary_errors: List of unitary error matrices to apply for the gate type.\n")

        .def("set_mixed_unitary_error", py::overload_cast<GateType, const std::vector<QStat> &, 
            const prob_vec &>(&MPSQVM::set_mixed_unitary_error),
            "Set mixed unitary errors with associated probabilities for the specified gate type.\n"
            "\n"
            "Args:\n"
            "     gate_type: Type of gate affected by the error.\n"
            "\n"
            "     unitary_errors: List of unitary error matrices.\n"
            "\n"
            "     probabilities: Probabilities associated with each unitary error.\n")

        /*readout error*/
        .def("set_readout_error", &MPSQVM::set_readout_error, 
            "readout_params"_a, "qubits"_a,
            "Set readout error parameters for the specified qubits.\n"
            "\n"
            "Args:\n"
            "     readout_params: Parameters defining the readout errors.\n"
            "\n"
            "     qubits: List of qubits to which the readout errors apply.\n",
            py::return_value_policy::reference)

        /*measurement error*/
        .def("set_measure_error", py::overload_cast<NOISE_MODEL, double>(&MPSQVM::set_measure_error),
            "Set the measurement error based on the specified noise model.\n"
            "\n"
            "Args:\n"
            "     noise_model: The type of noise model to apply.\n"
            "\n"
            "     error_rate: The rate of measurement error to be set.\n"
            )

        .def("set_measure_error", py::overload_cast<NOISE_MODEL, double, double, double>(&MPSQVM::set_measure_error),
            "Set the measurement error with multiple error rates for the specified noise model.\n"
            "\n"
            "Args:\n"
            "     noise_model: The type of noise model to apply.\n"
            "\n"
            "     error_rate1: First error rate.\n"
            "\n"
            "     error_rate2: Second error rate.\n"
            "\n"
            "     error_rate3: Third error rate.\n"
            )

        /*rotation error*/
        .def("set_rotation_error", &MPSQVM::set_rotation_error, 
            "param"_a, 
            py::return_value_policy::reference,
            "Set the rotation error parameters.\n"
            "\n"
            "Args:\n"
            "     param: The parameters defining the rotation error.\n"
            "\n"
            "Returns:\n"
            "     A reference to the updated instance of the class.\n"
            )

        /*reset error*/
        .def("set_reset_error", &MPSQVM::set_reset_error, 
            "reset_0_param"_a, "reset_1_param"_a, 
            py::return_value_policy::reference,
            "Set the reset error for the quantum state.\n"
            "\n"
            "Args:\n"
            "     reset_0_param: float, error probability for resetting qubit to 0.\n"
            "\n"
            "     reset_1_param: float, error probability for resetting qubit to 1.\n"
            "\n"
            "Returns:\n"
            "     None.\n"
            )

        /* bit-flip,phase-flip,bit-phase-flip,phase-damping,amplitude-damping,depolarizing*/
        .def("add_single_noise_model", py::overload_cast<NOISE_MODEL, GateType, double>(&MPSQVM::add_single_noise_model),
            "Add a noise model to a specific gate.\n"
            "\n"
            "Args:\n"
            "     noise_model: NOISE_MODEL, the type of noise model to apply.\n"
            "\n"
            "     gate_type: GateType, the type of gate affected by the noise.\n"
            "\n"
            "     error_rate: float, the rate of noise occurrence.\n"
            "\n"
            "Returns:\n"
            "     None.\n"
            )

        /*decoherence error*/
        .def("add_single_noise_model", py::overload_cast<NOISE_MODEL, GateType, double, double, double>(&MPSQVM::add_single_noise_model),
            "Add a noise model to a specific gate with multiple error rates.\n"
            "\n"
            "Args:\n"
            "     noise_model: NOISE_MODEL, the type of noise model to apply.\n"
            "\n"
            "     gate_type: GateType, the type of gate affected by the noise.\n"
            "\n"
            "     error_rate_1: float, the first error rate.\n"
            "\n"
            "     error_rate_2: float, the second error rate.\n"
            "\n"
            "     error_rate_3: float, the third error rate.\n"
            "\n"
            "Returns:\n"
            "     None.\n"
            );
// export_idealqvm_func<MPSQVM>::export_func(mps_qvm);

/*combine error*/

    py::enum_<RealChipType>(m, "real_chip_type", "origin quantum real chip type enum")
        .value("origin_wuyuan_d3", RealChipType::ORIGIN_WUYUAN_D3)
        .value("origin_wuyuan_d4", RealChipType::ORIGIN_WUYUAN_D4)
        .value("origin_wuyuan_d5", RealChipType::ORIGIN_WUYUAN_D5)
        .value("origin_72", RealChipType::ORIGIN_72)
        .export_values();

    py::enum_<EmMethod>(m, "em_method", "origin quantum real chip error_mitigation type")
        .value("ZNE", EmMethod::ZNE)
        .value("PEC", EmMethod::PEC)
        .value("READ_OUT", EmMethod::READ_OUT)
        .export_values();
#if defined(USE_QHETU)
    py::class_<QCloudService, QuantumMachine>(m, "QCloudService", "origin quantum cloud machine")
        .def(py::init<>())
        .def_readwrite("user_token", &QCloudService::m_user_token)
        .def_readwrite("inquire_url", &QCloudService::m_inquire_url)
        .def_readwrite("compute_url", &QCloudService::m_compute_url)
        .def_readwrite("estimate_url", &QCloudService::m_estimate_url)
        .def_readwrite("use_compress_data", &QCloudService::m_use_compress_data)
        .def_readwrite("configuration_header_data", &QCloudService::m_configuration_header_data)
        .def_readwrite("measure_qubits_num", &QCloudService::m_measure_qubits_num)
        .def_readwrite("batch_compute_url", &QCloudService::m_batch_compute_url)
        .def_readwrite("batch_inquire_url", &QCloudService::m_batch_inquire_url)


        .def_readwrite("pqc_init_url", &QCloudService::m_pqc_init_url)
        .def_readwrite("pqc_keys_url", &QCloudService::m_pqc_keys_url)
        .def_readwrite("pqc_compute_url", &QCloudService::m_pqc_compute_url)
        .def_readwrite("pqc_inquire_url", &QCloudService::m_pqc_inquire_url)
        .def_readwrite("pqc_batch_compute_url", &QCloudService::m_pqc_batch_compute_url)
        .def_readwrite("pqc_batch_inquire_url", &QCloudService::m_pqc_batch_inquire_url)

        .def_readwrite("big_data_batch_compute_url", &QCloudService::m_big_data_batch_compute_url)
        .def("init",
        &QCloudService::init,
            py::arg("user_token"),
            py::arg("is_logged") = false)

        .def("set_qcloud_url",
            &QCloudService::set_qcloud_url,
            py::arg("cloud_url"))

        .def("build_full_amplitude_measure",
            &QCloudService::build_full_amplitude_measure,
            py::arg("shots"))

        .def("set_noise_model", &QCloudService::set_noise_model)

        .def("build_noise_measure",
            &QCloudService::build_noise_measure,
            py::arg("shots"))

        .def("build_full_amplitude_pmeasure",
            &QCloudService::build_full_amplitude_pmeasure,
            py::arg("qubit_vec"))

        .def("build_partial_amplitude_pmeasure",
            &QCloudService::build_partial_amplitude_pmeasure,
            py::arg("amplitudes"))

        .def("build_single_amplitude_pmeasure",
            &QCloudService::build_single_amplitude_pmeasure,
            py::arg("amplitude"))

        .def("build_error_mitigation",
            [](QCloudService& service,
            int shots,
            int chip_id,
            std::vector<string> expectations,
            std::vector<double>& noise_strength,
            EmMethod qemMethod)
            {
                auto real_chip_type = static_cast<RealChipType>(chip_id);
                return service.build_error_mitigation(shots, real_chip_type, expectations, noise_strength, qemMethod);
            },
            py::arg("shots"),
            py::arg("chip_id"),
            py::arg("expectations"),
            py::arg("noise_strength"),
            py::arg("qemMethod"))

        .def("build_read_out_mitigation",
            [](QCloudService& service,
            int shots,
            int chip_id,
            std::vector<string> expectations,
            std::vector<double>& noise_strength,
            EmMethod qem_method)
            {
                auto real_chip_type = static_cast<RealChipType>(chip_id);
                return service.build_read_out_mitigation(shots, real_chip_type, expectations, noise_strength, qem_method);
            },
            py::arg("shots"),
            py::arg("chip_id"),
            py::arg("expectations"),
            py::arg("noise_strength"),
            py::arg("qem_method"))

        .def("build_real_chip_measure",
            [](QCloudService& service,
            int shots,
            int chip_id = (int)RealChipType::ORIGIN_WUYUAN_D5,
            bool is_amend = true,
            bool is_mapping = true,
            bool is_optimization = true)
            {
                auto real_chip_type = static_cast<RealChipType>(chip_id);
                return service.build_real_chip_measure(shots,
                real_chip_type,
                is_amend,
                is_mapping,
                is_optimization);
            },
            py::arg("shots"),
            py::arg("chip_id") = (int)RealChipType::ORIGIN_WUYUAN_D5,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true)

        .def("build_get_state_tomography_density",
            [](QCloudService& service,
            int shot,
            int chip_id = (int)RealChipType::ORIGIN_WUYUAN_D5,
            bool is_amend = true,
            bool is_mapping = true,
            bool is_optimization = true)
            {
                auto real_chip_type = static_cast<RealChipType>(chip_id);
                return service.build_get_state_tomography_density(shot,
                real_chip_type,
                is_amend,
                is_mapping,
                is_optimization);
            },
            py::arg("shot"),
            py::arg("chip_id") = (int)RealChipType::ORIGIN_WUYUAN_D5,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true)

    .def("build_get_state_fidelity",
        [](QCloudService& service,
        int shot,
        int chip_id = (int)RealChipType::ORIGIN_WUYUAN_D5,
        bool is_amend = true,
        bool is_mapping = true,
        bool is_optimization = true)
        {
            auto real_chip_type = static_cast<RealChipType>(chip_id);
            return service.build_get_state_fidelity(shot,
            real_chip_type,
            is_amend,
            is_mapping,
            is_optimization);
        },
        py::arg("shot"),
        py::arg("chip_id") = (int)RealChipType::ORIGIN_WUYUAN_D5,
        py::arg("is_amend") = true,
        py::arg("is_mapping") = true,
        py::arg("is_optimization") = true)

    .def("build_get_expectation",
        [](QCloudService& service,
        const QHamiltonian& hamiltonian,
        const Qnum& qubits)
        {
            return service.build_get_expectation(hamiltonian, qubits);
        },
        py::arg("hamiltonian"),
        py::arg("qubits"))


    .def("build_real_chip_measure_batch",
        [](QCloudService& service,
        std::vector<string>& originir_list,
        int shots,
        int chip_id = (int)RealChipType::ORIGIN_WUYUAN_D5,
        bool is_amend = true,
        bool is_mapping = true,
        bool is_optimization = true,
        bool enable_compress_check = false,
        std::string batch_id = "",
        int task_from = 4)
        {
            auto real_chip_type = static_cast<RealChipType>(chip_id);
            return service.build_real_chip_measure_batch(originir_list,
            shots,
            real_chip_type,
            is_amend,
            is_mapping,
            is_optimization,
            enable_compress_check,
            batch_id,
            task_from);
        },
        py::arg("prog_list"),
        py::arg("shots"),
        py::arg("chip_id") = (int)RealChipType::ORIGIN_WUYUAN_D5,
        py::arg("is_amend") = true,
        py::arg("is_mapping") = true,
        py::arg("is_optimization") = true,
        py::arg("enable_compress_check") = false,
        py::arg("batch_id") = "",
        py::arg("task_from") = 4)

        .def("build_real_chip_measure_batch",
            [](QCloudService& service,
            std::vector<QProg>& prog_vector,
            int shots,
            int chip_id = (int)RealChipType::ORIGIN_WUYUAN_D5,
            bool is_amend = true,
            bool is_mapping = true,
            bool is_optimization = true,
            bool enable_compress_check = false,
            std::string batch_id = "",
            int task_from = 4)
            {
                auto real_chip_type = static_cast<RealChipType>(chip_id);
                return service.build_real_chip_measure_batch(prog_vector,
                shots,
                real_chip_type,
                is_amend,
                is_mapping,
                is_optimization,
                enable_compress_check,
                batch_id,
                task_from);
            },
            py::arg("prog_list"),
            py::arg("shots"),
            py::arg("chip_id") = (int)RealChipType::ORIGIN_WUYUAN_D5,
            py::arg("is_amend") = true,
            py::arg("is_mapping") = true,
            py::arg("is_optimization") = true,
            py::arg("enable_compress_check") = false,
            py::arg("batch_id") = "",
            py::arg("task_from") = 4)

    .def("build_init_object",
        [](QCloudService& service,
        QProg& prog,
        std::string task_name = "QPanda Experiment",
        int task_from = 4)
        {
            return service.build_init_object(prog, task_name, task_from);
        },
        py::arg("prog"),
        py::arg("task_name") = "QPanda Experiment",
        py::arg("task_from") = 4)

        .def("build_init_object",
            [](QCloudService& service,
                std::string& prog_str,
                std::string task_name = "QPanda Experiment",
                int task_from = 4)
                {
                    return service.build_init_object(prog_str, task_name, task_from);
                },
                py::arg("prog_str"),
                py::arg("task_name") = "QPanda Experiment",
                py::arg("task_from") = 4)

            .def("parse_get_task_id", &QCloudService::parse_get_task_id)
            .def("query_prob_dict_result", &QCloudService::query_prob_dict_result)
            .def("query_comolex_result", &QCloudService::query_comolex_result)
            .def("query_prob_result", &QCloudService::query_prob_result)
            .def("query_prob_vec_result", &QCloudService::query_prob_vec_result)
            .def("query_qst_result", &QCloudService::query_qst_result)
            .def("query_state_dict_result", &QCloudService::query_state_dict_result)
            .def("query_prob_dict_result_batch", &QCloudService::query_prob_dict_result_batch)
                    
            .def("cyclic_query", [](QCloudService& service, const std::string& recv_json)
                {
                    std::string result_string = "";
                    bool is_retry_again = true;
                    service.cyclic_query(recv_json, is_retry_again, result_string);

        return std::make_tuple(is_retry_again, result_string);
    })
    .def("batch_cyclic_query", [](QCloudService& service, const std::string& recv_json)
    {
        std::vector<std::string> result_array;
        bool is_retry_again = true;
        service.batch_cyclic_query(recv_json, is_retry_again, result_array);

        return std::make_tuple(is_retry_again, result_array);
    })

            .def("sm4_encode", &QCloudService::sm4_encode,
                 "Encode data using SM4 algorithm",
                 py::arg("key"), py::arg("IV"), py::arg("data"))
            .def("sm4_decode", &QCloudService::sm4_decode,
                 "Decode data using SM4 algorithm",
                 py::arg("key"), py::arg("IV"), py::arg("enc_data"))
            .def("enc_hybrid", [](QCloudService& self, std::string_view pk_str, std::string& rdnum) 
                {
                    auto enc_data = self.enc_hybrid(pk_str, rdnum);
                    return py::make_tuple(enc_data[0], enc_data[1], enc_data[2], enc_data[3]);

                }, "Perform hybrid encryption and return a tuple",
                   py::arg("pk_str"), py::arg("rdnum"));
#endif

    py::implicitly_convertible<CPUQVM, QuantumMachine>();
    py::implicitly_convertible<GPUQVM, QuantumMachine>();
    py::implicitly_convertible<CPUSingleThreadQVM, QuantumMachine>();
    py::implicitly_convertible<NoiseQVM, QuantumMachine>();
    py::implicitly_convertible<SingleAmplitudeQVM, QuantumMachine>();
    py::implicitly_convertible<PartialAmplitudeQVM, QuantumMachine>();
    py::implicitly_convertible<Stabilizer, QuantumMachine>();
    py::implicitly_convertible<DensityMatrixSimulator, QuantumMachine>();

}
