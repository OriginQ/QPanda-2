#include "QPandaConfig.h"
#include "QPanda.h"
#include "template_generator.h"
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
#include "pybind11/numpy.h"


USING_QPANDA
using namespace std;
using namespace pybind11::literals;
using std::map;
namespace py = pybind11;

// template<>
// struct py::detail::type_caster<QVec>
//     : py::detail::list_caster<QVec, Qubit*> { };

void export_enum(py::module &);
void export_fundament_class(py::module &);
void export_noise_model(py::module &);
void export_core_class(py::module &);
void export_quantum_machine(py::module &);
void export_variational(py::module &);
void export_qalg(py::module &);
void export_components(py::module &);
void export_extension_class(py::module &);
void export_extension_funtion(py::module &);
void export_hamiltoniansimulation(py::module &);

PYBIND11_MODULE(pyQPanda, m)
{
    m.doc() = "A Quantum Program Development and Runtime Environment Kit, based on QPanda";

    /* beware of the declaration sequnce, type be used by ohters should be declare early */
    export_enum(m);
    export_fundament_class(m);
    export_noise_model(m);
    export_core_class(m);
    export_quantum_machine(m);
    export_variational(m);
    export_qalg(m);
    export_components(m);
    export_extension_class(m);
    export_extension_funtion(m);
    export_hamiltoniansimulation(m);

    /* include\Core.h */
    m.def("init", &init, py::arg_v("machine_type", QMachineType::CPU, "QMachineType.CPU"),
        "Init the global unique quantum machine at background.\n"
        "\n"
        "Args:\n"
        "    machine_type: quantum machine type, see pyQPanda.QMachineType\n"
        "\n"
        "Returns:\n"
        "    bool: true if initialization success\n"
    );

    m.def("finalize", py::overload_cast<>(&finalize),
        "Finalize the environment and destroy global unique quantum machine.\n"
        "\n"
        "Args:\n"
        "    none\n"
        "\n"
        "Returns:\n"
        "    none\n"
    );


    m.def("qAlloc",
        py::overload_cast<>(&qAlloc),
        "Create a qubit\n"
        "After init()\n"
        "\n"
        "Args:\n"
        "    None: No parameters are required for this function.\n"
        "\n"
        "Returns:\n"
        "    Qubit: A new qubit.\n"
        "\n"
        "    None: If the quantum machine has created the maximum number of qubits, which is 29.\n",
        py::return_value_policy::reference);

    m.def("qAlloc",
        py::overload_cast<size_t>(&qAlloc),
        py::arg("qubit_addr"),
        "Allocate a qubit\n"
        "After init()\n"
        "\n"
        "Args:\n"
        "    qubit_addr: The physical address of the qubit, should be in [0, 29).\n"
        "\n"
        "Returns:\n"
        "    Qubit: A new qubit.\n"
        "\n"
        "    None: If qubit_addr is invalid or if the maximum number of allowed qubits has been reached.\n",
        py::return_value_policy::reference);

    m.def("directly_run",
        &directlyRun,
        py::arg("qprog"),
        py::arg_v("noise_model", NoiseModel(), "NoiseModel()"),
        "Directly run a quantum program\n"
        "After init()\n"
        "\n"
        "Args:\n"
        "    qprog: The quantum program to be executed.\n"
        "\n"
        "    noise_model: The noise model to be used, default is no noise. The noise model only works on CPUQVM currently.\n"
        "\n"
        "Returns:\n"
        "    Dict[str, bool]: Result of the quantum program execution in one shot.\n"
        "                    The first element is the final qubit register state,\n"
        "                    and the second is its measurement probability.\n");

    m.def("qAlloc_many",
        [](size_t qubit_num)
        {
            auto qv = static_cast<std::vector<Qubit*>>(qAllocMany(qubit_num));
            return qv;
        },
        py::arg("qubit_num"),
            "Allocate several qubits\n"
            "After init()\n"
            "\n"
            "Args:\n"
            "    qubit_num: The number of qubits to be created.\n"
            "\n"
            "Returns:\n"
            "    list[pyQPanda.Qubit]: list of qubit.\n",
            py::return_value_policy::reference);

    m.def("cAlloc",
        py::overload_cast<>(&cAlloc),
        "Allocate a CBit\n"
        "After init()\n"
        "\n"
        "Args:\n"
        "    none\n"
        "\n"
        "Returns:\n"
        "    classic result cbit\n",
        py::return_value_policy::reference);

    m.def("cAlloc",
        py::overload_cast<size_t>(&cAlloc),
        py::arg("cbit_addr"),
        "Allocate a CBit\n"
        "After init()\n"
        "\n"
        "Args:\n"
        "    cbit_addr: cbit address, should be in [0,29).\n"
        "\n"
        "Returns:\n"
        "    classic result cbit\n",
        py::return_value_policy::reference);

    m.def("cAlloc_many",
        &cAllocMany,
        py::arg("cbit_num"),
        "Allocate several CBits\n"
        "After init()\n"
        "\n"
        "Args:\n"
        "    cbit_num: numbers of cbit want to be created.\n"
        "\n"
        "Returns:\n"
        "    list of cbit\n",
        py::return_value_policy::reference);

    m.def("cFree",
        &cFree,
        py::arg("cbit"),
        "Free a CBit\n"
        "\n"
        "Args:\n"
        "    CBit: The CBit to be freed.\n"
        "\n"
        "Returns:\n"
        "    none\n");

    m.def("cFree_all",
        py::overload_cast<>(&cFreeAll),
        "Free all CBits\n"
        "\n"
        "Args:\n"
        "    none\n"
        "\n"
        "Returns:\n"
        "    none\n");

    m.def("cFree_all",
        py::overload_cast<vector<ClassicalCondition> &>(&cFreeAll),
        py::arg("cbit_list"),
        "Free all CBits\n"
        "\n"
        "Args:\n"
        "    cbit_list: a list of cbits.\n"
        "\n"
        "Returns:\n"
        "    none\n");

    m.def("qFree",
        &qFree,
        py::arg("qubit"),
        "Free a qubit\n"
        "\n"
        "Args:\n"
        "    qubit: a qubit\n"
        "\n"
        "Returns:\n"
        "    None\n");

    m.def("qFree_all",
        py::overload_cast<>(&qFreeAll),
        "Free all qubits\n"
        "\n"
        "Args:\n"
        "    None\n"
        "\n"
        "Returns:\n"
        "    None\n");

    m.def("qFree_all",
        py::overload_cast<QVec &>(&qFreeAll),
        py::arg("qubit_list"),
        "Free a list of qubits\n"
        "\n"
        "Args:\n"
        "    qubit_list: A list of qubits to be freed.\n"
        "\n"
        "Returns:\n"
        "    None\n"
    );

    m.def("getstat",
        &getstat,
        "Get the status of the Quantum machine\n"
        "\n"
        "Args:\n"
        "    None\n"
        "\n"
        "Returns:\n"
        "    The status of the Quantum machine, see QMachineStatus.\n",
        py::return_value_policy::reference
    );

    m.def("get_allocate_qubits",
        []()
        {
            QVec qv;
            get_allocate_qubits(qv);
            return static_cast<std::vector<Qubit*>>(qv);
        },
        "Get allocated qubits of QuantumMachine\n"
        "\n"
        "Args:\n"
        "    None\n"
        "\n"
        "Returns:\n"
        "    A list of allocated qubits.\n",
        py::return_value_policy::reference
        );

    m.def(
        "get_allocate_cbits",
        []()
        {
            std::vector<ClassicalCondition> cv;
            get_allocate_cbits(cv);
            return cv;
        },
        "Get allocated cbits of QuantumMachine\n"
        "\n"
        "Args:\n"
        "    None\n"
        "\n"
        "Returns:\n"
        "    A list of allocated cbits.\n",
        py::return_value_policy::reference
        );

    m.def("get_tuple_list",
        &getProbTupleList,
        py::arg("qubit_list"),
        py::arg("select_max") = -1,
        "Get pmeasure result as tuple list\n"
        "\n"
        "Args:\n"
        "    qubit_list: pmeasure qubits list.\n"
        "\n"
        "    select_max: max returned element num in returned tuple, should be in [-1, 1<<len(qubit_list)],\n"
        "\n"
        "    default is -1, meaning no limit.\n"
        "\n"
        "Returns:\n"
        "    Measure result of quantum machine.\n",
        py::return_value_policy::reference);

    m.def("get_prob_list",
        &getProbList,
        py::arg("qubit_list"),
        py::arg("select_max") = -1,
        "Get pmeasure result as list\n"
        "\n"
        "Args:\n"
        "    qubit_list: pmeasure qubits list.\n"
        "\n"
        "    select_max: max returned element num in returned tuple, should be in [-1, 1<<len(qubit_list)],\n"
        "\n"
        "    default is -1, meaning no limit.\n"
        "\n"
        "Returns:\n"
        "    Measure result of quantum machine.\n",
        py::return_value_policy::reference);

    m.def("get_prob_dict",
        &getProbDict,
        py::arg("qubit_list"),
        py::arg("select_max") = -1,
        "Get pmeasure result as dict\n"
        "\n"
        "Args:\n"
        "    qubit_list: pmeasure qubits list.\n"
        "\n"
        "    select_max: max returned element num in returned tuple, should be in [-1, 1<<len(qubit_list)],\n"
        "\n"
        "    default is -1, meaning no limit.\n"
        "\n"
        "Returns:\n"
        "    Measure result of quantum machine.\n",
        py::return_value_policy::reference);

    m.def("prob_run_tuple_list",
        &probRunTupleList,
        py::arg("qptog"),
        py::arg("qubit_list"),
        py::arg("select_max") = -1,
        "Run quantum program and get pmeasure result as tuple list\n"
        "\n"
        "Args:\n"
        "    qprog: quantum program.\n"
        "\n"
        "    qubit_list: pmeasure qubits list.\n"
        "\n"
        "    select_max: max returned element num in returned tuple, should be in [-1, 1<<len(qubit_list)],\n"
        "\n"
        "    default is -1, meaning no limit.\n"
        "\n"
        "Returns:\n"
        "    Measure result of quantum machine.\n",
        py::return_value_policy::reference);

    m.def("prob_run_list",
        &probRunList,
        py::arg("qprog"),
        py::arg("qubit_list"),
        py::arg("select_max") = -1,
        "Run quantum program and get pmeasure result as list\n"
        "\n"
        "Args:\n"
        "    qprog: quantum program.\n"
        "\n"
        "    qubit_list: pmeasure qubits list.\n"
        "\n"
        "    select_max: max returned element num in returned tuple, should be in [-1, 1<<len(qubit_list)],\n"
        "\n"
        "    default is -1, meaning no limit.\n"
        "\n"
        "Returns:\n"
        "    Measure result of quantum machine.\n",
        py::return_value_policy::reference);


    m.def("prob_run_dict",
        &probRunDict,
        py::arg("qprog"),
        py::arg("qubit_list"),
        py::arg("select_max") = -1,
        "Run quantum program and get pmeasure result as dict\n"
        "\n"
        "Args:\n"
        "    qprog: quantum program.\n"
        "\n"
        "    qubit_list: pmeasure qubits list.\n"
        "\n"
        "    select_max: max returned element num in returned tuple, should be in [-1, 1<<len(qubit_list)],\n"
        "\n"
        "    default is -1, meaning no limit.\n"
        "\n"
        "Returns:\n"
        "    Measure result of quantum machine.\n",
        py::return_value_policy::reference);

    m.def(
        "run_with_configuration",
        [](QProg &prog, std::vector<ClassicalCondition> &cbits, int shots, const NoiseModel& model = NoiseModel())
        {
            return runWithConfiguration(prog, cbits, shots, model);
        },
        py::arg("program"),
            py::arg("cbit_list"),
            py::arg("shots"),
            py::arg_v("noise_model", NoiseModel(), "NoiseModel()"),
            "Run quantum program with configuration\n"
            "\n"
            "Args:\n"
            "    program: quantum program.\n"
            "\n"
            "    cbit_list: classical cbits list.\n"
            "\n"
            "    shots: number of times to repeat the quantum program.\n"
            "\n"
            "    noise_model: noise model; defaults to no noise. Noise model only works on CPUQVM now.\n"
            "\n"
            "Returns:\n"
            "    Result of quantum program execution in shots.\n"
            "  First is the final qubit register state, second is its hit count.\n");

    m.def(
        "run_with_configuration",
        [](QProg &prog, int shots, const NoiseModel& model = NoiseModel())
        {
            return runWithConfiguration(prog, shots, model);
        },
        py::arg("program"),
            py::arg("shots"),
            py::arg_v("noise_model", NoiseModel(), "NoiseModel()"),
            "Run quantum program with configuration.\n"
            "\n"
            "Args:\n"
            "    program: quantum program.\n"
            "\n"
            "    shots: repeat run quantum program times.\n"
            "\n"
            "    noise_model: noise model, default is no noise. Noise model only works on CPUQVM now.\n"
            "\n"
            "Returns:\n"
            "    tuple: result of quantum program execution in shots.\n"
            "\n"
            "    First is the final qubit register state, second is its hit shot.\n");

    m.def("quick_measure",
        &quickMeasure,
        py::arg("qubit_list"),
        py::arg("shots"),
        "Quick measure.\n"
        "\n"
        "Args:\n"
        "    qubit_list: qubit list to measure.\n"
        "\n"
        "    shots: the repeat number of measure operations.\n"
        "\n"
        "Returns:\n"
        "    result: result of quantum program execution.\n");

    m.def("accumulate_probabilities",
        &accumulateProbability,
        py::arg("probability_list"),
        "Accumulate the probability from a probability list.\n"
        "\n"
        "Args:\n"
        "    probability_list: measured result in probability list form.\n"
        "\n"
        "Returns:\n"
        "    accumulated_result: accumulated result.\n");

    m.def("accumulateProbability",
        &accumulateProbability,
        py::arg("probability_list"),
        "Accumulate the probability from a probability list.\n"
        "\n"
        "Args:\n"
        "    probability_list: measured result in probability list form.\n"
        "\n"
        "Returns:\n"
        "    accumulated_result: accumulated result.\n");

    m.def("accumulate_probability",
        &accumulateProbability,
        py::arg("probability_list"),
        "Accumulate the probability from a probability list.\n"
        "\n"
        "Args:\n"
        "    probability_list: measured result in probability list form.\n"
        "\n"
        "Returns:\n"
        "    accumulated_result: accumulated result.\n");

    m.def("get_qstate",
        &getQState,
        "Get quantum machine state vector.\n"
        "\n"
        "Args:\n"
        "    none: no parameters required.\n"
        "\n"
        "Returns:\n"
        "    list: state vector list result.\n"
        "\n"
        "Examples:\n"
        "    >>> print(machine.get_qstate())\n"
        "    [0.707+0j, 0.707+0j, 0, 0]\n");

    m.def("init_quantum_machine",
        /*
          pybind11 support C++ polymorphism
          when C++ func return a pointer/reference of base class but point to derived class object
          pybind11 will get it's runtime type info of derived class, convert pointer to derived class object and return python wrapper
        */
        &initQuantumMachine,
        py::arg_v("machine_type", QMachineType::CPU, "QMachineType.CPU,"),
        "Create and initialize a new quantum machine, and let it be a globally unique quantum machine.\n"
        "\n"
        "Args:\n"
        "    machine_type: quantum machine type, see pyQPanda.QMachineType.\n"
        "\n"
        "Returns:\n"
        "    object: the quantum machine, type depends on machine_type:\n"
        "\n"
        "        QMachineType.CPU               --> pyQPanda.CPUQVM\n"
        "\n"
        "        QMachineType.CPU_SINGLE_THREAD --> pyQPanda.CPUSingleThreadQVM\n"
        "\n"
        "        QMachineType.GPU               --> pyQPanda.GPUQVM (if pyQPanda is built with GPU)\n"
        "\n"
        "        QMachineType.NOISE             --> pyQPanda.NoiseQVM\n"
        "\n"
        "        return None if initialization fails.\n",
        /*
          if py::return_value_policy::reference, python object won't take ownership of returned C++ object, C++ should manage resources
        */
        py::return_value_policy::reference);

    /* see PyQuantumMachine to see how python polymorphism run as C++ like */
    m.def("destroy_quantum_machine",
        &destroyQuantumMachine,
        py::arg("machine"),
        "Destroy a quantum machine.\n"
        "\n"
        "Args:\n"
        "    machine: type should be one of CPUQVM, CPUSingleThreadQVM, GPUQVM, NoiseQVM.\n"
        "\n"
        "Returns:\n"
        "    None.\n");

    /* will delete */
    m.def(
        "originir_to_qprog",
        [](string file_path, QuantumMachine *qvm)
        {
            QVec qv;
            std::vector<ClassicalCondition> cv;
            return transformOriginIRToQProg(file_path, qvm, qv, cv);
        },
        py::arg("file_path"),
            py::arg("machine"),
            "Read an OriginIR file and transform it into QProg.\n"
            "\n"
            "Args:\n"
            "    file_path: OriginIR file path.\n"
            "\n"
            "    machine: initialized quantum machine.\n"
            "\n"
            "Returns:\n"
            "    Transformed QProg.\n",
            py::return_value_policy::automatic_reference);

    m.def(
        "convert_originir_to_qprog",
        [](std::string file_path, QuantumMachine *qvm)
        {
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
        py::arg("file_path"),
            py::arg("machine"),
            "Read an OriginIR file and transform it into QProg.\n"
            "\n"
            "Args:\n"
            "    file_path: OriginIR file path.\n"
            "\n"
            "    machine: initialized quantum machine.\n"
            "\n"
            "Returns:\n"
            "    A list containing QProg, qubit_list, and cbit_list.\n",
            py::return_value_policy::automatic_reference);

    m.def(
        "convert_originir_str_to_qprog",
        [](std::string originir_str, QuantumMachine *qvm)
        {
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
        py::arg("originir_str"),
            py::arg("machine"),
            "Transform OriginIR string into QProg.\n"
            "\n"
            "Args:\n"
            "    originir_str: OriginIR string.\n"
            "\n"
            "    machine: initialized quantum machine.\n"
            "\n"
            "Returns:\n"
            "    A list containing QProg, qubit_list, and cbit_list.\n",
            py::return_value_policy::automatic_reference);

    m.def(
        "convert_qasm_string_to_qprog",
        [](std::string qasm_str, QuantumMachine* qvm)
        {
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
        py::arg("qasm_str"),
        py::arg("machine"),
        "Trans QASM to QProg\n"
        "\n"
        "Args:\n"
        "    qasm_str: QASM string\n"
        "    machine: initialized quantum machine\n"
        "\n"
        "Returns:\n"
        "    list cotains QProg, qubit_list, cbit_list",
        py::return_value_policy::automatic_reference);

    m.def(
        "convert_qasm_to_qprog",
        [](std::string file_path, QuantumMachine* qvm)
        {
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
        py::arg("file_path"),
        py::arg("machine"),
        "Read QASM file and trans to QProg\n"
        "\n"
        "Args:\n"
        "    file_path: QASM file path\n"
        "    machine: initialized quantum machine\n"
        "\n"
        "Returns:\n"
        "    list cotains QProg, qubit_list, cbit_list"
        "Raises:\n"
        "    run_fail: An error occurred in convert_qasm_to_qprog\n",
        py::return_value_policy::automatic_reference);

    //
    m.def(
        "convert_qasm_string_to_originir",
        [](std::string qasm_str)
        {
            py::str originir_str = convert_qasm_string_to_originir(qasm_str);
            return originir_str;
        },
        py::arg("qasm_str"),
        "Trans QASM to OriginIR string\n"
        "\n"
        "Args:\n"
        "    qasm_str: QASM string\n"
        "\n"
        "Returns:\n"
        "    str originir_str",
        py::return_value_policy::automatic_reference);

    m.def(
        "convert_qasm_to_originir",
        [](std::string file_path)
        {
            py::str originir_str = convert_qasm_to_originir(file_path);
            return originir_str;
        },
        py::arg("file_path"),
        "Read QASM file and trans to OriginIR string\n"
        "\n"
        "Args:\n"
        "    file_path: QASM file path\n"
        "\n"
        "Returns:\n"
        "    str originir_str"
        "Raises:\n"
        "    run_fail: An error occurred in convert_qasm_to_originir\n",
        py::return_value_policy::automatic_reference);
    //convert pyquil to originir/qprog
    m.def(
        "convert_pyquil_string_to_originir",
        [](std::string pyquil_str)
        {
            py::str originir_str = convert_pyquil_string_to_originir(pyquil_str);
            return originir_str;
        },
        py::arg("pyquil_str"),
        "Trans pyquil to OriginIR string\n"
        "\n"
        "Args:\n"
        "    pyquil_str: pyquil string\n"
        "\n"
        "Returns:\n"
        "    str originir_str",
        py::return_value_policy::automatic_reference);

    m.def(
        "convert_pyquil_string_to_qprog",
        [](std::string pyquil_str, QuantumMachine* qvm)
        {
            py::list ret_data;
            QVec qv;
            std::vector<ClassicalCondition> cv;
            QProg prog = convert_pyquil_string_to_qprog(pyquil_str, qvm, qv, cv);
            py::list qubit_list;
            for (auto q : qv)
                qubit_list.append(q);

            ret_data.append(prog);
            ret_data.append(qubit_list);
            ret_data.append(cv);
            return ret_data;
        },
        py::arg("pyquil_str"),
        py::arg("machine"),
        "Trans pyquil to QProg\n"
        "\n"
        "Args:\n"
        "    pyquil_str: pyquil string\n"
        "    machine: initialized quantum machine\n"
        "\n"
        "Returns:\n"
        "    list cotains QProg, qubit_list, cbit_list",
        py::return_value_policy::automatic_reference);

    m.def(
        "convert_pyquil_file_to_originir",
        [](std::string file_path)
        {
            py::str originir_str = convert_pyquil_file_to_originir(file_path);
            return originir_str;
        },
        py::arg("file_path"),
        "Read pyquil file and trans to OriginIR string\n"
        "\n"
        "Args:\n"
        "    file_path: pyquil file path\n"
        "\n"
        "Returns:\n"
        "    str originir_str"
        "Raises:\n"
        "    run_fail: An error occurred in convert_pyquil_to_qprog\n",
        py::return_value_policy::automatic_reference);

    m.def(
        "convert_pyquil_file_to_qprog",
        [](std::string file_path, QuantumMachine* qvm)
        {
            py::list ret_data;
            QVec qv;
            std::vector<ClassicalCondition> cv;
            QProg prog = convert_pyquil_file_to_qprog(file_path, qvm, qv, cv);
            py::list qubit_list;
            for (auto q : qv)
                qubit_list.append(q);

            ret_data.append(prog);
            ret_data.append(qubit_list);
            ret_data.append(cv);
            return ret_data;
        },
        py::arg("file_path"),
        py::arg("machine"),
        "Read pyquil file and trans to QProg\n"
        "\n"
        "Args:\n"
        "    file_path: pyquil file path\n"
        "    machine: initialized quantum machine\n"
        "\n"
        "Returns:\n"
        "    list cotains QProg, qubit_list, cbit_list"
        "Raises:\n"
        "    run_fail: An error occurred in convert_pyquil_to_qprog\n",
        py::return_value_policy::automatic_reference);

    ///////////////

	/*will delete*/
	m.def("getAllocateCMem",
        &getAllocateCMemNum,
        "Deprecated, use get_allocate_cmem_num instead.\n"
        "\n"
        "Args:\n"
        "    none\n"
        "\n"
        "Returns:\n"
        "    allocate qubit num.\n");

    m.def("getAllocateQubitNum",
        &getAllocateQubitNum,
        "Deprecated, use get_allocate_qubit_num instead.\n"
        "\n"
        "Args:\n"
        "    none\n"
        "\n"
        "Returns:\n"
        "    allocate cbit num.\n");

    m.def("PMeasure",
        &PMeasure,
        "Deprecated, use pmeasure instead.\n"
        "\n"
        "Args:\n"
        "    QVec: pmeasure qubits list.\n"
        "\n"
        "    select_num: result select num.\n"
        "\n"
        "Returns:\n"
        "    result: pmeasure qubits result.\n");

    m.def("PMeasure_no_index",
        &PMeasure_no_index,
        "Deprecated, use pmeasure_no_index instead.\n"
        "\n"
        "Args:\n"
        "    QVec: pmeasure qubits list.\n"
        "\n"
        "Returns:\n"
        "    result: pmeasure qubits result.\n");

    /* new interface */
    m.def("get_allocate_qubit_num",
        &getAllocateQubitNum,
        "Get allocate qubit num.\n"
        "\n"
        "Args:\n"
        "    none.\n"
        "\n"
        "Returns:\n"
        "    qubit_num: allocate qubit num.\n",
        py::return_value_policy::automatic);

    m.def("get_allocate_cmem_num",
        &getAllocateCMem,
        "Get allocate cmem num.\n"
        "\n"
        "Args:\n"
        "    none.\n"
        "\n"
        "Returns:\n"
        "    cbit_num: allocate cbit num.\n",
        py::return_value_policy::automatic
    );

    m.def("pmeasure", &pMeasure,
        py::arg("qubit_list"),
        py::arg("select_max"),
        "Get the probability distribution over qubits.\n"
        "\n"
        "Args:\n"
        "    qubit_list: qubit list to measure.\n"
        "\n"
        "    select_max: max returned element num in returned tuple, should be in [-1, 1<<len(qubit_list)];\n"
        "\n"
        "    default is -1, means no limit.\n"
        "\n"
        "Returns:\n"
        "    Measure result of quantum machine in tuple form.\n",
        py::return_value_policy::automatic
    );

    m.def("pmeasure_no_index",
        &pMeasureNoIndex,
        py::arg("qubit_list"),
        "Get the probability distribution over qubits.\n"
        "\n"
        "Args:\n"
        "    qubit_list: qubit list to measure.\n"
        "\n"
        "Returns:\n"
        "    Measure result of quantum machine in list form.\n",
        py::return_value_policy::automatic
    );

    m.def("QOracle",
        py::overload_cast<const QVec &, const Eigen::MatrixXcd&, const double> (&QOracle),
        py::arg("qubit_list"),
        py::arg("matrix"),
        py::arg("tol") = 1e-10,
        "Generate QOracle Gate.\n"
        "\n"
        "Args:\n"
        "    qubit_list: gate in qubit list.\n"
        "\n"
        "    matrix: gate operator matrix.\n"
        "\n"
        "Returns:\n"
        "    Oracle gate.\n",
        py::return_value_policy::automatic
    );

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\QProgInfo\QGateCounter.h */
    /* getQGateNumber is a template function,*/
    m.def("get_qgate_num",
        py::overload_cast<QProg &>(&getQGateNum<QProg>),
        py::arg("quantum_prog"),
        "Count quantum gate number in the quantum program.\n"
        "\n"
        "Args:\n"
        "    quantum_prog: quantum program.\n"
        "\n"
        "Returns:\n"
        "    result: gate count.\n",
        py::return_value_policy::automatic
    );

    m.def("get_qgate_num",
        py::overload_cast<QCircuit &>(&getQGateNum<QCircuit>),
        py::arg("quantum_circuit"),
        "Count quantum gate number in the quantum circuit.\n"
        "\n"
        "Args:\n"
        "    quantum_circuit: quantum circuit.\n"
        "\n"
        "Returns:\n"
        "    result: gate count.\n",
        py::return_value_policy::automatic
    );

    m.def("count_gate",
        py::overload_cast<QProg &>(&getQGateNum<QProg>),
        py::arg("quantum_prog"),
        "Count quantum gate number in the quantum program.\n"
        "\n"
        "Args:\n"
        "    quantum_prog: quantum program.\n"
        "\n"
        "Returns:\n"
        "    result: gate count.\n",
        py::return_value_policy::automatic
    );

    m.def("count_gate",
        py::overload_cast<QCircuit &>(&getQGateNum<QCircuit>),
        py::arg("quantum_circuit"),
        "Count quantum gate number in the quantum circuit.\n"
        "\n"
        "Args:\n"
        "    quantum_circuit: quantum circuit.\n"
        "\n"
        "Returns:\n"
        "    result: gate count.\n",
        py::return_value_policy::automatic
    );

    m.def("count_qgate_num",
        [](QProg& prog, int type)
        {
            auto gate_type = static_cast<GateType>(type);
            return count_qgate_num<QProg>(prog, gate_type);
        },
        py::arg("prog"),
            py::arg("gate_type") = (int)GateType::GATE_UNDEFINED,
            "Count quantum gate number in the quantum program.\n"
            "\n"
            "Args:\n"
            "    prog: quantum program (QProg&).\n"
            "\n"
            "    gate_type: type of gate to count (const GateType).\n"
            "\n"
            "Returns:\n"
            "    result: number of quantum gates of the specified GateType.\n",
            py::return_value_policy::automatic
            );

    m.def("count_qgate_num",
        [](QCircuit& circuit, int type)
        {
            auto gate_type = static_cast<GateType>(type);
            return count_qgate_num<QCircuit>(circuit, gate_type);
        },
        py::arg("circuit"),
            py::arg("gate_type") = (int)GateType::GATE_UNDEFINED,
            "Count quantum gate number in the quantum circuit.\n"
            "\n"
            "Args:\n"
            "    circuit: quantum circuit (QCircuit&).\n"
            "\n"
            "    gate_type: type of gate to count (const GateType).\n"
            "\n"
            "Returns:\n"
            "    result: number of quantum gates of the specified GateType.\n",
            py::return_value_policy::automatic
            );
    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\QuantumCircuit\QProgram.h */
    m.def("CreateEmptyQProg",
        &CreateEmptyQProg,
        "Create an empty QProg container.\n"
        "\n"
        "Args:\n"
        "    none\n"
        "\n"
        "Returns:\n"
        "    an empty QProg.\n",
        py::return_value_policy::automatic
    );
    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\Compiler\QProgToOriginIR.h */
    export_transformQProgToOriginIR<QProg>(m);
    export_transformQProgToOriginIR<QCircuit>(m);
    export_transformQProgToOriginIR<QGate>(m);
    export_transformQProgToOriginIR<QIfProg>(m);
    export_transformQProgToOriginIR<QWhileProg>(m);
    export_transformQProgToOriginIR<QMeasure>(m);

    m.def("convert_qprog_to_originir",
        py::overload_cast<QProg &, QuantumMachine *>(&convert_qprog_to_originir<QProg>),
        py::arg("qprog"),
        py::arg("machine"),
        "Convert QProg to OriginIR string.\n"
        "\n"
        "Args:\n"
        "    qprog: quantum program (QProg&).\n"
        "\n"
        "    machine: quantum machine (QuantumMachine*).\n"
        "\n"
        "Returns:\n"
        "    originir: OriginIR string. For more information, see the OriginIR introduction:\n"
        "\n"
        "  https://pyqpanda-toturial.readthedocs.io/zh/latest\n",
        py::return_value_policy::automatic_reference
    );

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\Compiler\QProgToQuil.h */
    m.def("to_Quil",
        &transformQProgToQuil,
        py::arg("qprog"),
        py::arg("machine"),
        "Transform QProg to Quil instruction.\n"
        "\n"
        "Args:\n"
        "    qprog: quantum program (QProg).\n"
        "\n"
        "    machine: quantum machine (QuantumMachine*).\n"
        "\n"
        "Returns:\n"
        "    Quil instruction string.\n",
        py::return_value_policy::automatic_reference);

    m.def("transform_qprog_to_quil",
        &transformQProgToQuil,
        py::arg("qprog"),
        py::arg("machine"),
        "Transform QProg to Quil instruction.\n"
        "\n"
        "Args:\n"
        "    qprog: quantum program (QProg).\n"
        "\n"
        "    machine: quantum machine (QuantumMachine*).\n"
        "\n"
        "Returns:\n"
        "    Quil instruction string.\n",
        py::return_value_policy::automatic_reference);

    m.def("convert_qprog_to_quil",
        &convert_qprog_to_quil,
        py::arg("qprog"),
        py::arg("machine"),
        "Convert QProg to Quil instruction.\n"
        "\n"
        "Args:\n"
        "    qprog: quantum program (QProg).\n"
        "\n"
        "    machine: quantum machine (QuantumMachine*).\n"
        "\n"
        "Returns:\n"
        "    Quil instruction string.\n",
        py::return_value_policy::automatic_reference);

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\QProgInfo\QProgClockCycle.h */
    m.def("get_qprog_clock_cycle",
        &get_qprog_clock_cycle,
        py::arg("qprog"),
        py::arg("machine"),
        py::arg("optimize") = false,
        "Get Quantum Program Clock Cycle.\n"
        "\n"
        "Args:\n"
        "    qprog: quantum program (QProg).\n"
        "\n"
        "    machine: quantum machine (QuantumMachine*).\n"
        "\n"
        "    optimize: whether to optimize qprog (default is false).\n"
        "\n"
        "Returns:\n"
        "    QProg time consumed, no unit, not in seconds.\n",
        py::return_value_policy::automatic_reference);

    m.def(
        "get_clock_cycle",
        [](QProg prog)
        {
            extern QuantumMachine *global_quantum_machine;
            return getQProgClockCycle(prog, global_quantum_machine);
        },
        py::arg("qpog"),
            "Get quantum program clock cycle.\n"
            "\n"
            "Args:\n"
            "    qprog: quantum program (QProg).\n"
            "\n"
            "Returns:\n"
            "    clock_cycle.\n",
            py::return_value_policy::automatic_reference);

    py::class_<QProgInfoCount>(m, "ProgCount")
        .def(py::init<>())
        .def_readwrite("node_num", &QProgInfoCount::node_num,
            "Node number in the program count.\n"
            "\n"
            "This attribute represents the number of nodes in the quantum program.\n"
            "\n"
            "Returns:\n"
            "    An integer representing the node number.\n"
        )
        .def_readwrite("gate_num", &QProgInfoCount::gate_num,
            "Gate number in the program count.\n"
            "\n"
            "This attribute represents the number of gates in the quantum program.\n"
            "\n"
            "Returns:\n"
            "    An integer representing the gate number.\n"
        )
        .def_readwrite("layer_num", &QProgInfoCount::layer_num,
            "Layer number in the program count.\n"
            "\n"
            "This attribute represents the number of layers in the quantum program.\n"
            "\n"
            "Returns:\n"
            "    An integer representing the layer number.\n"
        )
        .def_readwrite("single_gate_num", &QProgInfoCount::single_gate_num,
            "Single gate number in the program count.\n"
            "\n"
            "This attribute represents the number of single gates in the quantum program.\n"
            "\n"
            "Returns:\n"
            "    An integer representing the single gate number.\n"
        )
        .def_readwrite("double_gate_num", &QProgInfoCount::double_gate_num,
            "Double gate number in the program count.\n"
            "\n"
            "This attribute represents the number of double gates in the quantum program.\n"
            "\n"
            "Returns:\n"
            "    An integer representing the double gate number.\n"
        )
        .def_readwrite("multi_control_gate_num", &QProgInfoCount::multi_control_gate_num,
            "Multi-control gate number in the program count.\n"
            "\n"
            "This attribute represents the number of multi-control gates in the quantum program.\n"
            "\n"
            "Returns:\n"
            "    An integer representing the multi-control gate number.\n"
        )
        .def_readwrite("single_gate_layer_num", &QProgInfoCount::single_gate_layer_num,
            "Single gate layer number in the program count.\n"
            "\n"
            "This attribute represents the number of layers containing single gates in the quantum program.\n"
            "\n"
            "Returns:\n"
            "    An integer representing the single gate layer number.\n"
        )
        .def_readwrite("double_gate_layer_num", &QProgInfoCount::double_gate_layer_num,
            "Double gate layer number in the program count.\n"
            "\n"
            "This attribute represents the number of layers containing double gates in the quantum program.\n"
            "\n"
            "Returns:\n"
            "    An integer representing the double gate layer number.\n"
        )
        .def_readwrite("selected_gate_nums", &QProgInfoCount::selected_gate_nums,
            "Selected gate numbers in the program count.\n"
            "\n"
            "This attribute represents the count of selected gates in the quantum program.\n"
            "\n"
            "Returns:\n"
            "    A list or array of integers representing the selected gate numbers.\n"
        );

    m.def("count_prog_info",
        py::overload_cast<QProg&, std::vector<GateType>>(&count_prog_info<QProg>),
        py::arg("node"),
        py::arg("selected_types") = std::vector<GateType>(),
        "Count quantum program information.\n"
        "\n"
        "Args:\n"
        "    node: quantum program (QProg).\n"
        "\n"
        "    selected_types: vector of selected GateType (default is empty).\n"
        "\n"
        "Returns:\n"
        "    ProgCount struct.\n",
        py::return_value_policy::automatic_reference);

    m.def("count_prog_info",
        py::overload_cast<QCircuit&, std::vector<GateType>>(&count_prog_info<QCircuit>),
        py::arg("node"),
        py::arg("selected_types") = std::vector<GateType>(),
        "Count quantum program information.\n"
        "\n"
        "Args:\n"
        "    node: quantum circuit (QCircuit).\n"
        "\n"
        "    selected_types: vector of selected GateType (default is empty).\n"
        "\n"
        "Returns:\n"
        "    ProgCount struct.\n",
        py::return_value_policy::automatic_reference);



    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\Compiler\QProgStored.h */
    m.def("transform_qprog_to_binary",
        py::overload_cast<QProg &, QuantumMachine *>(&transformQProgToBinary),
        py::arg("qprog"),
        py::arg("machine"),
        "Transform quantum program to binary data.\n"
        "\n"
        "Args:\n"
        "    qprog: quantum program (QProg).\n"
        "\n"
        "    machine: quantum machine.\n"
        "\n"
        "Returns:\n"
        "    binary data as a list.\n",
        py::return_value_policy::automatic_reference);

    m.def("transform_qprog_to_binary",
        py::overload_cast<QProg &, QuantumMachine *, const string &>(&transformQProgToBinary),
        py::arg("qprog"),
        py::arg("machine"),
        py::arg("fname"),
        "Save quantum program to file as binary data.\n"
        "\n"
        "Args:\n"
        "    qprog: quantum program (QProg).\n"
        "\n"
        "    machine: quantum machine.\n"
        "\n"
        "    fname: name of the file to save to.\n");

    m.def(
        "get_bin_data",
        [](QProg prog)
        {
            extern QuantumMachine *global_quantum_machine;
            return transformQProgToBinary(prog, global_quantum_machine);
        },
        py::arg("qprog"),
            "Get quantum program binary data.\n"
            "\n"
            "Args:\n"
            "    qprog: quantum program (QProg).\n"
            "\n"
            "Returns:\n"
            "    binary data as a list.\n",
            py::return_value_policy::automatic_reference);

    m.def(
        "bin_to_prog",
        [](const std::vector<uint8_t> &data, QVec &qubits, std::vector<ClassicalCondition> &cbits, QProg &prog)
        {
            extern QuantumMachine *global_quantum_machine;
            return binaryQProgDataParse(global_quantum_machine, data, qubits, cbits, prog);
        },
        py::arg("bin_data"),
            py::arg("qubit_list"),
            py::arg("cbit_list"),
            py::arg("qprog"),
            "Parse binary data to transform into a quantum program.\n"
            "\n"
            "Args:\n"
            "    bin_data: binary data that stores quantum program information.\n"
            "\n"
            "    qubit_list: list of quantum qubits.\n"
            "\n"
            "    cbit_list: list of classical bits.\n"
            "\n"
            "    qprog: quantum program.\n"
            "\n"
            "Returns:\n"
            "    prog: the parsed quantum program.\n",
            py::return_value_policy::automatic_reference);

    m.def(
        "get_bin_str",
        [](QProg prog, QuantumMachine *qvm)
        {
            auto data = transformQProgToBinary(prog, qvm);
            auto base64_data = Base64::encode(data.data(), data.size());
            std::string data_str(base64_data.begin(), base64_data.end());
            return data_str;
        },
        py::arg("qprog"),
            py::arg("machine"),
            "Transform a quantum program into a string representation.\n"
            "\n"
            "Args:\n"
            "    qprog: quantum program.\n"
            "\n"
            "    machine: quantum machine.\n"
            "\n"
            "Returns:\n"
            "    string: base64-encoded string of the binary representation.\n",
            py::return_value_policy::automatic);

    m.def("convert_qprog_to_binary",
        py::overload_cast<QProg &, QuantumMachine *>(&convert_qprog_to_binary),
        py::arg("qprog"),
        py::arg("machine"),
        "Convert a quantum program into binary data.\n"
        "\n"
        "Args:\n"
        "    qprog: quantum program.\n"
        "\n"
        "    machine: quantum machine.\n"
        "\n"
        "Returns:\n"
        "    string: binary data representation of the quantum program.\n",
        py::return_value_policy::automatic_reference
    );

    m.def("convert_qprog_to_binary",
        py::overload_cast<QProg &, QuantumMachine *, const string &>(&convert_qprog_to_binary),
        py::arg("qprog"),
        py::arg("machine"),
        py::arg("fname"),
        "Store the quantum program in a binary file.\n"
        "\n"
        "Args:\n"
        "    qprog: quantum program.\n"
        "\n"
        "    machine: quantum machine.\n"
        "\n"
        "    fname: name of the binary data file.\n"
        "\n"
        "Returns:\n"
        "    none: This function does not return a value.\n",
        py::return_value_policy::automatic_reference
    );

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\Compiler\QProgDataParse.h */
    m.def(
        "transform_binary_data_to_qprog",
        [](QuantumMachine *qm, std::vector<uint8_t> data)
        {
            QVec qubits;
            std::vector<ClassicalCondition> cbits;
            QProg prog;
            transformBinaryDataToQProg(qm, data, qubits, cbits, prog);
            return prog;
        },
        py::arg("machine"),
            py::arg("data"),
            "Parse binary data to transform it into a quantum program.\n"
            "\n"
            "Args:\n"
            "    machine: quantum machine.\n"
            "\n"
            "    data: list containing binary data from transform_qprog_to_binary().\n"
            "\n"
            "Returns:\n"
            "    QProg: the resulting quantum program.\n",
            py::return_value_policy::automatic_reference
            );

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\Compiler\QProgDataParse.h */
    m.def("transform_qprog_to_originir",
        py::overload_cast<QProg &, QuantumMachine *>(&transformQProgToOriginIR<QProg>),
        py::arg("qprog"),
        py::arg("machine"),
        "Transform a quantum program into an OriginIR instruction string.\n"
        "\n"
        "Args:\n"
        "    qprog: the quantum program (QProg).\n"
        "\n"
        "    machine: the quantum machine.\n"
        "\n"
        "Returns:\n"
        "    string: the resulting OriginIR instruction string.\n",
        py::return_value_policy::automatic_reference
    );

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\Compiler\OriginIRToQProg.h */
    m.def(
        "transform_originir_to_qprog",
        [](string file_path, QuantumMachine *qvm)
        {
            QVec qv;
            std::vector<ClassicalCondition> cv;
            return transformOriginIRToQProg(file_path, qvm, qv, cv);
        },
        py::arg("fname"),
            py::arg("machine"),
            "Transform OriginIR instruction from a file into a QProg.\n"
            "\n"
            "Args:\n"
            "    fname: file containing the OriginIR instructions.\n"
            "\n"
            "    machine: the quantum machine.\n"
            "\n"
            "Returns:\n"
            "    QProg: the resulting quantum program.\n",
            py::return_value_policy::automatic_reference
            );

    //---------------------------------------------------------------------------------------------------------------------
    /* new wrapper python interface */
    m.def(
        "apply_QGate",
        [](const QVec &qlist, const std::function<QGate(Qubit *)> &fun) -> QCircuit
        {
            QCircuit cir;
            for (auto &q : qlist)
            {
                cir << fun(q);
            }
            return cir;
        },
        py::arg("qubit_list"),
            py::arg("func_obj"),
            "Apply a quantum gate operation to a list of qubits.\n\n"
            "\n"
            "Args:\n"
            "    qubit_list: List of qubits to which the gate will be applied.\n"
            "\n"
            "    func_obj: A function object that takes a Qubit and returns a QGate.\n\n"
            "\n"
            "Returns:\n"
            "    QCircuit: The resulting circuit containing the QGate operations on all qubits.\n",
            py::return_value_policy::reference
            );

    m.def(
        "apply_QGate",
        [](const std::vector<int> &qlist_addr, const std::function<QGate(int)> &fun) -> QCircuit
        {
            QCircuit cir;
            for (auto &q : qlist_addr)
            {
                cir << fun(q);
            }
            return cir;
        },
        py::arg("qubit_addr_list"),
            py::arg("func_obj"),
            "Apply a quantum gate operation to a list of qubit addresses.\n\n"
            "\n"
            "Args:\n"
            "    qubit_addr_list: List of qubit addresses to which the gate will be applied.\n"
            "\n"
            "    func_obj: A function object that takes a qubit address (int) and returns a QGate.\n\n"
            "\n"
            "Returns:\n"
            "    QCircuit: The resulting circuit containing the QGate operations on all qubits.\n",
            py::return_value_policy::reference
            );

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\QuantumCircuit\ControlFlow.h */
    /* will delete */
    m.def("CreateWhileProg",
        &CreateWhileProg,
        py::arg("classical_condition"),
        py::arg("true_node"),
        "Create a WhileProg that executes while a classical condition is true.\n"
        "\n"
        "Args:\n"
        "    classical_condition: A classical condition representing the while-loop condition.\n"
        "\n"
        "    true_node: The quantum operations to execute while the condition is true.\n"
        "\n"
        "Returns:\n"
        "    WhileProg: The program that performs the specified operations while the condition holds.\n",
        py::return_value_policy::automatic
    );

    m.def("CreateIfProg",
        py::overload_cast<ClassicalCondition, QProg>(&CreateIfProg),
        py::arg("classical_condition"),
        py::arg("true_node"),
        "Create an IfProg that executes a quantum operation if a classical condition is true.\n"
        "\n"
        "Args:\n"
        "    classical_condition: A classical condition representing the if condition.\n"
        "\n"
        "    true_node: The quantum operations to execute if the condition is true.\n"
        "\n"
        "Returns:\n"
        "    IfProg: The program that performs the specified operations if the condition is true.\n",
        py::return_value_policy::automatic
    );

    m.def("CreateIfProg",
        py::overload_cast<ClassicalCondition, QProg, QProg>(&CreateIfProg),
        py::arg("classical_condition"),
        py::arg("true_node"),
        py::arg("false_node"),
        "Create an IfProg that executes one of two quantum operations based on a classical condition.\n"
        "\n"
        "Args:\n"
        "    classical_condition: A classical condition representing the if condition.\n"
        "\n"
        "    true_node: The quantum operations to execute if the condition is true.\n"
        "\n"
        "    false_node: The quantum operations to execute if the condition is false.\n"
        "\n"
        "Returns:\n"
        "    IfProg: The program that performs the specified operations based on the condition.\n",
        py::return_value_policy::automatic
    );

    /* new interface */
    m.def("create_while_prog",
        &createWhileProg,
        py::arg("classical_condition"),
        py::arg("true_node"),
        "Create a WhileProg.\n"
        "\n"
        "Args:\n"
        "    classical_condition: A quantum cbit representing the condition.\n"
        "\n"
        "    true_node: A quantum QWhile node that defines the operation to execute while the condition is true.\n"
        "\n"
        "Returns:\n"
        "    result: A WhileProg that executes the specified operations based on the condition.\n",
        py::return_value_policy::automatic);

    m.def("create_if_prog",
        py::overload_cast<ClassicalCondition, QProg>(&CreateIfProg),
        py::arg("classical_condition"),
        py::arg("true_node"),
        "Create a classical quantum IfProg.\n"
        "\n"
        "Args:\n"
        "    classical_condition: A quantum cbit representing the condition.\n"
        "\n"
        "    true_node: A quantum IfProg node that defines the operation to execute if the condition is true.\n"
        "\n"
        "Returns:\n"
        "    result: A classical quantum IfProg that executes based on the specified condition.\n",
        py::return_value_policy::automatic);

    m.def("create_if_prog",
        py::overload_cast<ClassicalCondition, QProg, QProg>(&CreateIfProg),
        py::arg("classical_condition"),
        py::arg("true_node"),
        py::arg("false_node"),
        "Create a classical quantum IfProg.\n"
        "\n"
        "Args:\n"
        "    classical_condition: A quantum cbit representing the condition.\n"
        "\n"
        "    true_node: A quantum IfProg node that defines the operation to execute if the condition is true.\n"
        "\n"
        "    false_node: A quantum IfProg node that defines the operation to execute if the condition is false.\n"
        "\n"
        "Returns:\n"
        "    result: A classical quantum IfProg that executes based on the specified condition.\n",
        py::return_value_policy::automatic);

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\QuantumCircuit\QCircuit.h */
    /* will delete */
    m.def("CreateEmptyCircuit",
        &CreateEmptyCircuit,
        "Create an empty QCircuit container.\n"
        "\n"
        "Args:\n"
        "    none\n"
        "\n"
        "Returns:\n"
        "    result: An empty QCircuit.\n",
        py::return_value_policy::automatic);

    /* new interface */
    m.def("create_empty_circuit",
        &createEmptyCircuit,
        "Create an empty QCircuit container.\n"
        "\n"
        "Args:\n"
        "    none\n"
        "\n"
        "Returns:\n"
        "    result: An empty QCircuit.\n",
        py::return_value_policy::automatic);

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\QuantumCircuit\QProgram.h */
    m.def("create_empty_qprog",
        &createEmptyQProg,
        "Create an empty QProg container.\n"
        "\n"
        "Args:\n"
        "    none.\n"
        "\n"
        "Returns:\n"
        "    an empty QProg.\n",
        py::return_value_policy::automatic);

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\QuantumCircuit\QuantumMeasure.h */
    m.def("Measure",
        py::overload_cast<Qubit *, ClassicalCondition>(&Measure),
        py::arg("qubit"),
        py::arg("cbit"),
        "Create a measure node.\n"
        "\n"
        "Args:\n"
        "    qubit: the qubit to be measured.\n"
        "\n"
        "    cbit: classical bit that stores the quantum measurement result.\n"
        "\n"
        "Returns:\n"
        "    a quantum measure node.\n",
        py::return_value_policy::automatic);

    m.def("Measure",
        py::overload_cast<Qubit *, CBit*>(&Measure),
        py::arg("qubit"),
        py::arg("cbit"),
        "Create a measure node.\n"
        "\n"
        "Args:\n"
        "    qubit: the qubit to be measured.\n"
        "\n"
        "    cbit: classical bit that stores the quantum measurement result.\n"
        "\n"
        "Returns:\n"
        "    a quantum measure node.\n",
        py::return_value_policy::automatic);

    m.def("Measure",
        py::overload_cast<int, int>(&Measure),
        py::arg("qubit_addr"),
        py::arg("cbit_addr"),
        "Create a measure node.\n"
        "\n"
        "Args:\n"
        "    qubit_addr: address of the qubit to be measured.\n"
        "\n"
        "    cbit_addr: address of the classical bit that stores the quantum measurement result.\n"
        "\n"
        "Returns:\n"
        "    a quantum measure node.\n",
        py::return_value_policy::automatic);

    m.def("measure_all",
        py::overload_cast<const QVec &, const std::vector<ClassicalCondition> &>(&MeasureAll),
        py::arg("qubit_list"),
        py::arg("cbit_list"),
        "Create a list of measure nodes.\n"
        "\n"
        "Args:\n"
        "    qubit_list: the qubits to be measured.\n"
        "\n"
        "    cbit_list: classical bits that store the quantum measurement results.\n"
        "\n"
        "Returns:\n"
        "    a list of measure nodes.\n",
        py::return_value_policy::automatic);

    m.def("measure_all",
        py::overload_cast<const std::vector<int> &, const std::vector<int> &>(&MeasureAll),
        py::arg("qubit_addr_list"),
        py::arg("cbit_addr_list"),
        "Create a list of measure nodes.\n"
        "\n"
        "Args:\n"
        "    qubit_addr_list: list of addresses of the qubits to be measured.\n"
        "\n"
        "    cbit_addr_list: list of addresses of the classical bits that store the quantum measurement results.\n"
        "\n"
        "Returns:\n"
        "    a list of measure nodes.\n",
        py::return_value_policy::automatic);

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\QuantumCircuit\QGate.h */

#define EXPORT_singleBitGate(gate_name)                            \
        m.def(#gate_name,                                              \
            py::overload_cast<Qubit *>(&gate_name),                  \
            py::arg("qubit"),                                        \
            "Create a " #gate_name " gate\n"                         \
            "\n"                                                     \
            "Args:\n"                                                \
            "    qubit : quantum gate operate qubit\n"     \
            "\n"                                                     \
            "Returns:\n"                                             \
            "    a " #gate_name " gate node\n"                       \
            "Raises:\n"                                              \
            "    run_fail: An error occurred in construct gate node\n", \
            py::return_value_policy::automatic);                     \
        m.def(#gate_name, \
            py::overload_cast<const QVec &>(&gate_name), \
            py::arg("qubit_list"), \
            "Create a " #gate_name " gate\n"                         \
            "\n"                                                     \
            "Args:\n"                                                \
            "    qubit_list: quantum gate operate qubits list\n"     \
            "\n" \
            "Returns:\n"                                             \
            "    a " #gate_name " gate node\n"                       \
            "Raises:\n"                                              \
            "    run_fail: An error occurred construct in gate node\n", \
            py::return_value_policy::automatic);                     \
        m.def(#gate_name, \
            py::overload_cast<int>(&gate_name), \
            py::arg("qubit_addr"), \
            "Create a " #gate_name " gate\n"                         \
            "\n"                                                     \
            "Args:\n"                                                \
            "    qubit_addr: quantum gate operate qubits addr\n"     \
            "\n"                                                     \
            "Returns:\n"                                             \
            "    a " #gate_name " gate node\n"                       \
            "Raises:\n"                                              \
            "    run_fail: An error occurred in construct gate node\n", \
            py::return_value_policy::automatic);                     \
        m.def(#gate_name, \
            py::overload_cast<const std::vector<int> &>(&gate_name), \
            py::arg("qubit_addr_list"), \
            "Create a " #gate_name " gate\n"                         \
            "\n" \
            "Args:\n"                                                \
            "    qubit_list_addr: quantum gate  qubits list addr\n"  \
            "\n"                                                     \
            "Returns:\n"                                             \
            "    a " #gate_name " gate node\n"                       \
            "Raises:\n"                                              \
            "    run_fail: An error occurred in construct gate node\n", \
            py::return_value_policy::automatic);

    EXPORT_singleBitGate(H);
    EXPORT_singleBitGate(T);
    EXPORT_singleBitGate(S);
    EXPORT_singleBitGate(I);
    EXPORT_singleBitGate(X);
    EXPORT_singleBitGate(Y);
    EXPORT_singleBitGate(Z);
    EXPORT_singleBitGate(X1);
    EXPORT_singleBitGate(Y1);
    EXPORT_singleBitGate(Z1);

    m.def("BARRIER",
        py::overload_cast<Qubit *>(&BARRIER),
        py::arg("qubit"),
        "Create a BARRIER gate for a specified qubit.\n"
        "\n"
        "Args:\n"
        "    qubit: the qubit to which the BARRIER will be applied.\n"
        "\n"
        "Returns:\n"
        "    A BARRIER node representing the operation.\n",
        py::return_value_policy::automatic
    );

    m.def("BARRIER",
        py::overload_cast<int>(&BARRIER),
        py::arg("qubit_list"),
        "Create a BARRIER gate for a list of qubits.\n"
        "\n"
        "Args:\n"
        "    qubit_list: integer representing the qubits to which the BARRIER will be applied.\n"
        "\n"
        "Returns:\n"
        "    A BARRIER node representing the operation.\n",
        py::return_value_policy::automatic
    );

    m.def("BARRIER",
        py::overload_cast<QVec>(&BARRIER),
        py::arg("qubit_list"),
        "Create a BARRIER gate for a list of qubits.\n"
        "\n"
        "Args:\n"
        "    qubit_list: a list of qubits to which the BARRIER will be applied.\n"
        "\n"
        "Returns:\n"
        "    A BARRIER node representing the operation.\n",
        py::return_value_policy::automatic
    );

    m.def("BARRIER",
        py::overload_cast<std::vector<int>>(&BARRIER),
        py::arg("qubit_addr_list"),
        "Create a BARRIER gate for a list of qubit addresses.\n"
        "\n"
        "Args:\n"
        "    qubit_addr_list: a list of integers representing the addresses of the qubits.\n"
        "\n"
        "Returns:\n"
        "    A BARRIER node representing the operation.\n",
        py::return_value_policy::automatic
    );

    TempHelper_RX<double>::export_singleBitGate(m);
    TempHelper_RY<double>::export_singleBitGate(m);
    TempHelper_RZ<double>::export_singleBitGate(m);
    TempHelper_P<double>::export_singleBitGate(m);
    TempHelper_U1<double>::export_singleBitGate(m);
    TempHelper_U2<double, double>::export_singleBitGate(m);
    TempHelper_U3<double, double, double>::export_singleBitGate(m);

    TempHelper_CNOT<>::export_doubleBitGate(m);
    TempHelper_CZ<>::export_doubleBitGate(m);

    m.def("U4",
        py::overload_cast<QStat &, Qubit *>(&U4),
        py::arg("matrix"),
        py::arg("qubit"),
        "Create a U4 gate.\n"
        "\n"
        "Args:\n"
        "    matrix: the U4 gate matrix to be applied.\n"
        "\n"
        "    qubit: the target qubit for the U4 gate.\n"
        "\n"
        "Returns:\n"
        "    A U4 node representing the operation.\n",
        py::return_value_policy::automatic
    );

    m.def("U4",
        py::overload_cast<double, double, double, double, Qubit *>(&U4),
        py::arg("alpha_angle"),
        py::arg("beta_angle"),
        py::arg("gamma_angle"),
        py::arg("delta_angle"),
        py::arg("qubit"),
        "Create a U4 gate.\n"
        "\n"
        "Args:\n"
        "    alpha_angle: the alpha angle for the U4 gate.\n"
        "\n"
        "    beta_angle: the beta angle for the U4 gate.\n"
        "\n"
        "    gamma_angle: the gamma angle for the U4 gate.\n"
        "\n"
        "    delta_angle: the delta angle for the U4 gate.\n"
        "\n"
        "    qubit: the target qubit for the U4 gate.\n"
        "\n"
        "Returns:\n"
        "    A U4 node representing the operation.\n",
        py::return_value_policy::automatic
    );

    m.def("U4",
        py::overload_cast<Qubit *, QStat &>(&U4),
        py::arg("qubit"),
        py::arg("matrix"),
        "Create a U4 gate.\n"
        "\n"
        "Args:\n"
        "    qubit: the target qubit for the U4 gate.\n"
        "\n"
        "    matrix: the U4 gate matrix to be applied.\n"
        "\n"
        "Returns:\n"
        "    A U4 node representing the operation.\n",
        py::return_value_policy::automatic
    );


    m.def("U4",
        py::overload_cast<const QVec &, QStat &>(&U4),
        py::arg("qubit_list"),
        py::arg("matrix"),
        "Create a U4 gate.\n"
        "\n"
        "Args:\n"
        "    qubit_list: the list of target qubits for the U4 gate.\n"
        "\n"
        "    matrix: the U4 gate matrix to be applied.\n"
        "\n"
        "Returns:\n"
        "    A U4 node representing the operation.\n",
        py::return_value_policy::automatic
    );

    m.def("U4",
        py::overload_cast<int, QStat &>(&U4),
        py::arg("qubit_addr"),
        py::arg("matrix"),
        "Create a U4 gate.\n"
        "\n"
        "Args:\n"
        "    qubit_addr: the address of the target qubit for the U4 gate.\n"
        "\n"
        "    matrix: the U4 gate matrix to be applied.\n"
        "\n"
        "Returns:\n"
        "    A U4 node representing the operation.\n",
        py::return_value_policy::automatic
    );

    m.def("U4",
        py::overload_cast<const std::vector<int> &, QStat &>(&U4),
        py::arg("qubit_addr_list"),
        py::arg("matrix"),
        "Create a U4 gate.\n"
        "\n"
        "Args:\n"
        "    qubit_addr_list: the list of addresses for the target qubits of the U4 gate.\n"
        "\n"
        "    matrix: the U4 gate matrix to be applied.\n"
        "\n"
        "Returns:\n"
        "    A U4 node representing the operation.\n",
        py::return_value_policy::automatic
    );

    m.def("U4",
        py::overload_cast<Qubit *, double, double, double, double>(&U4),
        py::arg("qubit"),
        py::arg("alpha_anlge"),
        py::arg("beta_anlge"),
        py::arg("gamma_anlge"),
        py::arg("delta_anlge"),
        "Create a U4 gate.\n"
        "\n"
        "Args:\n"
        "    qubit: the target qubit for the U4 gate.\n"
        "\n"
        "    alpha_angle: the alpha angle for the U4 gate.\n"
        "\n"
        "    beta_angle: the beta angle for the U4 gate.\n"
        "\n"
        "    gamma_angle: the gamma angle for the U4 gate.\n"
        "\n"
        "    delta_angle: the delta angle for the U4 gate.\n"
        "\n"
        "Returns:\n"
        "    A U4 node representing the operation.\n",
        py::return_value_policy::automatic
    );

    m.def("U4",
        py::overload_cast<const QVec &, double, double, double, double>(&U4),
        py::arg("qubit_list"),
        py::arg("alpha_angle"),
        py::arg("beta_angle"),
        py::arg("gamma_angle"),
        py::arg("delta_angle"),
        "Create a U4 gate.\n\n"
        "\n"
        "Args:\n"
        "    qubit_list: the list of target qubits for the U4 gate.\n"
        "\n"
        "    alpha_angle: the alpha angle for the U4 gate.\n"
        "\n"
        "    beta_angle: the beta angle for the U4 gate.\n"
        "\n"
        "    gamma_angle: the gamma angle for the U4 gate.\n"
        "\n"
        "    delta_angle: the delta angle for the U4 gate.\n"
        "\n"
        "Returns:\n"
        "    A U4 node representing the operation.",
        py::return_value_policy::automatic
    );

    m.def("U4",
        py::overload_cast<int, double, double, double, double>(&U4),
        py::arg("qubit_addr"),
        py::arg("alpha_anlge"),
        py::arg("beta_anlge"),
        py::arg("gamma_anlge"),
        py::arg("delta_anlge"),
        "Create a U4 gate.\n"
        "\n"
        "Args:\n"
        "    qubit_addr: the address of the target qubit for the U4 gate.\n"
        "\n"
        "    alpha_angle: the alpha angle for the U4 gate.\n"
        "\n"
        "    beta_angle: the beta angle for the U4 gate.\n"
        "\n"
        "    gamma_angle: the gamma angle for the U4 gate.\n"
        "\n"
        "    delta_angle: the delta angle for the U4 gate.\n"
        "\n"
        "Returns:\n"
        "    A U4 node representing the operation.",
        py::return_value_policy::automatic
    );

    m.def("U4",
        py::overload_cast<const std::vector<int> &, double, double, double, double>(&U4),
        py::arg("qubit_addr_list"),
        py::arg("alpha_anlge"),
        py::arg("beta_anlge"),
        py::arg("gamma_anlge"),
        py::arg("delta_anlge"),
        "Create a U4 gate.\n"
        "\n"
        "Args:\n"
        "    qubit_addr_list: the list of addresses of target qubits for the U4 gate.\n"
        "\n"
        "    alpha_angle: the alpha angle for the U4 gate.\n"
        "\n"
        "    beta_angle: the beta angle for the U4 gate.\n"
        "\n"
        "    gamma_angle: the gamma angle for the U4 gate.\n"
        "\n"
        "    delta_angle: the delta angle for the U4 gate.\n"
        "\n"
        "Returns:\n"
        "    A U4 node representing the operation.",
        py::return_value_policy::automatic
    );

    m.def("CU",
        py::overload_cast<double, double, double, double, Qubit *, Qubit *>(&CU),
        py::arg("alpha_angle"),
        py::arg("beta_angle"),
        py::arg("gamma_angle"),
        py::arg("delta_angle"),
        py::arg("control_qubit"),
        py::arg("target_qubit"),
        "Create a CU gate.\n"
        "\n"
        "Args:\n"
        "    alpha_angle: the alpha angle for the CU gate.\n"
        "\n"
        "    beta_angle: the beta angle for the CU gate.\n"
        "\n"
        "    gamma_angle: the gamma angle for the CU gate.\n"
        "\n"
        "    delta_angle: the delta angle for the CU gate.\n"
        "\n"
        "    control_qubit: the qubit that controls the operation.\n"
        "\n"
        "    target_qubit: the qubit that is affected by the control qubit.\n"
        "\n"
        "Returns:\n"
        "    A CU node representing the operation.",
        py::return_value_policy::automatic
    );

    m.def("CU",
        py::overload_cast<double, double, double, double, const QVec &, const QVec &>(&CU),
        py::arg("alpha_angle"),
        py::arg("beta_angle"),
        py::arg("gamma_angle"),
        py::arg("delta_angle"),
        py::arg("control_qubit_list"),
        py::arg("target_qubi_list"),
        "Create a CU gate.\n"
        "\n"
        "Args:\n"
        "    alpha_angle (double): U4 gate alpha angle.\n"
        "\n"
        "    beta_angle (double): U4 gate beta angle.\n"
        "\n"
        "    gamma_angle (double): U4 gate gamma angle.\n"
        "\n"
        "    delta_angle (double): U4 gate delta angle.\n"
        "\n"
        "    control_qubit_list (const QVec &): List of control qubits.\n"
        "\n"
        "    target_qubit_list (const QVec &): List of target qubits.\n"
        "\n"
        "Returns:\n"
        "    A CU node representing the operation.\n",
        py::return_value_policy::automatic
    );

    m.def("CU",
        py::overload_cast<QStat &, Qubit *, Qubit *>(&CU),
        py::arg("matrix"),
        py::arg("control_qubit"),
        py::arg("target_qubit"),
        "Create a CU gate.\n"
        "\n"
        "Args:\n"
        "    matrix (QStat &): The CU gate matrix.\n"
        "\n"
        "    control_qubit (Qubit *): The control qubit.\n"
        "\n"
        "    target_qubit (Qubit *): The target qubit.\n"
        "\n"
        "Returns:\n"
        "    A CU node representing the operation.\n",
        py::return_value_policy::automatic
    );

    m.def("CU",
        py::overload_cast<QStat &, const QVec &, const QVec &>(&CU),
        py::arg("matrix"),
        py::arg("control_qubit_list"),
        py::arg("target_qubit_list"),
        "Create a CU gate.\n"
        "\n"
        "Args:\n"
        "    matrix (QStat &): The CU gate matrix.\n"
        "\n"
        "    control_qubit_list (const QVec &): List of control qubits.\n"
        "\n"
        "    target_qubit_list (const QVec &): List of target qubits.\n"
        "\n"
        "Returns:\n"
        "    A CU node representing the operation.\n",
        py::return_value_policy::automatic
    );


    TempHelper_SWAP<>::export_doubleBitGate(m);
    TempHelper_iSWAP<>::export_doubleBitGate(m);
    TempHelper_iSWAP_2<double>::export_doubleBitGate(m);
    TempHelper_SqiSWAP<>::export_doubleBitGate(m);
    TempHelper_MS<>::export_doubleBitGate(m);

    TempHelper_CP<double>::export_doubleBitGate(m);
    TempHelper_CR<double>::export_doubleBitGate(m);

    m.def("CU",
        py::overload_cast<Qubit *, Qubit *, double, double, double, double>(&CU),
        py::arg("control_qubit"),
        py::arg("target_qubit"),
        py::arg("alpha_angle"),
        py::arg("beta_angle"),
        py::arg("gamma_angle"),
        py::arg("delta_angle"),
        "Create a CU gate.\n"
        "\n"
        "Args:\n"
        "    control_qubit (Qubit *): The control qubit.\n"
        "\n"
        "    target_qubit (Qubit *): The target qubit.\n"
        "\n"
        "    alpha_angle (double): U4 gate alpha angle.\n"
        "\n"
        "    beta_angle (double): U4 gate beta angle.\n"
        "\n"
        "    gamma_angle (double): U4 gate gamma angle.\n"
        "\n"
        "    delta_angle (double): U4 gate delta angle.\n"
        "\n"
        "Returns:\n"
        "    A CU node representing the operation.\n",
        py::return_value_policy::automatic
    );

    m.def("CU",
        py::overload_cast<const QVec &, const QVec &, double, double, double, double>(&CU),
        py::arg("control_qubit_list"),
        py::arg("target_qubit_list"),
        py::arg("alpha_angle"),
        py::arg("beta_angle"),
        py::arg("gamma_angle"),
        py::arg("delta_angle"),
        "Create a CU gate.\n"
        "\n"
        "Args:\n"
        "    control_qubit_list (QVec): Control qubit list.\n"
        "\n"
        "    target_qubit_list (QVec): Target qubit list.\n"
        "\n"
        "    alpha_angle (double): U4 gate alpha angle.\n"
        "\n"
        "    beta_angle (double): U4 gate beta angle.\n"
        "\n"
        "    gamma_angle (double): U4 gate gamma angle.\n"
        "\n"
        "    delta_angle (double): U4 gate delta angle.\n"
        "\n"
        "Returns:\n"
        "    CU node representing the operation.\n",
        py::return_value_policy::automatic
    );

    m.def("CU",
        py::overload_cast<int, int, double, double, double, double>(&CU),
        py::arg("control_qubit_addr"),
        py::arg("target_qubit_addr"),
        py::arg("alpha_angle"),
        py::arg("beta_angle"),
        py::arg("gamma_angle"),
        py::arg("delta_angle"),
        "Create a CU gate.\n"
        "\n"
        "Args:\n"
        "    control_qubit_addr (int): Address of the control qubit.\n"
        "\n"
        "    target_qubit_addr (int): Address of the target qubit.\n"
        "\n"
        "    alpha_angle (double): U4 gate alpha angle.\n"
        "\n"
        "    beta_angle (double): U4 gate beta angle.\n"
        "\n"
        "    gamma_angle (double): U4 gate gamma angle.\n"
        "\n"
        "    delta_angle (double): U4 gate delta angle.\n"
        "\n"
        "Returns:\n"
        "    A CU node representing the operation.\n",
        py::return_value_policy::automatic
    );

    m.def("CU",
        py::overload_cast<const std::vector<int> &, const std::vector<int> &, double, double, double, double>(&CU),
        py::arg("control_qubit_addr_list"),
        py::arg("target_qubit_addr_list"),
        py::arg("alpha_angle"),
        py::arg("beta_angle"),
        py::arg("gamma_angle"),
        py::arg("delta_angle"),
        "Create a CU gate.\n"
        "\n"
        "Args:\n"
        "    control_qubit_addr_list (std::vector<int>): List of control qubit addresses.\n"
        "\n"
        "    target_qubit_addr_list (std::vector<int>): List of target qubit addresses.\n"
        "\n"
        "    alpha_angle (double): U4 gate alpha angle.\n"
        "\n"
        "    beta_angle (double): U4 gate beta angle.\n"
        "\n"
        "    gamma_angle (double): U4 gate gamma angle.\n"
        "\n"
        "    delta_angle (double): U4 gate delta angle.\n"
        "\n"
        "Returns:\n"
        "    A CU node representing the operation.\n",
        py::return_value_policy::automatic
    );

    m.def("CU",
        py::overload_cast<Qubit *, Qubit *, QStat &>(&CU),
        py::arg("control_qubit"),
        py::arg("target_qubit"),
        py::arg("matrix"),
        "Create a CU gate.\n"
        "\n"
        "Args:\n"
        "    control_qubit (Qubit *): The control qubit.\n"
        "\n"
        "    target_qubit (Qubit *): The target qubit.\n"
        "\n"
        "    matrix (QStat &): The CU gate matrix.\n"
        "\n"
        "Returns:\n"
        "    A CU node representing the operation.\n",
        py::return_value_policy::automatic
    );

    m.def("CU",
        py::overload_cast<const QVec &, const QVec &, QStat &>(&CU),
        py::arg("control_qubit_list"),
        py::arg("target_qubit_list"),
        py::arg("matrix"),
        "Create a CU gate.\n"
        "\n"
        "Args:\n"
        "    control_qubit_list (const QVec &): List of control qubits.\n"
        "\n"
        "    target_qubit_list (const QVec &): List of target qubits.\n"
        "\n"
        "    matrix (QStat &): The CU gate matrix.\n"
        "\n"
        "Returns:\n"
        "    A CU node representing the operation.\n",
        py::return_value_policy::automatic
    );

    m.def("CU",
        py::overload_cast<int, int, QStat &>(&CU),
        py::arg("control_qubit_addr"),
        py::arg("target_qubit_addr"),
        py::arg("matrix"),
        "Create a CU gate.\n"
        "\n"
        "Args:\n"
        "    control_qubit_addr (int): Address of the control qubit.\n"
        "\n"
        "    target_qubit_addr (int): Address of the target qubit.\n"
        "\n"
        "    matrix (QStat &): The CU gate matrix.\n"
        "\n"
        "Returns:\n"
        "    A CU node representing the operation.\n",
        py::return_value_policy::automatic
    );

    m.def("CU",
        py::overload_cast<const std::vector<int> &, const std::vector<int> &, QStat &>(&CU),
        py::arg("control_qubit_addr_list"),
        py::arg("target_qubit_addr_list"),
        py::arg("matrix"),
        "Create a CU gate.\n"
        "\n"
        "Args:\n"
        "    control_qubit_addr_list (const std::vector<int> &): List of control qubit addresses.\n"
        "\n"
        "    target_qubit_addr_list (const std::vector<int> &): List of target qubit addresses.\n"
        "\n"
        "    matrix (QStat &): The CU gate matrix.\n"
        "\n"
        "Returns:\n"
        "    A CU node representing the operation.\n",
        py::return_value_policy::automatic
    );

#define EXPORT_MULTI_ROTATION_GATE_NTES(gate_name)                 \
    m.def(#gate_name,                                              \
        py::overload_cast<Qubit *, Qubit *, double>(&gate_name),   \
        py::arg("control_qubit"),                                  \
        py::arg("target_qubit"),                                   \
        py::arg("alpha_angle"),                                    \
        "Create a " #gate_name " gate\n"                           \
        "\n"                                                       \
        "Args:\n"                                                  \
        "    Qubit : control qubit\n"                              \
        "    Qubit : target qubit\n"                               \
        "    double: gate rotation angle theta\n"                  \
        "\n"                                                       \
        "Returns:\n"                                               \
        "    a " #gate_name " gate node\n"                         \
        "Raises:\n"                                                \
        "    run_fail: An error occurred in construct gate node\n",\
        py::return_value_policy::automatic);                       \
    m.def(#gate_name,                                              \
        py::overload_cast<const QVec &, const QVec &, double>(&gate_name), \
        py::arg("control_qubit_list"),                             \
        py::arg("target_qubit_list"),                              \
        py::arg("alpha_angle"),                                    \
        "Create a " #gate_name " gate\n"                           \
        "\n"                                                       \
        "Args:\n"                                                  \
        "    control_qubit_list : control qubit list\n"            \
        "    target_qubit_list : target qubit list\n"              \
        "    double: gate rotation angle theta\n"                  \
        "\n"                                                       \
        "Returns:\n"                                               \
        "    a " #gate_name " gate node\n"                         \
        "Raises:\n"                                                \
        "    run_fail: An error occurred in construct gate node\n",\
        py::return_value_policy::automatic);                       \
    m.def(#gate_name,                                              \
        py::overload_cast<int, int, double>(&gate_name),           \
        py::arg("control_qubit_addr"),                             \
        py::arg("target_qubit_addr"),                              \
        py::arg("alpha_angle"),                                    \
        "Create a " #gate_name " gate\n"                           \
        "\n"                                                       \
        "Args:\n"                                                  \
        "    qubit addr : control qubit addr \n"                   \
        "    qubit addr : target qubit addr \n"                    \
        "    double: gate rotation angle theta\n"                  \
        "\n"                                                       \
        "Returns:\n"                                               \
        "    a " #gate_name " gate node\n"                         \
        "Raises:\n"                                                \
        "    run_fail: An error occurred in construct gate node\n",\
        py::return_value_policy::automatic);                       \
    m.def(#gate_name,                                              \
        py::overload_cast<const std::vector<int> &, const std::vector<int> &, double>(&gate_name),           \
        py::arg("control_qubit_addr_list"),                        \
        py::arg("target_qubit_addr_list"),                         \
        py::arg("alpha_angle"),                                    \
        "Create a " #gate_name " gate\n"                           \
        "\n"                                                       \
        "Args:\n"                                                  \
        "    qubit addr list : control qubit addr list\n"          \
        "    qubit addr list : target qubit addr list\n"           \
        "    double: gate rotation angle theta\n"                  \
        "\n"                                                       \
        "Returns:\n"                                               \
        "    a " #gate_name " gate node\n"                         \
        "Raises:\n"                                                \
        "    run_fail: An error occurred in construct gate node\n",\
        py::return_value_policy::automatic);                       \

    EXPORT_MULTI_ROTATION_GATE_NTES(RXX);
    EXPORT_MULTI_ROTATION_GATE_NTES(RYY);
    EXPORT_MULTI_ROTATION_GATE_NTES(RZX);
    EXPORT_MULTI_ROTATION_GATE_NTES(RZZ);

    m.def("Toffoli",
        py::overload_cast<Qubit *, Qubit *, Qubit *>(&Toffoli),
        py::arg("control_qubit_first"),
        py::arg("control_qubit_second"),
        py::arg("target_qubit"),
        "Create a Toffoli gate.\n"
        "\n"
        "Args:\n"
        "    control_qubit_first (Qubit *): First control qubit.\n"
        "\n"
        "    control_qubit_second (Qubit *): Second control qubit.\n"
        "\n"
        "    target_qubit (Qubit *): Target qubit.\n"
        "\n"
        "Returns:\n"
        "    A Toffoli node representing the operation.\n",
        py::return_value_policy::automatic
    );

    m.def("Toffoli",
        py::overload_cast<int, int, int>(&Toffoli),
        py::arg("control_qubit_addr_first"),
        py::arg("control_qubit_addr_second"),
        py::arg("target_qubit_addr"),
        "Create a Toffoli gate.\n"
        "\n"
        "Args:\n"
        "    control_qubit_addr_first (int): Address of the first control qubit.\n"
        "\n"
        "    control_qubit_addr_second (int): Address of the second control qubit.\n"
        "\n"
        "    target_qubit_addr (int): Address of the target qubit.\n"
        "\n"
        "Returns:\n"
        "    A Toffoli node representing the operation.\n",
        py::return_value_policy::automatic
    );

    TempHelper_QDouble<QStat &>::export_doubleBitGate(m);

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\QuantumCircuit\QReset.h */
    m.def("Reset",
        py::overload_cast<Qubit *>(&Reset),
        py::arg("qubit"),
        "Create a Reset node.\n"
        "\n"
        "Args:\n"
        "    qubit (Qubit *): The qubit to be reset.\n"
        "\n"
        "Returns:\n"
        "    A Reset node representing the operation.\n",
        py::return_value_policy::automatic
    );

    m.def("Reset",
        py::overload_cast<int>(&Reset),
        py::arg("qubit_addr"),
        "Create a Reset node.\n"
        "\n"
        "Args:\n"
        "    qubit_addr (int): Address of the qubit to be reset.\n"
        "\n"
        "Returns:\n"
        "    A Reset node representing the operation.\n",
        py::return_value_policy::automatic
    );

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\Tools\QStatMatrix.h */
    m.def(
        "print_matrix",
        [](QStat &mat, const int precision)
        {
            auto mat_str = matrix_to_string(mat, precision);
            std::cout << mat_str << endl;
            return mat_str;
        },
        py::arg("matrix"),
            py::arg("precision") = 8,
            "Print matrix elements.\n"
            "\n"
            "Args:\n"
            "    matrix (QStat): The matrix to print.\n"
            "\n"
            "    precision (int, optional): Double value to string cutoff precision (default is 8).\n"
            "\n"
            "Returns:\n"
            "    A string representation of the matrix.\n",
            py::return_value_policy::automatic
            );

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\QProgInfo\QCircuitInfo.h */
    m.def("is_match_topology",
        &isMatchTopology,
        py::arg("gate"),
        py::arg("topo"),
        "Judge if the QGate matches the target topologic structure of the quantum circuit.\n"
        "\n"
        "Args:\n"
        "    gate (QGate): The quantum gate to evaluate.\n"
        "\n"
        "    topo: The target topologic structure of the quantum circuit.\n"
        "\n"
        "Returns:\n"
        "    bool: True if it matches, otherwise false.\n",
        py::return_value_policy::automatic
    );

    m.def(
        "get_adjacent_qgate_type",
        [](QProg &prog, NodeIter &node_iter)
        {
            std::vector<NodeInfo> adjacent_nodes;
            getAdjacentQGateType(prog, node_iter, adjacent_nodes);
            return adjacent_nodes;
        },
        py::arg("qprog"),
            py::arg("node_iter"),
            "Get the adjacent quantum gates' (the front one and the back one) type info from QProg.\n"
            "\n"
            "Args:\n"
            "    qprog: Target quantum program.\n"
            "\n"
            "    node_iter: Gate node iterator in qprog.\n"
            "\n"
            "Returns:\n"
            "    The front and back node info of node_iter in qprog.\n",
            py::return_value_policy::automatic
            );

    m.def("is_swappable",
        &isSwappable,
        py::arg("prog"),
        py::arg("nodeitr_1"),
        py::arg("nodeitr_2"),
        "Judge whether the specified two NodeIters in the quantum program can be exchanged.\n"
        "\n"
        "Args:\n"
        "    prog: Target quantum program.\n"
        "\n"
        "    nodeitr_1: Node iterator 1 in the quantum program.\n"
        "\n"
        "    nodeitr_2: Node iterator 2 in the quantum program.\n"
        "\n"
        "Returns:\n"
        "    bool: True if the two NodeIters can be exchanged, otherwise false.\n",
        py::return_value_policy::automatic
    );

    m.def("is_supported_qgate_type",
        &isSupportedGateType,
        py::arg("nodeitr"),
        "Judge if the target node is a QGate type.\n"
        "\n"
        "Args:\n"
        "    nodeitr: Node iterator in the quantum program.\n"
        "\n"
        "Returns:\n"
        "    bool: True if the target node is a QGate type, otherwise false.\n",
        py::return_value_policy::automatic);

    m.def("get_matrix",
        &getCircuitMatrix,
        py::arg("qprog"),
        py::arg("positive_seq") = false,
        py::arg_v("nodeitr_start", NodeIter(), "NodeIter()"),
        py::arg_v("nodeitr_end", NodeIter(), "NodeIter()"),
        "Get the target matrix between the input two NodeIters.\n"
        "\n"
        "Args:\n"
        "    qprog: Quantum program.\n"
        "\n"
        "    positive_seq: Qubit order of output matrix; true for positive sequence (q0q1q2),\n"
        "  false for inverted order (q2q1q0), default is false.\n"
        "\n"
        "    nodeitr_start: The start NodeIter.\n"
        "\n"
        "    nodeitr_end: The end NodeIter.\n"
        "\n"
        "Returns:\n"
        "    The target matrix including all the QGate's matrices (multiplied).\n",
        py::return_value_policy::automatic);

    m.def("get_unitary",
        &get_unitary,
        py::arg("qprog"),
        py::arg("positive_seq") = false,
        py::arg_v("nodeitr_start", NodeIter(), "NodeIter()"),
        py::arg_v("nodeitr_end", NodeIter(), "NodeIter()"),
        "Get the target unitary matrix between the input two NodeIters.\n"
        "\n"
        "Args:\n"
        "    qprog: Quantum program.\n"
        "\n"
        "    positive_seq: Qubit order of output matrix; true for positive sequence (q0q1q2),\n"
        "  false for inverted order (q2q1q0), default is false.\n"
        "\n"
        "    nodeitr_start: The start NodeIter.\n"
        "\n"
        "    nodeitr_end: The end NodeIter.\n"
        "\n"
        "Returns:\n"
        "    The target unitary matrix including all the QGate's matrices (multiplied).\n",
        py::return_value_policy::automatic);

    m.def(
        "get_all_used_qubits",
        [](QProg prog)
        {
            QVec vec_qubits_in_use;
            get_all_used_qubits(prog, vec_qubits_in_use);

            return static_cast<std::vector<Qubit*>>(vec_qubits_in_use);
        },
        py::arg("qprog"),
            "Get all the quantum bits used in the input program.\n"
            "\n"
            "Args:\n"
            "    qprog: A quantum program.\n"
            "\n"
            "Returns:\n"
            "    result: A list of all used qubits.\n",
            py::return_value_policy::automatic);

    m.def(
        "get_all_used_qubits_to_int",
        [](QProg prog)
        {
            std::vector<int> vec_qubits_in_use;
            get_all_used_qubits(prog, vec_qubits_in_use);
            return vec_qubits_in_use;
        },
        py::arg("qprog"),
            "Get the addresses of all used quantum bits in the input program.\n"
            "\n"
            "Args:\n"
            "    qprog: A quantum program.\n"
            "\n"
            "Returns:\n"
            "    result: A list of addresses of all used qubits.\n",
            py::return_value_policy::automatic);

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\QProgInfo\MetadataValidity.h */
    m.def(
        "validate_single_qgate_type",
        [](std::vector<string> single_gates)
        {
            py::list ret_date;
            std::vector<string> valid_gates;
            auto type = validateSingleQGateType(single_gates, valid_gates);
            ret_date.append(static_cast<SingleGateTransferType>(type));
            ret_date.append(valid_gates);
            return ret_date;
        },
        py::arg("gate_str_list"),
            "Get valid QGates and valid single bit QGate type.\n"
            "\n"
            "Args:\n"
            "    single_gates: A list of single gate strings.\n"
            "\n"
            "Returns:\n"
            "    result: A list containing the validated gate type and valid single gates.\n",
            py::return_value_policy::automatic);

    m.def(
        "validate_double_qgate_type",
        [](std::vector<string> double_gates)
        {
            py::list ret_data;
            std::vector<string> valid_gates;
            auto type = validateDoubleQGateType(double_gates, valid_gates);
            ret_data.append(static_cast<DoubleGateTransferType>(type));
            ret_data.append(valid_gates);
            return ret_data;
        },
        py::arg("gate_str_list"),
            "Get valid QGates and valid double bit QGate type.\n"
            "\n"
            "Args:\n"
            "    double_gates: A list of double gate strings.\n"
            "\n"
            "Returns:\n"
            "    result: A list containing the validated gate type and valid double gates.\n",
            py::return_value_policy::automatic_reference);

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\QProgInfo\QGateCompare.h */
    m.def("get_unsupport_qgate_num",
        &getUnsupportQGateNum<QProg>,
        py::arg("qprog"),
        py::arg("support_gates"),
        "Count the number of unsupported gates in a quantum program.\n"
        "\n"
        "Args:\n"
        "    qprog: The quantum program to analyze.\n"
        "\n"
        "    support_gates: A list of supported gates.\n"
        "\n"
        "Returns:\n"
        "    int: The number of unsupported gates in the quantum program.\n",
        py::return_value_policy::automatic);

    m.def("get_qgate_num",
        &getQGateNum<QProg>,
        py::arg("qprog"),
        "Count the number of quantum gates in a quantum program.\n"
        "\n"
        "Args:\n"
        "    qprog: The quantum program to analyze.\n"
        "\n"
        "Returns:\n"
        "    int: The number of quantum gates in the quantum program.\n",
        py::return_value_policy::automatic);

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\Tools\QProgFlattening.h */
    m.def(
        "flatten",
        [](QProg &prog)
        {
            flatten(prog);
        },
        py::arg("qprog"),
            "Flatten a quantum program in place.\n"
            "\n"
            "Args:\n"
            "    qprog: The quantum program to be flattened.\n"
            "\n"
            "Returns:\n"
            "    None: The function modifies the quantum program directly.\n",
            py::return_value_policy::automatic);

    m.def("flatten",
        py::overload_cast<QCircuit &>(&flatten),
        py::arg("qcircuit"),
        "Flatten a quantum circuit in place.\n"
        "\n"
        "Args:\n"
        "    qcircuit: The quantum circuit to be flattened.\n"
        "\n"
        "Returns:\n"
        "    None: The function modifies the circuit directly.\n",
        py::return_value_policy::automatic);

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\Compiler\QProgDataParse.h */
    m.def(
        "convert_binary_data_to_qprog",
        [](QuantumMachine *qm, std::vector<uint8_t> data)
        {
            QVec qubits;
            std::vector<ClassicalCondition> cbits;
            QProg prog;
            convert_binary_data_to_qprog(qm, data, qubits, cbits, prog);
            return prog;
        },
        py::arg("machine"),
            py::arg("data"),
            "Parse binary data into a quantum program.\n"
            "\n"
            "Args:\n"
            "    machine: The quantum machine used for execution.\n"
            "\n"
            "    data: The binary data representing the quantum program.\n"
            "\n"
            "Returns:\n"
            "    QProg: The generated quantum program.\n",
            py::return_value_policy::automatic_reference);

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\Compiler\QProgToQASM.h */
    m.def("convert_qprog_to_qasm",
        &convert_qprog_to_qasm,
        py::arg("qprog"),
        py::arg("machine"),
        "Convert a quantum program to a QASM instruction string.\n"
        "\n"
        "Args:\n"
        "    qprog: The quantum program to be converted.\n"
        "\n"
        "    machine: The quantum machine used for execution.\n"
        "\n"
        "Returns:\n"
        "    str: A QASM string representing the quantum program.\n",
        py::return_value_policy::automatic_reference);

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\QProgTransform\QProgToQGate.h */
    m.def("cast_qprog_qgate",
        &cast_qprog_qgate,
        py::arg("qprog"),
        "Cast a quantum program into a quantum gate.\n"
        "\n"
        "Args:\n"
        "    qprog: The quantum program to be cast.\n"
        "\n"
        "Returns:\n"
        "    None: This function does not return a value.\n",
        py::return_value_policy::automatic_reference);

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\QProgTransform\QProgToQMeasure.h */
    m.def("cast_qprog_qmeasure",
        &cast_qprog_qmeasure,
        py::arg("qprog"),
        "Cast a quantum program into a quantum measurement.\n"
        "\n"
        "Args:\n"
        "    qprog: The quantum program to be cast.\n"
        "\n"
        "Returns:\n"
        "    None: This function does not return a value.\n",
        py::return_value_policy::automatic_reference);

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\QProgTransform\QProgToQCircuit.h */
    m.def(
        "cast_qprog_qcircuit",
        [](QProg prog)
        {
            QCircuit cir;
            cast_qprog_qcircuit(prog, cir);
            return cir;
        },
        py::arg("qprog"),
            "Cast a quantum program into a quantum circuit.\n"
            "\n"
            "Args:\n"
            "    qprog: The quantum program to be cast.\n"
            "\n"
            "Returns:\n"
            "    QCircuit: The resulting quantum circuit.\n",
            py::return_value_policy::automatic_reference);

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\QProgTransform\TopologyMatch.h */
    m.def(
        "topology_match",
        [](QProg prog, QVec qv, QuantumMachine *qvm, const std::string conf)
        {
            py::list ret_data;
            QProg out_prog = topology_match(prog, qv, qvm, conf);
            py::list qubit_list;
            for (auto q : qv)
                qubit_list.append(q);


            ret_data.append(out_prog);
            ret_data.append(qubit_list);
            return ret_data;
        },
        py::arg("qprog"),
            py::arg("qubit_list"),
            py::arg("machine"),
            py::arg("confing_file") = CONFIG_PATH,
            "Judge whether a quantum program matches the topology of the physical qubits.\n"
            "\n"
            "Args:\n"
            "    qprog: The quantum program to be evaluated.\n"
            "\n"
            "    qubit_list: The list of qubits in the quantum program.\n"
            "\n"
            "    machine: The quantum machine used for execution.\n"
            "\n"
            "    confing_file: The configuration file path for matching (default: QPandaConfig.json).\n"
            "\n"
            "Returns:\n"
            "    list: Contains the resulting quantum program and the qubit list.\n",
            py::return_value_policy::automatic_reference);

    m.def("add",
        [](ClassicalCondition a, ClassicalCondition b)
        {
            return a + b;
        },
            "Add two ClassicalCondition objects.\n"
            "\n"
            "Args:\n"
            "    a: The first ClassicalCondition.\n"
            "\n"
            "    b: The second ClassicalCondition.\n"
            "\n"
            "Returns:\n"
            "    ClassicalCondition: The sum of the two conditions.\n"
            );

    m.def("add",
        [](ClassicalCondition a, cbit_size_t b)
        {
            return a + b;
        },
        "Add a ClassicalCondition and a bit size.\n"
            "\n"
            "Args:\n"
            "    a: The ClassicalCondition to which the bit size will be added.\n"
            "\n"
            "    b: The bit size to be added to the ClassicalCondition.\n"
            "\n"
            "Returns:\n"
            "    ClassicalCondition: The resulting ClassicalCondition after addition.\n"
            );

    m.def("add",
        [](cbit_size_t a, ClassicalCondition b)
        {
            return a + b;
        },
        "Add a bit size and a ClassicalCondition.\n"
            "\n"
            "Args:\n"
            "    a: The bit size to be added.\n"
            "\n"
            "    b: The ClassicalCondition to which the bit size will be added.\n"
            "\n"
            "Returns:\n"
            "    ClassicalCondition: The resulting ClassicalCondition after addition.\n"
            );

    m.def("sub",
        [](ClassicalCondition a, ClassicalCondition b)
        {
			  return a - b;
        },
        "Subtract one ClassicalCondition from another.\n"
            "\n"
            "Args:\n"
            "    a: The ClassicalCondition to subtract from.\n"
            "\n"
            "    b: The ClassicalCondition to subtract.\n"
            "\n"
            "Returns:\n"
            "    ClassicalCondition: The result of the subtraction.\n"
            );

    m.def("sub",
        [](ClassicalCondition a, cbit_size_t b)
        {
			  return a - b;
        },
        "Subtract a bit size from a ClassicalCondition.\n"
            "\n"
            "Args:\n"
            "    a: The ClassicalCondition from which the bit size will be subtracted.\n"
            "\n"
            "    b: The bit size to subtract from the ClassicalCondition.\n"
            "\n"
            "Returns:\n"
            "    ClassicalCondition: The resulting ClassicalCondition after subtraction.\n"
            );

    m.def("sub",
        [](cbit_size_t a, ClassicalCondition b)
        {
			  return a - b;
        },
        "Subtract a ClassicalCondition from a bit size.\n"
            "\n"
            "Args:\n"
            "    a: The bit size to subtract from.\n"
            "\n"
            "    b: The ClassicalCondition to be subtracted.\n"
            "\n"
            "Returns:\n"
            "    ClassicalCondition: The resulting ClassicalCondition after subtraction.\n"
            );

    m.def("mul",
        [](ClassicalCondition a, ClassicalCondition b)
        {
            return a * b;
        },
        "Multiply two ClassicalConditions.\n"
            "\n"
            "Args:\n"
            "    a: The first ClassicalCondition.\n"
            "\n"
            "    b: The second ClassicalCondition to multiply with.\n"
            "\n"
            "Returns:\n"
            "    ClassicalCondition: The result of the multiplication.\n"
            );

    m.def("mul",
        [](ClassicalCondition a, cbit_size_t b)
        {
            return a * b;
        },
        "Multiply a ClassicalCondition by a bit size.\n"
            "\n"
            "Args:\n"
            "    a: The ClassicalCondition to be multiplied.\n"
            "\n"
            "    b: The bit size to multiply with the ClassicalCondition.\n"
            "\n"
            "Returns:\n"
            "    ClassicalCondition: The resulting ClassicalCondition after multiplication.\n"
            );

    m.def("mul",
        [](cbit_size_t a, ClassicalCondition b)
        {
            return a * b;
        },
        "Multiply a bit size by a ClassicalCondition.\n"
            "\n"
            "Args:\n"
            "    a: The bit size to be multiplied.\n"
            "\n"
            "    b: The ClassicalCondition to multiply with the bit size.\n"
            "\n"
            "Returns:\n"
            "    ClassicalCondition: The resulting ClassicalCondition after multiplication.\n"
            );

    m.def("div",
        [](ClassicalCondition a, ClassicalCondition b)
        {
            return a / b;
        },
        "Divide one ClassicalCondition by another.\n"
            "\n"
            "Args:\n"
            "    a: The numerator ClassicalCondition.\n"
            "\n"
            "    b: The denominator ClassicalCondition.\n"
            "\n"
            "Returns:\n"
            "    ClassicalCondition: The result of the division.\n"
            );

    m.def("div",
        [](ClassicalCondition a, cbit_size_t b)
        {
            return a / b;
        },
        "Divide a ClassicalCondition by a bit size.\n"
            "\n"
            "Args:\n"
            "    a: The ClassicalCondition (numerator).\n"
            "\n"
            "    b: The bit size (denominator).\n"
            "\n"
            "Returns:\n"
            "    ClassicalCondition: The result of the division.\n"
            );

    m.def("div",
        [](cbit_size_t a, ClassicalCondition b)
        {
            return a / b;
        },
        "Divide a bit size by a ClassicalCondition.\n"
            "\n"
            "Args:\n"
            "    a: The bit size (numerator).\n"
            "\n"
            "    b: The ClassicalCondition (denominator).\n"
            "\n"
            "Returns:\n"
            "    ClassicalCondition: The result of the division.\n"
            );

    m.def("equal",
        [](ClassicalCondition a, ClassicalCondition b)
        {
            return a == b;
        },
        "Check if two ClassicalConditions are equal.\n"
            "\n"
            "Args:\n"
            "    a: The first ClassicalCondition.\n"
            "\n"
            "    b: The second ClassicalCondition.\n"
            "\n"
            "Returns:\n"
            "    bool: True if both ClassicalConditions are equal, otherwise False.\n"
            );

    m.def("equal",
        [](ClassicalCondition a, cbit_size_t b)
        {
            return a == b;
        },
        "Check if a ClassicalCondition is equal to a bit size.\n"
            "\n"
            "Args:\n"
            "    a: The ClassicalCondition to compare.\n"
            "\n"
            "    b: The bit size to compare against.\n"
            "\n"
            "Returns:\n"
            "    bool: True if the ClassicalCondition is equal to the bit size, otherwise False.\n"
            );

    m.def("equal",
        [](cbit_size_t a, ClassicalCondition b)
        {
            return a == b;
        },
        "Check if a bit size is equal to a ClassicalCondition.\n"
            "\n"
            "Args:\n"
            "    a: The bit size to compare.\n"
            "\n"
            "    b: The ClassicalCondition to compare against.\n"
            "\n"
            "Returns:\n"
            "    bool: True if the bit size is equal to the ClassicalCondition, otherwise False.\n"
            );

    m.def("assign",
        [](ClassicalCondition &a, ClassicalCondition b)
        {
            return a = b;
        },
        "Assign the value of one ClassicalCondition to another.\n"
            "\n"
            "Args:\n"
            "    a: The ClassicalCondition to be assigned to (passed by reference).\n"
            "\n"
            "    b: The ClassicalCondition to assign from.\n"
            "\n"
            "Returns:\n"
            "    ClassicalCondition: The updated value of the first ClassicalCondition.\n"
            );


    m.def("assign",
        [](ClassicalCondition &a, cbit_size_t b)
        {
            return a = b;
        },
        "Assign a bit size value to a ClassicalCondition.\n"
            "\n"
            "Args:\n"
            "    a: The ClassicalCondition to be updated (passed by reference).\n"
            "\n"
            "    b: The bit size value to assign.\n"
            "\n"
            "Returns:\n"
            "    ClassicalCondition: The updated ClassicalCondition after assignment.\n"
            );
    //---------------------------------------------------------------------------------------------------------------------
    /* include\Components\MaxCutProblemGenerator\MaxCutProblemGenerator.h */
    m.def("vector_dot",
        &vector_dot,
        py::arg("x"),
        py::arg("y"),
        "Compute the inner product of two vectors.\n"
        "\n"
        "Args:\n"
        "    x: A list representing the first vector.\n"
        "\n"
        "    y: A list representing the second vector.\n"
        "\n"
        "Returns:\n"
        "    dot result: The dot product of vectors x and y.\n"
    );

    m.def("all_cut_of_graph",
        &all_cut_of_graph,
        py::arg("adjacent_matrix"),
        py::arg("all_cut_list"),
        py::arg("target_value_list"),
        "Generate a graph representation for the max cut problem.\n"
        "\n"
        "Args:\n"
        "    adjacent_matrix: The adjacency matrix for the quantum program.\n"
        "\n"
        "    all_cut_list: A list of all cut graphs in the quantum program.\n"
        "\n"
        "    target_value_list: A list of target cut values.\n"
        "\n"
        "Returns:\n"
        "    max value: The maximum value found from the cuts.\n"
    );

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\Tools\ProcessOnTraversing.h */
    m.def("circuit_layer",
        [](QProg prg)
        {
            py::list ret_data;
            auto layer_info = prog_layer(prg);
            std::vector<std::vector<NodeInfo>> tmp_layer(layer_info.size());
            size_t layer_index = 0;
            for (auto &cur_layer : layer_info)
            {
                for (auto &node_item : cur_layer)
                {
                    const pOptimizerNodeInfo &n = node_item.first;
                    // single gate first
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
        },
        py::arg("qprog"),
            "Quantum circuit layering.\n"
            "\n"
            "Args:\n"
            "    QProg: Quantum program.\n"
            "\n"
            "Returns:\n"
            "    A list containing layer information and qubits/cbits in use.\n",
            py::return_value_policy::automatic);

	//---------------------------------------------------------------------------------------------------------------------
	/* include\Core\Utilities\QProgInfo\Visualization\QVisualization.h */
    m.def(
        "draw_qprog_text",
        [](QProg prg, uint32_t auto_wrap_len, const std::string& output_file, const NodeIter itr_start, const NodeIter itr_end, bool b_with_gate_params)
        {
            return draw_qprog(prg, PIC_TYPE::TEXT, false, b_with_gate_params, auto_wrap_len, output_file, itr_start, itr_end);
        },
        py::arg("qprog"),
        py::arg("auto_wrap_len") = 100,
        py::arg("output_file") = "QCircuitTextPic.txt",
        py::arg_v("itr_start", NodeIter(), "NodeIter()"),
        py::arg_v("itr_end", NodeIter(), "NodeIter()"),
        py::arg("b_with_gate_params"),
		"Convert a quantum prog/circuit to text-pic(UTF-8 code),\n"
		"and will save the text-pic in file named QCircuitTextPic.txt in the same time in current path"
        "\n"
        "Args:\n"
        "    QProg: quantum prog \n"
        "    auto_wrap_len: defaut is 100 \n"
        "    output_file: result output file name \n"
        "    itr_start: nodeiter start \n"
        "    itr_end: nodeiter end \n"
        "    b_with_gate_params: if b_with_gate_params is True, draw gate with gate's params.if b_with_gate_params is False, draw gate without gate's params.\n"
        "\n"
        "Returns:\n"
        "    result data tuple contains prog info\n"
        "Raises:\n"
        "    run_fail: An error occurred in get draw_qprog_text\n",
		py::return_value_policy::automatic);

	m.def(
		"draw_qprog_latex",
		[](QProg prg, uint32_t auto_wrap_len, const std::string &output_file, bool with_logo, const NodeIter itr_start, const NodeIter itr_end, bool b_with_gate_params)
		{
			return draw_qprog(prg, PIC_TYPE::LATEX, with_logo,b_with_gate_params, auto_wrap_len, output_file, itr_start, itr_end);
		},
		py::arg("prog"),
		py::arg("auto_wrap_len") = 100,
		py::arg("output_file") = "QCircuit.tex",
		py::arg("with_logo") = false,
		py::arg_v("itr_start", NodeIter(), "NodeIter()"),
		py::arg_v("itr_end", NodeIter(), "NodeIter()"),
        py::arg("b_with_gate_params"),
		"Convert a quantum prog/circuit to latex source code, and save the source code to file in current path with name QCircuit.tex"
        "\n"
        "Args:\n"
        "    QProg: quantum prog \n"
        "    auto_wrap_len: defaut is 100 \n"
        "    output_file: result output file name \n"
        "    itr_start: nodeiter start \n"
        "    itr_end: nodeiter end \n"
        "    b_with_gate_params: if b_with_gate_params is True, draw gate with gate's params.if b_with_gate_params is False, draw gate without gate's params.\n"
        "\n"
        "Returns:\n"
        "    result data tuple contains prog info\n"
        "Raises:\n"
        "    run_fail: An error occurred in get draw_qprog_text\n",
		py::return_value_policy::automatic);

	m.def(
		"draw_qprog_text_with_clock",
		[](QProg prog, const std::string config_data, uint32_t auto_wrap_len, const std::string &output_file, const NodeIter itr_start, const NodeIter itr_end, bool b_with_gate_params)
		{
			return draw_qprog_with_clock(prog, PIC_TYPE::TEXT, config_data, false,b_with_gate_params, auto_wrap_len, output_file, itr_start, itr_end);
		},
		py::arg("prog"),
		py::arg("config_data") = CONFIG_PATH,
		py::arg("auto_wrap_len") = 100,
		py::arg("output_file") = "QCircuitTextPic.txt",
		py::arg_v("itr_start", NodeIter(), "NodeIter()"),
		py::arg_v("itr_end", NodeIter(), "NodeIter()"),
        py::arg("b_with_gate_params"),
		"Convert a quantum prog/circuit to text-pic(UTF-8 code) with time sequence,\n"
		"and will save the text-pic in file named QCircuitTextPic.txt in the same time in current path"
        "\n"
        "Args:\n"
        "    QProg: quantum prog \n"
        "    auto_wrap_len: defaut is 100 \n"
        "    output_file: result output file name \n"
        "    itr_start: nodeiter start \n"
        "    itr_end: nodeiter end \n"
        "    b_with_gate_params: if b_with_gate_params is True, draw gate with gate's params.if b_with_gate_params is False, draw gate without gate's params.\n"
        "\n"
        "Returns:\n"
        "    result data tuple contains prog info\n"
        "Raises:\n"
        "    run_fail: An error occurred in get draw_qprog_text\n",
		py::return_value_policy::automatic);

	m.def(
		"draw_qprog_latex_with_clock",
		[](QProg prog, const std::string config_data, bool with_logo, uint32_t auto_wrap_len, const std::string &output_file, const NodeIter itr_start, const NodeIter itr_end, bool b_with_gate_params)
		{
			return draw_qprog_with_clock(prog, PIC_TYPE::LATEX, config_data, with_logo,b_with_gate_params, auto_wrap_len, output_file, itr_start, itr_end);
		},
		py::arg("prog"),
		py::arg("config_data") = CONFIG_PATH,
		py::arg("auto_wrap_len") = 100,
		py::arg("output_file") = "QCircuit.tex",
		py::arg("with_logo") = false,
		py::arg_v("itr_start", NodeIter(), "NodeIter()"),
		py::arg_v("itr_end", NodeIter(), "NodeIter()"),
        py::arg("b_with_gate_params"),
		"Convert a quantum prog/circuit to latex source code with time sequence, and save the source code to file in current path with name QCircuit.tex"
        "\n"
        "Args:\n"
        "    QProg: quantum prog \n"
        "    config_data: default config file is QPandaConfig.json \n"
        "    auto_wrap_len: defaut is 100 \n"
        "    output_file: result output file name \n"
        "    itr_start: nodeiter start \n"
        "    itr_end: nodeiter end \n"
        "    b_with_gate_params: if b_with_gate_params is True, draw gate with gate's params.if b_with_gate_params is False, draw gate without gate's params.\n"
        "\n"
        "Returns:\n"
        "    result data tuple contains prog info\n"
        "Raises:\n"
        "    run_fail: An error occurred in get draw_qprog_text\n",
		py::return_value_policy::automatic);

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\QProgInfo\Visualization\CharsTransform.h */
    m.def("fit_to_gbk",
        &fit_to_gbk,
        py::arg("utf8_str"),
        "Special character conversion.\n"
        "\n"
        "Args:\n"
        "    utf8_str: string using utf-8 encoding.\n"
        "\n"
        "Returns:\n"
        "    result: converted string.\n",
        py::return_value_policy::automatic);
    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\Tools\FillQProg.h */
    m.def(
        "fill_qprog_by_I",
        [](QProg &prg)
        {
            return fill_qprog_by_I(prg);
        },
        py::arg("qprog"),
            "Fill the input quantum program with I gates and return a new quantum program.\n"
            "\n"
            "Args:\n"
            "    qprog: the input quantum program.\n"
            "\n"
            "Returns:\n"
            "    a new quantum program filled with I gates.\n",
            py::return_value_policy::automatic);

    //#define QUERY_REPLACE(GRAPH_NODE,QUERY_NODE,REPLACE_NODE) \
    //    m.def("graph_query_replace", [](GRAPH_NODE &graph_node, QUERY_NODE &query_node,\
    //                                       REPLACE_NODE &replace_node, QuantumMachine *qvm)\
    //    {\
    //        QProg prog;\
    //        graph_query_replace(graph_node, query_node, replace_node, prog, qvm); \
    //        return prog;\
    //    },py::return_value_policy::automatic);

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\Compiler\QuantumChipAdapter.h */
    m.def(
        "quantum_chip_adapter",
        [](QProg prog, QuantumMachine *quantum_machine, bool b_mapping = true, const std::string config_data = CONFIG_PATH)
        {
            py::list ret_data;

            QVec new_qvec;
            quantum_chip_adapter(prog, quantum_machine, new_qvec, b_mapping, config_data);
            if (!b_mapping)
            {
                get_all_used_qubits(prog, new_qvec);
            }

            ret_data.append(prog);
            ret_data.append(static_cast<std::vector<Qubit*>>(new_qvec));
            return ret_data;
        },
        py::arg("qprog"),
            py::arg("machine"),
            py::arg("mapping") = true,
            py::arg("config_file") = CONFIG_PATH,
            "Perform adaptive conversion for the quantum chip.\n"
            "\n"
            "Args:\n"
            "    qprog: the quantum program.\n"
            "\n"
            "    machine: the quantum machine to be used.\n"
            "\n"
            "    mapping: whether to perform the mapping operation (default is true).\n"
            "\n"
            "    config_file: configuration file path (default is CONFIG_PATH).\n"
            "\n"
            "Returns:\n"
            "    a list containing the quantum program and the list of qubits after mapping; if mapping is false, the qubit list may be misoperated.\n",
            py::return_value_policy::automatic);

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\QProgTransform\TransformDecomposition.h */
    m.def(
        "decompose_multiple_control_qgate",
        [](QProg prog, QuantumMachine *quantum_machine, const std::string config_data = CONFIG_PATH)
        {
            decompose_multiple_control_qgate(prog, quantum_machine, config_data);
            return prog;
        },
        py::arg("qprog"),
            py::arg("machine"),
            py::arg("config_file") = CONFIG_PATH,
            "Decompose a multiple control quantum gate.\n"
            "\n"
            "Args:\n"
            "    qprog: the quantum program containing the gate to be decomposed.\n"
            "\n"
            "    machine: the quantum machine used for decomposition.\n"
            "\n"
            "    config_file: path to the configuration file (default is CONFIG_PATH).\n"
            "\n"
            "Returns:\n"
            "    the updated quantum program after the decomposition.\n",
            py::return_value_policy::automatic);

    /* #include "Core/Utilities/Tools/MultiControlGateDecomposition.h" */
    m.def("ldd_decompose", [](QProg prog)
        {
            return ldd_decompose(prog);
        },
        py::arg("qprog"),
            "Decompose a multiple control quantum gate using LDD.\n"
            "\n"
            "Args:\n"
            "    qprog: the quantum program to be decomposed.\n"
            "\n"
            "Returns:\n"
            "    the updated quantum program after decomposition.\n",
            py::return_value_policy::automatic);

    m.def(
        "transform_to_base_qgate",
        [](QProg prog, QuantumMachine *quantum_machine, const std::string config_data = CONFIG_PATH)
        {
            transform_to_base_qgate(prog, quantum_machine, config_data);
            return prog;
        },
        py::arg("qprog"),
            py::arg("machine"),
            py::arg("config_file") = CONFIG_PATH,
            "Convert quantum gates to basic quantum gates.\n"
            "\n"
            "Args:\n"
            "    qprog: the quantum program to be transformed.\n"
            "\n"
            "    machine: the quantum machine used for the transformation.\n"
            "\n"
            "    config_file: path to the configuration file (default is CONFIG_PATH).\n"
            "\n"
            "Returns:\n"
            "    the updated quantum program after transformation.\n",
            py::return_value_policy::automatic);

    m.def(
        "transform_to_base_qgate",
        [](QProg prog, QuantumMachine *quantum_machine,
            const std::vector<std::string>& convert_single_gates,
            const std::vector<std::string>& convert_double_gates)
        {
            std::vector<std::vector<std::string>> convert_gate_sets = { convert_single_gates ,convert_double_gates };
            transform_to_base_qgate_withinarg(prog, quantum_machine, convert_gate_sets);
            return prog;
        },
        py::arg("qprog"),
            py::arg("machine"),
            py::arg("convert_single_gates"),
            py::arg("convert_double_gates"),
            "Convert quantum gates to basic gates.\n"
            "\n"
            "Args:\n"
            "    qprog: the quantum program to transform.\n"
            "\n"
            "    machine: the quantum machine for the transformation.\n"
            "\n"
            "    convert_single_gates: a set of quantum single gates to convert.\n"
            "\n"
            "    convert_double_gates: a set of quantum double gates to convert.\n"
            "\n"
            "Returns:\n"
            "    the updated quantum program after the conversion.\n",
            py::return_value_policy::automatic);

    m.def(
        "circuit_optimizer",
        [](QProg prog, const std::vector<std::pair<QCircuit, QCircuit>> &optimizer_cir_vec, const std::vector<QCircuitOPtimizerMode> &mode_list = std::vector<QCircuitOPtimizerMode>(0))
        {
            int mode = 0;
            for (const auto &m : mode_list)
            {
                mode |= m;
            }
            cir_optimizer(prog, optimizer_cir_vec, mode);
            return prog;
        },
        py::arg("qprog"),
            py::arg("optimizer_cir_vec") = std::vector<std::pair<QCircuit, QCircuit>>(),
            py::arg("mode_list") = std::vector<QCircuitOPtimizerMode>(0),
            "Optimize a quantum circuit.\n"
            "\n"
            "Args:\n"
            "    qprog: the quantum program to optimize.\n"
            "\n"
            "    optimizer_cir_vec: a list of quantum circuits for optimization.\n"
            "\n"
            "    mode_list: a list of optimization modes.\n"
            "\n"
            "Returns:\n"
            "    the updated quantum program after optimization.\n",
            py::return_value_policy::automatic);

    m.def(
        "circuit_optimizer_by_config",
        [](QProg prog, const std::string config_data, const std::vector<QCircuitOPtimizerMode> &mode_list)
        {
            int mode = 0;
            for (const auto &m : mode_list)
            {
                mode |= m;
            }
            cir_optimizer_by_config(prog, config_data, mode);
            return prog;
        },
        py::arg("qprog"),
            py::arg("config_file") = CONFIG_PATH,
            py::arg("mode_list") = std::vector<QCircuitOPtimizerMode>(0),
            "Optimize a quantum circuit using configuration data.\n"
            "\n"
            "Args:\n"
            "    qprog: the quantum program to optimize.\n"
            "\n"
            "    config_file: configuration data for optimization.\n"
            "\n"
            "    mode_list: a list of optimization modes.\n"
            "\n"
            "Returns:\n"
            "    the updated quantum program after optimization.\n",
            py::return_value_policy::automatic);

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\Tools\Fidelity.h */
    m.def(
        "state_fidelity",
        [](const QStat &state1, const QStat &state2)
        {
            return state_fidelity(state1, state2);
        },
        py::arg("state1"),
            py::arg("state2"),
            "Compare two quantum states and calculate their fidelity.\n"
            "\n"
            "Args:\n"
            "    state1: first quantum state represented as a list.\n"
            "\n"
            "    state2: second quantum state represented as a list.\n"
            "\n"
            "Returns:\n"
            "    The fidelity between the two states, a value in the range [0, 1].\n",
            py::return_value_policy::automatic);

    m.def(
        "state_fidelity",
        [](const std::vector<QStat> &matrix1, const std::vector<QStat> &matrix2)
        {
            return state_fidelity(matrix1, matrix2);
        },
        py::arg("matrix1"),
            py::arg("matrix2"),
            "Compare two quantum state matrices and calculate their fidelity.\n"
            "\n"
            "   Args:\n"
            "       matrix1: first quantum state matrix.\n"
            "\n"
            "       matrix2: second quantum state matrix.\n"
            "\n"
            "   Returns:\n"
            "       The fidelity between the two matrices, a value in the range [0, 1].\n",
            py::return_value_policy::automatic);

    m.def(
        "state_fidelity",
        [](const QStat &state, const vector<QStat> &matrix)
        {
            return state_fidelity(state, matrix);
        },
        py::arg("state1"),
            py::arg("state2"),
            "Compare a quantum state with a state matrix and calculate their fidelity.\n"
            "\n"
            "Args:\n"
            "    state: a single quantum state represented as a list.\n"
            "\n"
            "    matrix: a quantum state matrix.\n"
            "\n"
            "Returns:\n"
            "    The fidelity between the state and the matrix, a value in the range [0, 1].\n",
            py::return_value_policy::automatic);

    m.def(
        "state_fidelity",
        [](const vector<QStat> &matrix, const QStat &state)
        {
            return state_fidelity(matrix, state);
        },
        py::arg("state1"),
            py::arg("state2"),
            "Compare a quantum state matrix with a quantum state and calculate their fidelity.\n"
            "\n"
            "Args:\n"
            "    matrix: a quantum state matrix.\n"
            "\n"
            "    state: a single quantum state represented as a list.\n"
            "\n"
            "Returns:\n"
            "    The fidelity between the matrix and the state, a value in the range [0, 1].\n",
            py::return_value_policy::automatic);

    m.def(
        "average_gate_fidelity",
        [](const QMatrixXcd &matrix, const QStat &state)
        {
            return average_gate_fidelity(matrix, state);
        },
        py::arg("state1"),
            py::arg("state2"),
            "Calculate the average gate fidelity between a quantum operation and a quantum state.\n"
            "\n"
            "Args:\n"
            "    matrix: a quantum operation represented as a matrix.\n"
            "\n"
            "    state: a single quantum state represented as a list.\n"
            "\n"
            "Returns:\n"
            "    The average gate fidelity, a value in the range [0, 1].\n",
            py::return_value_policy::automatic);

    m.def(
        "average_gate_fidelity",
        [](const QMatrixXcd &matrix, const QMatrixXcd &state)
        {
            return average_gate_fidelity(matrix, state);
        },
        py::arg("state1"),
            py::arg("state2"),
            "Calculate the average gate fidelity between two quantum operation matrices.\n"
            "\n"
            "Args:\n"
            "    matrix1: the first quantum operation represented as a matrix.\n"
            "\n"
            "    matrix2: the second quantum operation represented as a matrix.\n"
            "\n"
            "Returns:\n"
            "    The average gate fidelity, a value in the range [0, 1].\n",
            py::return_value_policy::automatic);

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\Tools\GetQubitTopology.h */
    m.def("get_circuit_optimal_topology",
        &get_circuit_optimal_topology,
        py::arg("qprog"),
        py::arg("machine"),
        py::arg("max_connect_degree"),
        py::arg("config_file") = CONFIG_PATH,
        "Retrieve the optimal topology of the input quantum circuit.\n"
        "\n"
        "Args:\n"
        "    qprog: The quantum program for which to determine the topology.\n"
        "\n"
        "    machine: The quantum machine used for execution.\n"
        "\n"
        "    max_connect_degree: The maximum allowable connection degree.\n"
        "\n"
        "    config_file: Path to the configuration file (default is CONFIG_PATH).\n"
        "\n"
        "Returns:\n"
        "    The topology program data.\n",
        py::return_value_policy::automatic);

    m.def("get_double_gate_block_topology",
        &get_double_gate_block_topology,
        py::arg("qprog"),
        "Retrieve the double gate block topology from the input quantum program.\n"
        "\n"
        "Args:\n"
        "    qprog: The quantum program for which to extract the double gate block topology.\n"
        "\n"
        "Returns:\n"
        "    The topology program data.\n",
        py::return_value_policy::automatic);

    m.def(
        "del_weak_edge",
        py::overload_cast<TopologyData &>(&del_weak_edge),
        py::arg("topo_data"),
        "Delete weakly connected edges from the quantum program topology.\n"
        "\n"
        "Args:\n"
        "    topo_data: The topology data of the quantum program.\n"
        "\n"
        "Returns:\n"
        "    None.\n",
        py::return_value_policy::automatic);

    m.def(
        "del_weak_edge2",
        [](TopologyData &topo_data, const size_t max_connect_degree, std::vector<int> &sub_graph_set)
        {
            py::list ret_data;

            std::vector<weight_edge> candidate_edges;
            std::vector<int> intermediary_points = del_weak_edge(topo_data, max_connect_degree, sub_graph_set, candidate_edges);

            ret_data.append(topo_data);
            ret_data.append(intermediary_points);
            ret_data.append(candidate_edges);
            return ret_data;
        },
        py::arg("topo_data"),
            py::arg("max_connect_degree"),
            py::arg("sub_graph_set"),
            "Delete weakly connected edges from the quantum program topology.\n"
            "\n"
            "Args:\n"
            "    topo_data: The topology data of the quantum program.\n"
            "\n"
            "    max_connect_degree: The maximum allowable connection degree.\n"
            "\n"
            "    sub_graph_set: A list of subgraph identifiers.\n"
            "\n"
            "Returns:\n"
            "    A list containing the updated topology data, intermediary points, and candidate edges.\n",
            py::return_value_policy::automatic);

    m.def(
        "del_weak_edge3",
        [](TopologyData &topo_data, std::vector<int> &sub_graph_set, const size_t max_connect_degree, const double lamda1, const double lamda2, const double lamda3)
        {
            py::list ret_data;

            std::vector<int> intermediary_points = del_weak_edge(topo_data, sub_graph_set, max_connect_degree, lamda1, lamda2, lamda3);

            ret_data.append(topo_data);
            ret_data.append(intermediary_points);
            return ret_data;
        },
        py::arg("topo_data"),
            py::arg("sub_graph_set"),
            py::arg("max_connect_degree"),
            py::arg("lamda1"),
            py::arg("lamda2"),
            py::arg("lamda3"),
            "Delete weakly connected edges based on specified parameters.\n"
            "\n"
            "Args:\n"
            "    topo_data: The topology data of the quantum program.\n"
            "\n"
            "    sub_graph_set: A list of subgraph identifiers.\n"
            "\n"
            "    max_connect_degree: The maximum allowable connection degree.\n"
            "\n"
            "    lamda1: Weight parameter for edge evaluation.\n"
            "\n"
            "    lamda2: Weight parameter for edge evaluation.\n"
            "\n"
            "    lamda3: Weight parameter for edge evaluation.\n"
            "\n"
            "Returns:\n"
            "    A list containing the updated topology data and intermediary points.\n",
            py::return_value_policy::automatic);

    m.def(
        "recover_edges",
        [](TopologyData &topo_data, const size_t max_connect_degree, std::vector<weight_edge> &candidate_edges)
        {
            recover_edges(topo_data, max_connect_degree, candidate_edges);
            return topo_data;
        },
        py::arg("topo_data"),
            py::arg("max_connect_degree"),
            py::arg("candidate_edges"),
            "Recover edges using the specified candidate edges.\n"
            "\n"
            "Args:\n"
            "    topo_data: The topology data of the quantum program.\n"
            "\n"
            "    max_connect_degree: The maximum allowed connection degree.\n"
            "\n"
            "    candidate_edges: A list of edges to consider for recovery.\n"
            "\n"
            "Returns:\n"
            "    The updated topology data after recovery.\n",
            py::return_value_policy::automatic);

    m.def("get_complex_points",
        &get_complex_points,
        py::arg("topo_data"),
        py::arg("max_connect_degree"),
        "Retrieve complex points from the given topology data.\n"
        "\n"
        "Args:\n"
        "    topo_data: The topology data of the quantum program.\n"
        "\n"
        "    max_connect_degree: The maximum allowable connection degree.\n"
        "\n"
        "Returns:\n"
        "    A list of complex points extracted from the topology data.\n",
        py::return_value_policy::automatic);

    py::enum_<ComplexVertexSplitMethod>(m, "ComplexVertexSplitMethod", "quantum complex vertex split method")
        .value("METHOD_UNDEFINED", ComplexVertexSplitMethod::METHOD_UNDEFINED)
        .value("LINEAR", ComplexVertexSplitMethod::LINEAR)
        .value("RING", ComplexVertexSplitMethod::RING)
        .export_values();

    m.def("split_complex_points",
        &split_complex_points,
        py::arg("complex_points"),
        py::arg("max_connect_degree"),
        py::arg("topo_data"),
        py::arg_v("split_method", ComplexVertexSplitMethod::LINEAR, "ComplexVertexSplitMethod.LINEAR"),
        "Split complex points into multiple discrete points.\n"
        "\n"
        "Args:\n"
        "    complex_points: A list of complex points to be split.\n"
        "\n"
        "    max_connect_degree: The maximum allowable connection degree.\n"
        "\n"
        "    topo_data: The topology data of the quantum program.\n"
        "\n"
        "    split_method: Method for splitting, as defined in ComplexVertexSplitMethod. Defaults to ComplexVertexSplitMethod.LINEAR.\n"
        "\n"
        "Returns:\n"
        "    None: The function modifies the input data in place.\n",
        py::return_value_policy::automatic);

    m.def("replace_complex_points",
        &replace_complex_points,
        py::arg("src_topo_data"),
        py::arg("max_connect_degree"),
        py::arg("sub_topo_vec"),
        "Replace complex points in the source topology with subgraphs.\n"
        "\n"
        "Args:\n"
        "    src_topo_data: The source topology data of the quantum program.\n"
        "\n"
        "    max_connect_degree: The maximum allowable connection degree.\n"
        "\n"
        "    sub_topo_vec: A list of sub-topologies to replace the complex points.\n"
        "\n"
        "Returns:\n"
        "    None: This function modifies the source topology in place.\n",
        py::return_value_policy::automatic);

    py::class_<CommProtocolConfig>(m, "CommProtocolConfig")
        .def(py::init<>())
        .def_readwrite("open_mapping", &CommProtocolConfig::open_mapping)
        .def_readwrite("open_error_mitigation", &CommProtocolConfig::open_error_mitigation)
        .def_readwrite("optimization_level", &CommProtocolConfig::optimization_level)
        .def_readwrite("circuits_num", &CommProtocolConfig::circuits_num)
        .def_readwrite("shots", &CommProtocolConfig::shots);

    m.def("get_sub_graph",
        &get_sub_graph,
        py::arg("topo_data"),
        "Retrieve a subgraph from the provided topology data.\n"
        "\n"
        "Args:\n"
        "    topo_data: The topology data of the quantum program.\n"
        "\n"
        "Returns:\n"
        "    sub graph: The extracted subgraph from the provided topology.\n",
        py::return_value_policy::automatic);

    m.def("comm_protocol_encode",
        [](QProg prog,
            CommProtocolConfig config = {})
        {
            auto encode_data = comm_protocol_encode(prog, config);
            return py::bytes(encode_data.data(), encode_data.size());
        },
        py::arg("prog"),
            py::arg("config") = CommProtocolConfig(),
            "Encode the communication protocol data into binary format.\n"
            "\n"
            "Args:\n"
            "    prog: The quantum program to be encoded.\n"
            "\n"
            "    config: The configuration for the communication protocol. Defaults to an empty configuration.\n"
            "\n"
            "Returns:\n"
            "    bytes: The encoded binary data representing the communication protocol.\n",
            py::return_value_policy::automatic);

    m.def("comm_protocol_encode",
        [](std::vector<QProg> prog_list,
            CommProtocolConfig config = {})
        {
            auto encode_data = comm_protocol_encode(prog_list, config);
            return py::bytes(encode_data.data(), encode_data.size());
        },
        py::arg("prog_list"),
            py::arg("config") = CommProtocolConfig(),
            "Encode a list of quantum programs into binary communication protocol data.\n"
            "\n"
            "Args:\n"
            "    prog_list: A list of quantum programs to be encoded.\n"
            "\n"
            "    config: The configuration for the communication protocol. Defaults to an empty configuration.\n"
            "\n"
            "Returns:\n"
            "    bytes: The encoded binary data representing the communication protocol.\n",
            py::return_value_policy::automatic);

    m.def("comm_protocol_decode",
        [](const py::bytes& byte_data, QuantumMachine* machine)
        {
            char* ptr;
            Py_ssize_t size;
            PyBytes_AsStringAndSize(byte_data.ptr(), &ptr, &size);

            auto encode_data = std::vector<char>(ptr, ptr + size);

            CommProtocolConfig config;
            auto decode_prog_list = comm_protocol_decode(config, encode_data, machine);
            return std::make_tuple(decode_prog_list, config);
        },
        py::arg("encode_data"),
            py::arg("machine"),
            "Decode binary data into a list of quantum programs using the communication protocol.\n"
            "\n"
            "Args:\n"
            "    encode_data: The encoded binary data representing quantum programs.\n"
            "\n"
            "    machine: A pointer to the QuantumMachine used for decoding.\n"
            "\n"
            "Returns:\n"
            "    tuple: A tuple containing the decoded program list and the communication protocol configuration.\n",
            py::return_value_policy::automatic);

    m.def("estimate_topology",
        &estimate_topology,
        py::arg("topo_data"),
        "Evaluate topology performance.\n"
        "\n"
        "Args:\n"
        "    topo_data: Quantum program topology data.\n"
        "\n"
        "Returns:\n"
        "    Result data.\n",
        py::return_value_policy::automatic);

    m.def("decompose_multiple_control_qgate",
        [](QProg prog, QuantumMachine *quantum_machine,
            const std::vector<std::string>& convert_single_gates,
            const std::vector<std::string>& convert_double_gates,
            bool b_transform_to_base_qgate = true)
        {
            std::vector<std::vector<std::string>> convert_gate_sets = { convert_single_gates ,convert_double_gates };
            decompose_multiple_control_qgate_withinarg(prog, quantum_machine, convert_gate_sets, b_transform_to_base_qgate);
            return prog;
        },
        py::arg("qprog"),
            py::arg("machine"),
            py::arg("convert_single_gates"),
            py::arg("convert_double_gates"),
            py::arg("b_transform_to_base_qgate") = true,
            "Decompose multiple control QGate.\n"
            "\n"
            "Args:\n"
            "    qprog: Quantum program.\n"
            "\n"
            "    machine: Quantum machine.\n"
            "\n"
            "    convert_single_gates: Sets of quantum single gates.\n"
            "\n"
            "    convert_double_gates: Sets of quantum double gates.\n"
            "\n"
            "    b_transform_to_base_qgate: Transform to base QGate sets.\n"
            "\n"
            "Returns:\n"
            "    A new program after decomposition.\n",
            py::return_value_policy::automatic);

    m.def("planarity_testing",
        &planarity_testing,
        py::arg("topo_data"),
        "Perform planarity testing.\n"
        "\n"
        "Args:\n"
        "    topo_data: Quantum program topology data.\n"
        "\n"
        "Returns:\n"
        "    Result data.\n",
        py::return_value_policy::automatic);

#if defined(USE_CURL) && defined(USE_QHETU)

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\QProgInfo\CrossEntropyBenchmarking.h */

    py::class_<QCloudTaskConfig>(m, "QCloudTaskConfig")
        .def(py::init<>())
        .def_readwrite("cloud_token", &QCloudTaskConfig::cloud_token)
        .def_readwrite("chip_id", &QCloudTaskConfig::chip_id)
        .def_readwrite("shots", &QCloudTaskConfig::shots)
        .def_readwrite("open_amend", &QCloudTaskConfig::open_amend)
        .def_readwrite("open_mapping", &QCloudTaskConfig::open_mapping)
        .def_readwrite("open_optimization", &QCloudTaskConfig::open_optimization);

    m.def("double_gate_xeb",
        [](QCloudTaskConfig config,
            int qubit_0,
            int qubit_1,
            const std::vector<int>& range,
            int num_circuits,
            GateType gt)
        {
            return double_gate_xeb(config, qubit_0, qubit_1, range, num_circuits, gt);
        },
        py::arg("config"),
            py::arg("qubit0"),
            py::arg("qubit1"),
            py::arg("clifford_range"),
            py::arg("num_circuits"),
            py::arg_v("gate_type", GateType::CZ_GATE, "GateType.CZ_GATE"),
            "double gate xeb\n"
            "\n"
            "Args:\n"
            "    qvm: quantum machine\n"
            "    qubit0: double qubit 0\n"
            "    qubit1: double qubit 1\n"
            "    clifford_range: clifford range list\n"
            "    num_circuits: the num of circuits\n"
            "    interleaved_gates: interleaved gates list\n"
            "\n"
            "Returns:\n"
            "    result data dict\n"
            "Raises:\n"
            "    run_fail: An error occurred in double_gate_xeb\n",
            py::return_value_policy::automatic);

    m.def("double_gate_xeb",
        [](QuantumMachine* qvm,
            Qubit* qbit0,
            Qubit* qbit1,
            const std::vector<int>& range,
            int num_circuits,
            int shots,
            int chip_id,
            GateType gt)
        {
            auto real_chip_type = static_cast<RealChipType>(chip_id);
            return double_gate_xeb(qvm, qbit0, qbit1, range, num_circuits, shots, real_chip_type, gt);
        },
        py::arg("qvm"),
            py::arg("qubit0"),
            py::arg("qubit1"),
            py::arg("clifford_range"),
            py::arg("num_circuits"),
            py::arg("shots"),
            py::arg("chip_id") = (int)RealChipType::ORIGIN_WUYUAN_D5,
            py::arg_v("gate_type", GateType::CZ_GATE, "GateType.CZ_GATE"),
            "double gate xeb\n"
            "\n"
            "Args:\n"
            "    qvm: quantum machine\n"
            "    qubit0: double qubit 0\n"
            "    qubit1: double qubit 1\n"
            "    clifford_range: clifford range list\n"
            "    num_circuits: the num of circuits\n"
            "    shots: measure shots\n"
            "    chip type: RealChipType\n"
            "    interleaved_gates: interleaved gates list\n"
            "\n"
            "Returns:\n"
            "    result data dict\n"
            "Raises:\n"
            "    run_fail: An error occurred in double_gate_xeb\n",
            py::return_value_policy::automatic);




    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\QProgInfo\QuantumVolume.h */
    m.def("calculate_quantum_volume",
        py::overload_cast<NoiseQVM *, std::vector<std::vector<int>>, int, int>(&calculate_quantum_volume),
        py::arg("noise_qvm"),
        py::arg("qubit_list"),
        py::arg("ntrials"),
        py::arg("shots") = 1000,
        "calculate quantum volume\n"
        "\n"
        "Args:\n"
        "    noise_qvm: noise quantum machine\n"
        "    qubit_list: qubit list \n"
        "    ntrials: ntrials\n"
        "    shots: measure shots\n"
        "\n"
        "Returns:\n"
        "    result data dict\n"
        "Raises:\n"
        "    run_fail: An error occurred in calculate_quantum_volume\n",
        py::return_value_policy::automatic);

    m.def("calculate_quantum_volume",
        py::overload_cast<QCloudMachine *, std::vector<std::vector<int>>, int, int>(&calculate_quantum_volume),
        py::arg("cloud_qvm"),
        py::arg("qubit_list"),
        py::arg("ntrials"),
        py::arg("shots") = 1000,
        "calculate quantum volume\n"
        "\n"
        "Args:\n"
        "    noise_qvm: noise quantum machine\n"
        "    qubit_list: qubit list \n"
        "    ntrials: ntrials\n"
        "    shots: measure shots\n"
        "\n"
        "Returns:\n"
        "    result data dict\n"
        "Raises:\n"
        "    run_fail: An error occurred in calculate_quantum_volume\n",
        py::return_value_policy::automatic);

    m.def("calculate_quantum_volume",
        py::overload_cast<QCloudTaskConfig, std::vector<std::vector<int>>, int>(&calculate_quantum_volume),
        py::arg("config"),
        py::arg("qubit_list"),
        py::arg("ntrials"),
        "calculate quantum volume\n"
        "\n"
        "Args:\n"
        "    config: QCloudTaskConfig\n"
        "    qubit_list: qubit list \n"
        "    ntrials: ntrials\n"
        "\n"
        "Returns:\n"
        "    result data dict\n"
        "Raises:\n"
        "    run_fail: An error occurred in calculate_quantum_volume\n",
        py::return_value_policy::automatic);

    //---------------------------------------------------------------------------------------------------------------------
    /* include\Core\Utilities\QProgInfo\RandomizedBenchmarking.h */
    m.def("single_qubit_rb",
        [](QuantumMachine* qvm,
            Qubit* qbit,
            const std::vector<int>& clifford_range,
            int num_circuits,
            int shots,
            int chip_id = (int)RealChipType::ORIGIN_WUYUAN_D5,
            const std::vector<QGate>& interleaved_gates = {})
        {
            auto real_chip_type = static_cast<RealChipType>(chip_id);
            return single_qubit_rb(qvm, qbit, clifford_range, num_circuits, shots, real_chip_type, interleaved_gates);
        },
        py::arg("qvm"),
            py::arg("qubit"),
            py::arg("clifford_range"),
            py::arg("num_circuits"),
            py::arg("shots"),
            py::arg("chip_id") = (int)RealChipType::ORIGIN_WUYUAN_D5,
            py::arg("interleaved_gates") = std::vector<QGate>(),
            "Single qubit rb with WU YUAN chip\n"
            "\n"
            "Args:\n"
            "    qvm: quantum machine\n"
            "    qubit: single qubit\n"
            "    clifford_range: clifford range list\n"
            "    num_circuits: the num of circuits\n"
            "    shots: measure shots\n"
            "    chip type: RealChipType\n"
            "    interleaved_gates: interleaved gates list\n"
            "\n"
            "Returns:\n"
            "    result data dict\n"
            "Raises:\n"
            "    run_fail: An error occurred in single_qubit_rb\n",
            py::return_value_policy::automatic);

    m.def("double_qubit_rb",
        [](QuantumMachine* qvm,
            Qubit* qbit0,
            Qubit* qbit1,
            const std::vector<int>& clifford_range,
            int num_circuits,
            int shots,
            int chip_id = (int)RealChipType::ORIGIN_WUYUAN_D5,
            const std::vector<QGate>& interleaved_gates = {})
        {
            auto real_chip_type = static_cast<RealChipType>(chip_id);
            return double_qubit_rb(qvm, qbit0, qbit1, clifford_range, num_circuits, shots, real_chip_type, interleaved_gates);
        },
        py::arg("qvm"),
            py::arg("qubit0"),
            py::arg("qubit1"),
            py::arg("clifford_range"),
            py::arg("num_circuits"),
            py::arg("shots"),
            py::arg("chip_id") = (int)RealChipType::ORIGIN_WUYUAN_D5,
            py::arg("interleaved_gates") = std::vector<QGate>(),
            "double qubit rb with WU YUAN chip"
            "\n"
            "Args:\n"
            "    qvm: quantum machine\n"
            "    qubit0: double qubit 0\n"
            "    qubit1: double qubit 1\n"
            "    clifford_range: clifford range list\n"
            "    num_circuits: the num of circuits\n"
            "    shots: measure shots\n"
            "    chip type: RealChipType\n"
            "    interleaved_gates: interleaved gates list\n"
            "\n"
            "Returns:\n"
            "    result data dict\n"
            "Raises:\n"
            "    run_fail: An error occurred in double_qubit_rb\n",
            py::return_value_policy::automatic);

    m.def("single_qubit_rb",
        [](QCloudTaskConfig config,
            int qbit,
            const std::vector<int>& clifford_range,
            int num_circuits,
            const std::vector<QGate>& interleaved_gates = {})
        {
            return single_qubit_rb(config, qbit, clifford_range, num_circuits, interleaved_gates);
        },
        py::arg("config"),
            py::arg("qubit"),
            py::arg("clifford_range"),
            py::arg("num_circuits"),
            py::arg("interleaved_gates") = std::vector<QGate>(),
            "Single qubit rb with origin chip\n"
            "\n"
            "Args:\n"
            "    config: quantum QCloudTaskConfig\n"
            "    qubit: single qubit\n"
            "    clifford_range: clifford range list\n"
            "    num_circuits: the num of circuits\n"
            "    interleaved_gates: interleaved gates list\n"
            "\n"
            "Returns:\n"
            "    result data dict\n"
            "Raises:\n"
            "    run_fail: An error occurred in single_qubit_rb\n",
            py::return_value_policy::automatic);

    m.def("double_qubit_rb",
        [](QCloudTaskConfig config,
            int qbit0,
            int qbit1,
            const std::vector<int>& clifford_range,
            int num_circuits,
            const std::vector<QGate>& interleaved_gates = {})
        {
            return double_qubit_rb(config, qbit0, qbit1, clifford_range, num_circuits, interleaved_gates);
        },
        py::arg("config"),
            py::arg("qubit0"),
            py::arg("qubit1"),
            py::arg("clifford_range"),
            py::arg("num_circuits"),
            py::arg("interleaved_gates") = std::vector<QGate>(),
            "double qubit rb with origin chip"
            "\n"
            "Args:\n"
            "    config: QCloudTaskConfig\n"
            "    qubit0: double qubit 0\n"
            "    qubit1: double qubit 1\n"
            "    clifford_range: clifford range list\n"
            "    num_circuits: the num of circuits\n"
            "    interleaved_gates: interleaved gates list\n"
            "\n"
            "Returns:\n"
            "    result data dict\n"
            "Raises:\n"
            "    run_fail: An error occurred in double_qubit_rb\n",
            py::return_value_policy::automatic);

#endif

    /* include\Core\Utilities\Tools\RandomCircuit.h */
    m.def("random_qprog",
        &random_qprog,
        py::arg("qubit_row"),
        py::arg("qubit_col"),
        py::arg("depth"),
        py::arg("qvm"),
        py::arg("qvec"),
        "Generate a random quantum program.\n"
        "\n"
        "Args:\n"
        "    qubit_row: Circuit qubit row value.\n"
        "\n"
        "    qubit_col: Circuit qubit column value.\n"
        "\n"
        "    depth: Circuit depth.\n"
        "\n"
        "    qvm: Quantum machine.\n"
        "\n"
        "    qvec: Output circuits for the random quantum program.\n"
        "\n"
        "Returns:\n"
        "    A random quantum program.\n");

    m.def("random_qcircuit",
        &random_qcircuit,
        py::arg("qvec"),
        py::arg("depth") = 100,
        py::arg("gate_type") = std::vector<std::string>(),
        "Generate a random quantum circuit.\n"
        "\n"
        "Args:\n"
        "    qvec: Output circuits for the random circuit.\n"
        "\n"
        "    depth: Circuit depth (default is 100).\n"
        "\n"
        "    gate_type: Types of gates to use (default is an empty vector).\n"
        "\n"
        "Returns:\n"
        "    A random quantum circuit.\n");


    m.def("prog_layer",
        &prog_layer,
        py::arg("prog"),
        "Process the given quantum program layer.\n"
        "\n"
        "Args:\n"
        "    prog: The quantum program to be processed.\n"
        "\n"
        "Returns:\n"
        "    Processed quantum program layer.\n",
        py::return_value_policy::automatic);

    m.def("remap",
        py::overload_cast<QProg, const QVec&, const std::vector<ClassicalCondition>&>(&remap),
        py::arg("prog"),
        py::arg("target_qlist"),
        py::arg("target_clist") = std::vector<ClassicalCondition>(),
        "Map the source quantum program to the target qubits.\n"
        "\n"
        "Args:\n"
        "    prog: Source quantum program.\n"
        "\n"
        "    target_qlist: Target qubits.\n"
        "\n"
        "    target_clist: Target classical bits (default is an empty vector).\n"
        "\n"
        "Returns:\n"
        "    The target quantum program.\n",
        py::return_value_policy::automatic);

    m.def("deep_copy",
        [](QProg& node) { return deepCopy(node); },
        py::arg("node"),
        "Create a deep copy of the given quantum program node.\n"
        "\n"
        "Args:\n"
        "    node: The quantum program node to copy.\n"
        "\n"
        "Returns:\n"
        "    A deep copy of the quantum program node.\n",
        py::return_value_policy::automatic);

    m.def("deep_copy",
        [](QCircuit& node) { return deepCopy(node); },
        py::arg("node"),
        "Create a deep copy of the given quantum program node.\n"
        "\n"
        "Args:\n"
        "    node: The quantum program node to copy.\n"
        "\n"
        "Returns:\n"
        "    A deep copy of the quantum program node.\n",
        py::return_value_policy::automatic);


    m.def("deep_copy",
        [](QGate& node) { return deepCopy(node); },
        py::arg("node"),
        "Create a deep copy of the given quantum program node.\n"
        "\n"
        "Args:\n"
        "    node: The quantum program node to copy.\n"
        "\n"
        "Returns:\n"
        "    A deep copy of the quantum program node.\n",
        py::return_value_policy::automatic);

    m.def("deep_copy",
        [](QMeasure& node) { return deepCopy(node); },
        py::arg("node"),
        "Create a deep copy of the given quantum program node.\n"
        "\n"
        "Args:\n"
        "    node: The quantum program node to copy.\n"
        "\n"
        "Returns:\n"
        "    A deep copy of the quantum program node.\n",
        py::return_value_policy::automatic);

    m.def("deep_copy",
        [](ClassicalProg& node) { return deepCopy(node); },
        py::arg("node"),
        "Create a deep copy of the given quantum program node.\n"
        "\n"
        "Args:\n"
        "    node: The quantum program node to copy.\n"
        "\n"
        "Returns:\n"
        "    A deep copy of the quantum program node.\n",
        py::return_value_policy::automatic);

    m.def("deep_copy",
        [](QIfProg& node) { return deepCopy(node); },
        py::arg("node"),
        "Create a deep copy of the given quantum program node.\n"
        "\n"
        "Args:\n"
        "    node: The quantum program node to copy.\n"
        "\n"
        "Returns:\n"
        "    A deep copy of the quantum program node.\n",
        py::return_value_policy::automatic);

    m.def("deep_copy",
        [](QWhileProg& node) { return deepCopy(node); },
        py::arg("node"),
        "Create a deep copy of the given quantum program node.\n"
        "\n"
        "Args:\n"
        "    node: The quantum program node to copy.\n"
        "\n"
        "Returns:\n"
        "    A deep copy of the quantum program node.\n",
        py::return_value_policy::automatic);

    // m.def("deep_copy", [](QCircuit& node) { return deepCopy(node); }, py::arg("node"), py::return_value_policy::automatic);
    // m.def("deep_copy", [](QGate& node) { return deepCopy(node); }, py::arg("node"), py::return_value_policy::automatic);
    // m.def("deep_copy", [](QMeasure& node) { return deepCopy(node); }, py::arg("node"), py::return_value_policy::automatic);
    // m.def("deep_copy", [](ClassicalProg& node) { return deepCopy(node); }, py::arg("node"), py::return_value_policy::automatic);
    // m.def("deep_copy", [](QIfProg& node) { return deepCopy(node); }, py::arg("node"), py::return_value_policy::automatic);
    // m.def("deep_copy", [](QWhileProg& node) { return deepCopy(node); }, py::arg("node"), py::return_value_policy::automatic);
    /* =============================test end =============================*/

    /*QUERY_REPLACE(QProg, QCircuit, QCircuit)
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
    QUERY_REPLACE(QGate, QGate, QGate);*/
#if 0
#endif
}
