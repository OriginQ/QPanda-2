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

	//---------------------------------------------------------------------------------------------------------------------
	/* include\Core.h */
	m.def("init",
		  &init,
		  py::arg_v("machine_type", QMachineType::CPU, "QMachineType.CPU"),
		  "Init the global unique quantum machine at background.\n"
		  "\n"
		  "Args:\n"
		  "    machine_type: quantum machine type, see pyQPanda.QMachineType\n"
		  "\n"
		  "Returns:\n"
		  "    bool: ture if initialization success");

	m.def("finalize",
        py::overload_cast<>(&finalize),
        "Finalize the environment and destory global unique quantum machine.\n"
        "\n"
        "Args:\n"
        "    none\n"
        "\n"
        "Returns:\n"
        "    none");

	m.def("qAlloc",
		  py::overload_cast<>(&qAlloc),
		  "Create a qubit\n"
		  "After init()\n"
		  "\n"
		  "Returns:\n"
		  "    a new qubit."
		  "    None, if quantum machine had created max number of qubit, which is 29",
		  py::return_value_policy::reference);

	m.def("qAlloc",
		  py::overload_cast<size_t>(&qAlloc),
		  py::arg("qubit_addr"),
		  "Allocate a qubits\n"
		  "After init()\n"
		  "\n"
		  "Args:\n"
		  "    qubit_addr: qubit physic address, should in [0,29)\n"
		  "\n"
		  "Returns:\n"
		  "    pyQPanda.Qubit: None, if qubit_addr error, or reached max number of allowed qubit",
		  py::return_value_policy::reference);

	m.def("directly_run",
		  &directlyRun,
		  py::arg("qprog"),
		  py::arg_v("noise_model", NoiseModel(), "NoiseModel()"),
		  "Directly run quantum prog\n"
		  "After init()\n"
		  "\n"
		  "Args:\n"
		  "    qprog: quantum program\n"
		  "    noise_model: noise model, default is no noise. noise model only work on CPUQVM now\n"
		  "\n"
		  "Returns:\n"
		  "    Dict[str, bool]: result of quantum program execution one shot.\n"
		  "                     first is the final qubit register state, second is it's measure probability");

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
		  "    qubit_num: numbers of qubit want to be created\n"
		  "\n"
		  "Returns:\n"
		  "    list[pyQPanda.Qubit]: list of qubit",
		  py::return_value_policy::reference);

	m.def("cAlloc",
		  py::overload_cast<>(&cAlloc),
		  "Allocate a CBit\n"
		  "After init()\n"
		  "\n"
		  "Returns:\n"
		  "    classic result cbit",
		  py::return_value_policy::reference);

	m.def("cAlloc",
		  py::overload_cast<size_t>(&cAlloc),
		  py::arg("cbit_addr"),
		  "Allocate a CBit\n"
		  "After init()\n"
		  "\n"
		  "Args:\n"
		  "    cbit_addr: cbit address, should in [0,29)"
		  "\n"
		  "Returns:\n"
		  "    classic result cbit",
		  py::return_value_policy::reference);

	m.def("cAlloc_many",
		  &cAllocMany,
		  py::arg("cbit_num"),
		  "Allocate several CBits\n"
		  "After init()\n"
		  "\n"
		  "Args:\n"
		  "    cbit_num: numbers of cbit want to be created\n"
		  "\n"
		  "Returns:\n"
		  "    list of cbit",
		  py::return_value_policy::reference);

	m.def("cFree",
            &cFree,
            py::arg("cbit"),
            "Free a CBit\n"
            "\n"
            "Args:\n"
            "    CBit: a CBit\n"
            "\n"
            "Returns:\n"
            "    none\n");

	m.def("cFree_all",
        py::overload_cast<>(&cFreeAll),
        "Free all cbits\n"
        "\n"
        "Args:\n"
        "    none\n"
        "\n"
        "Returns:\n"
        "    none\n");

	m.def("cFree_all",
		  py::overload_cast<vector<ClassicalCondition> &>(&cFreeAll),
		  py::arg("cbit_list"),
          "Free all cbits\n"
          "\n"
          "Args:\n"
          "    a list of cbits\n"
          "\n"
          "Returns:\n"
          "    none\n");

	m.def("qFree",
		  &qFree,
		  py::arg("qubit"),
          "Free a qubit\n"
          "\n"
          "Args:\n"
          "    Qubit: a qubit\n"
          "\n"
          "Returns:\n"
          "    none\n");

	m.def("qFree_all",
        py::overload_cast<>(&qFreeAll),
        "Free all qubits\n"
        "\n"
        "Args:\n"
        "    none\n"
        "\n"
        "Returns:\n"
        "    none\n");

	m.def("qFree_all",
        py::overload_cast<QVec &>(&qFreeAll),
        py::arg("qubit_list"),
        "Free a list of qubits\n"
        "\n"
        "Args:\n"
        "    a list of qubits\n"
        "\n"
        "Returns:\n"
        "    none\n");

	m.def("getstat",
        &getstat,
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
        py::return_value_policy::reference);

	m.def("get_allocate_qubits",
		[]()
		{
			QVec qv;
			get_allocate_qubits(qv);
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
        py::return_value_policy::reference);

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
        "    none\n"
        "\n"
        "Returns:\n"
        "    cbit list\n"
        "\n"
        "Raises:\n"
        "    run_fail: An error occurred in allocated cbits of QuantumMachine\n",
		py::return_value_policy::reference);

	m.def("get_tuple_list",
		  &getProbTupleList,
		  py::arg("qubit_list"),
		  py::arg("select_max") = -1,
		  "Get pmeasure result as tuple list\n"
		  "\n"
		  "Args:\n"
		  "    qubit_list: pmeasure qubits list\n"
		  "    select_max: max returned element num in returnd tuple, should in [-1, 1<<len(qubit_list)]\n"
		  "                default is -1, means no limit"
		  "\n"
		  "Returns:\n"
		  "    measure result of quantum machine\n"
          "Raises:\n"
          "    run_fail: An error occurred in get_tuple_list\n",
		  py::return_value_policy::reference);

	m.def("get_prob_list",
		  &getProbList,
		  py::arg("qubit_list"),
		  py::arg("select_max") = -1,
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
		  py::return_value_policy::reference);

	m.def("get_prob_dict",
		  &getProbDict,
		  py::arg("qubit_list"),
		  py::arg("select_max") = -1,
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
		  py::return_value_policy::reference);

	m.def("prob_run_tuple_list",
		  &probRunTupleList,
		  py::arg("qptog"),
		  py::arg("qubit_list"),
		  py::arg("select_max") = -1,
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
		  py::return_value_policy::reference);

	m.def("prob_run_list",
		  &probRunList,
		  py::arg("qprog"),
		  py::arg("qubit_list"),
		  py::arg("select_max") = -1,
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
		  py::return_value_policy::reference);

	m.def("prob_run_dict",
		  &probRunDict,
		  py::arg("qprog"),
		  py::arg("qubit_list"),
		  py::arg("select_max") = -1,
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
		  py::return_value_policy::reference);

    m.def(
        "run_with_configuration",
        [](QProg &prog, std::vector<ClassicalCondition> &cbits, int shots, const NoiseModel& model= NoiseModel())
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
        "    program: quantum program\n"
        "    cbit_list: classic cbits list\n"
        "    shots: repeate run quantum program times\n"
        "    noise_model: noise model, default is no noise. noise model only work on CPUQVM now\n"
        "\n"
        "Returns:\n"
        "    result of quantum program execution in shots.\n"
        "    first is the final qubit register state, second is it's hit shot"
        "Raises:\n"
        "    run_fail: An error occurred in measure quantum program\n");

    m.def(
        "run_with_configuration",
        [](QProg &prog, int shots, const NoiseModel& model = NoiseModel())
        {
        return runWithConfiguration(prog, shots, model);
        },
        py::arg("program"),
        py::arg("shots"),
        py::arg_v("noise_model", NoiseModel(), "NoiseModel()"),
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
        "    run_fail: An error occurred in measure quantum program\n");

	m.def("quick_measure",
		  &quickMeasure,
		  py::arg("qubit_list"),
		  py::arg("shots"),
		  "Quick measure\n"
		  "\n"
		  "Args:\n"
		  "    qubit_list: qubit list to measure\n"
		  "    shots: the repeat num  of measure operate\n"
		  "\n"
		  "Returns:\n"
		  "    result of quantum program"
          "Raises:\n"
          "    run_fail: An error occurred in measure quantum program\n");

	m.def("accumulate_probabilities",
		  &accumulateProbability,
		  py::arg("probability_list"),
		  "Accumulate the probability from a prob list\n"
		  "\n"
		  "Args:\n"
		  "    probability_list: measured result in probability list form\n"
		  "\n"
		  "Returns:\n"
		  "    accumulated result"
          "Raises:\n"
          "    run_fail: An error occurred in accumulate_probabilities\n");

	m.def("accumulateProbability",
		  &accumulateProbability,
		  py::arg("probability_list"),
		  "Accumulate the probability from a prob list\n"
		  "\n"
		  "Args:\n"
		  "    probability_list: measured result in probability list form\n"
		  "\n"
		  "Returns:\n"
		  "    accumulated result"
          "Raises:\n"
          "    run_fail: An error occurred in accumulateProbability\n");

	m.def("accumulate_probability",
		  &accumulateProbability,
		  py::arg("probability_list"),
		  "Accumulate the probability from a prob list\n"
		  "\n"
		  "Args:\n"
		  "    probability_list: measured result in probability list form\n"
		  "\n"
		  "Returns:\n"
		  "    accumulated result"
          "Raises:\n"
          "    run_fail: An error occurred in accumulate_probability\n");

	m.def("get_qstate",
        &getQState,
        "Get quantum machine state vector\n"
        "\n"
        "Args:\n"
        "    none\n"
        "\n"
        "Returns:\n"
        "    state vector list result"
        "Raises:\n"
        "    run_fail: An error occurred in get_qstate\n"
        "Examples:\n"
        "   >>> print(machine.get_qstate()"
        "   [0.707+0j, 0.707+0j, 0, 0])");

	m.def("init_quantum_machine",
		  /*
			pybind11 support C++ polymorphism
			when C++ func return a pointer/reference of base class but point to derived class object
			pybind11 will get it's runtime type info of derived class, convert pointer to derived class object and return python wrapper
		  */
		  &initQuantumMachine,
		  py::arg_v("machine_type", QMachineType::CPU, "QMachineType.CPU,"),
		  "Create and initialize a new quantum machine, and let it be global unique quantum machine\n"
		  "\n"
		  "Args:\n"
		  "    machine_type: quantum machine type, see pyQPanda.QMachineType\n"
		  "\n"
		  "Returns:\n"
		  "    the quantum machine, type is depend on machine_type\n"
		  "    QMachineType.CPU               --> pyQPanda.CPUQVM\n"
		  "    QMachineType.CPU_SINGLE_THREAD --> pyQPanda.CPUSingleThreadQVM\n"
		  "    QMachineType.GPU               --> pyQPanda.GPUQVM (if pyQPanda is build with GPU)\n"
		  "    QMachineType.NOISE             --> pyQPanda.NoiseQVM\n"
		  //   "    QMachineType.QCloud            --> pyQPanda.CloudQVM\n"
		  "    return None if initial machine faild\n"
          "Raises:\n"
          "    init_fail: An error occurred in init_quantum_machine\n",
		  /*
			if py::return_value_policy::reference, python object won't take ownership of returned C++ object, C++ should manage resources
		  */
		  py::return_value_policy::reference);

	/* see PyQuantumMachine to see how python polymorphism run as C++ like */
	m.def("destroy_quantum_machine",
		  &destroyQuantumMachine,
		  py::arg("machine"),
		  "Destroy a quantum machine\n"
		  "\n"
		  "Args:\n"
		  "    machine: type should be one of CPUQVM, CPUSingleThreadQVM, GPUQVM, NoiseQVM"
          "\n"
          "Returns:\n"
          "    none"
          "Raises:\n"
          "    run_fail: An error occurred in destroy_quantum_machine\n");

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
		"Read OriginIR file and trans to QProg\n"
		"\n"
		"Args:\n"
		"    file_path: OriginIR file path\n"
		"    machine: initialized quantum machine\n"
		"\n"
		"Returns:\n"
		"    Transformed QProg"
        "Raises:\n"
        "    run_fail: An error occurred in originir_to_qprog\n",
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
		"Read OriginIR file and trans to QProg\n"
		"\n"
		"Args:\n"
		"    file_path: OriginIR file path\n"
		"    machine: initialized quantum machine\n"
		"\n"
		"Returns:\n"
		"    list cotains QProg, qubit_list, cbit_list"
        "Raises:\n"
        "    run_fail: An error occurred in convert_originir_to_qprog\n",
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
		"Trans OriginIR to QProg\n"
		"\n"
		"Args:\n"
		"    originir_str: OriginIR string\n"
		"    machine: initialized quantum machine\n"
		"\n"
		"Returns:\n"
		"    list cotains QProg, qubit_list, cbit_list"
        "Raises:\n"
        "    run_fail: An error occurred in convert_originir_str_to_qprog\n",
		py::return_value_policy::automatic_reference);

	m.def(
		"convert_qasm_to_qprog",
		[](std::string file_path, QuantumMachine *qvm)
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

	m.def(
		"convert_qasm_string_to_qprog",
		[](std::string qasm_str, QuantumMachine *qvm)
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

	/*will delete*/
	m.def("getAllocateCMem",
        &getAllocateCMemNum,
        "Deprecated, use get_allocate_cmem_num instead"
        "\n"
        "Args:\n"
        "   none\n"
        "Returns:\n"
        "    allocate qubit num\n"
        "Raises:\n"
        "    run_fail: An error occurred in get_allocate_cmem_num\n");

	m.def("getAllocateQubitNum",
            &getAllocateQubitNum,
            "Deprecated, use get_allocate_qubit_num instead"
            "\n"
            "Args:\n"
            "   none\n"
            "Returns:\n"
            "    allocate cbit num\n"
            "Raises:\n"
            "    run_fail: An error occurred in get_allocate_qubit_num\n");

	m.def("PMeasure",
        &PMeasure,
        "Deprecated, use pmeasure instead\n"
        "\n"
        "Args:\n"
        "   QVec : pmeasure qubits list\n"
        "   select_num : result select num\n"
        "Returns:\n"
        "    result: pmeasure qubits result\n"
        "Raises:\n"
        "    run_fail: An error occurred in pmeasure\n");

	m.def("PMeasure_no_index",
        &PMeasure_no_index,
        "Deprecated, use pmeasure_no_index instead\n"
        "\n"
        "Args:\n"
        "   QVec : pmeasure qubits list\n"
        "Returns:\n"
        "    result: pmeasure qubits result\n"
        "Raises:\n"
        "    run_fail: An error occurred in pmeasure_no_index\n");

	/* new interface */
	m.def("get_allocate_qubit_num",
        &getAllocateQubitNum,
        "get allocate qubit num\n"
        "\n"
        "Args:\n"
        "   none\n"
        "Returns:\n"
        "    qubit_num : allocate qubit num\n"
        "Raises:\n"
        "    run_fail: An error occurred in get_allocate_qubit_num\n",
        py::return_value_policy::automatic);

	m.def("get_allocate_cmem_num",
        &getAllocateCMem,
        "get allocate cmem num"
        "\n"
        "Args:\n"
        "   none\n"
        "Returns:\n"
        "    qubit_num : allocate cbit num\n"
        "Raises:\n"
        "    run_fail: An error occurred in get_allocate_cmem_num\n",
        py::return_value_policy::automatic);

	m.def("pmeasure", &pMeasure,
		  py::arg("qubit_list"),
		  py::arg("select_max"),
		  "Get the probability distribution over qubits\n"
		  "\n"
		  "Args:\n"
		  "    qubit_list: qubit list to measure"
		  "    select_max: max returned element num in returnd tuple, should in [-1, 1<<len(qubit_list)]\n"
		  "                default is -1, means no limit\n"
		  "\n"
		  "Returns:\n"
		  "    measure result of quantum machine in tuple form",
		  py::return_value_policy::automatic);

	m.def("pmeasure_no_index",
		  &pMeasureNoIndex,
		  py::arg("qubit_list"),
		  "Get the probability distribution over qubits\n"
		  "\n"
		  "Args:\n"
		  "    qubit_list: qubit list to measure"
		  "\n"
		  "Returns:\n"
		  "    measure result of quantum machine in list form",
		  py::return_value_policy::automatic);

	m.def("QOracle",
		py::overload_cast<const QVec &, const QMatrixXcd &, const double> (&QOracle),
		  py::arg("qubit_list"),
		  py::arg("matrix"),
		  py::arg("tol")= 1e-10,
		  "Generate QOracle Gate\n"
		  "\n"
		  "Args:"
		  "    qubit_list: gate in qubit list\n"
		  "    matrix: gate operator matrix\n"
		  "\n"
		  "Return:\n"
		  "    Oracle gate");

	//---------------------------------------------------------------------------------------------------------------------
	/* include\Core\Utilities\QProgInfo\QGateCounter.h */
	/* getQGateNumber is a template function,*/
	m.def("get_qgate_num",
        py::overload_cast<QProg &>(&getQGateNum<QProg>),
        py::arg("quantum_prog"),
        "Count quantum gate num under quantum program, quantum circuit"
        "\n"
        "Args:\n"
        "   prog : quantum_prog\n"
        "Returns:\n"
        "    result: gate count\n"
        "Raises:\n"
        "    run_fail: An error occurred in get_qgate_num\n",
        py::return_value_policy::automatic);

	m.def("get_qgate_num",
        py::overload_cast<QCircuit &>(&getQGateNum<QCircuit>),
        py::arg("quantum_circuit"),
        "Count quantum gate num under quantum program, quantum circuit\n"
        "\n"
        "Args:\n"
        "   circuit : quantum_circuit\n"
        "Returns:\n"
        "    result: gate count\n"
        "Raises:\n"
        "    run_fail: An error occurred in get_qgate_num\n",
        py::return_value_policy::automatic);

	m.def("count_gate",
        py::overload_cast<QProg &>(&getQGateNum<QProg>),
        py::arg("quantum_prog"),
        "Count quantum gate num under quantum program, quantum circuit\n"
        "\n"
        "Args:\n"
        "   prog : quantum_prog\n"
        "Returns:\n"
        "    result: gate count\n"
        "Raises:\n"
        "    run_fail: An error occurred in get_qgate_num\n",
        py::return_value_policy::automatic);

	m.def("count_gate",
        py::overload_cast<QCircuit &>(&getQGateNum<QCircuit>),
        py::arg("quantum_circuit"),
        "Count quantum gate num under quantum program, quantum circuit"
        "\n"
        "Args:\n"
        "   circuit : quantum_circuit\n"
        "Returns:\n"
        "    result: gate count\n"
        "Raises:\n"
        "    run_fail: An error occurred in get_qgate_num\n",
        py::return_value_policy::automatic);

	/* new interface*/
	m.def("count_qgate_num",
		py::overload_cast<QProg&, const GateType> (&count_qgate_num<QProg>),
		py::arg("quantum_prog"),
		py::arg("gtype") = (int)GateType::GATE_UNDEFINED,
		"Count quantum gate num under quantum program\n"
        "\n"
		"Args:\n"
		"    quantum_prog: QProg&\n"
		"    gtype: const GateType\n"
		"\n"
		"Returns:\n"
		"    this GateType quantum gate num",
		py::return_value_policy::automatic);

	m.def("count_qgate_num",
		py::overload_cast<QCircuit&, const GateType> (&count_qgate_num<QCircuit>),
		py::arg("quantum_circuit"),
		py::arg("gtype") = (int)GateType::GATE_UNDEFINED,
		"Count quantum gate num under quantum circuit\n"
        "\n"
		"Args:\n"
		"    quantum_circuit: QCircuit&\n"
		"    gtype: const GateType\n"
		"\n"
		"Returns:\n"
		"    this GateType quantum gate num",
		py::return_value_policy::automatic);



	//---------------------------------------------------------------------------------------------------------------------
	/* include\Core\QuantumCircuit\QProgram.h */
	m.def("CreateEmptyQProg",
		  &CreateEmptyQProg,
		  "Create an empty QProg Container\n"
          "\n"
          "Args:\n"
          "    none\n"
		  "\n"
		  "Returns:\n"
		  "    a empty QProg",
		  py::return_value_policy::automatic);

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
        "Convert QProg to OriginIR string",
        "Args:\n"
        "    qprog: quantum prog\n"
        "    machine: quantum machine\n"
        "\n"
        "Returns:\n"
        "    originir : originir string , see originir indroduction :https://pyqpanda-toturial.readthedocs.io/zh/latest/QProgToOriginIR.html\n",
        py::return_value_policy::automatic_reference);

	//---------------------------------------------------------------------------------------------------------------------
	/* include\Core\Utilities\Compiler\QProgToQuil.h */
	m.def("to_Quil",
		  &transformQProgToQuil,
		  py::arg("qprog"),
		  py::arg("machine"),
		  "Transform QProg to Quil instruction\n"
		  "\n"
		  "Args:\n"
		  "    qprog: QProg\n"
		  "    machine: quantum machine\n"
		  "\n"
		  "Returns:\n"
		  "    Quil instruction string",
		  py::return_value_policy::automatic_reference);

	m.def("transform_qprog_to_quil",
		  &transformQProgToQuil,
		  py::arg("qprog"),
		  py::arg("machine"),
		  "Transform QProg to Quil instruction\n"
		  "\n"
		  "Args:\n"
		  "    qprog: QProg\n"
		  "    machine: quantum machine\n"
		  "\n"
		  "Returns:\n"
		  "    Quil instruction string",
		  py::return_value_policy::automatic_reference);

	m.def("convert_qprog_to_quil",
		  &convert_qprog_to_quil,
		  py::arg("qprog"),
		  py::arg("machine"),
          "convert QProg to Quil instruction\n"
          "\n"
          "Args:\n"
          "    qprog: QProg\n"
          "    machine: quantum machine\n"
          "\n"
          "Returns:\n"
          "    Quil instruction string",
		  py::return_value_policy::automatic_reference);

	//---------------------------------------------------------------------------------------------------------------------
	/* include\Core\Utilities\QProgInfo\QProgClockCycle.h */
	m.def("get_qprog_clock_cycle",
		  &get_qprog_clock_cycle,
		  py::arg("qprog"),
		  py::arg("machine"),
		  py::arg("optimize") = false,
		  "Get Quantum Program Clock Cycle\n"
		  "\n"
		  "Args:\n"
		  "    qprog: quantum program\n"
		  "    machine: quantum machine\n"
		  "    optimize: optimize qprog\n"
		  "\n"
		  "Returns:\n"
		  "    QProg time comsume, no unit, not in seconds",
		  py::return_value_policy::automatic_reference);

	m.def(
		"get_clock_cycle",
		[](QProg prog)
		{
			extern QuantumMachine *global_quantum_machine;
			return getQProgClockCycle(prog, global_quantum_machine);
		},
		py::arg("qpog"),
		"Get quantum program clock cycle"
        "\n"
        "Args:\n"
        "    qprog: QProg\n"
        "\n"
        "Returns:\n"
        "    clock_cycle",
		py::return_value_policy::automatic_reference);

	//---------------------------------------------------------------------------------------------------------------------
	/* include\Core\Utilities\Compiler\QProgStored.h */
	m.def("transform_qprog_to_binary",
		  py::overload_cast<QProg &, QuantumMachine *>(&transformQProgToBinary),
		  py::arg("qprog"),
		  py::arg("machine"),
		  "Get quantum program binary data\n"
		  "\n"
		  "Args:\n"
		  "    qprog: QProg\n"
		  "    machine: quantum machine\n"
		  "\n"
		  "Returns:\n"
		  "    binary data in list",
		  py::return_value_policy::automatic_reference);

	m.def("transform_qprog_to_binary",
		  py::overload_cast<QProg &, QuantumMachine *, const string &>(&transformQProgToBinary),
		  py::arg("qprog"),
		  py::arg("machine"),
		  py::arg("fname"),
		  "Save quantum program to file as binary data\n"
		  "\n"
		  "Args:\n"
		  "    qprog: QProg\n"
		  "    machine: quantum machine\n"
		  "    fname: save to file name\n");

	m.def(
		"get_bin_data",
		[](QProg prog)
		{
			extern QuantumMachine *global_quantum_machine;
			return transformQProgToBinary(prog, global_quantum_machine);
		},
		py::arg("qprog"),
        "Get quantum program binary data\n"
        "\n"
        "Args:\n"
        "    qprog: QProg\n"
        "    machine: quantum machine\n"
        "\n"
        "Returns:\n"
        "    binary data in list",
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
		"Parse binary data transfor to quantum program"
        "\n"
        "Args:\n"
        "    bin_data: binary data stores quantum prog information\n"
        "    qubit_list: quantum qubits list \n"
        "    cbit_list: quantum cbits list\n"
        "    qprog: quantum prog\n"
        "\n"
        "Returns:\n"
        "    prog\n",
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
		"Transfor quantum program to string"
        "\n"
        "Args:\n"
        "    machine: quantum machine\n"
        "    qprog: quantum prog\n"
        "\n"
        "Returns:\n"
        "    string for bin_str\n",
		py::return_value_policy::automatic);

	m.def("convert_qprog_to_binary",
        py::overload_cast<QProg &, QuantumMachine *>(&convert_qprog_to_binary),
        py::arg("qprog"),
        py::arg("machine"),
        "Trans quantum program to binary data"
        "\n"
        "Args:\n"
        "    machine: quantum machine\n"
        "    qprog: quantum prog\n"
        "\n"
        "Returns:\n"
        "    string for binary data\n",
        py::return_value_policy::automatic_reference);

	m.def("convert_qprog_to_binary",
        py::overload_cast<QProg &, QuantumMachine *, const string &>(&convert_qprog_to_binary),
        py::arg("qprog"),
        py::arg("machine"),
        py::arg("fname"),
        "Store quantum program in binary file "
        "\n"
        "Args:\n"
        "    machine: quantum machine\n"
        "    qprog: quantum prog\n"
        "    fname: binary data file name\n"
        "\n"
        "Returns:\n"
        "    none\n",
        py::return_value_policy::automatic_reference);

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
		"Parse binary data trans to quantum program\n"
		"\n"
		"Args:\n"
		"    machine: quantum machine\n"
		"    data: list contains binary data from transform_qprog_to_binary()\n"
		"\n"
		"Returns:\n"
		"    QProg",
		py::return_value_policy::automatic_reference);

	//---------------------------------------------------------------------------------------------------------------------
	/* include\Core\Utilities\Compiler\QProgDataParse.h */
	m.def("transform_qprog_to_originir",
		  py::overload_cast<QProg &, QuantumMachine *>(&transformQProgToOriginIR<QProg>),
		  py::arg("qprog"),
		  py::arg("machine"),
		  "Quantum program transform to OriginIR string\n"
		  "\n"
		  "Args:\n"
		  "    qprog: QProg\n"
		  "    machine: quantum machine\n"
		  "\n"
		  "Returns:\n"
		  "    OriginIR instruction string",
		  py::return_value_policy::automatic_reference);

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
		"Transform OriginIR to QProg\n"
		"\n"
		"Args:\n"
		"    fname: file sotred OriginIR instruction"
		"    machine: quantum machine\n"
		"\n"
		"Returns:\n"
		"    QProg",
		py::return_value_policy::automatic_reference);

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
		"Apply QGate to qubits\n"
		"\n"
		"Args:\n"
		"    qubit_list: qubit list\n"
		"    func_obj: QGate(Qubit) like function object accept Qubit as argument\n"
		"\n"
		"Returns:\n"
		"    QCircuit contain QGate operation on all qubit",
		py::return_value_policy::reference);

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
		"Apply QGate to qubits\n"
		"\n"
		"Args:\n"
		"    qubit_addr_list: qubit address list\n"
		"    func_obj: QGate(int) like function object accept Qubit address as argument\n"
		"\n"
		"Returns:\n"
		"    QCircuit contain QGate operation on all qubit",
		py::return_value_policy::reference);

	//---------------------------------------------------------------------------------------------------------------------
	/* include\Core\QuantumCircuit\ControlFlow.h */
	/* will delete */
	m.def("CreateWhileProg",
        &CreateWhileProg,
        py::arg("classical_condition"),
        py::arg("true_node"),
        "Create a WhileProg"
        "\n"
        "Args:\n"
        "    classical_condition: quatum cbit\n"
        "    true_node: quantum QWhile qnode\n"
        "\n"
        "Returns:\n"
        "    a WhileProg",
        py::return_value_policy::automatic);

	m.def("CreateIfProg",
        py::overload_cast<ClassicalCondition, QProg>(&CreateIfProg),
        py::arg("classical_condition"),
        py::arg("true_node"),
        "Create a classical quantum IfProg\n"
        "\n"
        "Args:\n"
        "    classical_condition: quatum cbit\n"
        "    true_node: quantum IfProg qnode\n"
        "\n"
        "Returns:\n"
        "    a classical quantum IfProg",
        py::return_value_policy::automatic);

	m.def("CreateIfProg",
        py::overload_cast<ClassicalCondition, QProg, QProg>(&CreateIfProg),
        py::arg("classical_condition"),
        py::arg("true_node"),
        py::arg("false_node"),
        "Create a classical quantum IfProg\n"
        "\n"
        "Args:\n"
        "    classical_condition: quatum cbit\n"
        "    true_node: quantum IfProg true qnode\n"
        "    false_node: quantum IfProg false qnode\n"
        "\n"
        "Returns:\n"
        "    a classical quantum IfProg",
        py::return_value_policy::automatic);

	/* new interface */
	m.def("create_while_prog",
		  &createWhileProg,
		  py::arg("classical_condition"),
		  py::arg("true_node"),
        "Create a WhileProg\n"
        "\n"
        "Args:\n"
        "    classical_condition: quatum cbit\n"
        "    true_node: quantum QWhile qnode\n"
        "\n"
        "Returns:\n"
        "    a WhileProg",
		  py::return_value_policy::automatic);

	m.def("create_if_prog",
		  py::overload_cast<ClassicalCondition, QProg>(&CreateIfProg),
		  py::arg("classical_condition"),
		  py::arg("true_node"),
        "Create a classical quantum IfProg\n"
        "\n"
        "Args:\n"
        "    classical_condition: quatum cbit\n"
        "    true_node: quantum IfProg qnode\n"
        "\n"
        "Returns:\n"
        "    a classical quantum IfProg",
		  py::return_value_policy::automatic);

	m.def("create_if_prog",
		  py::overload_cast<ClassicalCondition, QProg, QProg>(&CreateIfProg),
		  py::arg("classical_condition"),
		  py::arg("true_node"),
		  py::arg("false_node"),
        "Create a classical quantum IfProg\n"
        "\n"
        "Args:\n"
        "    classical_condition: quatum cbit\n"
        "    true_node: quantum IfProg true qnode\n"
        "    false_node: quantum IfProg false qnode\n"
        "\n"
        "Returns:\n"
        "    a classical quantum IfProg",
		  py::return_value_policy::automatic);

	//---------------------------------------------------------------------------------------------------------------------
	/* include\Core\QuantumCircuit\QCircuit.h */
	/* will delete */
	m.def("CreateEmptyCircuit",
		  &CreateEmptyCircuit,
		  "Create an empty QCircuit Container\n"
        "\n"
        "Args:\n"
        "    none\n"
        "\n"
        "Returns:\n"
        "    a empty QCircuit",
		  py::return_value_policy::automatic);

	/* new interface */
	m.def("create_empty_circuit",
		  &createEmptyCircuit,
        "Create an empty QCircuit Container\n"
        "\n"
        "Args:\n"
        "    none\n"
        "\n"
        "Returns:\n"
        "    a empty QCircuit",
		  py::return_value_policy::automatic);

	//---------------------------------------------------------------------------------------------------------------------
	/* include\Core\QuantumCircuit\QProgram.h */
	m.def("create_empty_qprog",
		  &createEmptyQProg,
        "Create an empty QProg Container\n"
        "\n"
        "Args:\n"
        "    none\n"
        "\n"
        "Returns:\n"
        "    a empty QProg",
		  py::return_value_policy::automatic);

	//---------------------------------------------------------------------------------------------------------------------
	/* include\Core\QuantumCircuit\QuantumMeasure.h */
	m.def("Measure",
		  py::overload_cast<Qubit *, ClassicalCondition>(&Measure),
		  py::arg("qubit"),
		  py::arg("cbit"),
        "Create an measure node\n"
        "\n"
        "Args:\n"
        "    qubit : measure qubit\n"
        "    cbit : cbit stores quantum measure result\n"
        "\n"
        "Returns:\n"
        "    a quantum measure node",
		  py::return_value_policy::automatic);

    m.def("Measure",
        py::overload_cast<Qubit *, CBit*>(&Measure),
        py::arg("qubit"),
        py::arg("cbit"),
        "Create an measure node\n"
        "\n"
        "Args:\n"
        "    qubit : measure qubit\n"
        "    cbit : cbit stores quantum measure result\n"
        "\n"
        "Returns:\n"
        "    a quantum measure node",
        py::return_value_policy::automatic);

	m.def("Measure",
		  py::overload_cast<int, int>(&Measure),
		  py::arg("qubit_addr"),
		  py::arg("cbit_addr"),
        "Create an measure node\n"
        "\n"
        "Args:\n"
        "    qubit : measure qubit\n"
        "    cbit : cbit stores quantum measure result\n"
        "\n"
        "Returns:\n"
        "    a quantum measure node",
		  py::return_value_policy::automatic);

	m.def("measure_all",
		  py::overload_cast<const QVec &, const std::vector<ClassicalCondition> &>(&MeasureAll),
		  py::arg("qubit_list"),
		  py::arg("cbit_list"),
        "Create a list of measure node\n"
        "\n"
        "Args:\n"
        "    qubit_list : measure qubits\n"
        "    cbit_list : cbits stores quantum measure result\n"
        "\n"
        "Returns:\n"
        "    a list of measure node",
		  py::return_value_policy::automatic);

	m.def("measure_all",
		  py::overload_cast<const std::vector<int> &, const std::vector<int> &>(&MeasureAll),
		  py::arg("qubit_addr_list"),
		  py::arg("cbit_addr_list"),
        "Create a list of measure node\n"
        "\n"
        "Args:\n"
        "    qubit_list : measure qubits list\n"
        "    cbit_list : cbits stores quantum measure result\n"
        "\n"
        "Returns:\n"
        "    a list of measure node",
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
        "Create a BARRIER gate\n"
        "\n"
        "Args:\n"
        "    qubit : measure qubit\n"
        "\n"
        "Returns:\n"
        "    a BARRIER node",
        py::return_value_policy::automatic);

	m.def("BARRIER",
        py::overload_cast<int>(&BARRIER),
        py::arg("qubit_list"),
        "Create a BARRIER gate\n"
        "\n"
        "Args:\n"
        "    qubit : measure qubit\n"
        "\n"
        "Returns:\n"
        "    a BARRIER node",
        py::return_value_policy::automatic);

	m.def("BARRIER",
        py::overload_cast<QVec>(&BARRIER),
        py::arg("qubit_list"),
        "Create a BARRIER gate\n"
        "\n"
        "Args:\n"
        "    qubit_list : measure qubits list\n"
        "\n"
        "Returns:\n"
        "    a BARRIER node",
        py::return_value_policy::automatic);

	m.def("BARRIER",
        py::overload_cast<std::vector<int>>(&BARRIER),
        py::arg("qubit_addr_list"),
        "Create a BARRIER gate\n"
        "\n"
        "Args:\n"
        "    qubit_list : measure qubits list\n"
        "\n"
        "Returns:\n"
        "    a BARRIER node",
        py::return_value_policy::automatic);

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
        "Create a U4 gate\n"
        "\n"
        "Args:\n"
        "    matrix : U4 gate matrix\n"
        "    qubit : U4 gate target qubit\n"
        "\n"
        "Returns:\n"
        "    a U4 node",
        py::return_value_policy::automatic);

	m.def("U4",
        py::overload_cast<double, double, double, double, Qubit *>(&U4),
        py::arg("alpha_angle"),
        py::arg("beta_angle"),
        py::arg("gamma_angle"),
        py::arg("delta_angle"),
        py::arg("qubit"),
        "Create a U4 gate\n"
        "\n"
        "Args:\n"
        "    double : u4 gate alpha angle\n"
        "    double : u4 gate beta angle\n"
        "    double : u4 gate gamma angle\n"
        "    double : u4 gate delta angle\n"
        "    qubit : U4 gate target qubit\n"
        "\n"
        "Returns:\n"
        "    a U4 node",
        py::return_value_policy::automatic);

	m.def("U4",
        py::overload_cast<Qubit *, QStat &>(&U4),
        py::arg("qubit"),
        py::arg("matrix"),
        "Create a U4 gate\n"
        "\n"
        "Args:\n"
        "    matrix : U4 gate matrix\n"
        "    qubit : U4 gate target qubit\n"
        "\n"
        "Returns:\n"
        "    a U4 node",
        py::return_value_policy::automatic);

	m.def("U4",
        py::overload_cast<const QVec &, QStat &>(&U4),
        py::arg("qubit_list"),
        py::arg("matrix"),
        "Create a U4 gate\n"
        "\n"
        "Args:\n"
        "    matrix : U4 gate matrix\n"
        "    qubit_list : U4 gate target qubit_list\n"
        "\n"
        "Returns:\n"
        "    a U4 node",
        py::return_value_policy::automatic);

	m.def("U4",
        py::overload_cast<int, QStat &>(&U4),
        py::arg("qubit_addr"),
        py::arg("matrix"),
        "Create a U4 gate\n"
        "\n"
        "Args:\n"
        "    matrix : U4 gate matrix\n"
        "    qubit : U4 gate target qubit\n"
        "\n"
        "Returns:\n"
        "    a U4 node",
        py::return_value_policy::automatic);

	m.def("U4",
        py::overload_cast<const std::vector<int> &, QStat &>(&U4),
        py::arg("qubit_addr_list"),
        py::arg("matrix"),
        "Create a U4 gate\n"
        "\n"
        "Args:\n"
        "    matrix : U4 gate matrix\n"
        "    qubit_list : U4 gate target qubit_list\n"
        "\n"
        "Returns:\n"
        "    a U4 node",
        py::return_value_policy::automatic);

	m.def("U4",
        py::overload_cast<Qubit *, double, double, double, double>(&U4),
        py::arg("qubit"),
        py::arg("alpha_anlge"),
        py::arg("beta_anlge"),
        py::arg("gamma_anlge"),
        py::arg("delta_anlge"),
        "Create a U4 gate\n"
        "\n"
        "Args:\n"
        "    double : u4 gate alpha angle\n"
        "    double : u4 gate beta angle\n"
        "    double : u4 gate gamma angle\n"
        "    double : u4 gate delta angle\n"
        "    qubit : U4 gate target qubit\n"
        "\n"
        "Returns:\n"
        "    a U4 node",
        py::return_value_policy::automatic);

	m.def("U4",
        py::overload_cast<const QVec &, double, double, double, double>(&U4),
        py::arg("qubit_list"),
        py::arg("alpha_angle"),
        py::arg("beta_angle"),
        py::arg("gamma_angle"),
        py::arg("delta_angle"),
        "Create a U4 gate\n"
        "\n"
        "Args:\n"
        "    double : u4 gate alpha angle\n"
        "    double : u4 gate beta angle\n"
        "    double : u4 gate gamma angle\n"
        "    double : u4 gate delta angle\n"
        "    qubit_list : U4 gate target qubit_list\n"
        "\n"
        "Returns:\n"
        "    a U4 node",
        py::return_value_policy::automatic);

	m.def("U4",
        py::overload_cast<int, double, double, double, double>(&U4),
        py::arg("qubit_addr"),
        py::arg("alpha_anlge"),
        py::arg("beta_anlge"),
        py::arg("gamma_anlge"),
        py::arg("delta_anlge"),
        "Create a U4 gate\n"
        "\n"
        "Args:\n"
        "    double : u4 gate alpha angle\n"
        "    double : u4 gate beta angle\n"
        "    double : u4 gate gamma angle\n"
        "    double : u4 gate delta angle\n"
        "    qubit : U4 gate target qubit\n"
        "\n"
        "Returns:\n"
        "    a U4 node",
        py::return_value_policy::automatic);

	m.def("U4",
        py::overload_cast<const std::vector<int> &, double, double, double, double>(&U4),
        py::arg("qubit_addr_list"),
        py::arg("alpha_anlge"),
        py::arg("beta_anlge"),
        py::arg("gamma_anlge"),
        py::arg("delta_anlge"),
        "Create a U4 gate\n"
        "\n"
        "Args:\n"
        "    double : u4 gate alpha angle\n"
        "    double : u4 gate beta angle\n"
        "    double : u4 gate gamma angle\n"
        "    double : u4 gate delta angle\n"
        "    qubit_addr_list : U4 gate target qubit_addr_list\n"
        "\n"
        "Returns:\n"
        "    a U4 node",
        py::return_value_policy::automatic);

	m.def("CU",
        py::overload_cast<double, double, double, double, Qubit *, Qubit *>(&CU),
        py::arg("alpha_angle"),
        py::arg("beta_angle"),
        py::arg("gamma_angle"),
        py::arg("delta_angle"),
        py::arg("control_qubit"),
        py::arg("target_qubit"),
        "Create a CU gate\n"
        "\n"
        "Args:\n"
        "    double : u4 gate alpha angle\n"
        "    double : u4 gate beta angle\n"
        "    double : u4 gate gamma angle\n"
        "    double : u4 gate delta angle\n"
        "    qubit : control qubit \n"
        "    qubit : target qubit\n"
        "\n"
        "Returns:\n"
        "    a CU node",
        py::return_value_policy::automatic);

	m.def("CU",
        py::overload_cast<double, double, double, double, const QVec &, const QVec &>(&CU),
        py::arg("alpha_angle"),
        py::arg("beta_angle"),
        py::arg("gamma_angle"),
        py::arg("delta_angle"),
        py::arg("control_qubit_list"),
        py::arg("target_qubi_list"),
        "Create a CU gate\n"
        "\n"
        "Args:\n"
        "    double : u4 gate alpha angle\n"
        "    double : u4 gate beta angle\n"
        "    double : u4 gate gamma angle\n"
        "    double : u4 gate delta angle\n"
        "    control_qubit_list : control qubit list \n"
        "    target_qubi_list : target qubit list\n"
        "\n"
        "Returns:\n"
        "    a CU node",
        py::return_value_policy::automatic);

	m.def("CU",
        py::overload_cast<QStat &, Qubit *, Qubit *>(&CU),
        py::arg("matrix"),
        py::arg("control_qubit"),
        py::arg("target_qubit"),
        "Create a CU gate\n"
        "\n"
        "Args:\n"
        "    matrix : CU gate matrix\n"
        "    qubit : control qubit \n"
        "    qubit : target qubit\n"
        "\n"
        "Returns:\n"
        "    a CU node",
        py::return_value_policy::automatic);

	m.def("CU",
        py::overload_cast<QStat &, const QVec &, const QVec &>(&CU),
        py::arg("matrix"),
        py::arg("control_qubit_list"),
        py::arg("target_qubit_list"),
        "Create a CU gate\n"
        "\n"
        "Args:\n"
        "    matrix : CU gate matrix\n"
        "    control_qubit_list : control qubit list \n"
        "    target_qubi_list : target qubit list\n"
        "\n"
        "Returns:\n"
        "    a CU node",
        py::return_value_policy::automatic);

	TempHelper_SWAP<>::export_doubleBitGate(m);
	TempHelper_iSWAP<>::export_doubleBitGate(m);
	TempHelper_iSWAP_2<double>::export_doubleBitGate(m);
	TempHelper_SqiSWAP<>::export_doubleBitGate(m);

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
        "Create a CU gate\n"
        "\n"
        "Args:\n"
        "    double : u4 gate alpha angle\n"
        "    double : u4 gate beta angle\n"
        "    double : u4 gate gamma angle\n"
        "    double : u4 gate delta angle\n"
        "    qubit : control qubit \n"
        "    qubit : target qubit\n"
        "\n"
        "Returns:\n"
        "    a CU node",
        py::return_value_policy::automatic);

	m.def("CU",
        py::overload_cast<const QVec &, const QVec &, double, double, double, double>(&CU),
        py::arg("control_qubit_list"),
        py::arg("target_qubit_list"),
        py::arg("alpha_angle"),
        py::arg("beta_angle"),
        py::arg("gamma_angle"),
        py::arg("delta_angle"),
        "Create a CU gate\n"
        "\n"
        "Args:\n"
        "    double : u4 gate alpha angle\n"
        "    double : u4 gate beta angle\n"
        "    double : u4 gate gamma angle\n"
        "    double : u4 gate delta angle\n"
        "    control_qubit_list : control qubit list \n"
        "    target_qubi_list : target qubit list\n"
        "\n"
        "Returns:\n"
        "    a CU node",
        py::return_value_policy::automatic);

	m.def("CU",
        py::overload_cast<int, int, double, double, double, double>(&CU),
        py::arg("control_qubit_addr"),
        py::arg("target_qubit_addr"),
        py::arg("alpha_angle"),
        py::arg("beta_angle"),
        py::arg("gamma_angle"),
        py::arg("delta_angle"),
        "Create a CU gate\n"
        "\n"
        "Args:\n"
        "    double : u4 gate alpha angle\n"
        "    double : u4 gate beta angle\n"
        "    double : u4 gate gamma angle\n"
        "    double : u4 gate delta angle\n"
        "    qubit addr: control qubit addr\n"
        "    qubit addr: target qubit addr\n"
        "\n"
        "Returns:\n"
        "    a CU node",
        py::return_value_policy::automatic);

	m.def("CU",
        py::overload_cast<const std::vector<int> &, const std::vector<int> &, double, double, double, double>(&CU),
        py::arg("control_qubit_addr_list"),
        py::arg("target_qubit_addr_list"),
        py::arg("alpha_angle"),
        py::arg("beta_angle"),
        py::arg("gamma_angle"),
        py::arg("delta_angle"),
        "Create a CU gate\n"
        "\n"
        "Args:\n"
        "    double : u4 gate alpha angle\n"
        "    double : u4 gate beta angle\n"
        "    double : u4 gate gamma angle\n"
        "    double : u4 gate delta angle\n"
        "    qubit addr list: control qubit addr list\n"
        "    qubit addr list: target qubit addr list\n"
        "\n"
        "Returns:\n"
        "    a CU node",
        py::return_value_policy::automatic);

	m.def("CU",
        py::overload_cast<Qubit *, Qubit *, QStat &>(&CU),
        py::arg("control_qubit"),
        py::arg("target_qubit"),
        py::arg("matrix"),
        "Create a CU gate\n"
        "\n"
        "Args:\n"
        "    matrix : CU gate matrix\n"
        "    qubit : control qubit \n"
        "    qubit : target qubit\n"
        "\n"
        "Returns:\n"
        "    a CU node",
        py::return_value_policy::automatic);

	m.def("CU",
        py::overload_cast<const QVec &, const QVec &, QStat &>(&CU),
        py::arg("control_qubit_list"),
        py::arg("target_qubit_list"),
        py::arg("matrix"),
        "Create a CU gate\n"
        "\n"
        "Args:\n"
        "    matrix : CU gate matrix\n"
        "    qubit list: control qubit list\n"
        "    qubit list: target qubit list\n"
        "\n"
        "Returns:\n"
        "    a CU node",
        py::return_value_policy::automatic);

	m.def("CU",
        py::overload_cast<int, int, QStat &>(&CU),
        py::arg("control_qubit_addr"),
        py::arg("target_qubit_addr"),
        py::arg("matrix"),
        "Create a CU gate\n"
        "\n"
        "Args:\n"
        "    matrix : CU gate matrix\n"
        "    qubit addr: control qubit addr\n"
        "    qubit addr: target qubit addr\n"
        "\n"
        "Returns:\n"
        "    a CU node",
        py::return_value_policy::automatic);

	m.def("CU",
        py::overload_cast<const std::vector<int> &, const std::vector<int> &, QStat &>(&CU),
        py::arg("control_qubit_addr_list"),
        py::arg("target_qubit_addr_list"),
        py::arg("matrix"),
        "Create a CU gate\n"
        "\n"
        "Args:\n"
        "    matrix : CU gate matrix\n"
        "    qubit addr list: control qubit addr list\n"
        "    qubit addr list: target qubit addr list\n"
        "\n"
        "Returns:\n"
        "    a CU node",
        py::return_value_policy::automatic);

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
		  "Create a Toffoli gate",
		  py::return_value_policy::automatic);

	m.def("Toffoli",
		  py::overload_cast<int, int, int>(&Toffoli),
		  py::arg("control_qubit_addr_first"),
		  py::arg("control_qubit_addr_second"),
		  py::arg("target_qubit_addr"),
		  "Create a Toffoli gate",
		  py::return_value_policy::automatic);

	TempHelper_QDouble<QStat &>::export_doubleBitGate(m);

	//---------------------------------------------------------------------------------------------------------------------
	/* include\Core\QuantumCircuit\QReset.h */
	m.def("Reset",
		  py::overload_cast<Qubit *>(&Reset),
		  py::arg("qubit"),
		  "Create a Reset node",
		  py::return_value_policy::automatic);

	m.def("Reset",
		  py::overload_cast<int>(&Reset),
		  py::arg("qubit_addr"),
		  "Create a Reset node",
		  py::return_value_policy::automatic);

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
		"Print matrix element\n"
		"\n"
		"Args:\n"
		"    matrix: matrix\n"
		"    precision: double value to string cutoff precision\n"
		"\n"
		"Returns:\n"
		"    string of matrix",
		py::return_value_policy::automatic);

	//---------------------------------------------------------------------------------------------------------------------
	/* include\Core\Utilities\QProgInfo\QCircuitInfo.h */
	m.def("is_match_topology",
		  &isMatchTopology,
		  py::arg("gate"),
		  py::arg("topo"),
		  "Judge the QGate if match the target topologic structure of quantum circuit\n"
		  "\n"
		  "Args:\n"
		  "    gate: QGate\n"
		  "    topo: the target topologic structure of quantum circuit\n"
		  "\n"
		  "Returns:\n"
		  "    true if match, else false",
		  py::return_value_policy::automatic);

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
		"Get the adjacent quantum gates's(the front one and the back one) typeinfo from QProg\n"
		"\n"
		"Args:\n"
		"    qprog: target quantum program\n"
		"    node_iter:  gate node iter in qprog\n"
		"\n"
		"Returns:\n"
		"    the front one and back node info of node_iter in qprog",
		py::return_value_policy::automatic);

	m.def("is_swappable",
        &isSwappable,
        py::arg("prog"),
        py::arg("nodeitr_1"),
        py::arg("nodeitr_2"),
        "Judge the specialed two NodeIters in qprog whether can be exchanged\n"
        "\n"
        "Args:\n"
        "    qprog: target quantum program\n"
        "    node_iter1:  node iter 1 in qprog\n"
        "    node_iter2:  node iter 2 in qprog\n"
        "\n"
        "Returns:\n"
        "    true ir false for two NodeIters in qprog whether can be exchanged",
        py::return_value_policy::automatic);

	m.def("is_supported_qgate_type",
        &isSupportedGateType,
        py::arg("nodeitr"),
        "Judge if the target node is a QGate type\n"
        "\n"
        "Args:\n"
        "    node_iter:  node iter  in qprog\n"
        "\n"
        "Returns:\n"
        "    true ir false if the target node is a QGate type",
        py::return_value_policy::automatic);

	m.def("get_matrix",
		  &getCircuitMatrix,
		  py::arg("qprog"),
		  py::arg("positive_seq") = false,
		  py::arg_v("nodeitr_start", NodeIter(), "NodeIter()"),
		  py::arg_v("nodeitr_end", NodeIter(), "NodeIter()"),
		  "Get the target matrix between the input two Nodeiters\n"
		  "\n"
		  "Args:\n"
		  "    qprog: quantum program\n"
		  "    positive_seq: Qubit order of output matrix\n"
		  "                  true for positive sequence(q0q1q2), false for inverted order(q2q1q0), default is false\n"
		  "    nodeiter_start: the start NodeIter\n"
		  "    nodeiter_end: the end NodeIter\n"
		  "\n"
		  "Returns:\n"
		  "    target matrix include all the QGate's matrix (multiply)",
		  py::return_value_policy::automatic);

    m.def("get_unitary",
        &get_unitary,
        py::arg("qprog"),
        py::arg("positive_seq") = false,
        py::arg_v("nodeitr_start", NodeIter(), "NodeIter()"),
        py::arg_v("nodeitr_end", NodeIter(), "NodeIter()"),
        "Get the target matrix between the input two Nodeiters\n"
        "\n"
        "Args:\n"
        "    qprog: quantum program\n"
        "    positive_seq: Qubit order of output matrix\n"
        "                  true for positive sequence(q0q1q2), false for inverted order(q2q1q0), default is false\n"
        "    nodeiter_start: the start NodeIter\n"
        "    nodeiter_end: the end NodeIter\n"
        "\n"
        "Returns:\n"
        "    target matrix include all the QGate's matrix (multiply)",
        py::return_value_policy::automatic);

	m.def(
        "get_all_used_qubits",
        [](QProg prog)
        {
	        QVec vec_qubits_in_use;
	        get_all_used_qubits(prog, vec_qubits_in_use);
	        return vec_qubits_in_use;
        },
        py::arg("qprog"),
        "Get all the used quantum bits in the input prog"
            "\n"
            "Args:\n"
            "    qprog: quantum program\n"
            "\n"
            "Returns:\n"
            "    all used qubits",
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
        "Get all the used quantum bits addr in the input prog"
        "\n"
        "Args:\n"
        "    qprog: quantum program\n"
        "\n"
        "Returns:\n"
        "    all used qubits",
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
        "Get valid QGates and valid single bit QGate type\n"
        "\n"
        "Args:\n"
        "    single_gates: single gates list\n"
        "\n"
        "Returns:\n"
        "    result list\n",
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
        "Get valid QGates and valid double bit QGate type\n"
        "\n"
        "Args:\n"
        "    double_gates: double gates list\n"
        "\n"
        "Returns:\n"
        "    result list\n",
		py::return_value_policy::automatic_reference);

	//---------------------------------------------------------------------------------------------------------------------
	/* include\Core\Utilities\QProgInfo\QGateCompare.h */
	m.def("get_unsupport_qgate_num",
		  &getUnsupportQGateNum<QProg>,
		  py::arg("qprog"),
		  py::arg("support_gates"),
		  "Count quantum program unsupported gate numner\n"
          "\n"
          "Args:\n"
          "    qprog: quantum prog\n"
          "    support_gates: support_gates\n"
          "\n"
          "Returns:\n"
          "    unsupported gate numner\n",
		  py::return_value_policy::automatic);

	m.def("get_qgate_num",
        &getQGateNum<QProg>,
        py::arg("qprog"),
        "Count quantum gate num under quantum program\n"
        "\n"
        "Args:\n"
        "    qprog: quantum prog\n"
        "\n"
        "Returns:\n"
        "    quantum gate num under quantum program\n",
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
        "Flatten quantum program\n"
        "\n"
        "Args:\n"
        "    qprog: quantum prog\n"
        "\n"
        "Returns:\n"
        "    none\n",
        py::return_value_policy::automatic);

	m.def("flatten",
        py::overload_cast<QCircuit &>(&flatten),
        py::arg("qcircuit"),
        "Flatten quantum circuit\n"
        "\n"
        "Args:\n"
        "    qprog: quantum circuit\n"
        "\n"
        "Returns:\n"
        "    none\n",
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
        "Parse  binary data to quantum program\n"
        "\n"
        "Args:\n"
        "    machine: quantum machine\n"
        "    data: quantum prog data\n"
        "\n"
        "Returns:\n"
        "    quantum prog\n"
        "Raises:\n"
        "    run_fail: An error occurred in convert_binary_data_to_qprog\n",
        py::return_value_policy::automatic_reference);

	//---------------------------------------------------------------------------------------------------------------------
	/* include\Core\Utilities\Compiler\QProgToQASM.h */
	m.def("convert_qprog_to_qasm",
        &convert_qprog_to_qasm,
        py::arg("qprog"),
        py::arg("machine"),
        "Convert QProg to QASM instruction string\n"
        "\n"
        "Args:\n"
        "    machine: quantum machine\n"
        "    qprog: quantum prog \n"
        "\n"
        "Returns:\n"
        "    qsm string stores prog\n"
        "Raises:\n"
        "    run_fail: An error occurred in convert_qprog_to_qasm\n",
        py::return_value_policy::automatic_reference);

	//---------------------------------------------------------------------------------------------------------------------
	/* include\Core\Utilities\QProgTransform\QProgToQGate.h */
	m.def("cast_qprog_qgate",
        &cast_qprog_qgate,
        py::arg("qprog"),
        "Cast QProg to QGate\n"
        "\n"
        "Args:\n"
        "    qprog: quantum prog \n"
        "\n"
        "Returns:\n"
        "    none\n"
        "Raises:\n"
        "    run_fail: An error occurred in cast_qprog_qgate\n",
        py::return_value_policy::automatic_reference);

	//---------------------------------------------------------------------------------------------------------------------
	/* include\Core\Utilities\QProgTransform\QProgToQMeasure.h */
	m.def("cast_qprog_qmeasure",
        &cast_qprog_qmeasure,
        py::arg("qprog"),
        "Cast QProg to QMeasure\n"
        "\n"
        "Args:\n"
        "    qprog: quantum prog \n"
        "\n"
        "Returns:\n"
        "    none\n"
        "Raises:\n"
        "    run_fail: An error occurred in cast_qprog_qmeasure\n",
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
        "Cast QProg to QCircuit\n"
        "\n"
        "Args:\n"
        "    qprog: quantum prog \n"
        "\n"
        "Returns:\n"
        "    none\n"
        "Raises:\n"
        "    run_fail: An error occurred in cast_qprog_qcircuit\n",
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
		"Judge QProg/QCircuit matches the topology of the physical qubits\n"
        "\n"
        "Args:\n"
        "    qprog: quantum prog \n"
        "    qubit_list: qubits list in quantum prog \n"
        "    machine: quantum machine \n"
        "    confing_file: match configfilepath, default is QPandaConfig.json \n"
        "\n"
        "Returns:\n"
        "    result data\n"
        "Raises:\n"
        "    run_fail: An error occurred in topology_match\n",
		py::return_value_policy::automatic_reference);

	m.def("add",
		  [](ClassicalCondition a, ClassicalCondition b)
		  {
			  return a + b;
		  });

	m.def("add",
		  [](ClassicalCondition a, cbit_size_t b)
		  {
			  return a + b;
		  });

	m.def("add",
		  [](cbit_size_t a, ClassicalCondition b)
		  {
			  return a + b;
		  });

	m.def("sub",
		  [](ClassicalCondition a, ClassicalCondition b)
		  {
			  return a - b;
		  });

	m.def("sub",
		  [](ClassicalCondition a, cbit_size_t b)
		  {
			  return a - b;
		  });

	m.def("sub",
		  [](cbit_size_t a, ClassicalCondition b)
		  {
			  return a - b;
		  });

	m.def("mul",
		  [](ClassicalCondition a, ClassicalCondition b)
		  {
			  return a * b;
		  });

	m.def("mul",
		  [](ClassicalCondition a, cbit_size_t b)
		  {
			  return a * b;
		  });

	m.def("mul",
		  [](cbit_size_t a, ClassicalCondition b)
		  {
			  return a * b;
		  });

	m.def("div",
		  [](ClassicalCondition a, ClassicalCondition b)
		  {
			  return a / b;
		  });

	m.def("div",
		  [](ClassicalCondition a, cbit_size_t b)
		  {
			  return a / b;
		  });

	m.def("div",
		  [](cbit_size_t a, ClassicalCondition b)
		  {
			  return a / b;
		  });

	m.def("equal",
		  [](ClassicalCondition a, ClassicalCondition b)
		  {
			  return a == b;
		  });

	m.def("equal",
		  [](ClassicalCondition a, cbit_size_t b)
		  {
			  return a == b;
		  });

	m.def("equal",
		  [](cbit_size_t a, ClassicalCondition b)
		  {
			  return a == b;
		  });

	m.def("assign",
		  [](ClassicalCondition &a, ClassicalCondition b)
		  {
			  return a = b;
		  });

	m.def("assign",
		  [](ClassicalCondition &a, cbit_size_t b)
		  {
			  return a = b;
		  });
	//---------------------------------------------------------------------------------------------------------------------
	/* include\Components\MaxCutProblemGenerator\MaxCutProblemGenerator.h */
	m.def("vector_dot",
        &vector_dot,
        py::arg("x"),
        py::arg("y"),
        "Inner product of vector x and y\n"
        "\n"
        "Args:\n"
        "    x: list x \n"
        "    y: list y  \n"
        "\n"
        "Returns:\n"
        "    dot result \n"
        "Raises:\n"
        "    run_fail: An error occurred in vector_dot\n");

	m.def("all_cut_of_graph",
        &all_cut_of_graph,
        py::arg("adjacent_matrix"),
        py::arg("all_cut_list"),
        py::arg("target_value_list"),
        "Generate graph of maxcut problem\n"
        "\n"
        "Args:\n"
        "    adjacent_matrix: adjacent_matrix for quantum prog \n"
        "    all_cut_list: all cut graph list in quantum prog \n"
        "    target_value_list: target cut value list \n"
        "\n"
        "Returns:\n"
        "    max value\n"
        "Raises:\n"
        "    run_fail: An error occurred in all_cut_of_graph\n");

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
		"Quantum circuit layering\n"
        "\n"
        "Args:\n"
        "    QProg: quantum prog \n"
        "\n"
        "Returns:\n"
        "    result data tuple contains layer info\n"
        "Raises:\n"
        "    run_fail: An error occurred in get circuit_layer\n",
		py::return_value_policy::automatic);

	//---------------------------------------------------------------------------------------------------------------------
	/* include\Core\Utilities\QProgInfo\Visualization\QVisualization.h */
	m.def(
		"draw_qprog_text",
		[](QProg prg, uint32_t auto_wrap_len, const std::string &output_file, const NodeIter itr_start, const NodeIter itr_end)
		{
			return draw_qprog(prg, PIC_TYPE::TEXT, false, auto_wrap_len, output_file, itr_start, itr_end);
		},
		py::arg("qprog"),
		py::arg("auto_wrap_len") = 100,
		py::arg("output_file") = "QCircuitTextPic.txt",
		py::arg_v("itr_start", NodeIter(), "NodeIter()"),
		py::arg_v("itr_end", NodeIter(), "NodeIter()"),
		"Convert a quantum prog/circuit to text-pic(UTF-8 code),\n"
		"and will save the text-pic in file named QCircuitTextPic.txt in the same time in current path"
        "\n"
        "Args:\n"
        "    QProg: quantum prog \n"
        "    auto_wrap_len: defaut is 100 \n"
        "    output_file: result output file name \n"
        "    itr_start: nodeiter start \n"
        "    itr_end: nodeiter end \n"
        "\n"
        "Returns:\n"
        "    result data tuple contains prog info\n"
        "Raises:\n"
        "    run_fail: An error occurred in get draw_qprog_text\n",
		py::return_value_policy::automatic);

	m.def(
		"draw_qprog_latex",
		[](QProg prg, uint32_t auto_wrap_len, const std::string &output_file, bool with_logo, const NodeIter itr_start, const NodeIter itr_end)
		{
			return draw_qprog(prg, PIC_TYPE::LATEX, with_logo, auto_wrap_len, output_file, itr_start, itr_end);
		},
		py::arg("prog"),
		py::arg("auto_wrap_len") = 100,
		py::arg("output_file") = "QCircuit.tex",
		py::arg("with_logo") = false,
		py::arg_v("itr_start", NodeIter(), "NodeIter()"),
		py::arg_v("itr_end", NodeIter(), "NodeIter()"),
		"Convert a quantum prog/circuit to latex source code, and save the source code to file in current path with name QCircuit.tex"
        "\n"
        "Args:\n"
        "    QProg: quantum prog \n"
        "    auto_wrap_len: defaut is 100 \n"
        "    output_file: result output file name \n"
        "    itr_start: nodeiter start \n"
        "    itr_end: nodeiter end \n"
        "\n"
        "Returns:\n"
        "    result data tuple contains prog info\n"
        "Raises:\n"
        "    run_fail: An error occurred in get draw_qprog_text\n",
		py::return_value_policy::automatic);

	m.def(
		"draw_qprog_text_with_clock",
		[](QProg prog, const std::string config_data, uint32_t auto_wrap_len, const std::string &output_file, const NodeIter itr_start, const NodeIter itr_end)
		{
			return draw_qprog_with_clock(prog, PIC_TYPE::TEXT, config_data, false, auto_wrap_len, output_file, itr_start, itr_end);
		},
		py::arg("prog"),
		py::arg("config_data") = CONFIG_PATH,
		py::arg("auto_wrap_len") = 100,
		py::arg("output_file") = "QCircuitTextPic.txt",
		py::arg_v("itr_start", NodeIter(), "NodeIter()"),
		py::arg_v("itr_end", NodeIter(), "NodeIter()"),
		"Convert a quantum prog/circuit to text-pic(UTF-8 code) with time sequence,\n"
		"and will save the text-pic in file named QCircuitTextPic.txt in the same time in current path"
        "\n"
        "Args:\n"
        "    QProg: quantum prog \n"
        "    auto_wrap_len: defaut is 100 \n"
        "    output_file: result output file name \n"
        "    itr_start: nodeiter start \n"
        "    itr_end: nodeiter end \n"
        "\n"
        "Returns:\n"
        "    result data tuple contains prog info\n"
        "Raises:\n"
        "    run_fail: An error occurred in get draw_qprog_text\n",
		py::return_value_policy::automatic);

	m.def(
		"draw_qprog_latex_with_clock",
		[](QProg prog, const std::string config_data, bool with_logo, uint32_t auto_wrap_len, const std::string &output_file, const NodeIter itr_start, const NodeIter itr_end)
		{
			return draw_qprog_with_clock(prog, PIC_TYPE::LATEX, config_data, with_logo, auto_wrap_len, output_file, itr_start, itr_end);
		},
		py::arg("prog"),
		py::arg("config_data") = CONFIG_PATH,
		py::arg("auto_wrap_len") = 100,
		py::arg("output_file") = "QCircuit.tex",
		py::arg("with_logo") = false,
		py::arg_v("itr_start", NodeIter(), "NodeIter()"),
		py::arg_v("itr_end", NodeIter(), "NodeIter()"),
		"Convert a quantum prog/circuit to latex source code with time sequence, and save the source code to file in current path with name QCircuit.tex"
        "\n"
        "Args:\n"
        "    QProg: quantum prog \n"
        "    config_data: default config file is QPandaConfig.json \n"
        "    auto_wrap_len: defaut is 100 \n"
        "    output_file: result output file name \n"
        "    itr_start: nodeiter start \n"
        "    itr_end: nodeiter end \n"
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
        "Special character conversion\n"
        "\n"
        "Args:\n"
        "    utf8_str: string using utf-8 encode \n"
        "\n"
        "Returns:\n"
        "    result string\n"
        "Raises:\n"
        "    run_fail: An error occurred in get fit_to_gbk\n",
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
		"Fill the input QProg by I gate, return a new quantum program\n"
        "\n"
        "Args:\n"
        "    prog: quantum prog \n"
        "\n"
        "Returns:\n"
        "    a new quantum program\n"
        "Raises:\n"
        "    run_fail: An error occurred in get fill_qprog_by_I\n",
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
			ret_data.append(new_qvec);
			return ret_data;
		},
		py::arg("qprog"),
		py::arg("machine"),
		py::arg("mapping") = true,
		py::arg("config_file") = CONFIG_PATH,
		"Quantum chip adaptive conversion\n"
		"\n"
		"Args:\n"
		"    qprog: quantum program\n"
		"    machine: quantum machine\n"
		"    mapping: whether or not perform the mapping operation\n"
		"    config_file: config file\n"
		"\n"
		"Returns:\n"
		"    list contains qprog and qubit_list after mapping, if mapping is false, the qubit_list may be misoperated",
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
		"Decompose multiple control QGate\n"
        "\n"
        "Args:\n"
        "    qprog: quantum program\n"
        "    machine: quantum machine\n"
        "    config_file: config file\n"
        "\n"
        "Returns:\n"
        "    a new prog after decomposition",
		py::return_value_policy::automatic);

    /* #include "Core/Utilities/Tools/MultiControlGateDecomposition.h" */
    m.def("ldd_decompose",[](QProg prog)
        {
            return ldd_decompose(prog);
        },
        py::arg("qprog"),
        "Decompose multiple control QGate\n"
        "\n"
        "Args:\n"
        "    qprog: quantum program\n"
        "\n"
        "Returns:\n"
        "    a new prog after decomposition",
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
		"Basic quantum - gate conversion\n"
        "\n"
        "Args:\n"
        "    qprog: quantum program\n"
        "    machine: quantum machine\n"
        "    config_file: config file\n"
        "\n"
        "Returns:\n"
        "    a new prog after transform_to_base_qgate",
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
		"Optimize QCircuit\n"
        "\n"
        "Args:\n"
        "    qprog: quantum program\n"
        "    optimizer_cir_vec: quantum circuit list \n"
        "    mode_list: optimize mode list\n"
        "\n"
        "Returns:\n"
        "    a new prog after optimize",
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
		"QCircuit optimizer\n"
        "\n"
        "Args:\n"
        "    qprog: quantum program\n"
        "    config_file: optimize config \n"
        "    mode_list: optimize mode list\n"
        "\n"
        "Returns:\n"
        "    a new prog after optimize",
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
		"compare two quantum states , Get the state fidelity\n"
        "\n"
        "Args:\n"
        "    state1: quantum state list 1\n"
        "    state2: quantum state list 2\n"
        "\n"
        "Returns:\n"
        "    state fidelity bewteen [0,1]",
		py::return_value_policy::automatic);

	m.def(
		"state_fidelity",
		[](const std::vector<QStat> &matrix1, const std::vector<QStat> &matrix2)
		{
			return state_fidelity(matrix1, matrix2);
		},
		py::arg("matrix1"),
		py::arg("matrix2"),
        "compare two quantum states matrix, Get the state fidelity\n"
        "\n"
        "Args:\n"
        "    state1: quantum state matrix 1\n"
        "    state2: quantum state matrix 2\n"
        "\n"
        "Returns:\n"
        "    state fidelity bewteen [0,1]",
		py::return_value_policy::automatic);

	m.def(
		"state_fidelity",
		[](const QStat &state, const vector<QStat> &matrix)
		{
			return state_fidelity(state, matrix);
		},
		py::arg("state1"),
		py::arg("state2"),
        "compare two quantum states , Get the state fidelity\n"
        "\n"
        "Args:\n"
        "    state1: quantum state list 1\n"
        "    state2: quantum state matrix 2\n"
        "\n"
        "Returns:\n"
        "    state fidelity bewteen [0,1]",
		py::return_value_policy::automatic);

	m.def(
		"state_fidelity",
		[](const vector<QStat> &matrix, const QStat &state)
		{
			return state_fidelity(matrix, state);
		},
		py::arg("state1"),
		py::arg("state2"),
        "compare two quantum states , Get the state fidelity\n"
        "\n"
        "Args:\n"
        "    state1: quantum state matrix 1\n"
        "    state2: quantum state list 2\n"
        "\n"
        "Returns:\n"
        "    state fidelity bewteen [0,1]",
		py::return_value_policy::automatic);

	m.def(
		"average_gate_fidelity",
		[](const QMatrixXcd &matrix, const QStat &state)
		{
			return average_gate_fidelity(matrix, state);
		},
		py::arg("state1"),
		py::arg("state2"),
        "compare two quantum states , Get the state fidelity\n"
        "\n"
        "Args:\n"
        "    state1: quantum state matrix 1\n"
        "    state2: quantum state list 2\n"
        "\n"
        "Returns:\n"
        "    state fidelity bewteen [0,1]",
		py::return_value_policy::automatic);

	m.def(
		"average_gate_fidelity",
		[](const QMatrixXcd &matrix, const QMatrixXcd &state)
		{
			return average_gate_fidelity(matrix, state);
		},
		py::arg("state1"),
		py::arg("state2"),
        "compare two quantum states , Get the state fidelity\n"
        "\n"
        "Args:\n"
        "    state1: quantum state matrix 1\n"
        "    state2: quantum state list 2\n"
        "\n"
        "Returns:\n"
        "    state fidelity bewteen [0,1]",
		py::return_value_policy::automatic);

	//---------------------------------------------------------------------------------------------------------------------
	/* include\Core\Utilities\Tools\GetQubitTopology.h */
	m.def("get_circuit_optimal_topology",
        &get_circuit_optimal_topology,
        py::arg("qprog"),
        py::arg("machine"),
        py::arg("max_connect_degree"),
        py::arg("config_file") = CONFIG_PATH,
        "Get the optimal topology of the input circuit\n"
        "\n"
        "Args:\n"
        "    qprog: quantum program\n"
        "    machine: quantum machine\n"
        "    max_connect_degree: max value of connect degree\n"
        "    config_file: config file\n"
        "\n"
        "Returns:\n"
        "    Topology prog Data"
        "Raises:\n"
        "    run_fail: An error occurred in get_circuit_optimal_topology\n",
        py::return_value_policy::automatic);

	m.def("get_double_gate_block_topology",
        &get_double_gate_block_topology,
        py::arg("qprog"),
        "get double gate block topology\n"
        "\n"
        "Args:\n"
        "    qprog: quantum program\n"
        "\n"
        "Returns:\n"
        "    Topology prog Data"
        "Raises:\n"
        "    run_fail: An error occurred in get_double_gate_block_topology\n",
        py::return_value_policy::automatic);

	m.def(
		"del_weak_edge",
		py::overload_cast<TopologyData &>(&del_weak_edge),
		py::arg("topo_data"),
		"Delete weakly connected edges\n"
        "\n"
        "Args:\n"
        "    topo_data: quantum program topo_data\n"
        "\n"
        "Returns:\n"
        "    none\n"
        "Raises:\n"
        "    run_fail: An error occurred in del_weak_edge\n",
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
		"Delete weakly connected edges\n"
        "\n"
        "Args:\n"
        "    topo_data: quantum program topo_data\n"
        "    max_connect_degree: max value of connect degree\n"
        "    sub_graph_set: sub graph set list\n"
        "\n"
        "Returns:\n"
        "    result data \n"
        "Raises:\n"
        "    run_fail: An error occurred in del_weak_edge2\n",
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
		"Delete weakly connected edges\n"
        "\n"
        "Args:\n"
        "    topo_data: quantum program topo_data\n"
        "    max_connect_degree: max value of connect degree\n"
        "    sub_graph_set: sub graph set list\n"
        "    lamda1: lamda1\n"
        "    lamda2: lamda2\n"
        "    lamda3: lamda3\n"
        "\n"
        "Returns:\n"
        "    result data \n"
        "Raises:\n"
        "    run_fail: An error occurred in del_weak_edge3\n",
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
		"Recover edges from the candidate edges\n"
        "\n"
        "Args:\n"
        "    topo_data: quantum program topo_data\n"
        "    max_connect_degree: max value of connect degree\n"
        "    candidate_edges: candidate edges\n"
        "\n"
        "Returns:\n"
        "    topo data \n"
        "Raises:\n"
        "    run_fail: An error occurred in recover_edges\n",
		py::return_value_policy::automatic);

	m.def("get_complex_points",
        &get_complex_points,
        py::arg("topo_data"),
        py::arg("max_connect_degree"),
        "Get complex points\n"
        "\n"
        "Args:\n"
        "    topo_data: quantum program topo_data\n"
        "    max_connect_degree: max value of connect degree\n"
        "\n"
        "Returns:\n"
        "    complex points list \n"
        "Raises:\n"
        "    run_fail: An error occurred in get_complex_points\n",
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
        "Splitting complex points into multiple points\n"
        "\n"
        "Args:\n"
        "    topo_data: quantum program topo_data\n"
        "    max_connect_degree: max value of connect degree\n"
        "    complex_points: complex points list\n"
        "    split_method: see ComplexVertexSplitMethod, default is ComplexVertexSplitMethod.LINEAR\n"
        "\n"
        "Returns:\n"
        "    none \n"
        "Raises:\n"
        "    run_fail: An error occurred in split_complex_points\n",
        py::return_value_policy::automatic);

	m.def("replace_complex_points",
        &replace_complex_points,
        py::arg("src_topo_data"),
        py::arg("max_connect_degree"),
        py::arg("sub_topo_vec"),
        "Replacing complex points with subgraphs\n"
        "\n"
        "Args:\n"
        "    src_topo_data: quantum program source topo data\n"
        "    max_connect_degree: max value of connect degree\n"
        "    sub_topo_vec: sub topo list\n"
        "\n"
        "Returns:\n"
        "    none \n"
        "Raises:\n"
        "    run_fail: An error occurred in replace_complex_points\n",
        py::return_value_policy::automatic);

	m.def("get_sub_graph",
        &get_sub_graph,
        py::arg("topo_data"),
        "Get sub graph\n"
        "\n"
        "Args:\n"
        "    topo_data: quantum program topo data\n"
        "\n"
        "Returns:\n"
        "    sub graph \n"
        "Raises:\n"
        "    run_fail: An error occurred in sub graph\n",
        py::return_value_policy::automatic);

	m.def("estimate_topology",
        &estimate_topology,
        py::arg("topo_data"),
        "Evaluate topology performance\n"
        "\n"
        "Args:\n"
        "    topo_data: quantum program topo data\n"
        "\n"
        "Returns:\n"
        "    result data \n"
        "Raises:\n"
        "    run_fail: An error occurred in estimate_topology\n",
        py::return_value_policy::automatic);

	m.def("planarity_testing",
        &planarity_testing,
        py::arg("topo_data"),
        "planarity testing\n"
        "\n"
        "Args:\n"
        "    topo_data: quantum program topo data\n"
        "\n"
        "Returns:\n"
        "    result data \n"
        "Raises:\n"
        "    run_fail: An error occurred in planarity_testing\n",
        py::return_value_policy::automatic);

    #if defined(USE_OPENSSL) && defined(USE_CURL)

	//---------------------------------------------------------------------------------------------------------------------
	/* include\Core\Utilities\QProgInfo\RandomizedBenchmarking.h */
	m.def("single_qubit_rb",
        py::overload_cast<NoiseQVM *, Qubit *, const std::vector<int> &, int, int, const std::vector<QGate> &>(&single_qubit_rb),
        py::arg("qvm"),
        py::arg("qubit"),
        py::arg("clifford_range"),
        py::arg("num_circuits"),
        py::arg("shots"),
        py::arg("interleaved_gates") = std::vector<QGate>(),
        "Single qubit rb with noise quantum virtual machine\n"
        "\n"
        "Args:\n"
        "    qvm: quantum machine\n"
        "    qubit: single qubit\n"
        "    clifford_range: clifford range list\n"
        "    num_circuits: the num of circuits\n"
        "    shots: measure shots\n"
        "    interleaved_gates: interleaved gates list\n"
        "\n"
        "Returns:\n"
        "    result data dict\n"
        "Raises:\n"
        "    run_fail: An error occurred in single_qubit_rb\n",
        py::return_value_policy::automatic);

	m.def("single_qubit_rb",
        [](QCloudMachine* qvm,
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
        py::overload_cast<NoiseQVM *, Qubit *, Qubit *, const std::vector<int> &, int, int, const std::vector<QGate> &>(&double_qubit_rb),
        py::arg("qvm"),
        py::arg("qubit0"),
        py::arg("qubit1"),
        py::arg("clifford_range"),
        py::arg("num_circuits"),
        py::arg("shots"),
        py::arg("interleaved_gates") = std::vector<QGate>(),
        "double qubit rb with noise quantum virtual machine\n"
        "\n"
        "Args:\n"
        "    qvm: quantum machine\n"
        "    qubit0: double qubit 0\n"
        "    qubit1: double qubit 1\n"
        "    clifford_range: clifford range list\n"
        "    num_circuits: the num of circuits\n"
        "    shots: measure shots\n"
        "    interleaved_gates: interleaved gates list\n"
        "\n"
        "Returns:\n"
        "    result data dict\n"
        "Raises:\n"
        "    run_fail: An error occurred in double_qubit_rb\n",
        py::return_value_policy::automatic);

	m.def("double_qubit_rb",
        [](QCloudMachine* qvm,
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

	//---------------------------------------------------------------------------------------------------------------------
	/* include\Core\Utilities\QProgInfo\CrossEntropyBenchmarking.h */
	m.def("double_gate_xeb",
        [](QCloudMachine* qvm,
           Qubit* qbit0,
           Qubit* qbit1,
           const std::vector<int>& range,
           int num_circuits,
           int shots,
           int chip_id = (int)RealChipType::ORIGIN_WUYUAN_D5,
           GateType gt = GateType::CZ_GATE)
        {
            auto real_chip_type = static_cast<RealChipType>(chip_id);
            return double_gate_xeb(qvm, qbit0, qbit1, range, num_circuits, shots, real_chip_type, gt);
        },
        py::arg("cloud_qvm"),
        py::arg("qubit0"),
        py::arg("qubit1"),
        py::arg("clifford_range"),
        py::arg("num_circuits"),
        py::arg("shots"),
        py::arg("chip_id") = (int)RealChipType::ORIGIN_WUYUAN_D5,
        py::arg_v("gate_type", GateType::CZ_GATE, "GateType.CZ_GATE"),
        "double gate xeb with WU YUAN chip\n"
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

	m.def("double_gate_xeb",
		py::overload_cast<NoiseQVM *, Qubit *, Qubit *, const std::vector<int> &, int, int, GateType>(&double_gate_xeb),
		py::arg("noise_qvm"),
		py::arg("qubit0"),
		py::arg("qubit1"),
		py::arg("clifford_range"),
		py::arg("num_circuits"),
		py::arg("shots"),
		py::arg_v("gate_type", GateType::CZ_GATE, "GateType.CZ_GATE"),
		"double gate xeb with WU YUAN chip\n"
        "\n"
        "Args:\n"
        "    qvm: quantum machine\n"
        "    qubit0: double qubit 0\n"
        "    qubit1: double qubit 1\n"
        "    clifford_range: clifford range list\n"
        "    num_circuits: the num of circuits\n"
        "    shots: measure shots\n"
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

    #endif

	//---------------------------------------------------------------------------------------------------------------------
	/* include\Core\Utilities\Tools\RandomCircuit.h */
	m.def("random_qprog",
        &random_qprog,
        py::arg("qubit_row"),
        py::arg("qubit_col"),
        py::arg("depth"),
        py::arg("qvm"),
        py::arg("qvec"),
        "Generate random quantum program\n"
        "\n"
        "Args:\n"
        "    qubit_row: circuit qubit row value\n"
        "    qubit_col: circuit qubit col value\n"
        "    depth: circuit depth\n"
        "    qvm: quantum machine\n"
        "    qvec: out put circuits for random qprog\n"
        "\n"
        "Returns:\n"
        "    random quantum program\n"
        "Raises:\n"
        "    run_fail: An error occurred in generate random qprog\n");

	m.def("random_qcircuit",
        &random_qcircuit,
        py::arg("qvec"),
        py::arg("depth") = 100,
        py::arg("gate_type") = std::vector<std::string>(),
        "Generate random quantum circuit\n"
        "\n"
        "Args:\n"
        "    qubit_row: circuit qubit row value\n"
        "    qubit_col: circuit qubit col value\n"
        "    depth: circuit depth\n"
        "    qvm: quantum machine\n"
        "    qvec: out put circuits for random circuit\n"
        "\n"
        "Returns:\n"
        "    random quantum program\n"
        "Raises:\n"
        "    run_fail: An error occurred in generate random circuit\n");

    m.def("deep_copy", [](QProg& node) { return deepCopy(node);}, py::arg("node"), py::return_value_policy::automatic);
    m.def("deep_copy", [](QCircuit& node) { return deepCopy(node); }, py::arg("node"), py::return_value_policy::automatic);
    m.def("deep_copy", [](QGate& node) { return deepCopy(node); }, py::arg("node"), py::return_value_policy::automatic);
    m.def("deep_copy", [](QMeasure& node) { return deepCopy(node); }, py::arg("node"), py::return_value_policy::automatic);
    m.def("deep_copy", [](ClassicalProg& node) { return deepCopy(node); }, py::arg("node"), py::return_value_policy::automatic);
    m.def("deep_copy", [](QIfProg& node) { return deepCopy(node); }, py::arg("node"), py::return_value_policy::automatic);
    m.def("deep_copy", [](QWhileProg& node) { return deepCopy(node); }, py::arg("node"), py::return_value_policy::automatic);
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
}