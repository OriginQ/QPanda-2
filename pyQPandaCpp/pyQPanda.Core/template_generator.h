#ifndef TEMPLATE_GENERATOR_H_
#define TEMPLATE_GENERATOR_H_


#include "QPanda.h"
#include "pybind11/pybind11.h"


USING_QPANDA
namespace py = pybind11;

/* Core\Utilities\Compiler\QProgToOriginIR.h */
template <typename T>
void export_transformQProgToOriginIR(py::module &m)
{
    m.def("to_originir",
        py::overload_cast<T &, QuantumMachine *>(&transformQProgToOriginIR<T>),
        py::arg("qprog"),
        py::arg("machine"),
        "Transform QProg to OriginIR string.\n"
        "\n"
        "Args:\n"
        "     qprog: T, the quantum program or circuit to transform.\n"
        "\n"
        "     machine: QuantumMachine*, the quantum machine being used.\n"
        "\n"
        "Returns:\n"
        "     OriginIR string, the transformed representation of the quantum program.\n",
        py::return_value_policy::automatic_reference);
}

/* include\Core\QuantumCircuit\QGate.h */
#define GEN_singleBitGate_TEMPLATE(gate_name, ...)                                     \
      template <typename... T>                                                         \
      class TempHelper_##gate_name                                                     \
      {                                                                                \
      public:                                                                          \
            static void export_singleBitGate(py::module &m)                            \
            {                                                                          \
                  m.def(#gate_name,                                                    \
                        py::overload_cast<Qubit *, T...>(&gate_name),                  \
                        py::arg("qubit"),                                              \
                        __VA_ARGS__,                                                   \
                        "Create a " #gate_name " gate"                                 \
                        "\n"                                                           \
                        "Args:\n"                                                      \
                        "    qubit : quantum gate operate qubit\n"                     \
                        "    args : quantum gate angles\n"                             \
                        "\n",                                                          \
                        "Returns:\n"                                                   \
                        "    a " #gate_name " gate node\n"                             \
                        "Raises:\n"                                                    \
                        "    run_fail: An error occurred in construct gate node\n",    \
                        py::return_value_policy::automatic);                           \
                  m.def(#gate_name,                                                    \
                        py::overload_cast<const QVec &, T...>(&gate_name),             \
                        py::arg("qubit_list"),                                         \
                        __VA_ARGS__,                                                   \
                        "Create a " #gate_name " gate"                                 \
                        "\n"                                                           \
                        "Args:\n"                                                      \
                        "    qubit_list: quantum gate operate qubits list\n"           \
                        "    args : quantum gate angles\n"                             \
                        "\n"                                                           \
                        "Returns:\n"                                                   \
                        "    a " #gate_name " gate node\n"                             \
                        "Raises:\n"                                                    \
                        "    run_fail: An error occurred construct in gate node\n",    \
                        py::return_value_policy::automatic);                           \
                  m.def(#gate_name,                                                    \
                        py::overload_cast<int, T...>(&gate_name),                      \
                        py::arg("qubit_addr"),                                         \
                        "Create a " #gate_name " gate"                                 \
                        "\n"                                                           \
                        "Args:\n"                                                      \
                        "    qubit_addr: quantum gate operate qubits addr\n"           \
                        "    args : quantum gate angles\n"                             \
                        "\n"                                                           \
                        "Returns:\n"                                                   \
                        "    a " #gate_name " gate node\n"                             \
                        "Raises:\n"                                                    \
                        "    run_fail: An error occurred in construct gate node\n",    \
                        __VA_ARGS__,                                                   \
                        py::return_value_policy::automatic);                           \
                  m.def(#gate_name,                                                    \
                        py::overload_cast<const std::vector<int> &, T...>(&gate_name), \
                        py::arg("qubit_addr_list"),                                    \
                        __VA_ARGS__,                                                   \
                        "Create a " #gate_name " gate"                                 \
                        "\n"                                                           \
                        "Args:\n"                                                      \
                        "    qubit_list_addr: quantum gate  qubits list addr\n"        \
                        "    args : quantum gate angles\n"                             \
                        "\n"                                                           \
                        "Returns:\n"                                                   \
                        "    a " #gate_name " gate node\n"                             \
                        "Raises:\n"                                                    \
                        "    run_fail: An error occurred in construct gate node\n",    \
                        py::return_value_policy::automatic);                           \
            }                                                                          \
      }

GEN_singleBitGate_TEMPLATE(RX, py::arg("angle"));
GEN_singleBitGate_TEMPLATE(RY, py::arg("angle"));
GEN_singleBitGate_TEMPLATE(RZ, py::arg("angle"));
GEN_singleBitGate_TEMPLATE(P, py::arg("angle"));
GEN_singleBitGate_TEMPLATE(U1, py::arg("angle"));
GEN_singleBitGate_TEMPLATE(U2, py::arg("phi_angle"), py::arg("lambda_angle"));
GEN_singleBitGate_TEMPLATE(U3, py::arg("theta_angle"), py::arg("phi_angle"), py::arg("lambda_angle"));

/**
 * @brief double bit gate pybind11 export template helper class
 *
 * @param helper name of helper class, will generate template class TempHelper_helper
 * @param gate_name name duoble bit gate
 * @param first_arg first bit args name, in string
 * @param second_arg second bit args name, in string
 * @param ... other args name, in pybind11 arg type
 */
#define GEN_doubleBitGate_TEMPLATE(helper, gate_name, first_arg, second_arg, ...)                                \
      template <typename... T>                                                                                   \
      class TempHelper_##helper                                                                                  \
      {                                                                                                          \
      public:                                                                                                    \
            static void export_doubleBitGate(py::module &m)                                                      \
            {                                                                                                    \
                  m.def(#gate_name,                                                                              \
                        py::overload_cast<Qubit *, Qubit *, T...>(&gate_name),                                   \
                        py::arg(first_arg),                                                                      \
                        py::arg(second_arg),                                                                     \
                        __VA_ARGS__,                                                                             \
                        "Create a " #gate_name " gate"                                                           \
                        "\n"                                                                                     \
                        "Args:\n"                                                                                \
                        "    control qubit : quantum gate control qubit\n"                                       \
                        "    target qubit : quantum gate target qubit\n"                                         \
                        "    args : quantum gate angles\n"                                                       \
                        "\n",                                                                                    \
                        "Returns:\n"                                                                             \
                        "    a " #gate_name " gate node\n"                                                       \
                        "Raises:\n"                                                                              \
                        "    run_fail: An error occurred in construct gate node\n",                              \
                        py::return_value_policy::automatic);                                                     \
                  m.def(#gate_name,                                                                              \
                        py::overload_cast<const QVec &, const QVec &, T...>(&gate_name),                         \
                        py::arg(first_arg "_list"),                                                              \
                        py::arg(second_arg "_list"),                                                             \
                        __VA_ARGS__,                                                                             \
                        "Create a " #gate_name " gate"                                                           \
                        "\n"                                                                                     \
                        "Args:\n"                                                                                \
                        "    control qubit : quantum gate control qubit\n"                                       \
                        "    target qubit : quantum gate target qubit\n"                                         \
                        "    args : quantum gate angles\n"                                                       \
                        "\n",                                                                                    \
                        "Returns:\n"                                                                             \
                        "    a " #gate_name " gate node\n"                                                       \
                        "Raises:\n"                                                                              \
                        "    run_fail: An error occurred in construct gate node\n",                              \
                        py::return_value_policy::automatic);                                                     \
                  m.def(#gate_name,                                                                              \
                        py::overload_cast<int, int, T...>(&gate_name),                                           \
                        py::arg(first_arg "_addr"),                                                              \
                        py::arg(second_arg "_addr"),                                                             \
                        "Create a " #gate_name " gate"                                                           \
                        "\n"                                                                                     \
                        "Args:\n"                                                                                \
                        "    control qubit : quantum gate control qubit\n"                                       \
                        "    target qubit : quantum gate target qubit\n"                                         \
                        "    args : quantum gate angles\n"                                                       \
                        "\n",                                                                                    \
                        "Returns:\n"                                                                             \
                        "    a " #gate_name " gate node\n"                                                       \
                        "Raises:\n"                                                                              \
                        "    run_fail: An error occurred in construct gate node\n",                              \
                        __VA_ARGS__,                                                                             \
                        py::return_value_policy::automatic);                                                     \
                  m.def(#gate_name,                                                                              \
                        py::overload_cast<const std::vector<int> &, const std::vector<int> &, T...>(&gate_name), \
                        py::arg(first_arg "_addr_list"),                                                         \
                        py::arg(second_arg "_addr_list"),                                                        \
                        __VA_ARGS__,                                                                             \
                        "Create a " #gate_name " gate"                                                           \
                        "\n"                                                                                     \
                        "Args:\n"                                                                                \
                        "    control qubit : quantum gate control qubit\n"                                       \
                        "    target qubit : quantum gate target qubit\n"                                         \
                        "    args : quantum gate angles\n"                                                       \
                        "\n",                                                                                    \
                        "Returns:\n"                                                                             \
                        "    a " #gate_name " gate node\n"                                                       \
                        "Raises:\n"                                                                              \
                        "    run_fail: An error occurred in construct gate node\n",                              \
                        py::return_value_policy::automatic);                                                     \
            }                                                                                                    \
      }

 /* linux not support empty __VA_ARGS__, use empty comment as position token. in pybind11 comment later will overwrite this empyt comment */
GEN_doubleBitGate_TEMPLATE(CNOT, CNOT, "control_qubit", "target_qubit", "");
GEN_doubleBitGate_TEMPLATE(CZ, CZ, "control_qubit", "target_qubit", "");
GEN_doubleBitGate_TEMPLATE(CP, CP, "control_qubit", "target_qubit", py::arg("theta_angle"));
GEN_doubleBitGate_TEMPLATE(CR, CR, "control_qubit", "target_qubit", py::arg("theta_angle"));

GEN_doubleBitGate_TEMPLATE(SWAP, SWAP, "first_qubit", "second_qubit", "");
GEN_doubleBitGate_TEMPLATE(iSWAP, iSWAP, "first_qubit", "second_qubit", "");
GEN_doubleBitGate_TEMPLATE(iSWAP_2, iSWAP, "first_qubit", "second_qubit", py::arg("theta_angle"));
GEN_doubleBitGate_TEMPLATE(SqiSWAP, SqiSWAP, "first_qubit", "second_qubit", "");
GEN_doubleBitGate_TEMPLATE(MS, MS, "first_qubit", "second_qubit", "");
GEN_doubleBitGate_TEMPLATE(QDouble, QDouble, "first_qubit", "second_qubit", py::arg("matrix"));

template <typename Cls_t>
class export_idealqvm_func
{
public:
    template <typename PyCls_t>
    static void export_func(PyCls_t& cls)
    {
        cls.def(py::init<>())
            .def("pmeasure",
                &Cls_t::PMeasure,
                py::arg("qubit_list"),
                py::arg("select_max") = -1,
                py::call_guard<py::gil_scoped_release>(),
                "Get the probability distribution over qubits.\n"
                "\n"
                "Args:\n"
                "     qubit_list: List of qubits to measure.\n"
                "\n"
                "     select_max: int, optional, maximum number of results to select (default: -1).\n"
                "\n"
                "Returns:\n"
                "     Probability distribution as a reference.\n",
                py::return_value_policy::reference)
            .def("pmeasure_no_index",
                &Cls_t::PMeasure_no_index,
                py::arg("qubit_list"),
                py::call_guard<py::gil_scoped_release>(),
                "Get the probability distribution over qubits without index.\n"
                "\n"
                "Args:\n"
                "     qubit_list: List of qubits to measure.\n"
                "\n"
                "Returns:\n"
                "     Probability distribution as a reference.\n",
                py::return_value_policy::reference)
            .def("get_prob_tuple_list",
                &Cls_t::getProbTupleList,
                py::arg("qubit_list"),
                py::arg("select_max") = -1,
                py::call_guard<py::gil_scoped_release>(),
                "Get a list of probability tuples for the specified qubits.\n"
                "\n"
                "Args:\n"
                "     qubit_list: List of qubits to measure.\n"
                "\n"
                "     select_max: int, optional, maximum number of tuples to return (default: -1).\n"
                "\n"
                "Returns:\n"
                "     List of probability tuples as a reference.\n",
                py::return_value_policy::reference)
            .def("get_prob_list",
                &Cls_t::getProbList,
                py::arg("qubit_list"),
                py::arg("select_max") = -1,
                py::call_guard<py::gil_scoped_release>(),
                "Get a list of probabilities for the specified qubits.\n"
                "\n"
                "Args:\n"
                "     qubit_list: List of qubits to measure.\n"
                "\n"
                "     select_max: int, optional, maximum number of probabilities to return (default: -1).\n"
                "\n"
                "Returns:\n"
                "     List of probabilities as a reference.\n",
                py::return_value_policy::reference)
            .def("get_prob_dict",
                &Cls_t::getProbDict,
                py::arg("qubit_list"),
                py::arg("select_max") = -1,
                py::call_guard<py::gil_scoped_release>(),
                "Get a dictionary of probabilities for the specified qubits.\n"
                "\n"
                "Args:\n"
                "     qubit_list: List of qubits to measure.\n"
                "\n"
                "     select_max: int, optional, maximum number of entries to return (default: -1).\n"
                "\n"
                "Returns:\n"
                "     Dictionary of probabilities as a reference.\n",
                py::return_value_policy::reference)

            .def("prob_run_tuple_list",
                py::overload_cast<QProg&, QVec, int>(&Cls_t::probRunTupleList),
                py::arg("program"),
                py::arg("qubit_list"),
                py::arg("select_max") = -1,
                py::call_guard<py::gil_scoped_release>(),
                "Execute a quantum program and get a list of probability tuples.\n"
                "\n"
                "Args:\n"
                "     program: The quantum program to execute.\n"
                "\n"
                "     qubit_list: List of qubits to measure.\n"
                "\n"
                "     select_max: int, optional, maximum number of tuples to return (default: -1).\n"
                "\n"
                "Returns:\n"
                "     List of probability tuples.\n",
                py::return_value_policy::automatic)

            .def("prob_run_tuple_list",
                py::overload_cast<QProg&, const std::vector<int>&, int>(&Cls_t::probRunTupleList),
                py::arg("program"),
                py::arg("qubit_addr_list"),
                py::arg("select_max") = -1,
                py::call_guard<py::gil_scoped_release>(),
                "Execute a quantum program and get a list of probability tuples using qubit addresses.\n"
                "\n"
                "Args:\n"
                "     program: The quantum program to execute.\n"
                "\n"
                "     qubit_addr_list: List of qubit addresses to measure.\n"
                "\n"
                "     select_max: int, optional, maximum number of tuples to return (default: -1).\n"
                "\n"
                "Returns:\n"
                "     List of probability tuples.\n",
                py::return_value_policy::automatic)

            .def("prob_run_list",
                py::overload_cast<QProg&, QVec, int>(&Cls_t::probRunList),
                py::arg("program"),
                py::arg("qubit_list"),
                py::arg("select_max") = -1,
                py::call_guard<py::gil_scoped_release>(),
                "Execute a quantum program and retrieve a list of probabilities for the specified qubits.\n"
                "\n"
                "Args:\n"
                "     program: The quantum program to execute.\n"
                "\n"
                "     qubit_list: List of qubits to measure.\n"
                "\n"
                "     select_max: int, optional, maximum number of probabilities to return (default: -1).\n"
                "\n"
                "Returns:\n"
                "     List of probabilities.\n",
                py::return_value_policy::automatic)

            .def("prob_run_list",
                py::overload_cast<QProg&, const std::vector<int>&, int>(&Cls_t::probRunList),
                py::arg("program"),
                py::arg("qubit_addr_list"),
                py::arg("select_max") = -1,
                py::call_guard<py::gil_scoped_release>(),
                "Execute a quantum program and retrieve a list of probabilities using qubit addresses.\n"
                "\n"
                "Args:\n"
                "     program: The quantum program to execute.\n"
                "\n"
                "     qubit_addr_list: List of qubit addresses to measure.\n"
                "\n"
                "     select_max: int, optional, maximum number of probabilities to return (default: -1).\n"
                "\n"
                "Returns:\n"
                "     List of probabilities.\n",
                py::return_value_policy::automatic)
            //.def("prob_run_list",
            //    py::overload_cast<std::vector<QProg>&, std::vector<QVec>&, std::vector<int>&, int>(&Cls_t::probRunList),
            //    py::arg("qprogs"),
            //    py::arg("qubit_list"),
            //    py::arg("select_max"),
            //    py::arg("select_thread") = 4,
            //    py::call_guard<py::gil_scoped_release>(),
            //    py::return_value_policy::automatic)
            .def("prob_run_dict",
                py::overload_cast<QProg&, QVec, int>(&Cls_t::probRunDict),
                py::arg("program"),
                py::arg("qubit_list"),
                py::arg("select_max") = -1,
                py::call_guard<py::gil_scoped_release>(),
                "Execute a quantum program and retrieve a dictionary of probabilities for the specified qubits.\n"
                "\n"
                "Args:\n"
                "     program: The quantum program to execute.\n"
                "\n"
                "     qubit_list: List of qubits to measure.\n"
                "\n"
                "     select_max: int, optional, maximum number of entries in the dictionary to return (default: -1).\n"
                "\n"
                "Returns:\n"
                "     Dictionary of probabilities.\n",
                py::return_value_policy::automatic)
            //.def("prob_run_dict",
            //    py::overload_cast<std::vector<QProg>&, std::vector<QVec>&, std::vector<int>&,int>(&Cls_t::probRunDict),
            //    py::arg("qprogs"),
            //    py::arg("qubit_list"),
            //    py::arg("select_max"),
            //    py::arg("select_thread") = 4,
            //    py::call_guard<py::gil_scoped_release>(),
            //    py::return_value_policy::automatic)
            .def("prob_run_dict",
                py::overload_cast<QProg&, const std::vector<int>&, int>(&Cls_t::probRunDict),
                py::arg("program"),
                py::arg("qubit_addr_list"),
                py::arg("select_max") = -1,
                py::call_guard<py::gil_scoped_release>(),
                "Execute a quantum program and retrieve a dictionary of probabilities using qubit addresses.\n"
                "\n"
                "Args:\n"
                "     program: The quantum program to execute.\n"
                "\n"
                "     qubit_addr_list: List of qubit addresses to measure.\n"
                "\n"
                "     select_max: int, optional, maximum number of entries in the dictionary to return (default: -1).\n"
                "\n"
                "Returns:\n"
                "     Dictionary of probabilities.\n",
                py::return_value_policy::automatic)

            .def("quick_measure",
                &Cls_t::quickMeasure,
                py::arg("qubit_list"),
                py::arg("shots"),
                py::call_guard<py::gil_scoped_release>(),
                "Perform a quick measurement on the specified qubits.\n"
                "\n"
                "Args:\n"
                "     qubit_list: List of qubits to measure.\n"
                "\n"
                "     shots: Number of measurement shots to perform.\n"
                "\n"
                "Returns:\n"
                "     Reference to the measurement results.\n",
                py::return_value_policy::reference);
    }
};

#endif