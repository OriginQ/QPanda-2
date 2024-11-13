#include "QPandaConfig.h"
#include "QPanda.h"
#include <math.h>
#include <map>
#include "Core/Utilities/Encode/Encode.h"
#include "QAlg/Error_mitigation/Correction.h"
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


using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;

USING_QPANDA

void export_core_class(py::module &m)
{
    py::class_<NodeInfo>(m, "NodeInfo", "Detailed information of a QProg node")
        .def(py::init<>())
        .def(py::init<const NodeIter, QVec, QVec, int, const bool>(),
            py::arg("iter"),
            py::arg("target_qubits"),
            py::arg("control_qubits"),
            py::arg("type"),
            py::arg("dagger"),
            "Initialize a quantum operation with the specified parameters.\n"
            "\n"
            "Args:\n"
            "     iter: The node iterator for the quantum operation.\n"
            "\n"
            "     target_qubits: The target qubits involved in the operation.\n"
            "\n"
            "     control_qubits: The control qubits for the operation.\n"
            "\n"
            "     type: An integer representing the type of operation.\n"
            "\n"
            "     dagger: A boolean indicating whether the operation is a dagger (adjoint) operation.\n"
            "\n"
            "Returns:\n"
            "     None: This function initializes the quantum operation.\n"
        )
        .def("reset", &NodeInfo::reset,
            "Reset the NodeInfo instance to its initial state.\n"
            "This function clears any current data in the NodeInfo and prepares it for reuse.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     None\n"
        )
        .def_readwrite("m_iter", &NodeInfo::m_iter,
            "The iterator associated with the NodeInfo instance.\n"
            "This attribute holds the current iterator for traversing nodes in the program.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     The current iterator.\n"
        )
        .def_readwrite("m_node_type", &NodeInfo::m_node_type,
            "The type of the node in the NodeInfo.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     The type of the node as a string.\n"
        )
        .def_readwrite("m_gate_type", &NodeInfo::m_gate_type,
            "The type of the gate in the NodeInfo.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     The type of the gate as a string.\n"
        )
        .def_readwrite("m_is_dagger", &NodeInfo::m_is_dagger,
            "Indicates whether the gate is a dagger (Hermitian conjugate).\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A boolean value indicating if the gate is a dagger.\n"
        )
        .def_readwrite("m_target_qubits", &NodeInfo::m_target_qubits,
            "The target qubits for the node in NodeInfo.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A list of target qubits as an array or vector.\n"
        )
        .def_readwrite("m_control_qubits", &NodeInfo::m_control_qubits,
            "The control qubits for the node in NodeInfo.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A list of control qubits as an array or vector.\n"
        )
        .def_readwrite("m_cbits", &NodeInfo::m_cbits,
            "The classical bits associated with the node in NodeInfo.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A list of classical bits as an array or vector.\n"
        )
        .def_readwrite("m_params", &NodeInfo::m_params,
            "The parameters associated with the node in NodeInfo.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A list of parameters as an array or vector.\n"
        )
        .def_readwrite("m_name", &NodeInfo::m_name,
            "The name of the node in NodeInfo.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     The name of the node as a string.\n"
        );



    py::class_<OriginCollection>(m, "OriginCollection", "A relatively free data collection class for saving data")
        .def(py::init<>())
        .def(py::init<std::string>(),
             py::arg("file_name"),
            "Construct a new OriginCollection by reading a JSON file.\n"
            "This function initializes the OriginCollection with data from the specified JSON file.\n"
            "\n"
            "Args:\n"
            "     file_name: The path to the JSON file to read.\n"
            "\n"
            "Returns:\n"
            "     An instance of OriginCollection.\n")
        .def(py::init<const OriginCollection &>(),
            "Construct a new OriginCollection by copying an existing instance.\n"
            "This function creates a new OriginCollection as a copy of the provided instance.\n"
            "\n"
            "Args:\n"
            "     other: The existing OriginCollection instance to copy.\n"
            "\n"
            "Returns:\n"
            "     An instance of OriginCollection.\n")
        .def("setNames",
             [](OriginCollection &c, py::args args)
             {
                 std::vector<std::string> all_key;
                 for (auto arg : args)
                 {
                     all_key.push_back(std::string(py::str(arg)));
                 }
                 c = all_key;
             },
            "Set names in the OriginCollection.\n"
            "This function accepts a variable number of name arguments and sets them in the OriginCollection.\n"
            "\n"
            "Args:\n"
            "     args: A variable number of name strings to be set.\n"
            "\n"
            "Returns:\n"
            "     None.\n")
        .def(
            "insertValue",
            [](OriginCollection &c, std::string key, py::args args)
            {
                int i = 1;
                auto vector = c.getKeyVector();
                c.addValue(vector[0], key);
                for (auto arg : args)
                {
                    c.addValue(vector[i], std::string(py::str(arg)));
                    i++;
                }
            },
            py::arg("key"),
            "Insert values into the OriginCollection under the specified key.\n"
            "This function adds the first value associated with the provided key\n"
            "and then inserts additional values from the provided arguments.\n"
            "\n"
            "Args:\n"
            "     key: The key under which to insert the values.\n"
            "\n"
            "     args: A variable number of values to be inserted.\n"
            "\n"
            "Returns:\n"
            "     None.\n")
        .def("getValue",
             &OriginCollection::getValue,
             py::arg("key_name"),
            "Get the value associated with the specified key name.\n"
            "This function retrieves the value stored in the OriginCollection for the given key.\n"
            "\n"
            "Args:\n"
            "     key_name: The name of the key whose value is to be retrieved.\n"
            "\n"
            "Returns:\n"
            "     The value associated with the specified key.\n")
        .def("getValueByKey",
             py::overload_cast<const std::string &>(&OriginCollection::getValueByKey),
             py::arg("key_value"),
            "Get the value associated with the specified key value.\n"
            "This function retrieves the value from the OriginCollection based on the provided key.\n"
            "\n"
            "Args:\n"
            "     key_value: The key whose corresponding value is to be retrieved.\n"
            "\n"
            "Returns:\n"
            "     The value associated with the specified key value.\n")
        .def("getValueByKey",
             py::overload_cast<int>(&OriginCollection::getValueByKey),
             py::arg("key_value"),
            "Retrieve the value associated with a specified key.\n"
            "This function returns the value that corresponds to the given key.\n"
            "\n"
            "Args:\n"
            "     key_value: The key for which to retrieve the associated value.\n"
            "\n"
            "Returns:\n"
            "     The value associated with the specified key.\n")
        .def("open",
             &OriginCollection::open,
             py::arg("file_name"),
            "Open and read the JSON file at the specified path.\n"
            "This function reads the contents of the JSON file provided.\n"
            "\n"
            "Args:\n"
            "     file_name: The path to the JSON file to be read.\n"
            "\n"
            "Returns:\n"
            "     None.\n")
        .def("write", &OriginCollection::write,
            "Write the current data to a JSON file.\n"
            "This function saves the current contents to a specified JSON file.\n"
            "\n"
            "Args:\n"
            "     None.\n"
            "\n"
            "Returns:\n"
            "     None.\n")
        .def("getJsonString", &OriginCollection::getJsonString,
            "Retrieve the JSON string representation of the OriginCollection.\n"
            "This function converts the collection's data into a JSON format string.\n"
            "\n"
            "Returns:\n"
            "     A string containing the JSON representation of the collection.\n")
        .def("getFilePath", &OriginCollection::getFilePath,
            "Retrieve the file path associated with the OriginCollection.\n"
            "This function returns the path to the file linked to the collection.\n"
            "\n"
            "Returns:\n"
            "     A string containing the file path.\n")
            .def("getKeyVector", &OriginCollection::getKeyVector,
            "Retrieve the vector of keys associated with the OriginCollection.\n"
            "This function returns a vector containing all the keys in the collection.\n"
            "\n"
            "Returns:\n"
            "     A vector of keys.\n");

    py::class_<QResult>(m, "QResult", "QResult abstract class, this class contains the result of the quantum measurement")
        .def("getResultMap", &QResult::getResultMap, py::return_value_policy::reference);

    py::class_<ClassicalProg>(m, "ClassicalProg", "quantum ClassicalProg")
        .def(py::init<ClassicalCondition &>());

    py::class_<QGate>(m, "QGate", "quantum gate node")
        .def(py::init(
            [](NodeIter &iter)
            {
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
            }),
            "Initialize a QGate instance based on the provided iterator.\n"
            "This constructor checks the validity of the iterator and ensures it points to a valid gate node.\n"
            "\n"
            "Args:\n"
            "     iter: A reference to a NodeIter that points to the node to be initialized.\n"
            "\n"
            "Returns:\n"
            "     A QGate instance initialized from the gate node.\n"
        )
        .def("dagger", &QGate::dagger,
            "Compute the dagger (Hermitian conjugate) of the QGate instance.\n"
            "This function returns the adjoint of the quantum gate, which is used in quantum mechanics for various calculations.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A new QGate instance representing the dagger of the current gate.\n"
        )
        .def("control", &QGate::control,
            py::arg("control_qubits"),
            "Get a controlled quantum gate based on the current QGate instance.\n"
            "This function creates a control version of the quantum gate using the specified control qubits.\n"
            "\n"
            "Args:\n"
            "     control_qubits: A list of qubits that serve as control qubits for the gate.\n"
            "\n"
            "Returns:\n"
            "     A new QGate instance representing the controlled gate.\n"
        )
        .def("is_dagger", &QGate::isDagger,
            "Check if the QGate instance is a dagger (Hermitian conjugate) of another gate.\n"
            "This function determines whether the current gate is the adjoint of its corresponding gate.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A boolean indicating whether the current gate is a dagger.\n"
        )
        .def("set_dagger", &QGate::setDagger,
            "Set the QGate instance to be a dagger (Hermitian conjugate) of another gate.\n"
            "This function configures the current gate to represent its adjoint.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     None\n"
        )
        .def("set_control", &QGate::setControl,
            "Set the control qubits for the QGate instance.\n"
            "This function specifies which qubits will act as control qubits for the gate.\n"
            "\n"
            "Args:\n"
            "     control_qubits: A list of qubits that will serve as control qubits.\n"
            "\n"
            "Returns:\n"
            "     None\n"
        )
        .def("get_target_qubit_num", &QGate::getTargetQubitNum,
            "Retrieve the number of target qubits for the QGate instance.\n"
            "This function returns the count of qubits that the quantum gate affects.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     An integer representing the number of target qubits.\n"
        )
        .def("get_control_qubit_num", &QGate::getControlQubitNum,
            "Retrieve the number of control qubits for the QGate instance.\n"
            "This function returns the count of qubits that act as control qubits for the gate.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     An integer representing the number of control qubits.\n"
        )
        .def("get_qubits",
            &QGate::getQuBitVector,
            py::arg("qubits"),
            "Get the qubit vector inside this quantum gate.\n"
            "\n"
            "Args:\n"
            "     qubits: The qubits output vector.\n"
            "\n"
            "Returns:\n"
            "     int: Size of the qubits.\n",
            py::return_value_policy::automatic
        )
        .def("get_control_qubits",
            &QGate::getControlVector,
            py::arg("control_qubits"),
            "Get the control vector from the current quantum gate node.\n"
            "\n"
            "Args:\n"
            "     control_qubits: The control qubits output vector.\n"
            "\n"
            "Returns:\n"
            "     int: Size of the control qubits.\n",
            py::return_value_policy::automatic
        )
        .def("gate_type",
            [](QGate& qgate)
            {
                return qgate.getQGate()->getGateType();
            },
            "Get the type of the quantum gate.\n"
            "\n"
            "Args:\n"
            "     qgate: The quantum gate instance.\n"
            "\n"
            "Returns:\n"
            "     The type of the quantum gate.\n"
        )
        .def("gate_matrix",
            [](QGate& qgate) {
                QStat matrix;
                qgate.getQGate()->getMatrix(matrix);
                return matrix;
            },
            py::return_value_policy::automatic,
            "Get the matrix representation of the quantum gate.\n"
            "\n"
            "Args:\n"
            "     qgate: The quantum gate instance.\n"
            "\n"
            "Returns:\n"
            "     QStat: The matrix representation of the quantum gate.\n"
        );

    /* QIfProg and QWhileProg will use QProg type, so we declare QProg type here, define it's function later, like C++ Forward declarations */
    py::class_<QProg> PyQProgClass(m, "QProg", "Quantum program,can construct quantum circuit,data struct is linked list");

    py::class_<QIfProg>(m, "QIfProg", "quantum if prog node")
        .def(py::init(
            [](NodeIter &iter)
            {
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
            }),
            "Constructor for QIfProg.\n"
            "\n"
            "Args:\n"
            "     iter: An iterator to a node.\n"
            "\n"
            "Raises:\n"
            "     runtime_error: If the iterator is null or the node type is incorrect.\n"
        )
        .def(py::init<ClassicalCondition, QProg>(),
             py::arg("classical_cond"),
             py::arg("true_branch_qprog"),
            "Constructor for initializing with a classical condition and a quantum program.\n"
            "\n"
            "Args:\n"
            "     classical_cond: The classical condition to evaluate.\n"
            "\n"
            "     true_branch_qprog: The quantum program to execute if the condition is true.\n"
        )
        .def(py::init<ClassicalCondition, QProg, QProg>(),
             py::arg("classical_cond"),
             py::arg("true_branch_qprog"),
             py::arg("false_branch_qprog"),
            "Constructor for initializing with a classical condition and two quantum programs.\n"
            "\n"
            "Args:\n"
            "     classical_cond: The classical condition to evaluate.\n"
            "\n"
            "     true_branch_qprog: The quantum program to execute if the condition is true.\n"
            "\n"
            "     false_branch_qprog: The quantum program to execute if the condition is false.\n"
        )
        .def(
            "get_true_branch",
            [](QIfProg &self)
            {
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
            py::return_value_policy::automatic,
            "Get the quantum program corresponding to the true branch.\n"
            "\n"
            "Returns:\n"
            "     QProg: The quantum program for the true branch.\n"
            "\n"
            "Raises:\n"
            "     runtime_error: If the true branch is null or has an incorrect node type.\n"
        )
        .def(
            "get_false_branch",
            [](QIfProg &self)
            {
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
            py::return_value_policy::automatic,
            "Get the quantum program corresponding to the false branch.\n"
            "\n"
            "Returns:\n"
            "     QProg: The quantum program for the false branch, or an empty QProg if null.\n"
            "\n"
            "Raises:\n"
            "     runtime_error: If the false branch has an incorrect node type.\n"
            )
            
        .def("get_classical_condition", &QIfProg::getClassicalCondition, py::return_value_policy::automatic,
            "Retrieve the classical condition associated with the quantum if program.\n"
            "\n"
            "Returns:\n"
            "     The classical condition object used in the if statement.\n"
            );

    py::class_<QWhileProg>(m, "QWhileProg", "quantum while node")
        .def(py::init(
            [](NodeIter& iter)
            {
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
            }),
            "Initialize QWhileProg from a node iterator.\n"
            "\n"
            "Args:\n"
            "     iter (NodeIter&): The iterator pointing to the node.\n"
            "\n"
            "Raises:\n"
            "     runtime_error: If the iterator is null or the node type is incorrect.\n"
        )
        .def(py::init<ClassicalCondition, QProg>())
        .def(
            "get_true_branch",
            [](QWhileProg& self)
            {
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
            py::return_value_policy::automatic,
            "Retrieve the quantum program corresponding to the true branch.\n"
            "\n"
            "Returns:\n"
            "     QProg: The quantum program for the true branch.\n"
            "\n"
            "Raises:\n"
            "     runtime_error: If the true branch is null or has an incorrect node type.\n"
        )
        .def("get_classical_condition",
            &QWhileProg::getClassicalCondition,
            py::return_value_policy::automatic,
            "Retrieve the classical condition associated with the while program.\n"
            "\n"
            "Returns:\n"
            "     The classical condition object used in the while statement.\n"
        );

    py::class_<QMeasure>(m, "QMeasure", "quantum measure node")
        .def(py::init(
            [](NodeIter &iter)
            {
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
            }),
            "Initialize QMeasure from a node iterator.\n"
            "\n"
            "Args:\n"
            "     iter (NodeIter&): The iterator pointing to the node.\n"
            "\n"
            "Returns:\n"
            "     QMeasure: The initialized measurement object.\n"
            "\n"
            "Raises:\n"
            "     runtime_error: If the iterator is null or the node type is incorrect.\n"
        );

    py::class_<QReset>(m, "QReset", "quantum reset node")
        .def(py::init(
            [](NodeIter &iter)
            {
                if (!(*iter))
                {
                    QCERR("iter is null");
                    throw runtime_error("iter is null");
                }

                if (RESET_NODE == (*iter)->getNodeType())
                {
                    auto gate_node = std::dynamic_pointer_cast<AbstractQuantumReset>(*iter);
                    return QReset(gate_node);
                }
                else
                {
                    QCERR("node type error");
                    throw runtime_error("node type error");
                }
            }),
            "Initialize QReset from a node iterator.\n"
            "\n"
            "Args:\n"
            "     iter (NodeIter&): The iterator pointing to the node.\n"
            "\n"
            "Returns:\n"
            "     QReset: The initialized reset object.\n"
            "\n"
            "Raises:\n"
            "     runtime_error: If the iterator is null or the node type is incorrect.\n"
        );

    py::class_<QCircuit>(m, "QCircuit", "quantum circuit node")
        .def(py::init<>())
        .def(py::init(
            [](NodeIter &iter)
            {
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
            }),
            "Initialize QCircuit from a node iterator.\n"
            "\n"
            "Args:\n"
            "     iter (NodeIter&): The iterator pointing to the node.\n"
            "\n"
            "Returns:\n"
            "     QCircuit: The initialized quantum circuit object.\n"
            "\n"
            "Raises:\n"
            "     runtime_error: If the iterator is null or the node type is incorrect.\n"
        )

        .def("__lshift__", &QCircuit::operator<<<QCircuit>, py::return_value_policy::reference,
            "Left shift operator for QCircuit.\n"
            "\n"
            "Args:\n"
            "     other (QCircuit): The circuit to be combined with this circuit.\n"
            "\n"
            "Returns:\n"
            "     QCircuit: A new circuit resulting from the left shift operation.\n"
        )
        .def("__lshift__", &QCircuit::operator<<<QGate>, py::return_value_policy::reference,
            "Left shift operator for QCircuit with a QGate.\n"
            "\n"
            "Args:\n"
            "     other (QGate): The gate to be added to this circuit.\n"
            "\n"
            "Returns:\n"
            "     QCircuit: A new circuit resulting from the left shift operation with the gate.\n"
        )

        .def("is_empty", &QCircuit::is_empty, py::return_value_policy::automatic,
            "Check if the circuit is empty.\n"
            "\n"
            "Returns:\n"
            "     bool: True if the circuit has no gates; otherwise, False.\n"
        )

        .def("insert", &QCircuit::operator<<<QCircuit>, py::return_value_policy::reference,
            "Insert another QCircuit into this circuit.\n"
            "\n"
            "Args:\n"
            "     other (QCircuit): The circuit to be inserted.\n"
            "\n"
            "Returns:\n"
            "     QCircuit: A reference to this circuit after insertion.\n"
        )
        .def("insert", &QCircuit::operator<<<QGate>, py::return_value_policy::reference,
            "Insert a QGate into this circuit.\n"
            "\n"
            "Args:\n"
            "     gate (QGate): The gate to be inserted.\n"
            "\n"
            "Returns:\n"
            "     QCircuit: A reference to this circuit after the gate insertion.\n"
        )
        .def("dagger", &QCircuit::dagger, py::return_value_policy::automatic,
            "Compute the adjoint (dagger) of the circuit.\n"
            "\n"
            "Returns:\n"
            "     QCircuit: The adjoint of this circuit.\n"
        )
        .def("control", &QCircuit::control,
             py::arg("control_qubits"),
             py::return_value_policy::automatic,
            "Apply a control operation to the circuit.\n"
            "\n"
            "Args:\n"
            "     control_qubits (list): A list of qubits that will act as control qubits.\n"
            "\n"
            "Returns:\n"
            "     QCircuit: The circuit with the control operation applied.\n"
        )
        .def("set_dagger", &QCircuit::setDagger,
            "Set the dagger property of the circuit.\n"
            "\n"
            "This method modifies the circuit to represent its adjoint.\n"
        )
        .def("set_control", &QCircuit::setControl,
             py::arg("control_qubits"),
            "Set control qubits for the circuit.\n"
            "\n"
            "Args:\n"
            "     control_qubits (list): A list of qubits to be set as control qubits.\n"
        )
        .def("begin", &QCircuit::getFirstNodeIter, py::return_value_policy::reference,
            "Get an iterator to the first node in the circuit.\n"
            "\n"
            "Returns:\n"
            "     Iterator: An iterator pointing to the first node.\n"
        )
        .def("end", &QCircuit::getEndNodeIter, py::return_value_policy::reference,
            "Get an iterator to the end of the circuit.\n"
            "\n"
            "Returns:\n"
            "     Iterator: An iterator pointing to the end of the nodes.\n"
        )
        .def("last", &QCircuit::getLastNodeIter, py::return_value_policy::reference,
            "Get an iterator to the last node in the circuit.\n"
            "\n"
            "Returns:\n"
            "     Iterator: An iterator pointing to the last node.\n"
        )
        .def("head", &QCircuit::getHeadNodeIter, py::return_value_policy::reference,
            "Get an iterator to the head of the circuit.\n"
            "\n"
            "Returns:\n"
            "     Iterator: An iterator pointing to the head node.\n"
        )
        .def(
            "__str__",
            [](QCircuit &p)
            {
                return draw_qprog(p);
            },
            py::return_value_policy::reference);


    PyQProgClass.def(py::init<>())
        .def(py::init<QProg &>(),
            "Construct a program node.\n"
            "\n"
            "Args:\n"
            "     quantum_prog: The quantum program reference.\n"
            "\n"
            "Returns:\n"
            "     A new program node.\n"
        )
        .def(py::init<QCircuit &>(),
            "Construct a program node from a QCircuit node.\n"
            "\n"
            "Args:\n"
            "     qcircuit: The QCircuit reference.\n"
            "\n"
            "Returns:\n"
            "     A new program node.\n"
        )
        .def(py::init<QIfProg &>(),
            "Construct a program node from a QIfProg node.\n"
            "\n"
            "Args:\n"
            "     qifprog: The QIfProg reference.\n"
            "\n"
            "Returns:\n"
            "     A new program node.\n"
        )
        .def(py::init<QWhileProg &>(),
            "Construct a program node from a QWhileProg node.\n"
            "\n"
            "Args:\n"
            "     qwhileprog: The QWhileProg reference.\n"
            "\n"
            "Returns:\n"
            "     A new program node.\n"
        )
        .def(py::init<QGate &>(),
            "Construct a program node from a QGate node.\n"
            "\n"
            "Args:\n"
            "     qgate: The QGate reference.\n"
            "\n"
            "Returns:\n"
            "     A new program node.\n"
        )
        .def(py::init<QMeasure &>(),
            "Construct a program node from a QMeasure node.\n"
            "\n"
            "Args:\n"
            "     qmeasure: The QMeasure reference.\n"
            "\n"
            "Returns:\n"
            "     A new program node.\n"
        )
        .def(py::init<QReset &>(),
            "Construct a program node from a QReset node.\n"
            "\n"
            "Args:\n"
            "     qreset: The QReset reference.\n"
            "\n"
            "Returns:\n"
            "     A new program node.\n"
        )
        .def(py::init<ClassicalCondition &>(),
            "Construct a program node from a ClassicalCondition node.\n"
            "\n"
            "Args:\n"
            "     classical_condition: The ClassicalCondition reference.\n"
            "\n"
            "Returns:\n"
            "     A new program node.\n"
        )
        .def(py::init(
            [](NodeIter &iter)
            {
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
            }),
            "Construct a program node from a node iterator.\n"
            "\n"
            "Args:\n"
            "     iter: The iterator for the node.\n"
            "\n"
            "Returns:\n"
            "     A new program node.\n"
        )
        /* template function should be instance explicit */
        .def("__lshift__", &QProg::operator<<<QProg>, py::return_value_policy::reference)
        .def("__lshift__", &QProg::operator<<<QGate>, py::return_value_policy::reference)
        .def("__lshift__", &QProg::operator<<<QCircuit>, py::return_value_policy::reference)
        .def("__lshift__", &QProg::operator<<<QIfProg>, py::return_value_policy::reference)
        .def("__lshift__", &QProg::operator<<<QWhileProg>, py::return_value_policy::reference)
        .def("__lshift__", &QProg::operator<<<QMeasure>, py::return_value_policy::reference)
        .def("__lshift__", &QProg::operator<<<QReset>, py::return_value_policy::reference)
        .def("__lshift__", &QProg::operator<<<ClassicalCondition>, py::return_value_policy::reference)

        .def("is_empty", &QProg::is_empty, py::return_value_policy::automatic_reference)
        .def("get_max_qubit_addr", &QProg::get_max_qubit_addr, py::return_value_policy::automatic_reference)
        .def("get_qgate_num", &QProg::get_qgate_num, py::return_value_policy::automatic_reference)
        .def("get_used_qubits", 
            [](QProg &self, QVec& qv) 
            {
                self.get_used_qubits(qv);
                return qv;
                
            },
            py::arg("qubit_vector"),
            "Get a list of qubits used in the program.\n"
            "\n"
            "Args:\n"
            "     qubit_vector: The vector to store the used qubits.\n"
            "\n"
            "Returns:\n"
            "     A reference to the updated qubit vector.\n",
            py::return_value_policy::reference
            )
        .def("get_used_cbits",
             [](QProg &self, std::vector<ClassicalCondition>& cv)
             {
                self.get_used_cbits(cv);
                return cv;

             },
             py::arg("cbit_vector"),
             "Get a list of classical bits used in the program.\n"
            "\n"
             "Args:\n"
             "     cbit_vector: The vector to store the used classical bits.\n"
            "\n"
             "Returns:\n"
             "     A reference to the updated classical bit vector.\n",
             py::return_value_policy::reference
             )
        .def("is_measure_last_pos", &QProg::is_measure_last_pos,
            "Check if the last operation in the program is a measurement.\n"
            "\n"
            "Returns:\n"
            "     True if the last operation is a measurement, otherwise False.\n"
        )
        .def("insert", &QProg::operator<<<QProg>,
            "Insert a program into the current program.\n"
            "\n"
            "Args:\n"
            "     program: The program to be inserted.\n"
            "\n"
            "Returns:\n"
            "     A reference to the updated program after insertion.\n",
            py::return_value_policy::reference
        )
        .def("insert", &QProg::operator<<<QGate>, py::return_value_policy::reference,
            "Insert a gate into the current program.\n"
            "\n"
            "Args:\n"
            "     gate: The gate to be inserted.\n"
            "\n"
            "Returns:\n"
            "     A reference to the updated program after insertion.\n"
            )
        .def("insert", &QProg::operator<<<QCircuit>, py::return_value_policy::reference,
            "Insert a circuit into the current program.\n"
            "\n"
            "Args:\n"
            "     circuit: The circuit to be inserted.\n"
            "\n"
            "Returns:\n"
            "     A reference to the updated program after insertion.\n")
        .def("insert", &QProg::operator<<<QIfProg>, py::return_value_policy::reference,
            "Insert an if program into the current program.\n"
            "\n"
            "Args:\n"
            "     if_program: The if program to be inserted.\n"
            "\n"
            "Returns:\n"
            "     A reference to the updated program after insertion.\n")
        .def("insert", &QProg::operator<<<QWhileProg>, py::return_value_policy::reference,
            "Insert a QWhileProg into the program.\n"
            "\n"
            "Returns:\n"
            "     A reference to the updated program.\n")
        .def("insert", &QProg::operator<<<QMeasure>, py::return_value_policy::reference,
            "Insert a QMeasure into the program.\n"
            "\n"
            "Returns:\n"
            "     A reference to the updated program.\n")
        .def("insert", &QProg::operator<<<QReset>, py::return_value_policy::reference,
            "Insert a QReset into the program.\n"
            "\n"
            "Args:\n"
            "     reset_op: The reset operation to be inserted.\n"
            "\n"
            "Returns:\n"
            "     A reference to the updated program.\n")
        .def("insert", &QProg::operator<<<ClassicalCondition>, py::return_value_policy::reference,
            "Insert a ClassicalCondition into the program.\n"
            "\n"
            "Args:\n"
            "     condition_op: The classical condition operation to be inserted.\n"
            "\n"
            "Returns:\n"
            "     A reference to the updated program.\n")
        .def("begin", &QProg::getFirstNodeIter, py::return_value_policy::reference,
            "Get an iterator to the first node in the program.\n"
            "\n"
            "Returns:\n"
            "     A reference to the iterator pointing to the first node.\n")
        .def("end", &QProg::getEndNodeIter, py::return_value_policy::reference,
            "Get an iterator to the end of the program.\n"
            "\n"
            "Returns:\n"
            "     A reference to the iterator pointing past the last node.\n")
        .def("head", &QProg::getHeadNodeIter, py::return_value_policy::reference,
            "Get an iterator to the head node of the program.\n"
            "\n"
            "Returns:\n"
            "     A reference to the iterator pointing to the head node.\n")
        .def("last", &QProg::getLastNodeIter, py::return_value_policy::reference,
            "Get an iterator to the last node in the program.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A reference to the iterator pointing to the last node.\n")
        .def(
            "__str__",
            [](QProg &p)
            {
                return draw_qprog(p);
            },
            py::return_value_policy::reference,
            "Get a string representation of the quantum program.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A string representing the quantum program.\n");

    py::implicitly_convertible<QGate, QProg>();
    py::implicitly_convertible<QCircuit, QProg>();
    py::implicitly_convertible<QIfProg, QProg>();
    py::implicitly_convertible<QWhileProg, QProg>();
    py::implicitly_convertible<QMeasure, QProg>();
    py::implicitly_convertible<QReset, QProg>();
    py::implicitly_convertible<ClassicalCondition, QProg>();

    /* hide */
    py::class_<HadamardQCircuit, QCircuit>(m, "hadamard_circuit","hadamard circuit class")
        .def(py::init<QVec &>());

    py::class_<Encode>(m, "Encode", "quantum amplitude encode")
        .def(py::init<>())
        .def("amplitude_encode",
             py::overload_cast<const QVec &, const std::vector<double> &>(&Encode::amplitude_encode),
             py::arg("qubit"),
             py::arg("data").noconvert(),
             py::return_value_policy::automatic,
            "Perform amplitude encoding on the given qubits.\n"
            "\n"
            "Args:\n"
            "     qubit: The quantum vector to be encoded.\n"
            "\n"
            "     data: The classical data to be encoded.\n"
            "\n"
            "Returns:\n"
            "     An encoded quantum state.\n")
        .def("amplitude_encode",
            py::overload_cast<const QVec&, const std::vector<std::complex<double>>&>(&Encode::amplitude_encode),
            py::arg("qubit"),
            py::arg("data").noconvert(),
            py::return_value_policy::automatic,
            "Perform amplitude encoding using complex numbers on the given qubits.\n"
            "\n"
            "Args:\n"
            "     qubit: The quantum vector to be encoded.\n"
            "\n"
            "     data: The classical complex data to be encoded.\n"
            "\n"
            "Returns:\n"
            "     An encoded quantum state.\n")
        .def("amplitude_encode_recursive",
             py::overload_cast<const QVec &, const std::vector<double> &>(&Encode::amplitude_encode_recursive),
             py::arg("qubit"),
             py::arg("data").noconvert(),
             py::return_value_policy::automatic,
            "Perform recursive amplitude encoding on the given qubits.\n"
            "\n"
            "Args:\n"
            "     qubit: The quantum vector to be encoded.\n"
            "\n"
            "     data: The classical data to be encoded.\n"
            "\n"
            "Returns:\n"
            "     An encoded quantum state.\n")
        .def("amplitude_encode_recursive",
             py::overload_cast<const QVec &, const QStat &>(&Encode::amplitude_encode_recursive),
             py::arg("qubit"),
             py::arg("data").noconvert(),
            py::return_value_policy::automatic,
            "Encode by amplitude recursively.\n"
            "\n"
            "Args:\n"
            "     QVec: qubits\n"
            "\n"
            "     QStat: amplitude\n"
            "\n"
            "Returns:\n"
            "     circuit\n")
        .def("angle_encode",
             &Encode::angle_encode,
             py::arg("qubit"),
             py::arg("data"),
             py::arg_v("gate_type", GateType::RY_GATE, "GateType.RY_GATE"),
            "Encode by angle.\n"
            "\n"
            "Args:\n"
            "     QVec: qubits \n"
            "\n"
            "     prob_vec: data \n"
            "\n"
            "Returns:\n"
            "     circuit.\n",
            py::return_value_policy::automatic)
        .def("dense_angle_encode",
             &Encode::dense_angle_encode,
             py::arg("qubit"),
             py::arg("data"),
            "Encode by dense angle.\n"
            "\n"
            "Args:\n"
            "     QVec: qubits \n"
            "\n"
            "     prob_vec: data \n"
            "\n"
            "Returns:\n"
            "     circuit\n",
            py::return_value_policy::automatic)
        .def("iqp_encode",
             &Encode::iqp_encode,
             py::arg("qubit"),
             py::arg("data"),
             py::arg("control_list") = std::vector<std::pair<int, int>>{},
             py::arg("bool_inverse") = false,
             py::arg("repeats") = 1,
            "Encode by IQP.\n"
            "\n"
            "Args:\n"
            "     QVec: qubits \n"
            "\n"
            "     prob_vec: data \n"
            "\n"
            "     list: control_list \n"
            "\n"
            "     bool: bool_inverse \n"
            "\n"
            "     int: repeats \n"
            "\n"
            "Returns:\n"
            "     circuit.\n",
            py::return_value_policy::automatic)
        .def("basic_encode",
             &Encode::basic_encode,
             py::arg("qubit"),
             py::arg("data"),
            "Basic encoding.\n"
            "\n"
            "Args:\n"
            "     QVec: qubits \n"
            "\n"
            "     string: data \n"
            "\n"
            "Returns:\n"
            "     circuit\n",
            py::return_value_policy::automatic)
        .def("dc_amplitude_encode",
             &Encode::dc_amplitude_encode,
             py::arg("qubit"),
             py::arg("data"),
            "Encode by DC amplitude.\n"
            "\n"
            "Args:\n"
            "     QVec: qubits \n"
            "\n"
            "     QStat: amplitude \n"
            "\n"
            "Returns:\n"
            "     circuit\n",
            py::return_value_policy::automatic)
        .def("schmidt_encode",
             &Encode::schmidt_encode,
             py::arg("qubit"),
             py::arg("data"),
             py::arg("cutoff"),
             "Encode by schmidt.\n"
            "\n"
             "Args:\n"
             "     QVec: qubits \n"
             "\n"
             "     QStat: amplitude \n"
             "\n"
             "     double: cutoff \n"
            "\n"
             "Returns:\n"
             "     circuit\n",
             py::return_value_policy::automatic)
        .def("bid_amplitude_encode",
             &Encode::bid_amplitude_encode,
             py::arg("qubit"),
             py::arg("data"),
             py::arg("split") = 0,
             "Encode by bid.\n"
             "\n"
             "Args:\n"
             "     QVec: qubits \n"
             "\n"
             "     QStat: amplitude \n"
             "\n"
             "     split: int \n"
            "\n"
             "Returns:\n"
             "     circuit\n",
             py::return_value_policy::automatic)
        .def("ds_quantum_state_preparation",
             py::overload_cast<const QVec &, const std::map<std::string, double> &>(&Encode::ds_quantum_state_preparation),
             py::arg("qubit"),
             py::arg("data").noconvert(),
             "Prepare a quantum state.\n"
             "\n"
             "Args:\n"
             "     QVec: qubits \n"
             "\n"
             "     std::map<std::string, double>: state parameters \n"
            "\n"
             "Returns:\n"
             "     circuit\n",
             py::return_value_policy::automatic)
        .def("ds_quantum_state_preparation",
             py::overload_cast<const QVec &, const std::map<std::string, std::complex<double>> &>(&Encode::ds_quantum_state_preparation),
             py::arg("qubit"),
             py::arg("data").noconvert(),
             "Prepare a quantum state.\n"
             "\n"
             "Args:\n"
             "     QVec: qubits \n"
             "\n"
             "     std::map<std::string, std::complex<double>>: state parameters \n"
             "\n"
             "Returns:\n"
             "     circuit\n",
             py::return_value_policy::automatic)
        .def("ds_quantum_state_preparation",
             py::overload_cast<const QVec &, const std::vector<double> &>(&Encode::ds_quantum_state_preparation),
             py::arg("qubit"),
             py::arg("data").noconvert(),
             py::return_value_policy::automatic,
             "Prepare a quantum state.\n"
            "\n"
             "Args:\n"
             "     QVec: qubits \n"
             "\n"
             "     std::vector<double>: state parameters \n"
            "\n"
             "Returns:\n"
             "     circuit\n"
             )
        .def("ds_quantum_state_preparation",
             py::overload_cast<const QVec &, const std::vector<std::complex<double>> &>(&Encode::ds_quantum_state_preparation),
             py::arg("qubit"),
             py::arg("data").noconvert(),
             py::return_value_policy::automatic,
             "Prepare a quantum state.\n"
            "\n"
             "Args:\n"
             "     QVec: qubits \n"
             "\n"
             "     std::vector<std::complex<double>>: state parameters \n"
            "\n"
             "Returns:\n"
             "     circuit\n"
             )
        .def("sparse_isometry",
             py::overload_cast<const QVec &, const std::map<std::string, double> &>(&Encode::sparse_isometry),
             py::arg("qubit"),
             py::arg("data").noconvert(),
             py::return_value_policy::automatic,
             "Perform a sparse isometry operation.\n"
            "\n"
             "Args:\n"
             "     QVec: qubits \n"
             "\n"
             "     std::map<std::string, double>: parameters for the isometry \n"
            "\n"
             "Returns:\n"
             "     circuit\n"
             )
        .def("sparse_isometry",
             py::overload_cast<const QVec &, const std::map<std::string, complex<double>> &>(&Encode::sparse_isometry),
             py::arg("qubit"),
             py::arg("data").noconvert(),
             py::return_value_policy::automatic,
             "Perform a sparse isometry operation.\n"
            "\n"
             "Args:\n"
             "     QVec: qubits \n"
             "\n"
             "     std::map<std::string, std::complex<double>>: parameters for the isometry \n"
            "\n"
             "Returns:\n"
             "     circuit\n"
             )
        .def("sparse_isometry",
             py::overload_cast<const QVec &, const std::vector<double> &>(&Encode::sparse_isometry),
             py::arg("qubit"),
             py::arg("data").noconvert(),
             py::return_value_policy::automatic,
            "Perform a sparse isometry operation.\n"
            "\n"
            "Args:\n"
            "     QVec: qubits \n"
            "\n"
            "     std::vector<double>: parameters for the isometry \n"
            "\n"
            "Returns:\n"
            "     circuit\n"
        )
        .def("sparse_isometry",
             py::overload_cast<const QVec &, const std::vector<complex<double>> &>(&Encode::sparse_isometry),
             py::arg("qubit"),
             py::arg("data").noconvert(),
             py::return_value_policy::automatic,
             "Perform a sparse isometry operation.\n"
             "\n"
             "Args:\n"
             "     QVec: qubits \n"
             "\n"
             "     std::vector<std::complex<double>>: parameters for the isometry \n"
            "\n"
             "Returns:\n"
             "     circuit\n"
             )
        .def("efficient_sparse",
             py::overload_cast<const QVec &, const std::map<std::string, double> &>(&Encode::efficient_sparse),
             py::arg("qubit"),
             py::arg("data").noconvert(),
             py::return_value_policy::automatic,
             "Perform an efficient sparse operation.\n"
            "\n"
             "Args:\n"
             "     QVec: qubits \n"
             "\n"
             "     std::map<std::string, double>: parameters for the operation \n"
            "\n"
             "Returns:\n"
             "     circuit\n"
             )
        .def("efficient_sparse",
             py::overload_cast<const QVec &, const std::map<std::string, complex<double>> &>(&Encode::efficient_sparse),
             py::arg("qubit"),
             py::arg("data").noconvert(),
             py::return_value_policy::automatic,
             "Perform an efficient sparse operation.\n"
            "\n"
             "Args:\n"
             "     QVec: qubits \n"
             "\n"
             "     std::map<std::string, std::complex<double>>: parameters for the operation \n"
            "\n"
             "Returns:\n"
             "     circuit\n"
             )
        .def("efficient_sparse",
             py::overload_cast<const QVec &, const std::vector<double> &>(&Encode::efficient_sparse),
             py::arg("qubit"),
             py::arg("data").noconvert(),
             py::return_value_policy::automatic,
            "Perform an efficient sparse operation.\n"
            "\n"
            "Args:\n"
            "     QVec: qubits \n"
            "\n"
            "     std::vector<double>: parameters for the operation \n"
            "\n"
            "Returns:\n"
            "     circuit\n"
        )
        .def("efficient_sparse",
             py::overload_cast<const QVec &, const std::vector<complex<double>> &>(&Encode::efficient_sparse),
             py::arg("qubit"),
             py::arg("data").noconvert(),
             py::return_value_policy::automatic,
            "Perform an efficient sparse operation.\n"
            "\n"
            "Args:\n"
            "     QVec: qubits \n"
            "\n"
            "     std::vector<std::complex<double>>: parameters for the operation \n"
            "\n"
            "Returns:\n"
            "     circuit\n"
        )
        .def("approx_mps", [](Encode &self, const QVec &q, const std::vector<double>& data,
                const int &layers = 3, const int &step = 100, const bool double2float = false) {
            if (double2float) {
                vector<float>data_tmp(data.size());
                for (int i = 0; i < data_tmp.size(); ++i) {
                    data_tmp[i] = static_cast<float>(data[i]);
                }
                return self.approx_mps_encode<float>(q, data_tmp, layers, step);
            }
            return self.approx_mps_encode<double>(q, data, layers, step);

        },
        py::arg("qubit"),
        py::arg("data").noconvert(),
        py::arg("layers") = 3,
        py::arg("sweeps") = 100,
        py::arg("double2float") = false,
        py::return_value_policy::automatic,
        "Approximate Matrix Product State encoding.\n"
        "\n"
        "Args:\n"
        "     QVec: qubits \n"
        "\n"
        "     std::vector<double>: input data \n"
        "\n"
        "     int: number of layers for encoding (default: 3)\n"
        "\n"
        "     int: number of sweeps for optimization (default: 100)\n"
        "\n"
        "     bool: flag to convert double data to float (default: false)\n"
        "\n"
        "Returns:\n"
        "     Encoded circuit based on input parameters.\n"
        "\n"
        "Raises:\n"
        "     run_fail: An error occurred during the encoding process.\n"
        )
        .def("approx_mps",
        [](Encode self, const QVec &q, const std::vector<qcomplex_t>& data,
            const int &layers = 3, const int &step = 100) {
            return self.approx_mps_encode<qcomplex_t>(q, data, layers, step);
        },
        py::arg("qubit"),
        py::arg("data").noconvert(),
        py::arg("layers") = 3,
        py::arg("sweeps") = 100,
        "Approximate Matrix Product State encoding.\n"
        "\n"
        "Args:\n"
        "     QVec: qubits \n"
        "\n"
        "     std::vector<qcomplex_t>: input data \n"
        "\n"
        "     int: number of layers (default: 3)\n"
        "\n"
        "     int: number of steps (default: 100)\n"
        "\n"
        "Returns:\n"
        "     Encoded circuit.\n",
        py::return_value_policy::automatic
        )
        
        .def("get_circuit", &Encode::get_circuit,
            "Retrieve the circuit from the encoder.\n"
            "\n"
            "Returns:\n"
            "     The corresponding circuit object.\n"
        )
        .def("get_out_qubits", &Encode::get_out_qubits,
            "Retrieve the output qubits from the encoder.\n"
            "\n"
            "Returns:\n"
            "     A vector of output qubits.\n"
        )
        .def("get_fidelity",
            py::overload_cast<const vector<double> &>(&Encode::get_fidelity),
            py::arg("data").noconvert(),
            py::return_value_policy::automatic,
            "Calculate the fidelity based on the provided data.\n"
            "\n"
            "Args:\n"
            "     data: A vector of doubles representing the input data.\n"
            "\n"
            "Returns:\n"
            "     The calculated fidelity value.\n"
            )
        .def("get_fidelity",
            py::overload_cast<const vector<qcomplex_t> &>(&Encode::get_fidelity),
            py::arg("data").noconvert(),
            py::return_value_policy::automatic,
            "Calculate the fidelity based on the provided complex data.\n"
            "\n"
            "Args:\n"
            "     data: A vector of qcomplex_t representing the input data.\n"
            "\n"
            "Returns:\n"
            "     The calculated fidelity value.\n"
            )
        .def("get_fidelity",
            py::overload_cast<const vector<float> &>(&Encode::get_fidelity),
            py::arg("data").noconvert(),
            py::return_value_policy::automatic,
            "Calculate the fidelity based on the provided float data.\n"
            "\n"
            "Args:\n"
            "     data: A vector of floats representing the input data.\n"
            "\n"
            "Returns:\n"
            "     The calculated fidelity value.\n"
            );

    /* use std::unique_ptr with py::nodelete as python object holder guarantee python object not own the C++ object */
    py::class_<OriginQubitPool, std::unique_ptr<OriginQubitPool, py::nodelete>>(m, "OriginQubitPool", "quantum qubit pool")
        .def(py::init(
            []()
            {
                return std::unique_ptr<OriginQubitPool, py::nodelete>(OriginQubitPool::get_instance());
            }),
            "Initialize the OriginQubitPool singleton instance.\n"
            "\n"
            "Returns:\n"
            "     A reference to the existing OriginQubitPool instance.\n"
        )
        .def("get_capacity", &OriginQubitPool::get_capacity,
            "Get the capacity of the OriginQubitPool.\n"
            "\n"
            "Returns:\n"
            "     An integer representing the capacity of the pool.\n"
        )
        .def("set_capacity", &OriginQubitPool::set_capacity,
            "Set the capacity of the OriginQubitPool.\n"
            "\n"
            "Args:\n"
            "     capacity: An integer representing the new capacity to be set.\n"
        )
        .def("get_qubit_by_addr",
             &OriginQubitPool::get_qubit_by_addr,
             py::arg("qubit_addr"),
             /* py::return_value_policy::reference_internal will guarantee the class life time longer than returned reference */
             py::return_value_policy::reference_internal,
            "Retrieve a qubit from the pool using its address.\n"
            "\n"
            "Args:\n"
            "     qubit_addr: The address of the qubit to retrieve.\n"
            "Returns:\n"
            "     A reference to the requested qubit.\n"
        )
        .def("clearAll", &OriginQubitPool::clearAll,
            "Clear all qubits from the OriginQubitPool.\n"
            "This method removes all qubits, resetting the pool to its initial state.\n"
        )
        .def("getMaxQubit", &OriginQubitPool::getMaxQubit,
            "Retrieve the maximum qubit from the OriginQubitPool.\n"
            "\n"
            "Returns:\n"
            "     The maximum qubit available in the pool.\n"
        )
        .def("getIdleQubit", &OriginQubitPool::getIdleQubit,
            "Retrieve an idle qubit from the OriginQubitPool.\n"
            "\n"
            "Returns:\n"
            "     An idle qubit if available, otherwise may return a null reference or indicate no idle qubits.\n"
        )
        .def("get_max_usedqubit_addr", &OriginQubitPool::get_max_usedqubit_addr,
            "Retrieve the address of the maximum used qubit in the OriginQubitPool.\n"
            "\n"
            "Returns:\n"
            "     The address of the maximum used qubit, or an indication if no qubits are in use.\n"
        )
        .def("allocateQubitThroughPhyAddress",
             &OriginQubitPool::allocateQubitThroughPhyAddress,
             py::arg("qubit_addr"),
             py::return_value_policy::reference_internal,
            "Allocate a qubit using its physical address.\n"
            "\n"
            "Args:\n"
            "     qubit_addr: The physical address of the qubit to allocate.\n"
            "\n"
            "Returns:\n"
            "     A reference to the allocated qubit.\n"
        )
        .def("allocateQubitThroughVirAddress",
             &OriginQubitPool::allocateQubitThroughVirAddress,
             py::arg("qubit_num"),
             py::return_value_policy::reference_internal,
            "Allocate a qubit using its virtual address.\n"
            "\n"
            "Args:\n"
            "     qubit_num: The virtual address of the qubit to allocate.\n"
            "\n"
            "Returns:\n"
            "     A reference to the allocated qubit.\n"
        )
        .def("getPhysicalQubitAddr",
             &OriginQubitPool::getPhysicalQubitAddr,
             py::arg("qubit"),
            "Retrieve the physical address of a specified qubit.\n"
            "\n"
            "Args:\n"
            "     qubit: The qubit for which to retrieve the physical address.\n"
            "\n"
            "Returns:\n"
            "     The physical address of the specified qubit.\n")
        .def("getVirtualQubitAddress",
             &OriginQubitPool::getVirtualQubitAddress,
             py::arg("qubit"),
            "Retrieve the virtual address of a specified qubit.\n"
            "\n"
            "Args:\n"
            "     qubit: The qubit for which to retrieve the virtual address.\n"
            "\n"
            "Returns:\n"
            "     The virtual address of the specified qubit.\n")
        .def(
            "get_allocate_qubits",
            [](OriginQubitPool &self)
            {
                QVec qubits;
                self.get_allocate_qubits(qubits);
                return qubits;
            },
            "Retrieve currently allocated qubits.\n"
            "\n"
            "Returns:\n"
            "     A reference to the vector of currently allocated qubits.\n",
            py::return_value_policy::reference_internal)

        .def("qAlloc", &OriginQubitPool::qAlloc,
            "Allocate a qubit.\n"
            "\n"
            "Returns:\n"
            "     A reference to the allocated qubit.\n",
            py::return_value_policy::reference_internal)
        .def("qAlloc_many",
            [](OriginQubitPool &self, size_t qubit_num)
            {
                auto qv = static_cast<std::vector<Qubit*>>(self.qAllocMany(qubit_num));
                return qv;
            },
             py::arg("qubit_num"),
            "Allocate a list of qubits.\n"
            "\n"
            "Args:\n"
            "     qubit_num: The number of qubits to allocate.\n"
            "\n"
            "Returns:\n"
            "     A reference to the vector of allocated qubits.\n",
            py::return_value_policy::reference_internal)
        .def("qFree", &OriginQubitPool::qFree,
            "Free a previously allocated qubit.\n"
            "\n"
            "Args:\n"
            "     qubit: The qubit to be freed.")
        .def("qFree_all", py::overload_cast<QVec &>(&OriginQubitPool::qFreeAll),
            "Free all qubits in the specified vector.\n"
            "\n"
            "Args:\n"
            "     qubits: A vector of qubits to be freed.\n")
        .def("qFree_all", py::overload_cast<>(&OriginQubitPool::qFreeAll),
            "Free all allocated qubits in the pool.\n"
            "This method releases all qubits that have been allocated previously.\n");

    py::class_<OriginCMem, std::unique_ptr<OriginCMem, py::nodelete>>(m, "OriginCMem", "origin quantum cmem")
        .def(py::init(
            []()
            {
                return std::unique_ptr<OriginCMem, py::nodelete>(OriginCMem::get_instance());
            }),
            "Create an instance of OriginCMem.\n"
            "This constructor returns a singleton instance of OriginCMem.\n")
        .def("get_capacity", &OriginCMem::get_capacity,
            "Get the capacity of the memory.\n"
            "\n"
            "Returns:\n"
            "     The total capacity of the memory in terms of qubits.\n")
        .def("set_capacity", &OriginCMem::set_capacity,
            "Set the capacity of the memory.\n"
            "\n"
            "Args:\n"
            "     capacity: The new capacity for the memory in terms of qubits.\n")
        .def("get_cbit_by_addr",
            &OriginCMem::get_cbit_by_addr,
            py::arg("cbit_addr"),
            py::return_value_policy::reference_internal,
            "Get a classical bit by its address.\n"
            "\n"
            "Args:\n"
            "     cbit_addr: The address of the classical bit.\n"
            "\n"
            "Returns:\n"
            "     A reference to the classical bit associated with the given address\n.")
        .def("Allocate_CBit", py::overload_cast<>(&OriginCMem::Allocate_CBit), py::return_value_policy::reference_internal,
            "Allocate a classical bit.\n"
            "This method allocates a new classical bit and returns a reference to it.\n")
        .def("Allocate_CBit",
            py::overload_cast<size_t>(&OriginCMem::Allocate_CBit),
            py::arg("cbit_num"),
            py::return_value_policy::reference_internal,
            "Allocate a specified number of classical bits.\n"
            "\n"
            "Args:\n"
            "     cbit_num: The number of classical bits to allocate.\n"
            "\n"
            "Returns:\n"
            "     A reference to the allocated classical bits.\n")
        .def("getMaxMem", &OriginCMem::getMaxMem,
            "Get the maximum memory capacity.\n"
            "\n"
            "Returns:\n"
            "     The maximum memory capacity in terms of qubits.")
        .def("getIdleMem", &OriginCMem::getIdleMem,
            "Get the amount of idle memory currently available.\n"
            "\n"
            "Returns:\n"
            "     The amount of idle memory in terms of qubits.\n")
        .def("Free_CBit",
            &OriginCMem::Free_CBit,
            py::arg("cbit"),
            "Free a previously allocated classical bit.\n"
            "\n"
            "Args:\n"
            "     cbit: The classical bit to be freed.")
        .def("clearAll", &OriginCMem::clearAll,
            "Clear all allocated classical bits.\n"
            "This method releases all resources associated with classical bits.\n")

        .def(
            "get_allocate_cbits",
            [](OriginCMem& self)
            {
                vector<ClassicalCondition> cc_vec;
                self.get_allocate_cbits(cc_vec);
                return cc_vec;
            },
            "Get allocate cbits.",
            py::return_value_policy::reference_internal,
            "Retrieve allocated classical bits.\n"
            "Returns a vector of ClassicalCondition representing the allocated cbits.\n")

        .def("cAlloc", py::overload_cast<>(&OriginCMem::cAlloc),
            "Allocate memory for classical bits.\n"
            "This method initializes or resets the memory allocation for classical bits.\n")
        .def("cAlloc",
            py::overload_cast<size_t>(&OriginCMem::cAlloc),
            "Allocate memory for classical bits.\n"
            "This method initializes or resets the memory allocation for classical bits.\n")
        .def("cAlloc_many",
            &OriginCMem::cAllocMany,
            py::arg("count"),
            "Allocate memory for multiple classical bits.\n"
            "\n"
            "Args:\n"
            "     count: The number of classical bits to allocate.\n")
        .def("cFree",
            &OriginCMem::cFree,
            py::arg("classical_cond"),
            "Free the allocated memory for a classical condition.\n"
            "\n"
            "Args:\n"
            "     classical_cond: The classical condition to be freed.\n")
        .def("cFree_all",
            py::overload_cast<std::vector<ClassicalCondition> &>(&OriginCMem::cFreeAll),
            py::arg("classical_cond_list"),
            "Free memory for a list of classical conditions.\n"
            "\n"
            "Args:\n"
            "     classical_cond_list: A vector of classical conditions to be freed.\n")
        .def("cFree_all", py::overload_cast<>(&OriginCMem::cFreeAll),
            "Free all allocated classical memory.\n"
            "This method releases all memory associated with classical conditions.\n");

    py::implicitly_convertible<ClassicalCondition, ClassicalProg>();
    py::implicitly_convertible<cbit_size_t, ClassicalCondition>();

    /*******************************************************************
     *                           QProgDAG
     */
    py::class_<QProgDAGEdge>(m, "QProgDAGEdge", "quantum prog dag edge")
        .def(py::init<uint32_t, uint32_t, uint32_t>(),
             py::arg("from_arg"),
             py::arg("to_arg"),
             py::arg("qubit_arg"),
            "Initialize a quantum program DAG edge.\n"
            "\n"
            "Args:\n"
            "     from_arg: The starting node of the edge.\n"
            "\n"
            "     to_arg: The ending node of the edge.\n"
            "\n"
            "     qubit_arg: The qubit associated with the edge.\n")
        .def_readwrite("m_from", &QProgDAGEdge::m_from)
        .def_readwrite("m_to", &QProgDAGEdge::m_to)
        .def_readwrite("m_qubit", &QProgDAGEdge::m_qubit);

    py::class_<QProgDAGVertex>(m, "QProgDAGVertex", "quantum prog dag vertex node")
        .def(py::init<>(),
            "Initialize a quantum program DAG vertex.\n"
            "This vertex represents a node in the quantum program's directed acyclic graph (DAG).")
        .def_readwrite("m_id", &QProgDAGVertex::m_id)
        .def_readwrite("m_type", &QProgDAGVertex::m_type)
        .def_readwrite("m_layer", &QProgDAGVertex::m_layer)
        .def_readwrite("m_pre_node", &QProgDAGVertex::m_pre_node)
        .def_readwrite("m_succ_node", &QProgDAGVertex::m_succ_node)
        .def(
            "is_dagger",
            [](QProgDAGVertex& self)
            {
                return self.m_node->m_dagger;
            },
            py::return_value_policy::reference,
            "Check if the vertex is a dagger operation.")
        .def(
            "get_iter",
            [](QProgDAGVertex& self)
            {
                return self.m_node->m_itr;
            },
            py::return_value_policy::reference_internal,
            "Retrieve the iterator associated with the vertex.")
        .def(
            "get_qubits_vec",
            [](QProgDAGVertex& self)
            {
                return self.m_node->m_qubits_vec;
            },
            py::return_value_policy::reference_internal,
            "Retrieve the vector of qubits associated with the vertex.")
        .def(
            "get_control_vec",
            [](QProgDAGVertex& self)
            {
                return self.m_node->m_control_vec;
            },
            py::return_value_policy::reference_internal,
            "Retrieve the vector of control qubits associated with the vertex.");

    py::class_<QProgDAG>(m, "QProgDAG", "quantum prog dag class")
        .def(py::init<>())
        .def("get_vertex_set",
            py::overload_cast<>(&QProgDAG::get_vertex),
            /* as vector is copied, but element of containor is reference */
            py::return_value_policy::reference_internal,
            "Retrieve the set of vertices in the quantum program DAG.\n"
            "\n"
            "Args:\n"
            "     QVec: The set of vertices.\n"
            "\n"
            "Returns:\n"
            "     QVec: A reference to the vector of vertices in the DAG.\n"
        )
        .def("get_target_vertex",
            py::overload_cast<const size_t>(&QProgDAG::get_vertex, py::const_),
            py::arg("vertice_num"),
            py::return_value_policy::reference_internal,
            "Retrieve a target vertex from the quantum program DAG.\n"
            "\n"
            "Args:\n"
            "     vertice_num: The index of the vertex to retrieve.\n"
            "\n"
            "Returns:\n"
            "     QVertex: A reference to the specified vertex in the DAG.\n"
        )
        .def("get_edges",
            [](QProgDAG& self)
            {
                auto edges_set = self.get_edges();
                std::vector<QProgDAGEdge> edges_vec;
                for (const auto& _e : edges_set)
                {
                    edges_vec.emplace_back(_e);
                }
                return edges_vec;
            },
            py::return_value_policy::automatic,
            "Retrieve the set of edges in the quantum program DAG.\n"
            "\n"
            "Returns:\n"
            "     List[QProgDAGEdge]: A list of edges in the DAG.\n"
        );

    m.def("prog_to_dag",
        [](QProg prog)
        {
            QProgToDAG prog_to_dag;
            auto dag = QProgDAG();
            prog_to_dag.traversal(prog, dag);

            return dag;
        },
        py::arg("prog"),
        py::return_value_policy::reference,
        "Convert a quantum program into a directed acyclic graph (DAG).\n"
        "\n"
        "Args:\n"
        "     prog: The quantum program to be converted.\n"
        "\n"
        "Returns:\n"
        "     QProgDAG: A reference to the resulting DAG.\n"
    );

    py::class_<QuantumStateTomography>(m, "QuantumStateTomography", "quantum state tomography class")
        .def(py::init<>())
        .def("combine_qprogs",
             py::overload_cast<const QProg &, const QVec &>(&QuantumStateTomography::combine_qprogs<QProg>),
             py::arg("circuit"),
             py::arg("qlist"),
                 "Return a list of quantum state tomography quantum programs.\n"
                 "\n"
                 "Args:\n"
                 "     circuit: The quantum circuit to be combined.\n"
                 "\n"
                 "     qlist: The list of qubits involved.\n"
                 "\n"
                 "Returns:\n"
                 "     A reference to the combined quantum programs.\n",
                 py::return_value_policy::reference_internal)

        .def("combine_qprogs",
             py::overload_cast<const QCircuit &, const QVec &>(&QuantumStateTomography::combine_qprogs<QCircuit>),
             py::arg("circuit"), py::arg("qlist"),
            "Return a list of quantum state tomography quantum programs.\n"
            "\n"
            "Args:\n"
            "     circuit: The quantum circuit to be combined.\n"
            "\n"
            "     qlist: The list of qubits involved.\n"
            "\n"
            "Returns:\n"
            "     A reference to the combined quantum programs.\n",
            py::return_value_policy::reference_internal)
        .def("combine_qprogs",
             py::overload_cast<const QProg &, const std::vector<size_t> &>(&QuantumStateTomography::combine_qprogs<QProg>),
             py::arg("circuit"),
             py::arg("qlist"),
            "Return a list of quantum state tomography quantum programs.\n"
            "\n"
            "Args:\n"
            "     circuit: The quantum circuit to be combined.\n"
            "\n"
            "     qlist: A vector of indices representing the qubits involved.\n"
            "\n"
            "Returns:\n"
            "     A reference to the combined quantum programs.\n",
            py::return_value_policy::reference_internal)
        .def("combine_qprogs",
             py::overload_cast<const QCircuit &, const std::vector<size_t> &>(&QuantumStateTomography::combine_qprogs<QCircuit>),
             py::arg("circuit"),
             py::arg("qlist"),
            "Return a list of quantum state tomography quantum programs.\n"
            "\n"
            "Args:\n"
            "     circuit: The quantum circuit to be combined.\n"
            "\n"
            "     qlist: A vector of indices representing the qubits involved.\n"
            "\n"
            "Returns:\n"
            "     A reference to the combined quantum programs.\n",
            py::return_value_policy::reference_internal)
        .def("exec",
             &QuantumStateTomography::exec,
             py::arg("qm"),
             py::arg("shots"),
                 "Run state tomography quantum programs.\n"
                 "\n"
                 "Args:\n"
                 "     qm: The quantum machine to execute the programs on.\n"
                 "\n"
                 "     shots: The number of shots for the execution.\n"
                 "\n"
                 "Returns:\n"
                 "     A reference to the execution results.\n",
                 py::return_value_policy::reference_internal)
        .def("set_qprog_results",
             py::overload_cast<size_t, const std::vector<std::map<std::string, double>> &>(&QuantumStateTomography::set_qprog_results),
             py::arg("qlist"),
             py::arg("results"),
            "Set the results of combined quantum programs.\n"
            "\n"
            "Args:\n"
            "     qlist: The index of the qubit list.\n"
            "\n"
            "     results: A vector of maps containing the result data.\n"
            "\n"
            "Returns:\n"
            "     A reference to the updated state.\n"
            "\n"
            "Raises:\n"
            "     run_fail: An error occurred while setting the results.\n",
            py::return_value_policy::reference_internal)
        .def("caculate_tomography_density",
             &QuantumStateTomography::caculate_tomography_density,
            "Calculate the tomography density.\n"
            "\n"
            "Returns:\n"
            "     A reference to the calculated density matrix.\n",
            py::return_value_policy::reference_internal);

    py::class_<Fusion>(m, "Fusion", "quantum fusion operation")
        .def(py::init<>())
        .def("aggregate_operations",
             py::overload_cast<QCircuit &>(&Fusion::aggregate_operations),
             py::arg("circuit"),
             py::return_value_policy::automatic,
            "Aggregate operations into the provided quantum circuit.\n"
            "\n"
            "Args:\n"
            "     circuit: The quantum circuit to which operations will be added.\n"
            "\n"
            "Returns:\n"
            "     A reference to the modified circuit.\n"
        )
        .def("aggregate_operations",
             py::overload_cast<QProg &>(&Fusion::aggregate_operations),
             py::arg("qprog"),
             py::return_value_policy::automatic,
            "Aggregate operations into the provided quantum program.\n"
            "\n"
            "Args:\n"
            "     qprog: The quantum program to which operations will be added.\n"
            "\n"
            "Returns:\n"
            "     A reference to the modified program.\n"
        );

    py::class_<LatexMatrix>(m, "LatexMatrix",
                            "Generate quantum circuits latex src code can be compiled on latex package 'qcircuit'\n"
                            "circuits element treated as matrix element in latex syntax\n"
                            "\n"
                            "qcircuit package tutorial [https://physics.unm.edu/CQuIC/Qcircuit/Qtutorial.pdf]")
        .def(py::init<>())
        .def("set_label",
             &LatexMatrix::set_label,
             py::arg("qubit_label"),
             py::arg("cbit_label") = LatexMatrix::Label(),
             py::arg("time_seq_label") = "",
             py::arg("head") = true,
            "Set label at the leftmost head column or rightmost tail column.\n"
            "Labels can be reset at any time.\n"
            "\n"
            "Args:\n"
            "     qubit_label: Label for the qubit wire's leftmost head label, specified in LaTeX syntax.If not given, the row will remain empty (e.g., {0: 'q_{1}', 2:'q_{2}'}).\n"
            "\n"
            "     cbit_label: Classic label string, supports LaTeX formatting.\n"
            "\n"
            "     time_seq_label: If given, sets the time sequence label.\n"
            "\n"
            "     head: If true, appends the label at the head; if false, appends at the tail.\n"
            "\n"
            "Returns:\n"
            "     None, as the function modifies the matrix in place.\n"
        )
        .def("set_logo",
             &LatexMatrix::set_logo,
             py::arg("logo") = "",
            "Add a logo string.\n"
            "\n"
            "Args:\n"
            "     logo: The logo string to be added. If not provided, the logo will be set to an empty string.\n"
            "\n"
            "Returns:\n"
            "     None, as the function modifies the matrix in place.\n"
        )
        .def("insert_gate",
             &LatexMatrix::insert_gate,
             py::arg("target_rows"),
             py::arg("ctrl_rows"),
             py::arg("from_col"),
             py::arg("gate_type"),
             py::arg("gate_name") = "",
             py::arg("dagger") = false,
             py::arg("param") = "",
            "Insert a gate into the circuit.\n"
            "\n"
            "Args:\n"
            "     target_rows: Gate target rows of the LaTeX matrix.\n"
            "\n"
            "     ctrl_rows: Control rows for the gate.\n"
            "\n"
            "     from_col: Desired column position for the gate; if space is insufficient, a suitable column will be found.\n"
            "\n"
            "     gate_type: Enum type of LATEX_GATE_TYPE.\n"
            "\n"
            "     gate_name: Name of the gate (default: '').\n"
            "\n"
            "     dagger: Flag indicating if the gate is a dagger (default: false).\n"
            "\n"
            "     param: Parameter string for the gate (default: '').\n"
            "\n"
            "Returns:\n"
            "     int: Actual column number where the gate is placed.\n")
        .def("insert_barrier",
             &LatexMatrix::insert_barrier,
             py::arg("rows"),
             py::arg("from_col"),
            "Insert a barrier into the circuit.\n"
            "\n"
            "Args:\n"
            "     rows: The rows of the LaTeX matrix where the barrier is applied.\n"
            "\n"
            "     from_col: Desired column position for the barrier; if space is insufficient, a suitable column will be found.\n"
            "\n"
            "Returns:\n"
            "     int: Actual column number where the barrier is placed.\n")
        .def("insert_measure",
             &LatexMatrix::insert_measure,
             py::arg("q_row"),
             py::arg("c_row"),
             py::arg("from_col"),
             py::arg("cbit_id"),
            "Insert a measurement operation into the circuit.\n"
            "\n"
            "Args:\n"
            "     q_row: The row of the qubit being measured.\n"
            "\n"
            "     c_row: The row of the classical bit that will store the measurement result.\n"
            "\n"
            "     from_col: The desired column position for the measurement.\n"
            "\n"
            "Returns:\n"
            "     None, as the function modifies the matrix in place.\n")
        .def("insert_reset",
             &LatexMatrix::insert_reset,
             py::arg("q_row"),
             py::arg("from_col"),
            "Insert a reset operation into the circuit.\n"
            "\n"
            "Args:\n"
            "     q_row: The row of the qubit to be reset.\n"
            "\n"
            "     from_col: The desired column position for the reset.\n"
            "\n"
            "Returns:\n"
            "     None, as the function modifies the matrix in place.\n")
        .def("insert_timeseq",
             &LatexMatrix::insert_time_seq,
             py::arg("t_col"),
             py::arg("time_seq"),
            "Insert a time sequence into the circuit.\n"
            "\n"
            "Args:\n"
            "     t_col: The column position where the time sequence will be inserted.\n"
            "\n"
            "     time_seq: The time sequence data to be inserted.\n"
            "\n"
            "Warning:\n"
            "     This function does not check for column number validity, which may cause overwriting.\n"
            "\n"
            "     Users must ensure the column number is managed correctly to avoid conflicts.\n")
        .def("str",
             &LatexMatrix::str,
             py::arg("with_time") = false,
             "Return the final LaTeX source code representation of the matrix.\n"
             "\n"
             "Args:\n"
             "     with_time: A boolean flag indicating whether to include timing information in the output.\n"
             "\n"
             "Returns:\n"
             "     str: The LaTeX source code as a string. This method can be called at any time to obtain the current state of the matrix.\n")
        .def("__str__", &LatexMatrix::str);

    return;
}
