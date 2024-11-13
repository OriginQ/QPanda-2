#include "QPanda.h"
#include "pybind11/pybind11.h"
#include "pybind11/operators.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"


USING_QPANDA
namespace py = pybind11;

#define BIND_CLASSICALCOND_OPERATOR_OVERLOAD(OP) def(py::self OP py::self)           \
                                                     .def(py::self OP cbit_size_t()) \
                                                     .def(cbit_size_t() OP py::self)

/* declare class types common used by others */
void export_fundament_class(py::module &m)
{
    py::class_<PhysicalQubit>(m, "PhysicalQubit", "Physical Qubit abstract class")
        .def("getQubitAddr", &PhysicalQubit::getQubitAddr, py::return_value_policy::reference_internal,
            "Retrieve the address of the physical qubit.\n"
            "\n"
            "This function returns the address of the physical qubit associated with the instance.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     The address of the physical qubit.\n");

    py::class_<Qubit>(m, "Qubit", "Qubit abstract class")
        .def("getPhysicalQubitPtr", &Qubit::getPhysicalQubitPtr, py::return_value_policy::reference_internal,
            "Retrieve a pointer to the associated physical qubit.\n"
            "\n"
            "This function returns a pointer to the physical qubit that corresponds to this qubit instance.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A pointer to the associated physical qubit.\n")
        .def("get_phy_addr", &Qubit::get_phy_addr, py::return_value_policy::reference_internal,
            "Retrieve the physical address of the qubit.\n"
            "\n"
            "This function returns the physical address associated with this qubit instance.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     The physical address of the qubit.\n");

    py::class_<QVec>(m, "QVec", "Qubit vector basic class")
        .def(py::init<>(),
            "Default constructor for QVec.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A new instance of QVec.\n")
        .def(py::init<std::vector<Qubit*>&>(),
            py::arg("qubit_list"),
            "Constructor that initializes QVec with a list of qubits.\n"
            "\n"
            "Args:\n"
            "     qubit_list: A reference to a vector of pointers to Qubit instances.\n"
            "\n"
            "Returns:\n"
            "     A new instance of QVec initialized with the provided qubits.\n")
        .def(py::init<const QVec&>(),
            py::arg("qvec"),
            "Copy constructor for QVec.\n"
            "\n"
            "Args:\n"
            "     qvec: A reference to an existing QVec instance.\n"
            "\n"
            "Returns:\n"
            "     A new instance of QVec that is a copy of the provided instance.\n")
        .def(py::init<Qubit*>(),
            py::arg("qubit"),
            "Constructor that initializes QVec with a single qubit.\n"
            "\n"
            "Args:\n"
            "     qubit: A pointer to a Qubit instance.\n"
            "\n"
            "Returns:\n"
            "     A new instance of QVec initialized with the provided qubit.\n")
        .def("__getitem__",
            py::overload_cast<size_t>(&QVec::operator[]),
            py::arg("qubit_addr"),
            py::return_value_policy::reference,
            "Retrieve a qubit by its index.\n"
            "\n"
            "Args:\n"
            "     qubit_addr: The index of the qubit to retrieve.\n"
            "\n"
            "Returns:\n"
            "     A reference to the Qubit at the specified index.\n")
        .def("__getitem__",
            py::overload_cast<ClassicalCondition&>(&QVec::operator[]),
            py::arg("classical_cond"),
            py::return_value_policy::reference,
            "Retrieve a qubit based on a classical condition.\n"
            "\n"
            "Args:\n"
            "     classical_cond: The classical condition to evaluate.\n"
            "\n"
            "Returns:\n"
            "     A reference to the Qubit associated with the provided condition.\n")
        .def("__len__", &QVec::size,
            "Get the number of qubits in the QVec.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     The count of qubits in the QVec.\n")
        .def(
            "to_list",
            [](QVec& self)
            {
                std::vector<Qubit*> q_list;
                for (const auto& item : self)
                {
                    q_list.push_back(item);
                }
                return q_list;
            },
            py::return_value_policy::reference,
            "Convert the QVec to a standard vector of qubit pointers.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A vector containing pointers to the qubits in the QVec.\n")
        .def("append",
            py::overload_cast<Qubit* const&>(&QVec::push_back),
            py::arg("qubit"),
            "Add a qubit to the end of the QVec.\n"
            "\n"
            "Args:\n"
            "     qubit: A pointer to the Qubit to be added.\n"
            "\n"
            "Returns:\n"
            "     None.\n")
        .def("pop", &QVec::pop_back,
            "Remove and return the last qubit from the QVec.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A pointer to the removed Qubit.\n");

    py::class_<CBit>(m, "CBit", "quantum classical bit")
            .def("getName", &CBit::getName,
                "Retrieve the name of the classical bit.\n"
                "\n"
                "Args:\n"
                "     None\n"
                "\n"
                "Returns:\n"
                "     The name of the CBit as a string.\n"
            );

    py::class_<ClassicalCondition>(m, "ClassicalCondition", "Classical condition class  Proxy class of cexpr class")
        .def("get_val", &ClassicalCondition::get_val, "get value",
            "Retrieve the current value of the classical condition.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     The value of the ClassicalCondition.\n"
        )
        .def("set_val", &ClassicalCondition::set_val, "set value",
            "Set a new value for the classical condition.\n"
            "\n"
            "Args:\n"
            "     value: The new value to set.\n"
            "\n"
            "Returns:\n"
            "     None\n"
        )
        .def("c_and", py::overload_cast<cbit_size_t>(&ClassicalCondition::c_and),
            "Perform a logical AND operation with a classical bit.\n"
            "\n"
            "Args:\n"
            "     cbit: The classical bit size to perform AND with.\n"
            "\n"
            "Returns:\n"
            "     The result of the AND operation.\n"
        )
        .def("c_and", py::overload_cast<const ClassicalCondition &>(&ClassicalCondition::c_and),
            "Perform a logical AND operation with another ClassicalCondition.\n"
            "\n"
            "Args:\n"
            "     other: Another ClassicalCondition to perform AND with.\n"
            "\n"
            "Returns:\n"
            "     The result of the AND operation.\n"
        )
        .def("c_or", py::overload_cast<cbit_size_t>(&ClassicalCondition::c_or),
            "Perform a logical OR operation with a classical bit.\n"
            "\n"
            "Args:\n"
            "     cbit: The classical bit size to perform OR with.\n"
            "\n"
            "Returns:\n"
            "     The result of the OR operation.\n"
        )
        .def("c_or", py::overload_cast<const ClassicalCondition &>(&ClassicalCondition::c_or),
            "Perform a logical OR operation with another ClassicalCondition.\n"
            "\n"
            "Args:\n"
            "     other: Another ClassicalCondition to perform OR with.\n"
            "\n"
            "Returns:\n"
            "     The result of the OR operation.\n"
        )
        .def("c_not", &ClassicalCondition::c_not,
            "Perform a logical NOT operation on the classical condition.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     The result of the NOT operation.\n"
        )
        .BIND_CLASSICALCOND_OPERATOR_OVERLOAD(< )
        .BIND_CLASSICALCOND_OPERATOR_OVERLOAD(<= )
        .BIND_CLASSICALCOND_OPERATOR_OVERLOAD(> )
        .BIND_CLASSICALCOND_OPERATOR_OVERLOAD(>= )
        .BIND_CLASSICALCOND_OPERATOR_OVERLOAD(+)
        .BIND_CLASSICALCOND_OPERATOR_OVERLOAD(-)
        .BIND_CLASSICALCOND_OPERATOR_OVERLOAD(*)
        .BIND_CLASSICALCOND_OPERATOR_OVERLOAD(/ )
        .BIND_CLASSICALCOND_OPERATOR_OVERLOAD(== );

    py::class_<NodeIter>(m, "NodeIter", "quantum node iter")
        .def(py::init<>(),
            "Initialize a new NodeIter instance.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A new instance of NodeIter.\n"
        )
        .def(py::self == py::self,
            "Check equality between two NodeIter instances.\n"
            "\n"
            "Args:\n"
            "     other: Another NodeIter instance.\n"
            "\n"
            "Returns:\n"
            "     True if equal, false otherwise.\n"
        )
        .def(py::self != py::self,
            "Check inequality between two NodeIter instances.\n"
            "\n"
            "Args:\n"
            "     other: Another NodeIter instance.\n"
            "\n"
            "Returns:\n"
            "     True if not equal, false otherwise.\n"
        )
        .def("get_next", &NodeIter::getNextIter, py::return_value_policy::automatic,
            "Get the next node iterator.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     The next NodeIter instance.\n"
        )
        .def("get_pre", &NodeIter::getPreIter, py::return_value_policy::reference,
            "Get the previous node iterator.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     The previous NodeIter instance.\n"
        )
        .def("get_node_type",
            [](NodeIter& iter)
            {
                auto node_type = (*iter)->getNodeType();
                return node_type;
            },
            py::return_value_policy::automatic,
            "Retrieve the type of the current node.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     The type of the node.\n"
            );

    py::implicitly_convertible<Qubit *, QVec>();
    py::implicitly_convertible<std::vector<Qubit *>, QVec>();
}