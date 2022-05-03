#include "pybind11/pybind11.h"
#include "pybind11/operators.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "QPanda.h"

USING_QPANDA
namespace py = pybind11;

#define BIND_CLASSICALCOND_OPERATOR_OVERLOAD(OP) def(py::self OP py::self)           \
                                                     .def(py::self OP cbit_size_t()) \
                                                     .def(cbit_size_t() OP py::self)

/* declare class types common used by others */
void export_fundament_class(py::module &m)
{
    py::class_<PhysicalQubit>(m, "PhysicalQubit", "Physical Qubit abstract class")
        .def("getQubitAddr", &PhysicalQubit::getQubitAddr, py::return_value_policy::reference_internal);

    py::class_<Qubit>(m, "Qubit", "Qubit abstract class")
        .def("getPhysicalQubitPtr", &Qubit::getPhysicalQubitPtr, py::return_value_policy::reference_internal)
        .def("get_phy_addr", &Qubit::get_phy_addr, py::return_value_policy::reference_internal);

    py::class_<QVec>(m, "QVec", "Qubit vector basic class")
        .def(py::init<>())
        .def(py::init<std::vector<Qubit *> &>(),
             py::arg("qubit_list"))
        .def(py::init<const QVec &>(),
             py::arg("qvec"))
        .def(py::init<Qubit *>(),
             py::arg("qubit"))
        .def("__getitem__",
             py::overload_cast<size_t>(&QVec::operator[]),
             py::arg("qubit_addr"),
             py::return_value_policy::reference)
        .def("__getitem__",
             py::overload_cast<ClassicalCondition &>(&QVec::operator[]),
             py::arg("classical_cond"),
             py::return_value_policy::reference)
        .def("__len__", &QVec::size)
        .def(
            "to_list",
            [](QVec &self)
            {
                std::vector<Qubit *> q_list;
                for (const auto &item : self)
                {
                    q_list.push_back(item);
                }
                return q_list;
            },
            py::return_value_policy::reference)
        .def("append",
             py::overload_cast<Qubit *const &>(&QVec::push_back),
             py::arg("qubit"))
        .def("pop", &QVec::pop_back);

    py::class_<CBit>(m, "CBit")
        .def("getName", &CBit::getName);

    py::class_<ClassicalCondition>(m, "ClassicalCondition", "Classical condition class  Proxy class of cexpr class")
        .def("get_val", &ClassicalCondition::get_val, "get value")
        .def("set_val", &ClassicalCondition::set_val, "set value")
        .BIND_CLASSICALCOND_OPERATOR_OVERLOAD(<)
        .BIND_CLASSICALCOND_OPERATOR_OVERLOAD(<=)
        .BIND_CLASSICALCOND_OPERATOR_OVERLOAD(>)
        .BIND_CLASSICALCOND_OPERATOR_OVERLOAD(>=)
        .BIND_CLASSICALCOND_OPERATOR_OVERLOAD(+)
        .BIND_CLASSICALCOND_OPERATOR_OVERLOAD(-)
        .BIND_CLASSICALCOND_OPERATOR_OVERLOAD(*)
        .BIND_CLASSICALCOND_OPERATOR_OVERLOAD(/)
        .BIND_CLASSICALCOND_OPERATOR_OVERLOAD(==);

    py::class_<NodeIter>(m, "NodeIter")
        .def(py::init<>())
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("get_next", &NodeIter::getNextIter, py::return_value_policy::automatic)
        .def("get_pre", &NodeIter::getPreIter, py::return_value_policy::reference)
        .def(
            "get_node_type",
            [](NodeIter &iter)
            {
                auto node_type = (*iter)->getNodeType();
                return node_type;
            },
            py::return_value_policy::automatic);
    
    py::implicitly_convertible<Qubit *, QVec>();
    py::implicitly_convertible<std::vector<Qubit *>, QVec>();
}