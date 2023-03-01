#include "Components/Operator/FermionOperator.h"
#include <math.h>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/chrono.h"
#include "Variational/utils.h"
#include "Variational/Optimizer.h"
#include "Variational/complex_var.h"
#include "pybind11/eigen.h"
#include "pybind11/operators.h"

USING_QPANDA
using namespace std;
using namespace pybind11::literals;
using std::map;
namespace py = pybind11;
namespace Var = QPanda::Variational;

#define BIND_VAR_OPERATOR_OVERLOAD(OP) def(py::self OP py::self)      \
                                           .def(py::self OP double()) \
                                           .def(double() OP py::self)

namespace QPanda
{
    namespace Variational
    {
        const var py_stack(int axis, std::vector<var> &args)
        {
            std::vector<std::shared_ptr<impl>> vimpl;
            for (auto &arg : args)
                vimpl.push_back(arg.pimpl);
            Var::var res(make_shared<impl_stack>(axis, args));
            for (const std::shared_ptr<impl> &_impl : vimpl)
            {
                _impl->parents.push_back(res.pimpl);
            }
            return res;
        }
    } // namespace QPanda
} // namespace Variational

void export_variational(py::module &m)
{
    m.def("VQG_I_batch", &Var::VQG_I_batch, "variational quantum I batch gates");
    m.def("VQG_H_batch", &Var::VQG_H_batch, "variational quantum H batch gates");
    m.def("VQG_T_batch", &Var::VQG_T_batch, "variational quantum T batch gates");
    m.def("VQG_S_batch", &Var::VQG_S_batch, "variational quantum S batch gates");
    m.def("VQG_X_batch", &Var::VQG_X_batch, "variational quantum X batch gates");
    m.def("VQG_Y_batch", &Var::VQG_Y_batch, "variational quantum Y batch gates");
    m.def("VQG_Z_batch", &Var::VQG_Z_batch, "variational quantum Z batch gates");
    m.def("VQG_X1_batch", &Var::VQG_X1_batch, "variational quantum X1 batch gates");
    m.def("VQG_Y1_batch", &Var::VQG_Y1_batch, "variational quantum Y1 batch gates");
    m.def("VQG_Z1_batch", &Var::VQG_Z1_batch, "variational quantum Z1 batch gates");
    m.def("VQG_U1_batch", &Var::VQG_U1_batch, "variational quantum U1 batch gates");
    m.def("VQG_U2_batch", &Var::VQG_U2_batch, "variational quantum U2 batch gates");
    m.def("VQG_U3_batch", &Var::VQG_U3_batch, "variational quantum U3 batch gates");
    m.def("VQG_U4_batch", &Var::VQG_U4_batch, "variational quantum U4 batch gates");
    m.def("VQG_CU_batch", &Var::VQG_CU_batch, "variational quantum CU batch gates");
    m.def("VQG_CZ_batch", &Var::VQG_CZ_batch, "variational quantum CZ batch gates");
    m.def("VQG_CNOT_batch", &Var::VQG_CNOT_batch, "variational quantum CNOT batch gates");
    m.def("VQG_SWAP_batch", &Var::VQG_SWAP_batch, "variational quantum SWAP batch gates");
    m.def("VQG_iSWAP_batch", &Var::VQG_iSWAP_batch, "variational quantum iSWAP batch gates");
    m.def("VQG_SqiSWAP_batch", &Var::VQG_SqiSWAP_batch, "variational quantum SqiSWAP batch gates");

    py::class_<Var::var>(m, "var", "quantum variational class")
        .def(py::init<double>())
        .def(py::init<py::EigenDRef<Eigen::MatrixXd>>())
        .def(py::init<double, bool>())
        .def(py::init<py::EigenDRef<Eigen::MatrixXd>, bool>())

        .def("get_value", &Var::var::getValue)
        .def("set_value", py::overload_cast<const MatrixXd &>(&Var::var::setValue))
        .def("set_value", py::overload_cast<const double &>(&Var::var::setValue))
        .def("clone", &Var::var::clone)
        .BIND_VAR_OPERATOR_OVERLOAD(+)
        .BIND_VAR_OPERATOR_OVERLOAD(-)
        .BIND_VAR_OPERATOR_OVERLOAD(*)
        .BIND_VAR_OPERATOR_OVERLOAD(/)

        .def(
            "__getitem__",
            [](Var::var &v, int idx)
            { return v[idx]; },
            py::is_operator())
        .def(py::self == py::self);

    py::class_<Var::VariationalQuantumGate>(m, "VariationalQuantumGate", "variational quantum gate base class")
        .def("feed", py::overload_cast<std::map<size_t, double>>(&Var::VariationalQuantumGate::feed))
        .def("get_vars", &Var::VariationalQuantumGate::get_vars, py::return_value_policy::reference)
        .def("get_constants", &Var::VariationalQuantumGate::get_constants, py::return_value_policy::reference)
        .def("set_dagger", &Var::VariationalQuantumGate::set_dagger, py::return_value_policy::automatic)
        .def("set_control", &Var::VariationalQuantumGate::set_control, py::return_value_policy::automatic)
        .def("is_dagger", &Var::VariationalQuantumGate::is_dagger, py::return_value_policy::automatic)
        .def("get_control_qubit", &Var::VariationalQuantumGate::get_control_qubit, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_I, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_I",
        "variational quantum  I gate class")
        .def(py::init<QPanda::Qubit *>())
        .def("dagger", &Var::VariationalQuantumGate_I::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_I::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_H, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_H",
        "variational quantum  H gate class")
        .def(py::init<QPanda::Qubit *>())
        .def("dagger", &Var::VariationalQuantumGate_H::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_H::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_S, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_S",
        "variational quantum S gate class")
        .def(py::init<QPanda::Qubit *>())
        .def("dagger", &Var::VariationalQuantumGate_S::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_S::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_T, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_T",
        "variational quantum T gate class")
        .def(py::init<QPanda::Qubit *>())
        .def("dagger", &Var::VariationalQuantumGate_T::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_T::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_X, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_X",
        "variational quantum X gate class")
        .def(py::init<QPanda::Qubit *>())
        .def("dagger", &Var::VariationalQuantumGate_X::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_X::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_Y, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_Y",
        "variational quantum Y gate class")
        .def(py::init<QPanda::Qubit *>())
        .def("dagger", &Var::VariationalQuantumGate_Y::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_Y::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_Z, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_Z",
        "variational quantum Z gate class")
        .def(py::init<QPanda::Qubit *>())
        .def("dagger", &Var::VariationalQuantumGate_Z::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_Z::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_X1, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_X1",
        "variational quantum X1 gate class")
        .def(py::init<QPanda::Qubit *>())
        .def("dagger", &Var::VariationalQuantumGate_X1::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_X1::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_Y1, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_Y1",
        "variational quantum Y1 gate class")
        .def(py::init<QPanda::Qubit *>())
        .def("dagger", &Var::VariationalQuantumGate_Y1::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_Y1::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_Z1, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_Z1",
        "variational quantum Z1 gate class")
        .def(py::init<QPanda::Qubit *>())
        .def("dagger", &Var::VariationalQuantumGate_Z1::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_Z1::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_RX, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_RX",
        "variational quantum RX gate class")
        .def(py::init<QPanda::Qubit *, Var::var>())
        .def(py::init<QPanda::Qubit *, double>())
        .def("dagger", &Var::VariationalQuantumGate_RX::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_RX::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_RY, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_RY",
        "variational quantum RY gate class")
        .def(py::init<QPanda::Qubit *, Var::var>())
        .def(py::init<QPanda::Qubit *, double>())
        .def("dagger", &Var::VariationalQuantumGate_RY::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_RY::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_RZ, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_RZ",
        "variational quantum RZ gate class")
        .def(py::init<QPanda::Qubit *, Var::var>())
        .def(py::init<QPanda::Qubit *, double>())
        .def("dagger", &Var::VariationalQuantumGate_RZ::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_RZ::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_U1, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_U1",
        "variational quantum U1 gate class")
        .def(py::init<QPanda::Qubit *, Var::var>())
        .def(py::init<QPanda::Qubit *, double>())
        .def("dagger", &Var::VariationalQuantumGate_U1::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_U1::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_U2, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_U2",
        "variational quantum U2 gate class")
        .def(py::init<QPanda::Qubit *, Var::var, Var::var>())
        .def(py::init<QPanda::Qubit *, double, double>())
        .def("dagger", &Var::VariationalQuantumGate_U2::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_U2::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_U3, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_U3",
        "variational quantum U3 gate class")
        .def(py::init<QPanda::Qubit *, Var::var, Var::var, Var::var>())
        .def(py::init<QPanda::Qubit *, double, double, double>())
        .def("dagger", &Var::VariationalQuantumGate_U3::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_U3::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_U4, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_U4",
        "variational quantum U4 gate class")
        .def(py::init<QPanda::Qubit *, Var::var, Var::var, Var::var, Var::var>())
        .def(py::init<QPanda::Qubit *, double, double, double, double>())
        .def("dagger", &Var::VariationalQuantumGate_U4::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_U4::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_CR, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_CR",
        "variational quantum CR gate class")
        .def(py::init<QPanda::Qubit *, QPanda::Qubit *, double>())
        .def(py::init<QPanda::Qubit *, QPanda::Qubit *, Var::var>())
        .def(py::init<Var::VariationalQuantumGate_CR &>())
        .def("dagger", &Var::VariationalQuantumGate_CR::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_CR::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_CRX, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_CRX",
        "variational quantum CRX gate class")
        .def(py::init<QPanda::Qubit *, QVec &, double>())
        .def(py::init<QPanda::Qubit *, QVec &, Var::var>())
        .def(py::init<Var::VariationalQuantumGate_CRX &>())
        .def("dagger", &Var::VariationalQuantumGate_CRX::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_CRX::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_CRY, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_CRY",
        "variational quantum CRY gate class")
        .def(py::init<QPanda::Qubit *, QVec &, double>())
        .def(py::init<QPanda::Qubit *, QVec &, Var::var>())
        .def(py::init<Var::VariationalQuantumGate_CRY &>())
        .def("dagger", &Var::VariationalQuantumGate_CRY::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_CRY::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_CRZ, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_CRZ",
        "variational quantum CRZ gate class")
        .def(py::init<QPanda::Qubit *, QVec &, double>())
        .def(py::init<QPanda::Qubit *, QVec &, Var::var>())
        .def(py::init<Var::VariationalQuantumGate_CRZ &>())
        .def("dagger", &Var::VariationalQuantumGate_CRZ::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_CRZ::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_CNOT, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_CNOT",
        "variational quantum CNOT gate class")
        .def(py::init<QPanda::Qubit *, QPanda::Qubit *>())
        .def("dagger", &Var::VariationalQuantumGate_CNOT::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_CNOT::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_CZ, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_CZ",
        "variational quantum CZ gate class")
        .def(py::init<QPanda::Qubit *, QPanda::Qubit *>())
        .def("dagger", &Var::VariationalQuantumGate_CZ::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_CZ::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_SWAP, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_SWAP",
        "variational quantum SWAP gate class")
        .def(py::init<QPanda::Qubit *, QPanda::Qubit *>())
        .def("dagger", &Var::VariationalQuantumGate_SWAP::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_SWAP::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_iSWAP, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_iSWAP",
        "variational quantum iSWAP gate class")
        .def(py::init<QPanda::Qubit *, QPanda::Qubit *>())
        .def("dagger", &Var::VariationalQuantumGate_iSWAP::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_iSWAP::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_SqiSWAP, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_SqiSWAP",
        "variational quantum SqiSWAP gate class")
        .def(py::init<QPanda::Qubit *, QPanda::Qubit *>())
        .def("dagger", &Var::VariationalQuantumGate_SqiSWAP::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_SqiSWAP::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumGate_CU, Var::VariationalQuantumGate>(m, 
        "VariationalQuantumGate_CU",
        "variational quantum CU gate class")
        .def(py::init<QPanda::Qubit *, QPanda::Qubit *, double, double, double, double>())
        .def(py::init<QPanda::Qubit *, QPanda::Qubit *, Var::var, Var::var, Var::var, Var::var>())
        .def(py::init<Var::VariationalQuantumGate_CU &>())
        .def("dagger", &Var::VariationalQuantumGate_CU::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumGate_CU::control, py::return_value_policy::automatic);

    py::class_<Var::VariationalQuantumCircuit>(m, "VariationalQuantumCircuit", "variational quantum CIRCUIT class")
        .def(py::init<>())
        .def(py::init<QCircuit>())
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_I>, py::return_value_policy::reference)
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_H>, py::return_value_policy::reference)
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_X>, py::return_value_policy::reference)
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_Y>, py::return_value_policy::reference)
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_T>, py::return_value_policy::reference)
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_S>, py::return_value_policy::reference)
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_Z>, py::return_value_policy::reference)
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_X1>, py::return_value_policy::reference)
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_Y1>, py::return_value_policy::reference)
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_Z1>, py::return_value_policy::reference)
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_U1>, py::return_value_policy::reference)
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_U2>, py::return_value_policy::reference)
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_U3>, py::return_value_policy::reference)
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_U4>, py::return_value_policy::reference)
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_RX>, py::return_value_policy::reference)
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_RY>, py::return_value_policy::reference)
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_RZ>, py::return_value_policy::reference)
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_CNOT>, py::return_value_policy::reference)
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_CR>, py::return_value_policy::reference)
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_CZ>, py::return_value_policy::reference)
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_CRX>, py::return_value_policy::reference)
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_CRY>, py::return_value_policy::reference)
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_CRZ>, py::return_value_policy::reference)
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_SWAP>, py::return_value_policy::reference)
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_iSWAP>, py::return_value_policy::reference)
        .def("__lshift__", &Var::VariationalQuantumCircuit::operator<<<Var::VQG_SqiSWAP>, py::return_value_policy::reference)

        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_I>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_H>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_X>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_Y>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_T>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_S>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_Z>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_X1>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_Y1>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_Z1>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_U1>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_U2>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_U3>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_U4>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_RX>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_RY>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_RZ>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_CNOT>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_CR>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_CZ>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_CRX>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_CRY>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_CRZ>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_SWAP>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_iSWAP>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_SqiSWAP>, py::return_value_policy::reference)

        .def("insert", py::overload_cast<Var::VariationalQuantumCircuit>(&Var::VariationalQuantumCircuit::insert<Var::VariationalQuantumCircuit>), py::return_value_policy::reference)
        .def("insert", py::overload_cast<QCircuit>(&Var::VariationalQuantumCircuit::insert<QCircuit>), py::return_value_policy::reference)
        .def("insert", py::overload_cast<QGate>(&Var::VariationalQuantumCircuit::insert<QGate>), py::return_value_policy::reference)

        .def("__lshift__", py::overload_cast<Var::VariationalQuantumCircuit>(&Var::VariationalQuantumCircuit::operator<<<Var::VariationalQuantumCircuit>), py::return_value_policy::reference)
        .def("__lshift__", py::overload_cast<QCircuit>(&Var::VariationalQuantumCircuit::operator<<<QCircuit>), py::return_value_policy::reference)
        .def("__lshift__", py::overload_cast<QGate>(&Var::VariationalQuantumCircuit::operator<<<QGate>), py::return_value_policy::reference)

        .def("feed", py::overload_cast<>(&Var::VariationalQuantumCircuit::feed))
        .def("feed", py::overload_cast<const std::vector<std::tuple<weak_ptr<Var::VariationalQuantumGate>, size_t, double>>>(&Var::VariationalQuantumCircuit::feed, py::const_))
        .def("dagger", &Var::VariationalQuantumCircuit::dagger, py::return_value_policy::automatic)
        .def("control", &Var::VariationalQuantumCircuit::control, py::return_value_policy::automatic)
        .def("set_dagger", &Var::VariationalQuantumCircuit::set_dagger, py::return_value_policy::automatic)
        .def("set_control", &Var::VariationalQuantumCircuit::set_control, py::return_value_policy::automatic)
        .def("is_dagger", &Var::VariationalQuantumCircuit::is_dagger, py::return_value_policy::automatic)
        .def("get_control_qubit", &Var::VariationalQuantumCircuit::get_control_qubit, py::return_value_policy::automatic);

    py::class_<Var::expression>(m, "expression", "variational quantum expression class")
        .def(py::init<Var::var>())
        .def("find_leaves", &Var::expression::findLeaves)
        .def("find_non_consts", py::overload_cast<const std::vector<Var::var> &>(&Var::expression::findNonConsts))
        .def("propagate", py::overload_cast<>(&Var::expression::propagate))
        .def("propagate", py::overload_cast<const std::vector<Var::var> &>(&Var::expression::propagate))
        .def("backprop", py::overload_cast<std::unordered_map<Var::var, MatrixXd> &>(&Var::expression::backpropagate))
        .def("backprop", py::overload_cast<std::unordered_map<Var::var, MatrixXd> &, const std::unordered_set<Var::var> &>(&Var::expression::backpropagate))
        .def("get_root", &Var::expression::getRoot);

    m.def("eval", Var::eval);
    m.def("eval", [](Var::var v)
          { return eval(v, true); });

    m.def("_back", py::overload_cast<Var::expression &, std::unordered_map<Var::var, MatrixXd> &, const std::unordered_set<Var::var> &>(&Var::back));
    m.def("_back", py::overload_cast<Var::expression &, std::unordered_map<Var::var, MatrixXd> &>(&Var::back));
    m.def("_back", py::overload_cast<const Var::var &, std::unordered_map<Var::var, MatrixXd> &>(&Var::back));
    m.def("_back", py::overload_cast<const Var::var &, std::unordered_map<Var::var, MatrixXd> &, const std::unordered_set<Var::var> &>(&Var::back));

    m.def("exp", Var::exp);
    m.def("log", Var::log);
    m.def("poly", Var::poly);
    m.def("dot", Var::dot);

    // add trigonometric function
    m.def("sin", Var::sin);
    m.def("cos", Var::cos);
    m.def("tan", Var::tan);
    m.def("asin", Var::asin);
    m.def("acos", Var::acos);
    m.def("atan", Var::atan);
    m.def("inverse", Var::inverse);
    m.def("transpose", Var::transpose);
    m.def("sum", Var::sum);

    m.def("stack",
          [](int axis, py::args args)
          {
              std::vector<Var::var> vars;
              for (auto arg : args)
              {
                  vars.push_back(py::cast<Var::var>(arg));
              }
              return py_stack(axis, vars);
          });
    m.def("sigmoid", Var::sigmoid);
    m.def("softmax", Var::softmax);
    m.def("crossEntropy", Var::crossEntropy);
    m.def("dropout", Var::dropout);

    m.def("qop",
          py::overload_cast<Var::VariationalQuantumCircuit &, QPanda::PauliOperator, QPanda::QuantumMachine *, std::vector<Qubit *>>(&Var::qop),
          "VariationalQuantumCircuit"_a,
          "Hamiltonian"_a,
          "QuantumMachine"_a,
          "qubitList"_a);
    m.def("qop",
          py::overload_cast<Var::VariationalQuantumCircuit &, QPanda::PauliOperator, QPanda::QuantumMachine *, std::map<size_t, Qubit *>>(&Var::qop),
          "VariationalQuantumCircuit"_a,
          "Hamiltonian"_a,
          "QuantumMachine"_a,
          "qubitList"_a);
    m.def("qop_pmeasure", Var::qop_pmeasure);

    py::implicitly_convertible<double, Var::var>();

    py::class_<Var::Optimizer>(m, "Optimizer", "variational quantum Optimizer class")
        .def("get_variables", &Var::Optimizer::get_variables)
        .def("get_loss", &Var::Optimizer::get_loss)
        .def("run", &Var::Optimizer::run);

    py::enum_<Var::OptimizerMode>(m, "OptimizerMode", "variational quantum OptimizerMode");

    py::class_<Var::VanillaGradientDescentOptimizer, std::shared_ptr<Var::VanillaGradientDescentOptimizer>>(m, 
        "VanillaGradientDescentOptimizer", "variational quantum VanillaGradientDescentOptimizer")
        .def(py::init<>(
            [](Var::var lost_function,
               double learning_rate = 0.01,
               double stop_condition = 1.e-6,
               Var::OptimizerMode mode = Var::OptimizerMode::MINIMIZE)
            { return Var::VanillaGradientDescentOptimizer(lost_function, learning_rate, stop_condition, mode); }))
        .def("minimize", &Var::VanillaGradientDescentOptimizer::minimize)
        .def("get_variables", &Var::VanillaGradientDescentOptimizer::get_variables)
        .def("get_loss", &Var::VanillaGradientDescentOptimizer::get_loss)
        .def("run", &Var::VanillaGradientDescentOptimizer::run);

    py::class_<Var::MomentumOptimizer, std::shared_ptr<Var::MomentumOptimizer>>(m, "MomentumOptimizer", "variational quantum MomentumOptimizer")
        .def(py::init<>(
            [](Var::var lost,
               double learning_rate = 0.01,
               double momentum = 0.9)
            { return Var::MomentumOptimizer(lost, learning_rate, momentum); }))
        .def("minimize", &Var::MomentumOptimizer::minimize)
        .def("get_variables", &Var::MomentumOptimizer::get_variables)
        .def("get_loss", &Var::MomentumOptimizer::get_loss)
        .def("run", &Var::MomentumOptimizer::run);

    py::class_<Var::AdaGradOptimizer, std::shared_ptr<Var::AdaGradOptimizer>>(m, 
        "AdaGradOptimizer", "variational quantum AdaGradOptimizer")
        .def(py::init<>(
            [](Var::var lost,
               double learning_rate = 0.01,
               double initial_accumulator_value = 0.0,
               double epsilon = 1e-10)
            { return Var::AdaGradOptimizer(lost,
                                           learning_rate, initial_accumulator_value, epsilon); }))
        .def("minimize", &Var::AdaGradOptimizer::minimize)
        .def("get_variables", &Var::AdaGradOptimizer::get_variables)
        .def("get_loss", &Var::AdaGradOptimizer::get_loss)
        .def("run", &Var::AdaGradOptimizer::run);

    py::class_<Var::RMSPropOptimizer, std::shared_ptr<Var::RMSPropOptimizer>>(m, 
        "RMSPropOptimizer", "variational quantum RMSPropOptimizer")
        .def(py::init<>(
            [](Var::var lost,
               double learning_rate = 0.001,
               double decay = 0.9,
               double epsilon = 1e-10)
            { return Var::RMSPropOptimizer(lost, learning_rate, decay, epsilon); }))
        .def("minimize", &Var::RMSPropOptimizer::minimize)
        .def("get_variables", &Var::RMSPropOptimizer::get_variables)
        .def("get_loss", &Var::RMSPropOptimizer::get_loss)
        .def("run", &Var::RMSPropOptimizer::run);

    py::class_<Var::AdamOptimizer, std::shared_ptr<Var::AdamOptimizer>>(m, 
        "AdamOptimizer", "variational quantum AdamOptimizer")
        .def(py::init<>(
            [](Var::var lost,
               double learning_rate = 0.001,
               double beta1 = 0.9,
               double beta2 = 0.999,
               double epsilon = 1e-8)
            { return Var::AdamOptimizer(lost, learning_rate, beta1, beta2, epsilon); }))
        .def("minimize", &Var::AdamOptimizer::minimize)
        .def("get_variables", &Var::AdamOptimizer::get_variables)
        .def("get_loss", &Var::AdamOptimizer::get_loss)
        .def("run", &Var::AdamOptimizer::run);
}