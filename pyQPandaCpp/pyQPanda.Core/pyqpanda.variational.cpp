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

#define GET_FEED_PTR_NO_OFFSET(ptr_name, classname) \
    QPanda::QGate(classname::*ptr_name)() \
    = &classname::feed

#define GET_FEED_PTR_WITH_OFFSET(ptr_name, classname) \
    QPanda::QGate(classname::*ptr_name)( \
        std::map<size_t, double>) \
    = &classname::feed

#define BIND_VAR_OPERATOR_OVERLOAD(OP) .def(py::self OP py::self)\
                                       .def(py::self OP double())\
                                       .def(double() OP py::self)

namespace QPanda {
    namespace Variational {
        const var py_stack(int axis, std::vector<var>& args)
        {
            std::vector<std::shared_ptr<impl>> vimpl;
            for (auto &arg : args)
                vimpl.push_back(arg.pimpl);
            Var::var res(make_shared<impl_stack>(axis, args));
            for (const std::shared_ptr<impl>& _impl : vimpl) {
                _impl->parents.push_back(res.pimpl);
            }
            return res;
        }
    } // namespace QPanda
} // namespace Variational


void init_variational(py::module & m)
{
    py::class_<Var::var>(m, "var")
        .def(py::init<double>())
        .def(py::init<py::EigenDRef<Eigen::MatrixXd>>())
        .def(py::init<double, bool>())
        .def(py::init<py::EigenDRef<Eigen::MatrixXd>, bool>())
        .def("get_value", &Var::var::getValue)
        .def("set_value", &Var::var::setValue)
        .def("clone", &Var::var::clone)
        BIND_VAR_OPERATOR_OVERLOAD(+)
        BIND_VAR_OPERATOR_OVERLOAD(-)
        BIND_VAR_OPERATOR_OVERLOAD(*)
        BIND_VAR_OPERATOR_OVERLOAD(/)
        .def("__getitem__", [](Var::var& v, int idx) {return v[idx]; }, py::is_operator())
        .def(py::self == py::self);

    py::class_<Var::VariationalQuantumGate>
        (m, "VariationalQuantumGate")
        .def("get_vars", &Var::VariationalQuantumGate::get_vars, py::return_value_policy::reference)
		.def("get_constants", &Var::VariationalQuantumGate::get_constants, py::return_value_policy::reference)
		.def("set_dagger",&Var::VariationalQuantumGate::set_dagger,py::return_value_policy::automatic)
		.def("set_control", &Var::VariationalQuantumGate::set_control, py::return_value_policy::automatic)
		.def("is_dagger", &Var::VariationalQuantumGate::is_dagger, py::return_value_policy::automatic)
		.def("get_control_qubit", &Var::VariationalQuantumGate::get_control_qubit, py::return_value_policy::automatic);

    GET_FEED_PTR_NO_OFFSET(feed_vqg_h_no_ptr, Var::VariationalQuantumGate_H);

    py::class_<Var::VariationalQuantumGate_H, Var::VariationalQuantumGate>
        (m, "VariationalQuantumGate_H")
        .def(py::init<QPanda::Qubit*>())
        .def("feed", feed_vqg_h_no_ptr)
		.def("dagger", &Var::VariationalQuantumGate_H::dagger, py::return_value_policy::automatic)
		.def("control", &Var::VariationalQuantumGate_H::control, py::return_value_policy::automatic);

    GET_FEED_PTR_NO_OFFSET(feed_vqg_x_no_ptr, Var::VariationalQuantumGate_X);

    py::class_<Var::VariationalQuantumGate_X, Var::VariationalQuantumGate>
        (m, "VariationalQuantumGate_X")
        .def(py::init<QPanda::Qubit*>())
        .def("feed", feed_vqg_x_no_ptr)
		.def("dagger", &Var::VariationalQuantumGate_X::dagger, py::return_value_policy::automatic)
		.def("control", &Var::VariationalQuantumGate_X::control, py::return_value_policy::automatic);

    GET_FEED_PTR_NO_OFFSET(feed_vqg_rx_no_ptr, Var::VariationalQuantumGate_RX);
    GET_FEED_PTR_WITH_OFFSET(feed_vqg_rx_with_ptr, Var::VariationalQuantumGate_RX);

    py::class_<Var::VariationalQuantumGate_RX, Var::VariationalQuantumGate>
        (m, "VariationalQuantumGate_RX")
        .def(py::init<QPanda::Qubit*, Var::var>())
        .def(py::init<QPanda::Qubit*, double>())
        .def("feed", feed_vqg_rx_no_ptr)
        .def("feed", feed_vqg_rx_with_ptr)
		.def("dagger", &Var::VariationalQuantumGate_RX::dagger, py::return_value_policy::automatic)
		.def("control", &Var::VariationalQuantumGate_RX::control, py::return_value_policy::automatic);

    GET_FEED_PTR_NO_OFFSET(feed_vqg_ry_no_ptr, Var::VariationalQuantumGate_RY);
    GET_FEED_PTR_WITH_OFFSET(feed_vqg_ry_with_ptr, Var::VariationalQuantumGate_RY);

    py::class_<Var::VariationalQuantumGate_RY, Var::VariationalQuantumGate>
        (m, "VariationalQuantumGate_RY")
        .def(py::init<QPanda::Qubit*, Var::var>())
        .def(py::init<QPanda::Qubit*, double>())
        .def("feed", feed_vqg_ry_no_ptr)
        .def("feed", feed_vqg_ry_with_ptr)
		.def("dagger", &Var::VariationalQuantumGate_RY::dagger, py::return_value_policy::automatic)
		.def("control", &Var::VariationalQuantumGate_RY::control, py::return_value_policy::automatic);

    GET_FEED_PTR_NO_OFFSET(feed_vqg_rz_no_ptr, Var::VariationalQuantumGate_RZ);
    GET_FEED_PTR_WITH_OFFSET(feed_vqg_rz_with_ptr, Var::VariationalQuantumGate_RZ);

    py::class_<Var::VariationalQuantumGate_RZ, Var::VariationalQuantumGate>
        (m, "VariationalQuantumGate_RZ")
        .def(py::init<QPanda::Qubit*, Var::var>())
        .def(py::init<QPanda::Qubit*, double>())
        .def("feed", feed_vqg_rz_no_ptr)
        .def("feed", feed_vqg_rz_with_ptr)
		.def("dagger", &Var::VariationalQuantumGate_RZ::dagger, py::return_value_policy::automatic)
		.def("control", &Var::VariationalQuantumGate_RZ::control, py::return_value_policy::automatic);

    GET_FEED_PTR_NO_OFFSET(feed_vqg_crx_no_ptr, Var::VariationalQuantumGate_CRX);
	GET_FEED_PTR_WITH_OFFSET(feed_vqg_crx_with_ptr, Var::VariationalQuantumGate_CRX);
    py::class_<Var::VariationalQuantumGate_CRX, Var::VariationalQuantumGate>
        (m, "VariationalQuantumGate_CRX")
        .def(py::init<QPanda::Qubit*, QVec &, double>())
		.def(py::init<QPanda::Qubit*, QVec &, Var::var>())
        .def(py::init<Var::VariationalQuantumGate_CRX &>())
        .def("feed", feed_vqg_crx_no_ptr)
		.def("feed", feed_vqg_crx_with_ptr)
		.def("dagger", &Var::VariationalQuantumGate_CRX::dagger, py::return_value_policy::automatic)
		.def("control", &Var::VariationalQuantumGate_CRX::control, py::return_value_policy::automatic);


    GET_FEED_PTR_NO_OFFSET(feed_vqg_cry_no_ptr, Var::VariationalQuantumGate_CRY);
	GET_FEED_PTR_WITH_OFFSET(feed_vqg_cry_with_ptr, Var::VariationalQuantumGate_CRY);
    py::class_<Var::VariationalQuantumGate_CRY, Var::VariationalQuantumGate>
        (m, "VariationalQuantumGate_CRY")
        .def(py::init<QPanda::Qubit*, QVec &, double>())
		.def(py::init<QPanda::Qubit*, QVec &, Var::var>())
        .def(py::init<Var::VariationalQuantumGate_CRY &>())
        .def("feed", feed_vqg_cry_no_ptr)
		.def("feed", feed_vqg_cry_with_ptr)
		.def("dagger", &Var::VariationalQuantumGate_CRY::dagger, py::return_value_policy::automatic)
		.def("control", &Var::VariationalQuantumGate_CRY::control, py::return_value_policy::automatic);

    GET_FEED_PTR_NO_OFFSET(feed_vqg_crz_no_ptr, Var::VariationalQuantumGate_CRZ);
	GET_FEED_PTR_WITH_OFFSET(feed_vqg_crz_with_ptr, Var::VariationalQuantumGate_CRZ);
    py::class_<Var::VariationalQuantumGate_CRZ, Var::VariationalQuantumGate>
        (m, "VariationalQuantumGate_CRZ")
        .def(py::init<QPanda::Qubit*, QVec &, double>())
		.def(py::init<QPanda::Qubit*, QVec &, Var::var>())
        .def(py::init<Var::VariationalQuantumGate_CRZ &>())
        .def("feed", feed_vqg_crz_no_ptr)
		.def("feed", feed_vqg_crz_with_ptr)
		.def("dagger", &Var::VariationalQuantumGate_CRZ::dagger, py::return_value_policy::automatic)
		.def("control", &Var::VariationalQuantumGate_CRZ::control, py::return_value_policy::automatic);



    GET_FEED_PTR_NO_OFFSET(feed_vqg_cnot_no_ptr, Var::VariationalQuantumGate_CNOT);

	py::class_<Var::VariationalQuantumGate_CNOT, Var::VariationalQuantumGate>
		(m, "VariationalQuantumGate_CNOT")
		.def(py::init<QPanda::Qubit*, QPanda::Qubit*>())
		.def("feed", feed_vqg_cnot_no_ptr)
		.def("dagger", &Var::VariationalQuantumGate_CNOT::dagger, py::return_value_policy::automatic)
		.def("control", &Var::VariationalQuantumGate_CNOT::control, py::return_value_policy::automatic);

    GET_FEED_PTR_NO_OFFSET(feed_vqg_cz_no_ptr, Var::VariationalQuantumGate_CZ);

	py::class_<Var::VariationalQuantumGate_CZ, Var::VariationalQuantumGate>
		(m, "VariationalQuantumGate_CZ")
		.def(py::init<QPanda::Qubit*, QPanda::Qubit*>())
		.def("feed", feed_vqg_cz_no_ptr)
		.def("dagger", &Var::VariationalQuantumGate_CZ::dagger, py::return_value_policy::automatic)
		.def("control", &Var::VariationalQuantumGate_CZ::control, py::return_value_policy::automatic);

    QCircuit(Var::VariationalQuantumCircuit::*feed_vqc_with_ptr)
        (const std::vector<std::tuple<weak_ptr<Var::VariationalQuantumGate>,
            size_t, double>>) const
        = &Var::VariationalQuantumCircuit::feed;

    QCircuit(Var::VariationalQuantumCircuit::*feed_vqc_no_ptr)() 
        = &Var::VariationalQuantumCircuit::feed;

    Var::VariationalQuantumCircuit& (Var::VariationalQuantumCircuit::*insert_vqc_vqc)
        (Var::VariationalQuantumCircuit)
        = &Var::VariationalQuantumCircuit::insert;

    Var::VariationalQuantumCircuit& (Var::VariationalQuantumCircuit::*insert_vqc_qc)
        (QCircuit) = &Var::VariationalQuantumCircuit::insert;

    Var::VariationalQuantumCircuit& (Var::VariationalQuantumCircuit::*insert_vqc_qg)
        (QGate&) = &Var::VariationalQuantumCircuit::insert;



    py::class_<Var::VariationalQuantumCircuit>
        (m, "VariationalQuantumCircuit")
        .def(py::init<>())
        .def(py::init<QCircuit>())
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_H>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_X>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_RX>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_RY>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_RZ>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_CNOT>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_CZ>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_CRX>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_CRY>, py::return_value_policy::reference)
        .def("insert", &Var::VariationalQuantumCircuit::insert<Var::VQG_CRZ>, py::return_value_policy::reference)
        .def("insert", insert_vqc_vqc, py::return_value_policy::reference)
        .def("insert", insert_vqc_qc, py::return_value_policy::reference)
        .def("insert", insert_vqc_qg, py::return_value_policy::reference)
        .def("feed", feed_vqc_no_ptr)
        .def("feed", feed_vqc_with_ptr)
		.def("dagger", &Var::VariationalQuantumCircuit::dagger, py::return_value_policy::automatic)
		.def("control", &Var::VariationalQuantumCircuit::control, py::return_value_policy::automatic)
		.def("set_dagger", &Var::VariationalQuantumCircuit::set_dagger, py::return_value_policy::automatic)
		.def("set_control", &Var::VariationalQuantumCircuit::set_control, py::return_value_policy::automatic)
		.def("is_dagger", &Var::VariationalQuantumCircuit::is_dagger, py::return_value_policy::automatic)
		.def("get_control_qubit", &Var::VariationalQuantumCircuit::get_control_qubit, py::return_value_policy::automatic);

    py::class_<Var::expression>(m, "expression")
        .def(py::init<Var::var>())
        .def("find_leaves", &Var::expression::findLeaves)
        .def("find_non_consts", (std::unordered_set<Var::var>(Var::expression::*)(const std::vector<Var::var>&)) &Var::expression::findNonConsts)
        .def("propagate", (MatrixXd(Var::expression::*)()) &Var::expression::propagate)
        .def("propagate", (MatrixXd(Var::expression::*)(const std::vector<Var::var>&))
            &Var::expression::propagate)
        .def("backprop", (void(Var::expression::*)(std::unordered_map<Var::var, MatrixXd>&))
            &Var::expression::backpropagate)
        .def("backprop", (void(Var::expression::*)(std::unordered_map<Var::var, MatrixXd>&,
            const std::unordered_set<Var::var>&))&Var::expression::backpropagate)
        .def("get_root", &Var::expression::getRoot);
    m.def("eval", Var::eval);
    m.def("eval", [](Var::var v) {return eval(v, true); });

    m.def("_back", [](Var::expression& exp,
        std::unordered_map<Var::var, MatrixXd>& derivatives,
        const std::unordered_set<Var::var>& leaves) {
        Var::back(exp, derivatives, leaves);
        return derivatives;
    });
    m.def("_back", [](Var::expression& exp,
        std::unordered_map<Var::var, MatrixXd>& derivatives) {
        Var::back(exp, derivatives);
        return derivatives;
    });
    m.def("_back", [](const Var::var& v,
        std::unordered_map<Var::var, MatrixXd>& derivatives) {
        Var::back(v, derivatives);
        return derivatives;
    });
    m.def("_back", [](const Var::var& v,
        std::unordered_map<Var::var, MatrixXd>& derivatives,
        const std::unordered_set<Var::var>& leaves) {
        Var::back(v, derivatives, leaves);
        return derivatives;
    });

    m.def("exp", Var::exp);
    m.def("log", Var::log);
    m.def("poly", Var::poly);
    m.def("dot", Var::dot);
    m.def("inverse", Var::inverse);
    m.def("transpose", Var::transpose);
    m.def("sum", Var::sum);
    m.def("stack", [](int axis, py::args args) {
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
    const Var::var(*qop_plain)(Var::VariationalQuantumCircuit&,
        QPanda::PauliOperator,
        QPanda::QuantumMachine*,
        std::vector<Qubit*>) = Var::qop;

    const Var::var(*qop_map)(Var::VariationalQuantumCircuit&,
        QPanda::PauliOperator,
        QPanda::QuantumMachine*,
        std::map<size_t, Qubit*>) = Var::qop;
    m.def("qop", qop_plain, "VariationalQuantumCircuit"_a, "Hamiltonian"_a, "QuantumMachine"_a, "qubitList"_a);
    m.def("qop", qop_map, "VariationalQuantumCircuit"_a, "Hamiltonian"_a, "QuantumMachine"_a, "qubitList"_a);
    m.def("qop_pmeasure", Var::qop_pmeasure);
    py::implicitly_convertible<double, Var::var>();


    py::class_<Var::Optimizer>(m, "Optimizer")
        .def("get_variables", &Var::Optimizer::get_variables)
        .def("get_loss", &Var::Optimizer::get_loss)
        .def("run", &Var::Optimizer::run);


    py::enum_<Var::OptimizerMode>(m, "OptimizerMode");

    py::class_<Var::VanillaGradientDescentOptimizer, std::shared_ptr<Var::VanillaGradientDescentOptimizer>>
        (m, "VanillaGradientDescentOptimizer")
        .def(py::init<>([](
            Var::var lost_function,
            double learning_rate = 0.01,
            double stop_condition = 1.e-6,
            Var::OptimizerMode mode = Var::OptimizerMode::MINIMIZE) {
        return Var::VanillaGradientDescentOptimizer(lost_function,
            learning_rate, stop_condition, mode);
    }))
        .def("minimize", &Var::VanillaGradientDescentOptimizer::minimize)
        .def("get_variables", &Var::VanillaGradientDescentOptimizer::get_variables)
        .def("get_loss", &Var::VanillaGradientDescentOptimizer::get_loss)
        .def("run", &Var::VanillaGradientDescentOptimizer::run);


    py::class_<Var::MomentumOptimizer, std::shared_ptr<Var::MomentumOptimizer>>
        (m, "MomentumOptimizer")
        .def(py::init<>([](
            Var::var lost,
            double learning_rate = 0.01,
            double momentum = 0.9) {
        return Var::MomentumOptimizer(lost,
            learning_rate, momentum);
    }))
        .def("minimize", &Var::MomentumOptimizer::minimize)
        .def("get_variables", &Var::MomentumOptimizer::get_variables)
        .def("get_loss", &Var::MomentumOptimizer::get_loss)
        .def("run", &Var::MomentumOptimizer::run);

    py::class_<Var::AdaGradOptimizer, std::shared_ptr<Var::AdaGradOptimizer>>
        (m, "AdaGradOptimizer")
        .def(py::init<>([](
            Var::var lost,
            double learning_rate = 0.01,
            double initial_accumulator_value = 0.0,
            double epsilon = 1e-10) {
        return Var::AdaGradOptimizer(lost,
            learning_rate, initial_accumulator_value, epsilon);
    }))
        .def("minimize", &Var::AdaGradOptimizer::minimize)
        .def("get_variables", &Var::AdaGradOptimizer::get_variables)
        .def("get_loss", &Var::AdaGradOptimizer::get_loss)
        .def("run", &Var::AdaGradOptimizer::run);

    py::class_<Var::RMSPropOptimizer, std::shared_ptr<Var::RMSPropOptimizer>>
        (m, "RMSPropOptimizer")
        .def(py::init<>([](
            Var::var lost,
            double learning_rate = 0.001,
            double decay = 0.9,
            double epsilon = 1e-10) {
        return Var::RMSPropOptimizer(lost,
            learning_rate, decay, epsilon);
    }))
        .def("minimize", &Var::RMSPropOptimizer::minimize)
        .def("get_variables", &Var::RMSPropOptimizer::get_variables)
        .def("get_loss", &Var::RMSPropOptimizer::get_loss)
        .def("run", &Var::RMSPropOptimizer::run);


    py::class_<Var::AdamOptimizer, std::shared_ptr<Var::AdamOptimizer>>
        (m, "AdamOptimizer")
        .def(py::init<>([](
            Var::var lost,
            double learning_rate = 0.001,
            double beta1 = 0.9,
            double beta2 = 0.999,
            double epsilon = 1e-8) {
        return Var::AdamOptimizer(lost,
            learning_rate, beta1, beta2, epsilon);
    }))
        .def("minimize", &Var::AdamOptimizer::minimize)
        .def("get_variables", &Var::AdamOptimizer::get_variables)
        .def("get_loss", &Var::AdamOptimizer::get_loss)
        .def("run", &Var::AdamOptimizer::run);

    py::class_<complex_var>(m, "complex_var")
        .def(py::init<>())
        .def(py::init<Var::var>())
        .def(py::init<Var::var, Var::var>())
        .def("real", &complex_var::real)
        .def("imag", &complex_var::imag)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self);
        
    py::implicitly_convertible<Var::var, complex_var>();

}