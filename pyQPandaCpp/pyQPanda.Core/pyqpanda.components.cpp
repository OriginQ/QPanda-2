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
#include "QPanda.h"


USING_QPANDA
using namespace std;
using namespace pybind11::literals;
using std::map;
namespace py = pybind11;

void init_components(py::module& m)
{
    py::enum_<OptimizerType>(m, "OptimizerType", py::arithmetic())
        .value("NELDER_MEAD", OptimizerType::NELDER_MEAD)
        .value("POWELL", OptimizerType::POWELL)
        .value("GRADIENT", OptimizerType::GRADIENT)
        .export_values();

    py::class_<AbstractOptimizer>(m, "AbstractOptimizer")
        .def("registerFunc", &AbstractOptimizer::registerFunc)
        .def("setXatol", &AbstractOptimizer::setXatol)
        .def("exec", &AbstractOptimizer::exec)
        .def("getResult", &AbstractOptimizer::getResult)
        .def("setAdaptive", &AbstractOptimizer::setAdaptive)
        .def("setDisp", &AbstractOptimizer::setDisp)
        .def("setFatol", &AbstractOptimizer::setFatol)
        .def("setMaxFCalls", &AbstractOptimizer::setMaxFCalls)
        .def("setMaxIter", &AbstractOptimizer::setMaxIter)
        .def("setRestoreFromCacheFile", &AbstractOptimizer::setRestoreFromCacheFile)
        .def("setCacheFile", &AbstractOptimizer::setCacheFile);

    py::class_<OptimizerFactory>(m, "OptimizerFactory")
        .def(py::init<>())
        .def("makeOptimizer", py::overload_cast<const OptimizerType&>
            (&OptimizerFactory::makeOptimizer), "Please input OptimizerType ")
        .def("makeOptimizer", py::overload_cast<const std::string&>
            (&OptimizerFactory::makeOptimizer), "Please input the Optimizer's name(string)")
        ;

    py::class_<QOptimizationResult>(m, "QOptimizationResult")
        .def(py::init<std::string&, size_t&, size_t&, std::string&, double&, vector_d&>())
        .def_readwrite("message", &QOptimizationResult::message)
        .def_readwrite("fcalls", &QOptimizationResult::fcalls)
        .def_readwrite("fun_val", &QOptimizationResult::fun_val)
        .def_readwrite("iters", &QOptimizationResult::iters)
        .def_readwrite("key", &QOptimizationResult::key)
        .def_readwrite("para", &QOptimizationResult::para);
        
    py::class_<NodeSortProblemGenerator>(m, "NodeSortProblemGenerator")
        .def(py::init<>())
        .def("set_problem_graph", &NodeSortProblemGenerator::setProblemGraph)
        .def("exec", &NodeSortProblemGenerator::exec)
        .def("get_ansatz", &NodeSortProblemGenerator::getAnsatz)
        .def("get_Hamiltonian", &NodeSortProblemGenerator::getHamiltonian);
}
