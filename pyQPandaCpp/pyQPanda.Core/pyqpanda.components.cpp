#include "QPanda.h"
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


USING_QPANDA
using namespace std;
using namespace pybind11::literals;
using std::map;
namespace py = pybind11;

void export_components(py::module &m)
{
    py::enum_<OptimizerType>(m, "OptimizerType", "quantum OptimizerType"
        , py::arithmetic())
        .value("NELDER_MEAD", OptimizerType::NELDER_MEAD)
        .value("POWELL", OptimizerType::POWELL)
        .value("GRADIENT", OptimizerType::GRADIENT)
        .export_values();

    py::class_<AbstractOptimizer>(m, "AbstractOptimizer", "quantum AbstractOptimizer class")
        .def("registerFunc", &AbstractOptimizer::registerFunc,
            "Register an optimization function.\n"
            "\n"
            "Args:\n"
            "     func: The optimization function to be registered.\n"
            "\n"
            "Returns:\n"
            "     None")
        .def("setXatol", &AbstractOptimizer::setXatol,
            "Set the absolute tolerance for optimization.\n"
            "\n"
            "Args:\n"
            "     atol: The absolute tolerance value to be set.\n"
            "\n"
            "Returns:\n"
            "     None")
        .def("exec", &AbstractOptimizer::exec,
            "Execute the optimization process.\n"
            "\n"
            "Args:\n"
            "     None: This method takes no parameters.\n"
            "\n"
            "Returns:\n"
            "     result: The result of the optimization process.")
        .def("getResult", &AbstractOptimizer::getResult,
            "Retrieve the result of the last optimization.\n"
            "\n"
            "Args:\n"
            "     None: This method takes no parameters.\n"
            "\n"
            "Returns:\n"
            "     result: The result of the last optimization.")
        .def("setAdaptive", &AbstractOptimizer::setAdaptive,
            "Set whether the optimizer should use adaptive methods.\n"
            "\n"
            "Args:\n"
            "     adaptive: A boolean indicating whether to enable adaptive optimization.\n"
            "\n"
            "Returns:\n"
            "     None")
        .def("setDisp", &AbstractOptimizer::setDisp,
            "Set the display flag for the optimizer.\n"
            "\n"
            "Args:\n"
            "     disp: A boolean indicating whether to display optimization progress.\n"
            "\n"
            "Returns:\n"
            "     None")
        .def("setFatol", &AbstractOptimizer::setFatol,
            "Set the function absolute tolerance for optimization.\n"
            "\n"
            "Args:\n"
            "     fatol: The function absolute tolerance value to be set.\n"
            "\n"
            "Returns:\n"
            "     None")
        .def("setMaxFCalls", &AbstractOptimizer::setMaxFCalls,
            "Set the maximum number of function calls allowed during optimization.\n"
            "\n"
            "Args:\n"
            "     max_calls: The maximum number of function calls to be set.\n"
            "\n"
            "Returns:\n"
            "     None")
        .def("setMaxIter", &AbstractOptimizer::setMaxIter,
            "Set the maximum number of iterations allowed during optimization.\n"
            "\n"
            "Args:\n"
            "     max_iter: The maximum number of iterations to be set.\n"
            "\n"
            "Returns:\n"
            "     None")
        .def("setRestoreFromCacheFile", &AbstractOptimizer::setRestoreFromCacheFile,
            "Set whether to restore the optimization state from a cache file.\n"
            "Args:\n"
            "\n"
            "     cache_file: A string representing the path to the cache file.\n"
            "\n"
            "Returns:\n"
            "     None\n"
        )
        .def("setCacheFile", &AbstractOptimizer::setCacheFile,
            "Set the path for the cache file used in optimization.\n"
            "\n"
            "Args:\n"
            "     cache_file: A string representing the path to the cache file.\n"
            "\n"
            "Returns:\n"
            "     None\n"
        );

    py::class_<OptimizerFactory>(m, "OptimizerFactory", "quantum OptimizerFactory class")
        .def(py::init<>())
        .def("makeOptimizer", py::overload_cast<const OptimizerType &>(&OptimizerFactory::makeOptimizer),
            "Create an optimizer of the specified type.\n"
            "\n"
            "Args:\n"
            "     optimizer_type: An instance of OptimizerType indicating the desired optimizer.\n"
            "\n"
            "Returns:\n"
            "     An instance of the created optimizer.\n"
        )
        .def("makeOptimizer", py::overload_cast<const std::string &>(&OptimizerFactory::makeOptimizer),
            "Create an optimizer using its name.\n"
            "\n"
            "Args:\n"
            "     optimizer_name: A string representing the name of the desired optimizer.\n"
            "\n"
            "Returns:\n"
            "     An instance of the created optimizer.\n"
        );

    py::class_<QOptimizationResult>(m, "QOptimizationResult", "quantum QOptimizationResult class")
        .def(py::init<std::string &, size_t &, size_t &, std::string &, double &, vector_d &>(),
            "Initialize a QOptimizationResult instance.\n"
            "\n"
            "Args:\n"
            "     message: A string containing the result message.\n"
            "\n"
            "     iteration: A size_t representing the iteration count.\n"
            "\n"
            "     total_iterations: A size_t representing the total iterations.\n"
            "\n"
            "     status: A string indicating the optimization status.\n"
            "\n"
            "     value: A double representing the optimization value.\n"
            "\n"
            "     results: A vector of doubles for the optimization results.\n"
        )
        .def_readwrite("message", &QOptimizationResult::message,
            "A string containing the result message of the optimization.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A string that provides details about the optimization result.\n"
        )
        .def_readwrite("fcalls", &QOptimizationResult::fcalls,
            "An integer representing the number of function calls made during optimization.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     An integer indicating the count of function calls.\n"
        )
        .def_readwrite("fun_val", &QOptimizationResult::fun_val,
            "A double representing the value of the objective function at the optimal point.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A double indicating the function value.\n"
        )
        .def_readwrite("iters", &QOptimizationResult::iters,
            "An integer representing the number of iterations performed during optimization.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     An integer indicating the iteration count.\n"
        )
        .def_readwrite("key", &QOptimizationResult::key,
            "A string representing a unique identifier for the optimization result.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A string that serves as the key for the result.\n"
        )
        .def_readwrite("para", &QOptimizationResult::para,
            "A vector of doubles representing the parameters used in the optimization.\n"
            "\n"
            "Args:\n"
            "     None\n"
            "\n"
            "Returns:\n"
            "     A vector of doubles containing the optimization parameters.\n"
        );
}
