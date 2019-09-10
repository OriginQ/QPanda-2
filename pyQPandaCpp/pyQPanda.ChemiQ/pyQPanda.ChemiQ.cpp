#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "ChemiQ/ChemiQ.h"

namespace py = pybind11;

using namespace QPanda;

void initChemiQUtil(py::module&);

PYBIND11_MODULE(pyQPandaChemiQ, m)
{
    m.doc() = "QPanda ChemiQ";

    initChemiQUtil(m);

    py::enum_<UccType>(m, "UccType", py::arithmetic())
        .value("UCCS", UccType::UCCS)
        .value("UCCSD", UccType::UCCSD)
        .export_values();

    py::enum_<TransFormType>(m, "TransFormType", py::arithmetic())
        .value("Jordan_Wigner", TransFormType::Jordan_Wigner)
        .value("Bravyi_Ktaev", TransFormType::Bravyi_Ktaev)
		.value("Parity", TransFormType::Parity)
        .export_values();

    py::class_<ChemiQ>(m, "ChemiQ")
        .def(py::init<>())
        //.def("setMoleculeHamiltonian", &ChemiQ::setMoleculeHamiltonian)
        //.def("setMoleculesHamiltonian", &ChemiQ::setMoleculesHamiltonian)
        .def("initialize", &ChemiQ::initialize)
        .def("finalize", &ChemiQ::finalize)
        .def("setMolecule", &ChemiQ::setMolecule)
        .def("setMolecules", &ChemiQ::setMolecules)
        .def("setMultiplicity", &ChemiQ::setMultiplicity)
        .def("setCharge", &ChemiQ::setCharge)
        .def("setBasis", &ChemiQ::setBasis)
        .def("setTransformType", &ChemiQ::setTransformType)
        .def("setUccType", &ChemiQ::setUccType)
        .def("setOptimizerType", &ChemiQ::setOptimizerType)
        .def("setOptimizerIterNum", &ChemiQ::setOptimizerIterNum)
        .def("setOptimizerFuncCallNum", &ChemiQ::setOptimizerFuncCallNum)
        .def("setOptimizerXatol", &ChemiQ::setOptimizerXatol)
        .def("setOptimizerFatol", &ChemiQ::setOptimizerFatol)
        .def("setOptimizerDisp", &ChemiQ::setOptimizerDisp)
        .def("setLearningRate", &ChemiQ::setLearningRate)
        .def("setEvolutionTime", &ChemiQ::setEvolutionTime)
        .def("setHamiltonianSimulationSlices",
            &ChemiQ::setHamiltonianSimulationSlices)
        .def("setSaveDataDir", &ChemiQ::setSaveDataDir)
        .def("setRandomPara", &ChemiQ::setRandomPara)
        .def("setDefaultOptimizedPara", &ChemiQ::setDefaultOptimizedPara)
        .def("setToGetHamiltonianFromFile", &ChemiQ::setToGetHamiltonianFromFile)
        .def("setHamiltonianGenerationOnly", &ChemiQ::setHamiltonianGenerationOnly)
        .def("exec", &ChemiQ::exec)
        .def("getLastError", &ChemiQ::getLastError)
        .def("getEnergies", &ChemiQ::getEnergies);
}
