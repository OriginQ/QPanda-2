#include <math.h>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "QAlg/ChemiQ/ChemiQ.h"

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
		.def("exec", &ChemiQ::exec)
		.def("initialize", &ChemiQ::initialize)
		.def("finalize", &ChemiQ::finalize)

		/*will delete*/
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
        .def("getLastError", &ChemiQ::getLastError)
        .def("getEnergies", &ChemiQ::getEnergies)
		/*new interface*/
		.def("set_molecule", &ChemiQ::setMolecule)
		.def("set_molecules", &ChemiQ::setMolecules)
		.def("set_multiplicity", &ChemiQ::setMultiplicity)
		.def("set_charge", &ChemiQ::setCharge)
		.def("set_basis", &ChemiQ::setBasis)
		.def("set_transform_type", &ChemiQ::setTransformType)
		.def("set_ucc_type", &ChemiQ::setUccType)
		.def("set_optimizer_type", &ChemiQ::setOptimizerType)
		.def("set_optimizer_iter_num", &ChemiQ::setOptimizerIterNum)
		.def("set_optimizer_func_call_num", &ChemiQ::setOptimizerFuncCallNum)
		.def("set_optimizer_xatol", &ChemiQ::setOptimizerXatol)
		.def("set_optimizer_fatol", &ChemiQ::setOptimizerFatol)
		.def("set_optimizer_disp", &ChemiQ::setOptimizerDisp)
		.def("set_learning_rate", &ChemiQ::setLearningRate)
		.def("set_evolution_time", &ChemiQ::setEvolutionTime)
		.def("set_hamiltonian_simulation_slices",
			&ChemiQ::setHamiltonianSimulationSlices)
		.def("set_save_data_dir", &ChemiQ::setSaveDataDir)
		.def("set_random_para", &ChemiQ::setRandomPara)
		.def("set_default_optimized_para", &ChemiQ::setDefaultOptimizedPara)
		.def("set_to_get_hamiltonian_from_file", &ChemiQ::setToGetHamiltonianFromFile)
		.def("set_hamiltonian_generation_only", &ChemiQ::setHamiltonianGenerationOnly)
		.def("exec", &ChemiQ::exec)
		.def("get_last_error", &ChemiQ::getLastError)
		.def("get_energies", &ChemiQ::getEnergies);
}
