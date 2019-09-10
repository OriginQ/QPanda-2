#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "ChemiQ/ChemiqUtil.h"

USING_QPANDA
namespace py = pybind11;

auto JW_Normal = [](const FermionOperator& fermion)
{
    return JordanWignerTransform(fermion);
};

auto JW_Var = [](const VarFermionOperator& fermion)
{
    return JordanWignerTransform(fermion);
};

auto getCCS_Normal = [](size_t qn, size_t en, const vector_d& para_vec)
{
    return getCCS(qn, en, para_vec);
};

auto getCCS_Var = [](size_t qn, size_t en, var& para_vec)
{
    return getCCS(qn, en, para_vec);
};

auto getCCSD_Normal = [](size_t qn, size_t en, const vector_d& para_vec)
{
    return getCCSD(qn, en, para_vec);
};

auto getCCSD_Var = [](size_t qn, size_t en, var& para_vec)
{
    return getCCSD(qn, en, para_vec);
};

auto transCC2UCC_Normal = [](const PauliOperator& cc)
{
    return transCC2UCC(cc);
};

auto transCC2UCC_Var = [](const VarPauliOperator& cc)
{
    return transCC2UCC(cc);
};

auto simulateHamiltonian_Var = [](
    std::vector<Qubit*> &qubit_vec,
    VarPauliOperator& pauli,
    double t,
    size_t slices
    )
{
    auto tmp_vec = QVec(qubit_vec);
    return simulateHamiltonian(tmp_vec, pauli, t, slices);
};

void initChemiQUtil(py::module& m)
{
    m.def("getElectronNum", 
        &getElectronNum,
        "get the electron number of the atom.");
    m.def("JordanWignerTransform",
        JW_Normal,
        "Jordan-Wigner transform from FermionOperator to PauliOperator.");
    m.def("JordanWignerTransformVar",
        JW_Var,
        "Jordan-Wigner transform from VarFermionOperator to VarPauliOperator.");
    m.def("getCCS_N_Trem",
        getCCS_N_Trem,
        "get CCS term number.");
    m.def("getCCSD_N_Trem",
        getCCSD_N_Trem,
        "get CCSD term number.");
    m.def("getCCS_Normal",
        getCCS_Normal,
        "get Coupled cluster single model.");
    m.def("getCCS_Var",
        getCCS_Var,
        "get Coupled cluster single model with variational parameters.");
    m.def("getCCSD_Normal",
        getCCSD_Normal,
        "get Coupled cluster single and double model.");
    m.def("getCCSD_Var",
        getCCSD_Var,
        "get Coupled cluster single and double model "
        "with variational parameters.");
    m.def("transCC2UCC_Normal",
        transCC2UCC_Normal,
        "Generate Hamiltonian form of unitary coupled cluster based on coupled "
        "cluster, H = 1j * (T-dagger(T)), then exp(-jHt) = exp(T-dagger(T)).");
    m.def("transCC2UCC_Var",
        transCC2UCC_Var,
        "Generate Hamiltonian form of unitary coupled cluster based on coupled "
        "cluster, H = 1j * (T-dagger(T)), then exp(-jHt) = exp(T-dagger(T)).");
    m.def("simulateHamiltonian_Var",
        simulateHamiltonian_Var,
        "Simulate a general case of hamiltonian by Trotter-Suzuki "
        "approximation.U = exp(-iHt) = (exp(-iH1t/n) * exp(-iH2t/n))^n");
    m.def("parsePsi4DataToFermion",
        parsePsi4DataToFermion,
        "Parse psi4 data to fermion operator.");
}
