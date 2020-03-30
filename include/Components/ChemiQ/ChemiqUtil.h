#ifndef CHEMIQUTIL_H
#define CHEMIQUTIL_H

#include "Components/DataStruct.h"
#include "Components/Operator/PauliOperator.h"
#include "Components/Operator/FermionOperator.h"
#include "Core/Variational/VarPauliOperator.h"
#include "Core/Variational/VarFermionOperator.h"

QPANDA_BEGIN

using namespace Variational;

/**
* @brief  get the electron number of the atom.
* @ingroup ChemiQ
* @param[in] std::string& atom
* @return size_t atom's electorn number
*/
DLLEXPORT 
size_t getElectronNum(const std::string &atom);

/**
* @brief  Jordan-Wigner transform of one fermion term, like "3+ 1 2+ 0".
* @ingroup ChemiQ
* @param[in] OrbitalActVec& fermion term
* @return PauliOperator
* @see OrbitalActVec
* @see PauliOperator
*/
DLLEXPORT 
PauliOperator JordanWignerTransform(const OrbitalActVec &fermion_item);

/**
* @brief  Jordan-Wigner transform from FermionOperator to PauliOperator.
* @ingroup ChemiQ
* @param[in] FermionOperator& fermion operator
* @return PauliOperator
* @see FermionOperator
* @see PauliOperator
*/
DLLEXPORT
PauliOperator JordanWignerTransform(const FermionOperator &fermion);

/**
* @brief  Jordan-Wigner transform from VarFermionOperator to VarPauliOperator.
* @ingroup ChemiQ
* @param[in] VarFermionOperator& variational fermion operator
* @return VarPauliOperator
* @see VarFermionOperator
* @see VarPauliOperator
*/
DLLEXPORT
VarPauliOperator JordanWignerTransform(const VarFermionOperator &fermion);

/**
* @brief  Parity transform of one fermion term, like "3+ 1 2+ 0".
* @ingroup ChemiQ
* @param[in] OrbitalActVec& fermion term
* @param[in] size_t maxqubit
* @return PauliOperator
* @see OrbitalActVec
* @see PauliOperator
*/
DLLEXPORT
PauliOperator ParityTransform(const OrbitalActVec &fermion_item, size_t maxqubit);

/**
* @brief  Parity transform from FermionOperator to PauliOperator.
* @ingroup ChemiQ
* @param[in] FermionOperator& fermion operator
* @return PauliOperator
* @see FermionOperator
* @see PauliOperator
*/
DLLEXPORT
PauliOperator ParityTransform(const FermionOperator &fermio);

/**
* @brief  Parity transform from VarFermionOperator to VarPauliOperator.
* @ingroup ChemiQ
* @param[in] VarFermionOperator& variational fermion operator
* @return VarPauliOperator
* @see VarFermionOperator
* @see VarPauliOperator
*/
DLLEXPORT
VarPauliOperator ParityTransform(const VarFermionOperator &fermion);

/**
* @brief BKMatrix required by  BravyiKitaev transform 
* @ingroup ChemiQ
* @param[in] size_t qn quantum number
*/
DLLEXPORT
std::vector<Eigen::MatrixXi> BKMatrix(size_t qn);

/**
* @brief  BravyiKitaev transform of one fermion term, like "3+ 1 2+ 0".
* @ingroup ChemiQ
* @param[in] OrbitalActVec& fermion term
* @param[in] size_t maxqubit
* @param[in] std::vector<Eigen::MatrixXi> BK
* @return PauliOperator
* @see OrbitalActVec
* @see PauliOperator
*/
DLLEXPORT
PauliOperator BravyiKitaevTransform(const OrbitalActVec &fermion_item, size_t maxqubit, std::vector<Eigen::MatrixXi> BK);

/**
* @brief  BravyiKitaev transform from FermionOperator to PauliOperator.
* @param[in] FermionOperator& fermion operator
* @param[in] std::vector<Eigen::MatrixXi> BK
* @return PauliOperator
* @see FermionOperator
* @see PauliOperator
*/
DLLEXPORT
PauliOperator BravyiKitaevTransform(const FermionOperator &fermion, std::vector<Eigen::MatrixXi> BK);

/**
* @brief  BravyiKitaev transform from VarFermionOperator to VarPauliOperator.
* @ingroup ChemiQ
* @param[in] VarFermionOperator& variational fermion operator
* @param[in] std::vector<Eigen::MatrixXi> BK
* @return VarPauliOperator
* @see VarFermionOperator
* @see VarPauliOperator
*/
DLLEXPORT
VarPauliOperator BravyiKitaevTransform(const VarFermionOperator &fermion, std::vector<Eigen::MatrixXi> BK);

/**
* @brief  get CCS term number.
* @ingroup ChemiQ
* @param[in] size_t quantum number(orbital number)
* @param[in] size_t electron number
* @return size_t CCS term number
* @note Coupled cluster single model.
*       e.g. 4 qubits, 2 electrons
*       then 0 and 1 are occupied,just consider 0->2,0->3,1->2,1->3
*/
DLLEXPORT
size_t getCCS_N_Trem(size_t qn, size_t en);

/**
* @brief  get CCSD term number.
* @ingroup ChemiQ
* @param[in] size_t quantum number(orbital number)
* @param[in] size_t electron number
* @return size_t CCSD term number
* @note Coupled cluster single and double model.
*       e.g. 4 qubits, 2 electrons
*       then 0 and 1 are occupied,just consider 0->2,0->3,1->2,1->3,01->23
*/
DLLEXPORT
size_t getCCSD_N_Trem(size_t qn, size_t en);

/**
* @brief  get Coupled cluster single model.
* @ingroup ChemiQ
* @param[in] size_t quantum number(orbital number)
* @param[in] size_t electron number
* @param[in] vector_d& parameters
* @return FermionOperator
* @note Coupled cluster single model.
*       e.g. 4 qubits, 2 electrons
*       then 0 and 1 are occupied,just consider 0->2,0->3,1->2,1->3.
*       returned FermionOperator like this:
*       {{"2+ 0":para0},{"3+ 0":para1},{"2+ 1":para2},{"3+ 1":para3}}
*/
DLLEXPORT
FermionOperator getCCS(
    size_t qn,
    size_t en,
    const vector_d &para_vec);

/**
* @brief  get Coupled cluster single model with variational parameters.
* @ingroup ChemiQ
* @param[in] size_t quantum number(orbital number)
* @param[in] size_t electron number
* @param[in] var parameters
* @return VarFermionOperator
* @note Coupled cluster single model.
*       e.g. 4 qubits, 2 electrons
*       then 0 and 1 are occupied,just consider 0->2,0->3,1->2,1->3.
*       returned FermionOperator like this:
*       {{"2+ 0":var[0]},{"3+ 0":var[1]},{"2+ 1":var[2]},{"3+ 1":var[3]}}
*/
DLLEXPORT
VarFermionOperator getCCS(
    size_t qn,
    size_t en,
    var &para);

/**
* @brief  get Coupled cluster single model with variational parameters.
* @ingroup ChemiQ
* @param[in] size_t quantum number(orbital number)
* @param[in] size_t electron number
* @param[in] std::vector<var>& parameters
* @return VarFermionOperator
* @note Coupled cluster single model.
*       e.g. 4 qubits, 2 electrons
*       then 0 and 1 are occupied,just consider 0->2,0->3,1->2,1->3.
*       returned FermionOperator like this:
*       {{"2+ 0":var[0]},{"3+ 0":var[1]},{"2+ 1":var[2]},{"3+ 1":var[3]}}
*/
DLLEXPORT
VarFermionOperator getCCS(
    size_t qn,
    size_t en,
    std::vector<var>& para);

/**
* @brief  get Coupled cluster single and double model.
* @ingroup ChemiQ
* @param[in] size_t quantum number(orbital number)
* @param[in] size_t electron number
* @param[in] vector_d& parameters
* @return FermionOperator
* @note Coupled cluster single and double model.
*       e.g. 4 qubits, 2 electrons
*       then 0 and 1 are occupied,just consider 0->2,0->3,1->2,1->3,01->23.
*       returned FermionOperator like this:
*       {{"2+ 0":para0},{"3+ 0":para1},{"2+ 1":para2},{"3+ 1":para3},
*       {"3+ 2+ 1 0":para5}}
*/
DLLEXPORT
FermionOperator getCCSD(
    size_t qn,
    size_t en,
    const vector_d &para_vec);

/**
* @brief  get Coupled cluster single and double model 
*         with variational parameters.
* @ingroup ChemiQ
* @param[in] size_t quantum number(orbital number)
* @param[in] size_t electron number
* @param[in] var& parameters
* @return VarFermionOperator
* @note Coupled cluster single and double model.
*       e.g. 4 qubits, 2 electrons
*       then 0 and 1 are occupied,just consider 0->2,0->3,1->2,1->3,01->23.
*       returned FermionOperator like this:
*       {{"2+ 0":var[0]},{"3+ 0":var[1]},{"2+ 1":var[2]},{"3+ 1":var[3]},
*       {"3+ 2+ 1 0":var[4]}}
*/
DLLEXPORT
VarFermionOperator getCCSD(
    size_t qn,
    size_t en,
    var &para);

/**
* @brief  get Coupled cluster single and double model 
*         with variational parameters.
* @ingroup ChemiQ
* @param[in] size_t quantum number(orbital number)
* @param[in] size_t electron number
* @param[in] std::vector<var>& parameters
* @return VarFermionOperator
* @note Coupled cluster single and double model.
*       e.g. 4 qubits, 2 electrons
*       then 0 and 1 are occupied,just consider 0->2,0->3,1->2,1->3,01->23.
*       returned FermionOperator like this:
*       {{"2+ 0":var[0]},{"3+ 0":var[1]},{"2+ 1":var[2]},{"3+ 1":var[3]},
*       {"3+ 2+ 1 0":var[4]}}
*/
DLLEXPORT
VarFermionOperator getCCSD(
    size_t qn,
    size_t en,
    std::vector<var>& para);

/**
* @brief  Generate Hamiltonian form of unitary coupled cluster based on coupled
*         cluster,H=1j*(T-dagger(T)), then exp(-jHt)=exp(T-dagger(T)).
* @ingroup ChemiQ
* @param[in] PauliOperator& pauli operator
* @return PauliOperator
* @see PauliOperator
*/
DLLEXPORT
PauliOperator transCC2UCC(const PauliOperator &cc);

/**
* @brief  Generate Hamiltonian form of unitary coupled cluster based on coupled
*         cluster,H=1j*(T-dagger(T)), then exp(-jHt)=exp(T-dagger(T)).
* @ingroup ChemiQ
* @param[in] VarPauliOperator& pauli operator
* @return VarPauliOperator
* @see VarPauliOperator
*/
DLLEXPORT
VarPauliOperator transCC2UCC(const VarPauliOperator &cc);

/**
* @brief  Simulate a general case of hamiltonian by Trotter-Suzuki
*         approximation. U=exp(-iHt)=(exp(-i H1 t/n)*exp(-i H2 t/n))^n
* @ingroup ChemiQ
* @param[in] QVec& the qubit needed to simulate the Hamiltonian
* @param[in] VarPauliOperator& Hamiltonian
* @param[in] double time
* @param[in] size_t the approximate slices
* @return VQC
* @see QVec
* @see VarPauliOperator
* @see QPanda::Variational::VQC
*/
DLLEXPORT
VQC simulateHamiltonian(
    QVec &qubit_vec,
    VarPauliOperator &pauli,
    double t,
    size_t slices);

/**
* @brief  Simulate a single term of Hamilonian like "X0 Y1 Z2" with
*         coefficient and time. U=exp(-it*coef*H)
* @ingroup ChemiQ
* @param[in] QVec& the qubit needed to simulate the Hamiltonian
* @param[in] QTerm& Hamiltonian term, string like "X0 Y1 Z2"
* @param[in] var& the coefficient of hamiltonian
* @param[in] double time
* @return VQC
* @see QVec
* @see QTerm
* @see QPanda::Variational::var
*/
DLLEXPORT
VQC simulateOneTerm(
    QVec &qubit_vec,
    const QTerm &hamiltonian_term,
    const var &coef,
    double t);

/**
* @brief  Simulating z-only term like H=coef * (Z0..Zn-1)
*         U=exp(-iHt)
* @ingroup ChemiQ
* @param[in] QVec& the qubit needed to simulate the Hamiltonian
* @param[in] var& the coefficient of hamiltonian
* @param[in] double time
* @return VQC
* @see QVec
* @see QPanda::Variational::var
*/
DLLEXPORT
VQC simulateZTerm(
    QVec &qubit_vec,
    const var &coef,
    double t);

/**
* @brief  Parse psi4 data to fermion operator.
* @ingroup ChemiQ
* @param[in] std::string& fermon str
* @return FermionOperator
* @see FermionOperator
*/
DLLEXPORT
FermionOperator parsePsi4DataToFermion(const std::string& data);

QPANDA_END
#endif // CHEMIQUTIL_H
