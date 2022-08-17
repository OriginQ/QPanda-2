/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

HamiltonianSimulation.h

Author: LiYe
Created in 2018-09-19


*/

#ifndef HAMILTONIANSIMULATION_H
#define HAMILTONIANSIMULATION_H

#include "Core/QuantumCircuit/QNode.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumMachine/QVec.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/Module/DataStruct.h"
#include "Core/Utilities/QProgInfo/ConfigMap.h"
#include "Core/Utilities/Tools/QMatrixDef.h"

#include <complex>

QPANDA_BEGIN


class QOperator : public QCircuit
{
protected:
    QCircuit m_pQuantumCircuitOperator;
public:
    QOperator();
    QOperator(QGate& gate);
    QOperator(QCircuit& circuit);
    QCircuit copy(QCircuit& circuit);
    QCircuit copy(QGate& gate);
    QCircuit copy(std::shared_ptr<AbstractQuantumCircuit> node);
 
    std::shared_ptr<AbstractQuantumCircuit> getImplementationPtr();
    QStat get_matrix();
    std::string to_instruction(std::string ir_type = "OriginIR");
    

};




QPANDA_END


namespace QPanda
{
    /*************************************************************
    * @brief Computing the matrix exponential using Pade approximation.
             U=exp(-iHt)
    * @ingroup HamiltonianSimulation
    * @param[in] const std::complex<double>& conf : i
    * @param[in] EigenMatrixComplex& Mat : the Matrix H
    * @param[in] number 
    * @return Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>
    * @note 
        ----------
        A : (N, N) array_like or sparse matrix
        Matrix to be exponentiated.

        Returns
        ------ -
        expm : (N, N) ndarray
        Matrix exponential of `A`.

        References
        ----------
        ..[1] Awad H.Al - Mohy and Nicholas J.Higham(2009)
        "A New Scaling and Squaring Algorithm for the Matrix Exponential."
        SIAM Journal on Matrix Analysis and Applications.
        31 (3).pp. 970 - 989. ISSN 1095 - 7162

        Euler's identity (exp(i*theta) = cos(theta) + i*sin(theta))
        applied to a matrix 
    *************************************************************/
    QMatrixXcd expMat(
        const qcomplex_t& conf,
        const QMatrixXcd& Mat,
        double number = PI);

    /**
    * @brief Simulating z-only term like H=coef * (Z0..Zn-1)
             U=exp(-iHt)
    * @ingroup HamiltonianSimulation
    * @param[in] std::vector<Qubit*>& the qubit needed to simulate the Hamiltonian
    * @param[in] double the coefficient of hamiltonian
    * @param[in] double time
    * @return QCircuit
    * @note Z-Hamiltonian spreads over the qubit_vec
    */
    QCircuit simulateZTerm(
        const std::vector<Qubit*>& qubit_vec,
        double coef,
        double t);

    /**
    * @brief Simulate a single term of Hamilonian like "X0 Y1 Z2" with
             coefficient and time. U=exp(-it*coef*H)
    * @ingroup HamiltonianSimulation
    * @param[in] std::vector<Qubit*>& the qubit needed to simulate the Hamiltonian
    * @param[in] QTerm& hamiltonian_term: string like "X0 Y1 Z2"
    * @param[in] double coef: the coefficient of hamiltonian
    * @param[in] double t time
    * @return QCircuit
    */
    QCircuit simulateOneTerm(
        const std::vector<Qubit*>& qubit_vec,
        const QTerm& hamiltonian_term,
        double coef,
        double t);

    /**
    * @brief Simulate a general case of hamiltonian by Trotter-Suzuki
             approximation. U=exp(-iHt)=(exp(-i H1 t/n)*exp(-i H2 t/n))^n
    * @ingroup HamiltonianSimulation
    * @param[in] std::vector<Qubit*>& qubit_vec: the qubit needed to simulate the Hamiltonian
    * @param[in] QHamiltonian& hamiltonian: Hamiltonian
    * @param[in] double t: time
    * @param[in] size_t slices: the approximate slices
    * @return QCircuit
    */
    QCircuit simulateHamiltonian(
        const std::vector<Qubit*>& qubit_vec,
        const QHamiltonian& hamiltonian,
        double t,
        size_t slices);

    /**
    * @brief Simulate hamiltonian consists of pauli-Z operators
    * @ingroup HamiltonianSimulation
    * @param[in] std::vector<Qubit*>& qubit_vec: the qubit needed to simulate the Hamiltonian
    * @param[in] QHamiltonian& hamiltonian: Hamiltonian
    * @param[in] double t: time
    * @return QCircuit
    */
    QCircuit simulatePauliZHamiltonian(
        const std::vector<Qubit*>& qubit_vec,
        const QHamiltonian& hamiltonian,
        double t);

    /**
    * @brief Apply single gates to all qubits in qubit_list
    * @ingroup HamiltonianSimulation
    * @param[in] std::string& gate: single gate name.
    * @param[in] std::vector<Qubit*>& qubit_vec: qubit vector
    * @return QCircuit
    */
    QCircuit applySingleGateToAll(
        const std::string& gate,
        const std::vector<Qubit*>& qubit_vec);

    /**
    * @brief Apply single gates to all qubits in qubit_list and insert into circuit.
    * @ingroup HamiltonianSimulation
    * @param[in] std::string& gate: single gate name.
    * @param[in] std::vector<Qubit*>& qubit_vec: qubit vector
    * @param[in] QCircuit& circuit: operated circuit.
    * @return
    */
    void applySingleGateToAll(
        const std::string& gate,
        const std::vector<Qubit*>& qubit_vec,
        QCircuit& circuit);

    /**
    * @brief Ising model
    * @ingroup HamiltonianSimulation
    * @param[in] std::vector<Qubit*>&  qubit_vec: qubit vector
    * @param[in] QGraph& graph: the problem graph
    * @param[in] vector_d& gamma: model para
    * @return QCircuit
    */
    QCircuit ising_model(
        const std::vector<Qubit*>& qubit_vec,
        const QGraph& graph,
        const vector_d& gamma);

    /**
    * @brief pauli X model
    * @ingroup HamiltonianSimulation
    * @param[in] std::vector<Qubit*>& qubit_vec: qubit vector
    * @param[in] vector_d& beta: model para
    * @return QCircuit
    */
    QCircuit pauliX_model(
        const std::vector<Qubit*>& qubit_vec,
        const vector_d& beta);

};

#endif // HAMILTONIANSIMULATION_H