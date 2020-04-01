/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

HamiltonianSimulation.h

Author: LiYe
Created in 2018-09-19


*/

#ifndef HAMILTONIANSIMULATION_H
#define HAMILTONIANSIMULATION_H

#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Components/DataStruct.h"

namespace QPanda
{
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
        const std::vector<Qubit*> &qubit_vec,
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
        const std::vector<Qubit*> &qubit_vec,
        const QTerm &hamiltonian_term,
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
        const std::vector<Qubit*> &qubit_vec,
        const QHamiltonian &hamiltonian,
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
        const std::vector<Qubit*> &qubit_vec,
        const QHamiltonian &hamiltonian,
        double t);

    /**
	* @brief Apply single gates to all qubits in qubit_list
	* @ingroup HamiltonianSimulation
	* @param[in] std::string& gate: single gate name.
	* @param[in] std::vector<Qubit*>& qubit_vec: qubit vector
	* @return QCircuit
    */
    QCircuit applySingleGateToAll(
        const std::string &gate,
        const std::vector<Qubit*> &qubit_vec);

    /**
	* @brief Apply single gates to all qubits in qubit_list and insert into circuit.
	* @ingroup HamiltonianSimulation
	* @param[in] std::string& gate: single gate name.
	* @param[in] std::vector<Qubit*>& qubit_vec: qubit vector
	* @param[in] QCircuit& circuit: operated circuit.
	* @return
    */
    void applySingleGateToAll(
        const std::string &gate,
        const std::vector<Qubit*> &qubit_vec,
        QCircuit &circuit);

    /**
	* @brief Ising model
	* @ingroup HamiltonianSimulation
	* @param[in] std::vector<Qubit*>&  qubit_vec: qubit vector
	* @param[in] QGraph& graph: the problem graph
	* @param[in] vector_d& gamma: model para
	* @return QCircuit 
    */
    QCircuit ising_model(
        const std::vector<Qubit*> &qubit_vec,
        const QGraph &graph,
        const vector_d &gamma);

    /**
	* @brief pauli X model
	* @ingroup HamiltonianSimulation
	* @param[in] std::vector<Qubit*>& qubit_vec: qubit vector
	* @param[in] vector_d& beta: model para
	* @return QCircuit
    */
    QCircuit pauliX_model(
        const std::vector<Qubit*> &qubit_vec,
        const vector_d &beta);
}

#endif // HAMILTONIANSIMULATION_H
