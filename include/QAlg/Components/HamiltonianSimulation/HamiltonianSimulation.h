/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

HamiltonianSimulation.h

Author: LiYe
Created in 2018-09-19


*/

#ifndef HAMILTONIANSIMULATION_H
#define HAMILTONIANSIMULATION_H

#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "QAlg/DataStruct.h"

namespace QPanda
{
    /*
    Simulating z-only term like H=coef * (Z0..Zn-1)
    U=exp(-iHt)

    param:
        qubit_vec: the qubit needed to simulate the Hamiltonian
        coef: the coefficient of hamiltonian
        t: time
    return:
        QCircuit

    Note:
        Z-Hamiltonian spreads over the qubit_vec
    */
    QCircuit simulateZTerm(
        const std::vector<Qubit*> &qubit_vec, 
        double coef,
        double t);
    /*
    Simulate a single term of Hamilonian like "X0 Y1 Z2" with
    coefficient and time. U=exp(-it*coef*H)

    param:
        qubit_vec: the qubit needed to simulate the Hamiltonian
        hamiltonian_term: string like "X0 Y1 Z2"
        coef: the coefficient of hamiltonian
        t: time
    return:
        QCircuit
    */
    QCircuit simulateOneTerm(
        const std::vector<Qubit*> &qubit_vec,
        const QTerm &hamiltonian_term,
        double coef,
        double t);

    /*
    Simulate a general case of hamiltonian by Trotter-Suzuki
    approximation. U=exp(-iHt)=(exp(-i H1 t/n)*exp(-i H2 t/n))^n

    param:
        qubit_vec: the qubit needed to simulate the Hamiltonian
        hamiltonian: Hamiltonian
        t: time
        slices: the approximate slices
    return:
        QCircuit

    */
    QCircuit simulateHamiltonian(
        const std::vector<Qubit*> &qubit_vec,
        const QHamiltonian &hamiltonian,
        double t,
        size_t slices);

    /*
    Simulate hamiltonian consists of pauli-Z operators

    param:
        qubit_vec: the qubit needed to simulate the Hamiltonian
        hamiltonian: Hamiltonian
        t: time
    return:
        QCircuit

    */
    QCircuit simulatePauliZHamiltonian(
        const std::vector<Qubit*> &qubit_vec,
        const QHamiltonian &hamiltonian,
        double t);

    /*
    Apply single gates to all qubits in qubit_list

    param:
        gate: single gate name.
        qubit_vec: qubit vector
    */
    QCircuit applySingleGateToAll(
        const std::string &gate,
        const std::vector<Qubit*> &qubit_vec);

    /*
    Apply single gates to all qubits in qubit_list and insert into circuit.

    param:
        gate: single gate name.
        qubit_vec: qubit vector
        circuit: operated circuit.
    */
    void applySingleGateToAll(
        const std::string &gate,
        const std::vector<Qubit*> &qubit_vec,
        QCircuit &circuit);

    /*
    Ising model

    param:
        qubit_vec: qubit vector
        graph: the problem graph
        gamma: model para
    */
    QCircuit ising_model(
        const std::vector<Qubit*> &qubit_vec,
        const QGraph &graph,
        const vector_d &gamma);

    /*
    pauli X model

    param:
        qubit_vec: qubit vector
        beta: model para
    */
    QCircuit pauliX_model(
        const std::vector<Qubit*> &qubit_vec,
        const vector_d &beta);
}

#endif // HAMILTONIANSIMULATION_H
