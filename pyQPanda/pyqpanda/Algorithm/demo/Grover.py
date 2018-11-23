'''
Easy demo for Grover algorithm\n
Copyright (C) Origin Quantum 2017-2018\n
Licensed Under Apache Licence 2.0
'''

from pyqpanda.Algorithm.fragments import (two_qubit_database,
                                         diffusion_operator,
                                         two_qubit_database)
from pyqpanda import *
from pyqpanda.utils import *

def Three_Qubit_Grover_Circuit(qubits, ancilla, data_pos):
    '''
    
    Three Qubit Grover Circuit Generator\n
    list<Qubit>, Qubit, int -> QCircuit\n

    Args:
        qubits:   working qubits (len=2)
        ancilla:  ancillary qubit
        data_pos: suppose the data position
    '''
    circ=QCircuit()
    
    
    circ.insert(NOT(ancilla)) \
       .insert(single_gate_apply_to_all(gate=H, qubit_list=qubits)) \
       .insert(H(ancilla)) \
       .insert(two_qubit_database(data_pos,qubits,ancilla)) \
       .insert(diffusion_operator(qubits))
    
    '''
    circ.insert(NOT(ancilla)) \
        .insert(H(qubits[0])).insert(H(qubits[1]))\
        .insert(H(ancilla))\
        .insert(two_qubit_database(data_pos,qubits,ancilla))\
        .insert(H(qubits[0])).insert(H(qubits[1]))\
        .insert(X(qubits[0])).insert(X(qubits[1]))\
        .insert(H(qubits[1])).insert(CNOT(qubits[0],qubits[1])).insert(H(qubits[1]))\
        .insert(X(qubits[0])).insert(X(qubits[1]))\
        .insert(H(qubits[0])).insert(H(qubits[1]))    
    '''
    return circ

def Three_Qubit_Grover_Demo():
    '''
    `QPanda Easy Demo`\n
    Quantum algorithm easy demo : Grover's algorithm
    '''

    print("Demo for Grover Search\n")
    for data_position in range(4):
        #initialization
        init()

        #allocating qubits/cbits
        q=qAlloc_many(2)
        anc=qAlloc()    # ancillary qubits
        c=cAlloc_many(2)

        prog=QProg()
        prog.insert(Three_Qubit_Grover_Circuit(qubits=q, ancilla=anc, data_pos=data_position))\
            .insert(meas_all(q,c))

        print("Data to search is located in", data_position)

        result=run_with_configuration(program=prog, shots=100, cbit_list=c)

        print("Quantum computer fetches result (100 times):", result)
        print("")
        finalize()

       