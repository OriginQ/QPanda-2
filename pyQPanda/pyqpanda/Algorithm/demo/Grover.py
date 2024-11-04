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
    """
    Constructs a three-qubit Grover circuit for searching an unsorted database.

        Args:
            qubits (list of Qubit): The two working qubits that will be used for the search process.
            ancilla (Qubit): An auxiliary qubit used to assist in the search algorithm.
            data_pos (int): The position of the marked item in the database.

        Returns:
            QCircuit: The constructed quantum circuit representing the Grover search algorithm.

        Description:
            This function generates a quantum circuit for the Grover's algorithm using three qubits:
            two working qubits and one ancilla qubit. The algorithm is designed to find the marked item in
            an unsorted database with a quadratic speedup compared to classical search methods.

            The circuit performs a series of quantum gates including:
            - NOT operation on the ancilla qubit.
            - Hadamard (H) gates on all qubits to create superposition.
            - A controlled phase shift (CNOT) gate that targets the ancilla qubit and the working qubits.
            - The diffusion operator (T) on the working qubits to amplify the amplitude of the marked state.
    """
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
    """
    Demo for the Three-Qubit Grover's Algorithm Implementation in pyQPanda.

    This function demonstrates the application of Grover's algorithm using a three-qubit system within the pyQPanda package. It initializes the quantum system, allocates qubits and ancillary qubits, constructs the Grover circuit, and performs measurements to find the marked state. The algorithm is executed for each of the four possible data positions, and the results are printed for analysis.

        Usage:
            The function runs Grover's algorithm on a quantum circuit simulator or quantum cloud service provided by pyQPanda.
            It prints the position of the data to be searched and the results of the algorithm after 100 iterations.

        Args:
            None

        Returns:
            None

        Example:
            Three_Qubit_Grover_Demo()

        Details:
            This function is designed to serve as an educational demonstration of Grover's algorithm in the context of pyQPanda.
            The function iterates over all possible data positions, applying the Grover circuit and measurement process for each.
            The `Three_Qubit_Grover_Circuit` function is used to construct the Grover circuit for the given data position.
            The `run_with_configuration` function executes the quantum program and returns the measurement results.
    """

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

       