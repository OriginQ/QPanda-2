'''
Quantum Algorithm Easy Demo: Deutsch-Jozsa Algorithm\n
This algorithm takes a function as the input. Task is to judge 
it whether "Constant" (same outputs with all inputs) or "Balanced"
(half outputs 0 and half outputs 1)\n
Copyright (C) Origin Quantum 2017-2018\n
Licensed Under Apache Licence 2.0
'''
from pyqpanda import *
from pyqpanda.utils import *
from pyqpanda.Algorithm.fragments import two_qubit_oracle

def Two_Qubit_DJ_Circuit(function='f(x)=0'):
    """
    Generates a 2-qubit DJ algorithm circuit using the pyQPanda library.

    This function allocates two qubits and one classical bit, constructs a
    quantum program with a series of quantum gates, and measures the outcome
    of the first qubit. The quantum program includes a two-qubit oracle
    based on the provided function, which is used to implement the DJ algorithm.

        Args:
            function (str): A string representing the function for the two-qubit oracle. Default is 'f(x)=0'.

        Returns:
        tuple: A tuple containing the quantum program (`QProg`) and the classical
               bit (`cbit`) associated with the measurement of the first qubit.
    """ 
    # Allocate 2 qubits
    qubits=qAlloc_many(2)
    # Allocate 1 cbit
    cbit=cAlloc()
    
    # Create Empty QProg
    dj_prog=QProg()
    
    # Insert
    dj_prog.insert(X(qubits[1]))\
           .insert(single_gate_apply_to_all(H,qubits)) \
           .insert(two_qubit_oracle(function,qubits)) \
           .insert(H(qubits[0]))\
           .insert(Measure(qubits[0],cbit))
    return dj_prog,cbit

def Two_Qubit_DJ_Demo():
    """
    Demonstrates the Deutsch-Jozsa algorithm using a two-qubit quantum circuit within the pyQPanda framework.

    This function initializes the quantum circuit, runs the Deutsch-Jozsa algorithm on four predefined functions,
    and prints the results. The functions considered are constant, balanced, and linear transformations.

    The demonstration is executed on a quantum circuit simulator or quantum cloud service, leveraging the
    capabilities of pyQPanda for quantum programming.

        Args:
            None

        Returns:
            None

    The algorithm performs the following steps for each function:
        1. Initializes the quantum circuit.
        2. Defines the quantum circuit corresponding to the function.
        3. Executes the circuit and measures the outcomes.
        4. Determines if the function is constant or balanced based on the measurement results.
        5. Finalizes the quantum circuit.

    The output includes the function being tested and whether it is classified as constant or balanced.
    """
    # Prepare functions
    function=['f(x)=0','f(x)=1','f(x)=x','f(x)=x+1']
    
    for i in range(4):
        init()
        print("Running on function: ", function[i])    

        # fetch programs and cbit (ready for readout)    
        prog,cbit=Two_Qubit_DJ_Circuit(function[i])

        # run and fetch results
        result=directly_run(prog)

        print("Measure Results:",result)

        # judge from the measure result 
        # 1 for constant and 0 for balanced
        if (getCBitValue(cbit)==One):
            print('So that',function[i], 'is a Constant function')
        else:
            print('So that',function[i], 'is a Balanced function')
        finalize()
        print("")
        