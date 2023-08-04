import pyqpanda.Algorithm.test as pa_test
import pyqpanda.Hamiltonian.test as ph_test

import pyqpanda.Algorithm.demo as pademo

from pyqpanda import *
from pyqpanda.utils import *

from pyqpanda.Hamiltonian.QubitOperator import PauliOperator

def full_test():
    """
    Calls the `full_test` method from the `pa_test` module, which is a part of the pyQPanda package.
    This package facilitates programming quantum computers using quantum circuits and gates,
    supporting execution on quantum circuit simulators, quantum virtual machines, or quantum cloud services.
    The function is situated within the `Algorithm.test.test_script` subpackage of pyQPanda's directory structure.
    """
    pa_test.full_test()

def Grover_demo():
    """
    Grover_demo

    Execute the Grover's search algorithm demonstration for three qubits using the `Three_Qubit_Grover_Demo` method from the `Grover` class within the `pademo` module.

    This function is intended for educational purposes and to showcase the implementation of the Grover's algorithm in the pyQPanda package, which facilitates quantum computing programming. It can be run on a quantum circuit simulator or through a quantum cloud service to visualize the algorithm's operation.

    Note: This function is a part of the `Algorithm.test.test_script` module within the `pyQPanda` package.
    """
    pademo.Grover.Three_Qubit_Grover_Demo()

def Deutsch_Jozsa_demo():
    """
    Perform a simple demonstration of the Deutsch-Jozsa algorithm using the Two-Qubit DJ Demo feature from the pyQPanda package.

    This function is designed to showcase the basic principles of the Deutsch-Jozsa algorithm, which is a quantum algorithm that
    distinguishes between two types of functions with a single qubit input. The demonstration is executed through the
    `Two_Qubit_DJ_Demo` method within the `Deustch_Jozsa` class of the `pademo` module.

    Note: This function is intended for educational purposes and does not require any additional parameters.
    """
    pademo.Deustch_Jozsa.Two_Qubit_DJ_Demo()

def common_test():
    """
    Run a common test suite for quantum programming utilities within the QPanda framework.

    This function initializes the quantum programming environment, allocates quantum and classical
    ubits, constructs a quantum program with an Hadamard gate applied to all qubits followed by
    measurements on a subset of classical bits. It then runs the program with a specified number
    of shots and prints the results. Finally, it cleans up the allocated resources.

    Functions called within this test:
    - init(): Sets up the quantum programming environment.
    - qAlloc_many(n): Allocates 'n' qubits.
    - cAlloc_many(n): Allocates 'n' classical bits.
    - single_gate_apply_to_all(gate, qubit_list): Applies a specified gate to all qubits in the list.
    - meas_all(qubit_list, cbit_list): Measures the qubits and stores the results in classical bits.
    - run_with_configuration(program, shots, cbit_list): Executes the quantum program with the given configuration.
    - finalize(): Frees the allocated resources and cleans up the quantum programming environment.

    This test is designed to validate the basic functionality of quantum programming utilities in
    the QPanda package, particularly the initialization, execution, and cleanup processes.
    """      
    init()
    q=qAlloc_many(10)
    c=cAlloc_many(10)

    prog=QProg()
    prog.insert(single_gate_apply_to_all(gate=H,qubit_list=q)) \
        .insert(meas_all(q,c))

    result=run_with_configuration(program=prog, shots=100,cbit_list=c[0:4])

    print(result)
    finalize()

# let's enjoy!
#common_test()

#Deutsch_Jozsa_demo()
#Grover_demo()
#full_test()
#ph_test.hamiltonian_test.hamiltonian_operator_overload()

h=PauliOperator({'X0':1,"X1":2})
h2=PauliOperator({'Y1':2,"X2 X3":5})
print(3*h*h2*3-h2)