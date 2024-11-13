from pyqpanda import *
from pyqpanda.utils import *

def parity_check_circuit(qubit_list):
    """
    Constructs a quantum circuit that applies a CNOT gate between all preceding qubits and the last qubit in the list.

        Args:
            qubit_list (list): A list of qubits where the last qubit will not be CNOT-ed with itself.

        Returns:
            QCircuit: A quantum circuit object representing the parity check circuit.
    """
    prog=QCircuit()
    for i in range(len(qubit_list)-1):
        prog.insert(CNOT(qubit_list[i],qubit_list[-1]))
    return prog

def two_qubit_oracle(function,qubits):
    """
    Constructs a two-qubit oracle circuit based on specified function.

    The oracle circuit is designed to evaluate the given function on the input qubits.

    Supported functions:
        - f(x) = x: Performs a CNOT gate between the specified qubits.
        - f(x) = x + 1 or f(x) = x XOR 1: Applies a sequence of single-qubit X gates and a CNOT gate.
        - f(x) = 0: Returns an empty quantum circuit.
        - f(x) = 1: Applies an X gate to the second qubit.

        Args:
            function (str): The function to be implemented by the oracle circuit.
            qubits (list): A list of two qubits on which the oracle operates.

        Returns:
            QuantumCircuit: The constructed quantum circuit representing the specified function.
    """
    if function=='f(x)=x':
        return CNOT(qubits[0],qubits[1])
    if function=='f(x)=x+1' or function=='f(x)=x XOR 1':
        return QCircuit().insert(X(qubits[0]))\
                         .insert(CNOT(qubits[0],qubits[1]))\
                         .insert(X(qubits[0]))
    if function=='f(x)=0':
        return QCircuit()
    if function=='f(x)=1':
        return X(qubits[1])     

def two_qubit_database(data_pos,addr,data):
    """
    Constructs a quantum circuit by applying a Toffoli gate with specified address and data conditions.
    
    The function creates a quantum circuit that applies a Toffoli gate to the qubits at the given address,
    modifying the state of the target qubit based on the values of the address and data qubits. The Toffoli
    gate's behavior is influenced by the position of the 'data_pos' qubit, which can be one of the four
    possible positions (0-3) within the quantum circuit.

        Args:
            data_pos (int): Specifies the position of the 'data' qubit within the circuit. It can be 0, 1, 2, or 3.
            addr (tuple): A tuple of two integers representing the address of the qubits where the Toffoli gate will be applied.
            data (int): The value of the 'data' qubit (either 0 or 1), which determines the control behavior of the Toffoli gate.

        Returns:
            QuantumCircuit: A quantum circuit object representing the constructed quantum circuit.

        Notes:
            When `data_pos` is 3, the Toffoli gate is applied directly to the specified qubits.
            For `data_pos` values of 1, 2, or 0, the function applies an X gate to the corresponding qubit(s) before and/or after the Toffoli gate,
            depending on the position of the 'data' qubit.
    """
    toffoli_gate=Toffoli(addr[0],addr[1],data)
    if data_pos==3:
        return toffoli_gate
    if data_pos==1:
        return QCircuit().insert(X(addr[0]))\
                         .insert(toffoli_gate)\
                         .insert(X(addr[0]))
    if data_pos==2:
        return QCircuit().insert(X(addr[1]))\
                         .insert(toffoli_gate)\
                         .insert(X(addr[1]))
    if data_pos==0:
        return QCircuit().insert(X(addr[0]))\
                         .insert(X(addr[1]))\
                         .insert(toffoli_gate)\
                         .insert(X(addr[0]))\
                         .insert(X(addr[1]))       

def diffusion_operator(qubits):
    """
    Applies the diffusion operator to a set of qubits in a quantum circuit.

    The diffusion operator is defined as the sum of the Hadamard gate applied to
    each qubit, followed by the controlled Z gate between the first and second qubits,
    and the Hadamard gate applied again to each qubit. The operation is equivalent to:

        2|s><s| - I

    where |s> is the standard basis state and I is the identity matrix.

        Args:
            qubits (list): A list of qubits on which the diffusion operator will be applied.

        Returns:
            QuantumCircuit: A QuantumCircuit object representing the diffusion operator applied
                        to the input qubits.
    """     
    return single_gate_apply_to_all(H, qubits) \
            .insert(single_gate_apply_to_all(X,qubits)) \
            .insert(Z(qubits[0]).control(qubits[1:(len(qubits))])) \
            .insert(single_gate_apply_to_all(X,qubits)) \
            .insert(single_gate_apply_to_all(H,qubits))

def set_zero(qubit,cbit):
    """
    Measure the specified qubit and set its state to zero by applying a NOT gate if the measurement result is 1.

        Args:
            qubit (int): The index of the qubit to be measured and potentially flipped.
            cbit (int): The index of the classical bit to store the measurement result.

        Returns:
            QProgram: A quantum program object representing the measurement and conditional NOT gate operation.

    This function is designed to be used within the pyQPanda package, which facilitates quantum computing with quantum circuits and gates.
    It can be executed on a quantum circuit simulator or a quantum cloud service.
    The function is located in the 'Algorithm.fragments.py' module within the 'pyQPanda.build.lib.pyqpanda' directory.
    """        
    prog=QProg()
    prog.insert(Measure(qubit,cbit))\
        .insert(CreateIfProg(
            bind_a_cbit(cbit),              #classical condition
            QCircuit(),                     #true branch
            QCircuit().insert(NOT(qubit))   #false branch
            ))



