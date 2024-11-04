'''
QPanda Utilities\n
Copyright (C) Origin Quantum 2017-2018\n
Licensed Under Apache Licence 2.0
'''
import pyqpanda as pywrap
from json import JSONEncoder

def single_gate_apply_to_all(gate,qubit_list):
    """
    Applies a specified quantum gate to each qubit within the provided list.

    This function is intended for use within the pyQPanda package, which facilitates
    quantum computing with quantum circuits and gates. It operates on a quantum circuit
    simulator or a quantum cloud service.

        Args:
            gate (pywrap.QGate): The quantum gate instance to be applied to each qubit.
            qubit_list (list of pyQPanda.Qubit): A list of qubits to which the gate will be applied.

        Returns:
            pywrap.QCircuit: A quantum circuit object with the applied gates.
    """
    qcirc=pywrap.QCircuit()
    for q in qubit_list:
        qcirc.insert(gate(q))
    return qcirc

def single_gate(gate,qubit,angle=None):
    """
    Constructs a quantum gate operation on a specified qubit.

    This function applies a quantum gate to a qubit, optionally rotating the gate's action around a specified angle for rotation gates.

    Args:
        gate (callable): A quantum gate represented as a callable that takes a qubit and an optional angle.
        qubit (int): The index of the qubit to which the gate will be applied.
        angle (float, optional): The rotation angle for rotation gates. Defaults to None, which means no rotation is applied.

    Returns:
        pyqpanda.QGate: The resulting quantum gate operation after applying to the qubit.

    Raises:
        pyqpanda.run_fail: If an error occurs while constructing the single gate node.
    """
    if angle is None:
        return gate(qubit)
    else:
        return gate(qubit,angle)

def meas_all(qubits, cbits):
    """
    Constructs a quantum program by measuring specified qubits and mapping their outcomes to classical bits.

    Args:
        qubits (list): A list of qubits to be measured.
        cbits (list): A list of classical bits where the measurement outcomes will be stored.

    Returns:
        QProg: A quantum program object representing the measurement operation.

    Raises:
        run_fail: If an error occurs while constructing the measure all node within the quantum program.
    """
    prog=pywrap.QProg()
    for i in range(len(qubits)):
        prog.insert(pywrap.Measure(qubits[i],cbits[i]))

    return prog

def get_fidelity(result, shots, target_result):
    """
    Calculate the fidelity between a given quantum state and a target state.

    Args:
        result (dict): A dictionary representing the current quantum state,
                       with terms as keys and probabilities as values.
        shots (int): The number of measurement shots taken to observe the current state.
        target_result (dict): A dictionary representing the target quantum state,
                              with terms as keys and probabilities as values.

    Returns:
        float: The fidelity between the current and target states, ranging from 0 to 1.

    Raises:
        run_fail: An error is encountered during the computation of fidelity.
    """
    correct_shots=0
    for term in target_result:
        if term in result:
            correct_shots+=result[term]
    return correct_shots / shots

""" Module that monkey-patches the json module when it's imported so
JSONEncoder.default() automatically checks to see if the object being encoded
is an instance of an user-defined type and, if so, returns its name or value
"""
_saved_default = JSONEncoder().default  # Save default method.

def _new_default(self, obj):
    """
    Converts a specified enum object from the pyQPanda package to its corresponding Python integer representation.

    Args:
        obj (enum): The enum object from the pyQPanda package to be converted. It can be either a QMachineType or a NoiseModel.

    Returns:
        int: The integer value of the converted enum object.

    Raises:
        TypeError: If the provided object is not an instance of QMachineType or NoiseModel.
    """
    if isinstance(obj, pywrap.QMachineType):
        return int(obj)  # Could also be obj.value
    elif isinstance(obj, pywrap.NoiseModel):
        return int(obj)  # Could also be obj.value
    else:
        return _saved_default

JSONEncoder.default = _new_default

_saved_default = JSONEncoder().default  # Save default method.

