'''
QPanda Utilities\n
Copyright (C) Origin Quantum 2017-2018\n
Licensed Under Apache Licence 2.0
'''
import pyqpanda as pywrap
from json import JSONEncoder

def single_gate_apply_to_all(gate,qubit_list):
    '''
    Apply single gates to all qubits in qubit_list
    QGate(callback), list<Qubit> -> QCircuit

    Args:
        gate : the quantum gate need to apply to all qubits
        qubit_list : qubit list

    Returns: 
        quantum circuit
    '''
    qcirc=pywrap.QCircuit()
    for q in qubit_list:
        qcirc.insert(gate(q))
    return qcirc

def single_gate(gate,qubit,angle=None):
    '''
    Apply a single gate to a qubit\n
    Gate(callback), Qubit, angle(optional) -> QGate

    Args:
        gate : the quantum gate need to apply qubit
        qubit : single qubit
        angle : theta for rotation gate

    Returns: 
        quantum circuit

    Raises:\n"
        run_fail: An error occurred in construct single gate node
    '''
    if angle is None:
        return gate(qubit)
    else:
        return gate(qubit,angle)

def meas_all(qubits, cbits):
    '''
    Measure qubits mapping to cbits\n
    list<Qubit>, list<CBit> -> QProg
    Args:
        qubit_list : measure qubits list 
        cbits_list : measure cbits list 

    Returns: 
        quantum prog

    Raises:
        run_fail: An error occurred in construct measure all node
    '''
    prog=pywrap.QProg()
    for i in range(len(qubits)):
        prog.insert(pywrap.Measure(qubits[i],cbits[i]))

    return prog

def get_fidelity(result, shots, target_result):
    '''
    get quantum state fidelity
    Args:
        result : current quantum state 
        shots : measure shots
        target_result : compared state

    Returns: 
        fidelity bewteen [0,1]

    Raises:\n"
        run_fail: An error occurred in get_fidelity
    '''
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
    '''
    convert enum to python int
    Args:
        obj : qpanda enum

    Returns:
        int
    '''
    if isinstance(obj, pywrap.QMachineType):
        return int(obj)  # Could also be obj.value
    elif isinstance(obj, pywrap.NoiseModel):
        return int(obj)  # Could also be obj.value
    else:
        return _saved_default

JSONEncoder.default = _new_default

_saved_default = JSONEncoder().default  # Save default method.

