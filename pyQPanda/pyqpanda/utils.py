'''
QPanda Utilities\n
Copyright (C) Origin Quantum 2017-2018\n
Licensed Under Apache Licence 2.0
'''
import pyqpanda as pywrap
from json import JSONEncoder

def single_gate_apply_to_all(gate,qubit_list):
    '''
    `Extended QPanda API`\n
    Apply single gates to all qubits in qubit_list
    QGate(callback), list<Qubit> -> QCircuit
    '''
    qcirc=pywrap.QCircuit()
    for q in qubit_list:
        qcirc.insert(gate(q))
    return qcirc

def single_gate(gate,qubit,angle=None):
    '''
    `Extended QPanda API`\n
    Apply a single gate to a qubit\n
    Gate(callback), Qubit, angle(optional) -> QGate
    '''
    if angle is None:
        return gate(qubit)
    else:
        return gate(qubit,angle)

def meas_all(qubits, cbits):
    '''
    `Extended QPanda API`\n
    Measure qubits mapping to cbits\n
    list<Qubit>, list<CBit> -> QProg
    '''
    prog=pywrap.QProg()
    for i in range(len(qubits)):
        prog.insert(pywrap.Measure(qubits[i],cbits[i]))

    return prog

def Toffoli(control1,control2,target):
    '''
    `Extended QPanda API`\n
    Create foffoli gate\n
    Qubit(control1), Qubit(control2), Qubit(target) -> QGate
    '''

    return pywrap.X(target).control([control1,control2])

def get_fidelity(result, shots, target_result):
    correct_shots=0
    for term in target_result:
        if term in result:
            correct_shots+=result[term]
    return correct_shots / shots

def exportToQCloud(output):
    print(output)


""" Module that monkey-patches the json module when it's imported so
JSONEncoder.default() automatically checks to see if the object being encoded
is an instance of an user-defined type and, if so, returns its name or value
"""
_saved_default = JSONEncoder().default  # Save default method.

def _new_default(self, obj):
    if isinstance(obj, pywrap.QMachineType):
        return int(obj)  # Could also be obj.value
    elif isinstance(obj, pywrap.NoiseModel):
        return int(obj)  # Could also be obj.value
    else:
        return _saved_default

JSONEncoder.default = _new_default

_saved_default = JSONEncoder().default  # Save default method.

