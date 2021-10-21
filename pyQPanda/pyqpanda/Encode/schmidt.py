import numpy as np
import scipy as sp
from pyqpanda import *
from pyqpanda.pyQPanda import *

# pylint: disable=maybe-no-member


def schmidt_encode(circuit,qubits,unit_vector):

    state = np.copy(unit_vector)

    size = len(state)
    n_qubits = np.log2(size)

    r = n_qubits % 2

    state.shape = (int(2**(n_qubits//2)), int(2**(n_qubits//2 + r)))

    u, d, v = np.linalg.svd(state)
    d = d / np.linalg.norm(d)

    A=qubits[0:(int)(n_qubits//2 + r)]
    B=qubits[(int)(n_qubits//2 + r):(int)(n_qubits)]
    if len(d) > 2:
        circ = schmidt_encode(circuit,B,d)
    else:
        if d[0] < 0:
            circuit << RY(B,2*np.pi- 2 * np.arccos(d[0]))
        else:
            circuit << RY(B, 2 * np.arccos(d[0]))

    for k in range(int(n_qubits//2)):
        circuit<<CNOT(B[k], A[k])
       
    # apply gate U to the first register
    unitary(circuit,B,u)

    # apply gate V to the second register
    unitary(circuit,A,v.T)

    return circuit
def _unitary(gate_list,qubits,circuit):
    if len(gate_list[0]) == 2:
        gate=np.array(gate_list,complex)
        circuit<<X(qubits[1])<<U4(list(gate[0][0])+list(gate[0][1]),qubits[0]).control(qubits[1])<<X(qubits[1])
        circuit<<U4(list(gate[1][0])+list(gate[1][1]),qubits[0]).control(qubits[1])
        return circuit
    return _qsd(circuit,qubits,*gate_list)

def _index(k, circuit, control_qubits, numberof_controls):
    binary_index = '{:0{}b}'.format(k, numberof_controls)
    for j, qbit in enumerate(reversed(control_qubits)):
        if binary_index[j] == '1':
            circuit << X(qbit)

def _qsd(circuit,qubits,gate1, gate2):
    n_qubits = int(np.log2(len(gate1))) + 1
    q=qubits[0:n_qubits]
    list_d, gate_v, gate_w = _compute_gates(gate1, gate2)
    unitary(circuit,q,gate_w)
    control_bits=q[0:-1]
    numberof_controls=len(control_bits)
    for k, angle in enumerate(reversed(list(-2*np.angle(list_d)))):
                _index(k, circuit, control_bits, numberof_controls)
                circuit<<RZ(qubits[n_qubits-1],angle).control(control_bits) 
                _index(k, circuit, control_bits, numberof_controls)
    unitary(circuit,q,gate_v)
    return circuit


def _compute_gates(gate1, gate2):

    d_square, gate_v = np.linalg.eig(gate1 @ gate2.conj().T)
    list_d = np.sqrt(d_square)
    gate_d = np.diag(list_d)
    gate_w = gate_d @ gate_v.conj().T @ gate2

    return list_d, gate_v, gate_w


def unitary(circuit,qubits,gate):
    """
    Implements a generic quantum computation from a
    unitary matrix gate using the cosine sine decomposition.
    """
    size = len(gate)
    if size > 2:
        n_qubits = int(np.log2(size))
        q=qubits[0:n_qubits]
        right_gates, theta, left_gates = \
            sp.linalg.cossin(gate, size/2, size/2, separate=True)

        _unitary(list(left_gates), q,circuit)
        control_bits=list(range(n_qubits-1))
        q1=[]
        for i in control_bits:
            q1.append(q[i])
        numberof_controls=len(control_bits)
        for k, angle in enumerate(reversed(list(2*theta))):
                _index(k, circuit, q1, numberof_controls)
                circuit<<RY(qubits[n_qubits-1],angle).control(q1) 
                _index(k, circuit, q1, numberof_controls)
        _unitary(list(right_gates),q,circuit)
        return circuit
    gate=np.array(gate,complex)
    circuit<<U4(list(gate[0])+list(gate[1]),qubits[0])
    return circuit