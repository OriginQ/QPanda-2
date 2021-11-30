import numpy as np
import scipy as sp
from pyqpanda import *
from pyqpanda.pyQPanda import *
from scipy import sparse

def gray_code(rank):

    def gray_code_recurse(g, rank):
        k = len(g)
        if rank <= 0:
            return

        for i in range(k - 1, -1, -1):
            char = "1" + g[i]
            g.append(char)
        for i in range(k - 1, -1, -1):
            g[i] = "0" + g[i]

        gray_code_recurse(g, rank - 1)

    g = ["0", "1"]
    gray_code_recurse(g, rank - 1)

    return g


def _matrix_M_entry(row, col):

    b_and_g = row & ((col >> 1) ^ col)
    sum_of_ones = 0
    while b_and_g > 0:
        if b_and_g & 0b1:
            sum_of_ones += 1

        b_and_g = b_and_g >> 1

    return (-1) ** sum_of_ones


def _compute_theta(alpha):

    ln = len(alpha)
    k = np.log2(ln)

    M_trans = np.zeros(shape=(ln, ln))
    for i in range(len(M_trans)):
        for j in range(len(M_trans[0])):
            M_trans[i, j] = _matrix_M_entry(j, i)

    theta = np.dot(M_trans, alpha.T).T

    return theta / 2 ** k


def _uniform_rotation_dagger(circuit, gate, alpha, control_qubits, target_qubit):

    theta = _compute_theta(alpha)

    gray_code_rank = len(control_qubits)

    if gray_code_rank == 0:
        if np.all(theta[..., 0] != 0.0):
            if gate=="RY":
                circuit<<RY(target_qubit,theta[..., 0])
            else:
                circuit<<RZ(target_qubit,theta[..., 0])
        return

    code = gray_code(gray_code_rank)
    num_selections = len(code)

    control_indices = [
        int(np.log2(int(code[i], 2) ^ int(code[(i + 1) % num_selections], 2)))
        for i in range(num_selections)
    ]

    for i, control_index in enumerate(control_indices):
        if np.all(theta[..., i] != 0.0):
            if gate=="RY":
                circuit<<RY(target_qubit,theta[..., i])
            else:
                circuit<<RZ(target_qubit,theta[..., i])
        circuit<<CNOT(control_qubits[control_index],target_qubit)    



def _get_alpha_z(omega, n, k):

    indices1 = [
        [(2 * j - 1) * 2 ** (k - 1) + l - 1 for l in range(1, 2 ** (k - 1) + 1)]
        for j in range(1, 2 ** (n - k) + 1)
    ]
    indices2 = [
        [(2 * j - 2) * 2 ** (k - 1) + l - 1 for l in range(1, 2 ** (k - 1) + 1)]
        for j in range(1, 2 ** (n - k) + 1)
    ]

    term1 = np.take(omega, indices=indices1)
    term2 = np.take(omega, indices=indices2)
    diff = (term1 - term2) / 2 ** (k - 1)

    return np.sum(diff, axis=1)


def _get_alpha_y(a, n, k):

    indices_numerator = [
        [(2 * (j + 1) - 1) * 2 ** (k - 1) + l for l in range(2 ** (k - 1))]
        for j in range(2 ** (n - k))
    ]
    
    numerator = np.take(a, indices=indices_numerator)
    numerator = np.sum(np.abs(numerator) ** 2, axis=1)

    indices_denominator = [[j * 2 ** k + l for l in range(2 ** k)] for j in range(2 ** (n - k))]
    denominator = np.take(a, indices=indices_denominator)
    denominator = np.sum(np.abs(denominator) ** 2, axis=1)


    with np.errstate(divide="ignore", invalid="ignore"):
        division = numerator / denominator

    return 2 * np.arcsin(np.sqrt(division))

def mottonen(circuit,qubits,state_vector):
    shape = np.shape(state_vector)

    if len(shape) != 1:
        raise ValueError(f"State vector must be a one-dimensional vector; got shape {shape}.")

    n_amplitudes = shape[0]
    if n_amplitudes != 2 ** len(qubits):
        raise ValueError(
            f"State vector must be of length {2 ** len(qubits)} or less; got length {n_amplitudes}."
        )

    norm = np.sum(np.abs(state_vector) ** 2)
    if not np.allclose(norm, 1.0, atol=1e-3):
        raise ValueError("State vector has to be of norm 1.0, got {}".format(norm))
    
    
    a = np.abs(state_vector)
    omega = np.angle(state_vector)

    qubits_reverse = qubits[::-1]


    for k in range(len(qubits_reverse), 0, -1):
        alpha_y_k = _get_alpha_y(a, len(qubits_reverse), k)
        alpha_y_k=np.array(alpha_y_k)
        control = qubits_reverse[k:]
        target = qubits_reverse[k - 1]
        _uniform_rotation_dagger(circuit,"RY", alpha_y_k, control, target)

    if not np.allclose(omega, 0):
        for k in range(len(qubits_reverse), 0, -1):
            alpha_z_k = _get_alpha_z(omega, len(qubits_reverse), k)
            alpha_y_k=np.array(alpha_z_k)
            control = qubits_reverse[k:]
            target = qubits_reverse[k - 1]
            if len(alpha_z_k) > 0:
                _uniform_rotation_dagger(circuit,"RZ", alpha_z_k, control, target)
 