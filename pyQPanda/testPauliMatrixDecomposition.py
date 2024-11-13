from pyqpanda import *
import numpy as np
from numpy.linalg import eig

Z=np.array([[1,0],[0,-1]])
X=np.array([[0,1],[1,0]])
Y=np.array([[0,-1j],[1j,0]])
I=np.eye(2)

pauli_mat={'':I,'X':X,'Y':Y,'Z':Z}

def krons(matlist):
    """
    Computes the Kronecker product of a sequence of matrices.

    This function sequentially computes the Kronecker product of a list of
    matrices, starting with an initial matrix of ones. The result is the
    Kronecker product of all matrices in the list.

    Parameters:
    matlist (list): A list of 2D numpy arrays (matrices).

    Returns:
    numpy.ndarray: The Kronecker product of the input matrices.

    Note: This function is designed to be used within the pyQPanda package,
    which facilitates programming quantum computers using quantum circuits and
    gates, and can be executed on a quantum circuit simulator or quantum cloud
    service.
    """
    mat=np.array([1])
    for term in matlist:
        mat=np.kron(mat,term)
        
    return mat

def get_dense_pauli(qindex, pauli_char, n_qubit):
    """
    Constructs a dense Pauli operator matrix for a given qubit index and Pauli character.
    
    This function generates a dense matrix representation of a Pauli operator acting on
    a quantum system with `n_qubit` qubits. The Pauli operator is specified by `pauli_char`,
    which can be one of 'X', 'Y', 'Z', or 'I' representing the Pauli-X, Pauli-Y, Pauli-Z, and
    identity operators, respectively. The operation is performed on the qubit at index `qindex`.
    
    Parameters:
    - qindex (int): The index of the qubit on which the Pauli operator is applied.
    - pauli_char (str): The character indicating the Pauli operator to be applied ('X', 'Y', 'Z', 'I').
    - n_qubit (int): The total number of qubits in the quantum system.
    
    Returns:
    - np.ndarray: A dense complex128 numpy array representing the Pauli operator matrix.
    
    The function is intended for use within the pyQPanda package, which is designed for
    programming quantum computers using quantum circuits and gates. It can be executed on a
    quantum circuit simulator, quantum virtual machine, or quantum cloud service.
    """
    result=np.array([1],dtype='complex128')
    for i in range(n_qubit):
        if i == qindex:
            result=np.kron(result, pauli_mat[pauli_char])
        else:
            result=np.kron(result,I)
    return result

def get_matrix(pauliOperator):
    """
    Computes the matrix representation of a given Pauli operator using the Hamiltonian conversion method.

    Parameters:
        pauliOperator (object): An instance of a Pauli operator that supports the `toHamiltonian` and `getMaxIndex`
                               methods, typically from the pyQPanda package.

    Returns:
        np.ndarray: A complex numpy array representing the matrix of the Pauli operator.

    The function initializes an identity matrix for the system of size `n_qubit`, where `n_qubit` is
    determined by adding 1 to the maximum index of the Pauli operator. It then iterates over the terms
    in the Hamiltonian representation of the Pauli operator, applying the corresponding Pauli operators
    to the identity matrix and accumulating the results.

    Note: This function assumes the existence of a helper function `get_dense_pauli`, which is responsible
    for generating the dense matrix representation of a Pauli operator acting on a single qubit.
    """
    op=pauliOperator.toHamiltonian(1)
    n_qubit=pauliOperator.getMaxIndex()+1
    
    #preparation for numpy array
    I_=np.eye(1<<n_qubit,dtype='complex128')
    result=np.zeros((1<<n_qubit,1<<n_qubit),dtype='complex128')
    
    for term in op:
        one_term_result=I_
        pauli_map=term[0]
        for k in pauli_map.keys():
            one_term_result=one_term_result.dot(get_dense_pauli(k, pauli_map[k],n_qubit))
        result+=one_term_result*term[1]
    return result

if __name__=="__main__":
    a = PauliOperator({"" : -0.042079,
                        "X0 X1 Y2 Y3" : -0.044750,
                        "X0 Y1 Y2 X3" : 0.044750,
                        "Y0 X1 X2 Y3" : 0.044750,
                        "Y0 Y1 X2 X3" : -0.044750,
                        "Z0" : 0.177713,
                        "Z0 Z1" : 0.170597,
                        "Z0 Z2" : 0.122933,
                        "Z0 Z3" : 0.167683,
                        "Z1" : 0.177713,
                        "Z1 Z2" : 0.167683,
                        "Z1 Z3" : 0.122933,
                        "Z2" : -0.242743,
                        "Z2 Z3" : 0.176276,
                        "Z3" : -0.242743})

    matrix = get_matrix(a)
    eigenvalue,featurevector=np.linalg.eig(matrix)
    print("eigenvalue=", eigenvalue)
    print("min eigenvalue=", np.min(eigenvalue))
    # print("featurevector=",featurevector)