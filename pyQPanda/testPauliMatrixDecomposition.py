from pyqpanda import *
import numpy as np
from numpy.linalg import eig

Z=np.array([[1,0],[0,-1]])
X=np.array([[0,1],[1,0]])
Y=np.array([[0,-1j],[1j,0]])
I=np.eye(2)

pauli_mat={'':I,'X':X,'Y':Y,'Z':Z}

def krons(matlist):
    mat=np.array([1])
    for term in matlist:
        mat=np.kron(mat,term)
        
    return mat

def get_dense_pauli(qindex, pauli_char, n_qubit):
    result=np.array([1],dtype='complex128')
    for i in range(n_qubit):
        if i == qindex:
            result=np.kron(result, pauli_mat[pauli_char])
        else:
            result=np.kron(result,I)
    return result

def get_matrix(pauliOperator):
    op=pauliOperator.toHamiltonian(1)
    n_qubit=pauliOperator.getMaxIndex()
    
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