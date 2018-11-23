from pyqpanda import *
from pyqpanda.utils import *
from pyqpanda.Hamiltonian import PauliOperator
import numpy as np
from pyqpanda import *
from numpy.linalg import eig
from copy import deepcopy
#from pyqpanda.Hamiltonian.QubitOperator import parse_pauli



def parity_check_circuit(qubit_list):
    '''CNOT all qubits (except last) with the last qubit'''
    prog=QCircuit()
    for i in range(len(qubit_list)-1):
        prog.insert(CNOT(qubit_list[i],qubit_list[-1]))
    return prog

def two_qubit_oracle(function,qubits):
    '''
    Two qubit oracle.\n
    Support functions:
        f(x)=0
        f(x)=1
        f(x)=x
        f(x)=x XOR 1 / f(x)=x+1
    '''
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
    '''
    Mapping the data in the "addr" qubits to "data" qubit\n
    data=database[addr]\n
    data=1 iff addr==data_pos\n
    data_pos ranges from 0~3
    '''
    toffoli_gate=Toffoli(addr[0],addr[1],data)
    if data_pos==3:
        return toffoli_gate
    if data_pos==1:
        return QCircuit().insert(NOT(addr[0]))\
                         .insert(toffoli_gate)\
                         .insert(NOT(addr[0]))
    if data_pos==2:
        return QCircuit().insert(NOT(addr[1]))\
                         .insert(toffoli_gate)\
                         .insert(NOT(addr[1]))
    if data_pos==0:
        return QCircuit().insert(NOT(addr[0]))\
                         .insert(NOT(addr[1]))\
                         .insert(toffoli_gate)\
                         .insert(NOT(addr[0]))\
                         .insert(NOT(addr[1]))       

def diffusion_operator(qubits):
    '''
    Diffusion operator.\n
    2|s><s|-I
    '''      
    return single_gate_apply_to_all(H, qubits) \
            .insert(single_gate_apply_to_all(X,qubits)) \
            .insert(Z(qubits[0]).control(qubits[1:(len(qubits))])) \
            .insert(single_gate_apply_to_all(X,qubits)) \
            .insert(single_gate_apply_to_all(H,qubits))

def set_zero(qubit,cbit):
    '''
    Measure a qubit and set to zero
    '''        
    prog=QProg()
    prog.insert(Measure(qubit,cbit))\
        .insert(CreateIfProg(
            bind_a_cbit(cbit),              #classical condition
            QCircuit(),                     #true branch
            QCircuit().insert(NOT(qubit))   #false branch
            ))


def parity_check(string, paulis):
    '''
    string has element '0' or '1',paulis is a str like 'X1 Y2 Z3'.
    parity check of partial element of string, number of paulis are 
    invloved position,
    to be repaired
    '''
    check=0
    tuplelist=PauliOperator.parse_pauli(paulis)
    qubit_idx_list=list()
    for term in tuplelist:
        qubit_idx_list.append(term[1])
    
    for i in qubit_idx_list:
        if string[i]=='1':
            check+=1
    
    return check%2      

def krons(matlist):
    mat=np.array([1])
    for term in matlist:
        mat=np.kron(mat,term)
        
    return mat

def get_dense_pauli(pauli_tuple, n_qubit):
    result=np.array([1],dtype='complex128')
    for i in range(n_qubit):
        if i == pauli_tuple[1]:
            result=np.kron(result, pauli_mat[pauli_tuple[0]])
        else:
            result=np.kron(result,I)
    return result

def get_matrix(pauliOperator):
    op=pauliOperator.ops
    n_qubit=pauliOperator.get_qubit_count()
    
    #preparation for numpy array
    I_=np.eye(1<<n_qubit,dtype='complex128')
    result=np.zeros((1<<n_qubit,1<<n_qubit),dtype='complex128')
    
    for term in op:
        one_term_result=deepcopy(I_)
        tuplelist=PauliOperator.parse_pauli(term)
        for pauli_tuple in tuplelist:
            one_term_result=one_term_result.dot(get_dense_pauli(pauli_tuple,n_qubit))
        result+=one_term_result*op[term]
    return result


def to_dense(pauliOperator):
        '''
        get eigenvalues and eigenvectors of PauliOperator
        '''
        op=pauliOperator.ops
        n_qubit=pauliOperator.get_qubit_count()
        #preparation for numpy array
        I_=np.eye(1<<n_qubit,dtype='complex128')
        result=np.zeros((1<<n_qubit,1<<n_qubit),dtype='complex128')
    
        for term in op:
            one_term_result=deepcopy(I_)
            tuplelist=PauliOperator.parse_pauli(term)
            for pauli_tuple in tuplelist:
                one_term_result=one_term_result.dot(get_dense_pauli(pauli_tuple,n_qubit))
            result+=one_term_result*op[term]
        eigval,_=eig(result)
        # return min(eigval).real
        return eig(result)