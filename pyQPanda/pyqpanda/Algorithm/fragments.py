from pyqpanda import *
from pyqpanda.utils import *

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



