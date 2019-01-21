'''
Quantum Algorithm Easy Demo: Deutsch-Jozsa Algorithm\n
This algorithm takes a function as the input. Task is to judge 
it whether "Constant" (same outputs with all inputs) or "Balanced"
(half outputs 0 and half outputs 1)\n
Copyright (C) Origin Quantum 2017-2018\n
Licensed Under Apache Licence 2.0
'''
from pyqpanda import *
from pyqpanda.utils import *
from pyqpanda.Algorithm.fragments import two_qubit_oracle

def Two_Qubit_DJ_Circuit(function='f(x)=0'):
    '''
    Create a 2-qubit DJ algorithm circuit
    '''    
    # Allocate 2 qubits
    qubits=qAlloc_many(2)
    # Allocate 1 cbit
    cbit=cAlloc()
    
    # Create Empty QProg
    dj_prog=QProg()
    
    # Insert
    dj_prog.insert(X(qubits[1]))\
           .insert(single_gate_apply_to_all(H,qubits)) \
           .insert(two_qubit_oracle(function,qubits)) \
           .insert(H(qubits[0]))\
           .insert(Measure(qubits[0],cbit))
    return dj_prog,cbit

def Two_Qubit_DJ_Demo():
    '''
    `QPanda Easy Demo`\n
    Quantum Algorithm Easy Demo: Deustch-Jozsa algorithm\n
    '''
    # Prepare functions
    function=['f(x)=0','f(x)=1','f(x)=x','f(x)=x+1']
    
    for i in range(4):
        init()
        print("Running on function: ", function[i])    

        # fetch programs and cbit (ready for readout)    
        prog,cbit=Two_Qubit_DJ_Circuit(function[i])

        # run and fetch results
        result=directly_run(prog)

        print("Measure Results:",result)

        # judge from the measure result 
        # 1 for constant and 0 for balanced
        if (getCBitValue(cbit)==One):
            print('So that',function[i], 'is a Constant function')
        else:
            print('So that',function[i], 'is a Balanced function')
        finalize()
        print("")
        