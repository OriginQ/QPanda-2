import pyqpanda.Algorithm.test as pa_test
import pyqpanda.Hamiltonian.test as ph_test

import pyqpanda.Algorithm.demo as pademo

from pyqpanda import *
from pyqpanda.utils import *

from pyqpanda.Hamiltonian.QubitOperator import PauliOperator

def full_test():
    pa_test.full_test()

def Grover_demo():
    '''
    `Easy Demo`
    Grover search demo
    '''
    pademo.Grover.Three_Qubit_Grover_Demo()

def Deutsch_Jozsa_demo():
    '''
    `Easy Demo`
    Deutsch Jozsa demo
    '''
    pademo.Deustch_Jozsa.Two_Qubit_DJ_Demo()

def common_test():
    '''
    Common test for QPanda Utilities
    '''        
    init()
    q=qAlloc_many(10)
    c=cAlloc_many(10)

    prog=QProg()
    prog.insert(single_gate_apply_to_all(gate=H,qubit_list=q)) \
        .insert(meas_all(q,c))

    result=run_with_configuration(program=prog, shots=100,cbit_list=c[0:4])

    print(result)
    finalize()

# let's enjoy!
#common_test()

#Deutsch_Jozsa_demo()
#Grover_demo()
#full_test()
#ph_test.hamiltonian_test.hamiltonian_operator_overload()

h=PauliOperator({'X0':1,"X1":2})
h2=PauliOperator({'Y1':2,"X2 X3":5})
print(3*h*h2*3-h2)