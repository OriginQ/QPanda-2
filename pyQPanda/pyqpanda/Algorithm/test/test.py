from pyqpanda.Hamiltonian import (PauliOperator, 
                                  chem_client)
                                  
import pyqpanda.Algorithm.test as pa_test
import pyqpanda.Hamiltonian.test as ph_test

import pyqpanda.Algorithm.demo as pademo

from pyqpanda import *
from pyqpanda.utils import *

from pyqpanda.Hamiltonian.QubitOperator import PauliOperator

from pyqpanda.Algorithm.fragments import *
from pyqpanda.Algorithm.hamiltonian_simulation import *

#h1=PauliOperator({'Z3':8,"Z2":4,'Z1':2,"Z0":1,})
#h2=PauliOperator({'Z7':8,"Z6":4,'Z5':2,"Z4":1,})
#h1=PauliOperator({"Z2":4,'Z1':2,"Z0":1,})
#h2=PauliOperator({"Z5":4,'Z4':2,"Z3":1,})
# h1=PauliOperator({"Z2":-2,'Z1':-1,"Z0":-0.5,'':3.5})
# h2=PauliOperator({'Z4':-1,"Z3":-0.5,'':1.5})
# #parse_pauli
# h3=PauliOperator({'':21})
# h=h2*h2
# print(h)

init()
prog=QProg()
q=qAlloc_many(20)
c=cAlloc_many(4)

#prog.insert(U4(q[0],1,2,3,4)).insert(U4(q[0],1.1,1.2,1.3,4.4))
prog.insert(single_gate_apply_to_all(H,q))
prog.insert(single_gate_apply_to_all(H,q))
prog.insert(single_gate_apply_to_all(H,q))
prog.insert(single_gate_apply_to_all(H,q))
prog.insert(single_gate_apply_to_all(H,q))
prog.insert(single_gate_apply_to_all(H,q))
prog.insert(single_gate_apply_to_all(H,q))
prog.insert(single_gate_apply_to_all(H,q))
result=prob_run(program=prog,noise=False,select_max=-1,qubit_list=q,dataType='list')
#result=run_with_configuration(program=prog, shots=100000, cbit_list=[c[0]])
print(result)
finalize()
#print(to_qrunes(prog))








