

import sys
import os.path
dir = os.path.abspath(__file__)
model_path = os.path.abspath(os.path.join(dir, "../.."))
sys.path.insert(0, model_path)

#from pyqpanda import *

from pyqpanda.pyQPanda import QMachineType, convert_pyquil_string_to_qprog, destroy_quantum_machine, init_quantum_machine
import unittest

class Test_PyquilToQProg(unittest.TestCase):
    pyquil_str_list = []
    XY_pyquil = """DECLARE ro BIT[3]
XY(3.14) 1 0
XY(6.28) 0 11
"""
    pyquil_str_list.append(XY_pyquil)

    PSWAP_pyquil = """DECLARE ro BIT[3]
DECLARE theta REAL[2]
PSWAP(3.14) 0 1
ISWAP 0 1
"""
    pyquil_str_list.append(PSWAP_pyquil)
    
    def test_pyquil_to_qprog(self,pyquil_str):
        print("### pyquil_str:\n",pyquil_str)
        machine = init_quantum_machine(QMachineType.CPU)
        qprog,qbits,cbits = convert_pyquil_string_to_qprog(pyquil_str,machine)
        print("### qprog:",end='\n')
        print(qprog,end='\n')
        print("### qbits:",end='\n')
        print(qbits)
        print("### cbits:",end='\n')
        print(cbits)
        
        destroy_quantum_machine(machine)


if __name__=="__main__":
    #unittest.main(verbosity=2)
    TQI = Test_PyquilToQProg()
    for pyquil_str in TQI.pyquil_str_list:
        TQI.test_pyquil_to_qprog(pyquil_str)
    

