from pyqpanda import *
import unittest

class Test_Quantum_OriginIR_Learning(unittest.TestCase):

    def test_qprog_to_originir(self):
        machine = init_quantum_machine(QMachineType.CPU)
        # f = open('testfile.txt', mode='w',encoding='utf-8')
        IR_string=("""QINIT 6 
    CREG 0 
    H q[0]
    H q[1]
    H q[2]
    H q[3]
    H q[4]
    H q[5]
    CNOT q[0],q[1]
    CNOT q[4],q[5]
    CNOT q[0],q[2]
    RZ q[1],(0.78539816)
    RZ q[4],(0.78539816)
    RZ q[5],(0.78539816)
    RZ q[0],(0.78539816)
    RZ q[2],(0.78539816)
    CNOT q[3],q[5]
    )""")

        # f.close()
        prog_trans = convert_originir_str_to_qprog(IR_string,machine)
        # print(to_originir(prog_trans,machine))

        destroy_quantum_machine(machine)


if __name__=="__main__":
    unittest.main(verbosity=2)
