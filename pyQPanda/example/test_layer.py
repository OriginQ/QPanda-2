import pyqpanda.pyQPanda as pq
from pyqpanda.Visualization.circuit_draw import *
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
import unittest

class InitQMachine:
    def __init__(self, machineType = pq.QMachineType.CPU):
        self.m_machine = pq.init_quantum_machine(machineType)

    def __del__(self):
        pq.destroy_quantum_machine(self.m_machine)



# def test_layer1():
#     init_machine = InitQMachine()
#     machine = init_machine.m_machine
#     q = machine.qAlloc_many(6)
#     c = machine.cAlloc_many(6)
#     prog = pq.QProg()
#     cir = pq.QCircuit()
#     cir.insert(pq.T(q[0])).insert(pq.iSWAP(q[1], q[5])).insert(pq.S(q[1])).insert(pq.CNOT(q[1], q[0]))
#     cir.insert(pq.CU(np.pi/3, 3, 4, 5, q[3], q[2]))
#     cir.insert(pq.CZ(q[0], q[2])).insert(pq.CU(np.pi/3, 3, 4, 5, q[5], q[2])).insert(pq.SWAP(q[1], q[0]))
#     cir.insert(pq.iSWAP(q[1], q[5])).insert(pq.iSWAP(q[1], q[5], 0.12345)).insert(pq.SqiSWAP(q[1], q[5]))
#     cir.set_control([q[4],q[3]])
#     cir.set_dagger(True)
#     prog.insert(cir)
#
#     #分层接口 circuit_layer
#     layer_info = pq.circuit_layer(prog)
#
#     #通过打印线路查看分层信息
#     qcd = MatplotlibDrawer(qregs = layer_info[1], cregs = layer_info[2], ops = layer_info[0], scale=0.7)
#     qcd.draw(filename='D:/test_cir_draw2.jpg')

class test_layer(unittest.TestCase):
    def test_layer1(self):
        init_machine = InitQMachine()
        machine = init_machine.m_machine
        q = machine.qAlloc_many(6)
        c = machine.cAlloc_many(6)
        prog = pq.QProg()
        cir = pq.QCircuit()
        cir.insert(pq.T(q[0])).insert(pq.iSWAP(q[1], q[5])).insert(pq.S(q[1])).insert(pq.CNOT(q[1], q[0]))
        cir.insert(pq.CU(np.pi / 3, 3, 4, 5, q[3], q[2]))
        cir.insert(pq.CZ(q[0], q[2])).insert(pq.CU(np.pi / 3, 3, 4, 5, q[5], q[2])).insert(pq.SWAP(q[1], q[0]))
        cir.insert(pq.iSWAP(q[1], q[5])).insert(pq.iSWAP(q[1], q[5], 0.12345)).insert(pq.SqiSWAP(q[1], q[5]))
        cir.set_control([q[4], q[3]])
        cir.set_dagger(True)
        prog.insert(cir)

        # 分层接口 circuit_layer
        layer_info = pq.circuit_layer(prog)

        # 通过打印线路查看分层信息
        qcd = MatplotlibDrawer(qregs=layer_info[1], cregs=layer_info[2], ops=layer_info[0], scale=0.7)
        # qcd.draw(filename='D:/test_cir_draw2.jpg')
        # print("cdhaoecfoiwejf")

if __name__ == '__main__':
    unittest.main(verbosity=2)