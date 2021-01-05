import pyqpanda.pyQPanda as pq
from pyqpanda.Visualization.circuit_draw import *
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt

class InitQMachine:
    def __init__(self, machineType = pq.QMachineType.CPU):
        self.m_machine = pq.init_quantum_machine(machineType)

    def __del__(self):
        pq.destroy_quantum_machine(self.m_machine)

def test_layer1():
    init_machine = InitQMachine()
    machine = init_machine.m_machine
    q = machine.qAlloc_many(6)
    c = machine.cAlloc_many(6)
    prog = pq.QProg()
    cir = pq.QCircuit()
    cir.insert(pq.T(q[0])).insert(pq.iSWAP(q[1], q[5], 3.2233233)).insert(pq.S(q[1])).insert(pq.CNOT(q[1], q[0]))
    cir.insert(pq.CU(np.pi/3, 3, 4, 5, q[1], q[2]))
    cir.insert(pq.CR(q[2], q[1], 0.00000334))
    cir.insert(pq.RX(q[2], np.pi/3)).insert(pq.RZ(q[2], np.pi/3)).insert(pq.RY(q[2], np.pi/3))
    cir.insert(pq.CZ(q[0], q[2])).insert(pq.CU(np.pi/3, 3, 4, 5, q[5], q[2])).insert(pq.SWAP(q[1], q[0]))
    cir.set_control([q[4],q[3]])
    cir.set_dagger(True)
    prog.insert(cir)

    # 打印多控门分解之前的量子线路
    draw_qprog(prog, 'pic', filename='D:/before_decompose_multiple_control_qgate.jpg', verbose=True)

    #多控门分解接口
    new_prog = pq.decompose_multiple_control_qgate(prog, machine)

    #打印多控门分解之后的量子线路
    draw_qprog(new_prog, 'pic', filename='D:/after_decompose_multiple_control_qgate.jpg', verbose=True)

if __name__=="__main__":
    test_layer1()
    print("Test over.")