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
    q = machine.qAlloc_many(4)
    c = machine.cAlloc_many(4)
    prog = pq.QProg()
    cir = pq.QCircuit()
    cir.insert(pq.RX(q[1], np.pi / 5)).insert(pq.RX(q[1], np.pi / 5))
    cir.insert(pq.CU(1, 2, 3, 4, q[1], q[0])).insert(pq.H(q[1])).insert(pq.X(q[2])).insert(pq.RZ(q[1], np.pi / 2)).insert(pq.Y(q[2]))
    cir.insert(pq.CR(q[0], q[3], np.pi / 2)).insert(pq.S(q[2])).insert(pq.S(q[1])).insert(pq.RZ(q[1], np.pi / 2)).insert(pq.RZ(q[1], np.pi / 2))
    cir.insert(pq.RZ(q[1], np.pi / 2)).insert(pq.RZ(q[1], np.pi / 2)).insert(pq.Y(q[0])).insert(pq.SWAP(q[3], q[1]))
    cir.insert(pq.CU(1, 2, 3, 4, q[1], q[0])).insert(pq.H(q[1])).insert(pq.X(q[2])).insert(pq.RX(q[1], np.pi / 2)).insert(pq.RX(q[1], np.pi / 2)).insert(pq.Y(q[2]))
    cir.insert(pq.CR(q[2], q[3], np.pi / 2)).insert(pq.CU(1, 2, 3, 4, q[1], q[0])).insert(pq.H(q[1])).insert(pq.X(q[2])).insert(pq.RZ(q[1], np.pi / 2)).insert(pq.Y(q[2]))

    cir2 = pq.QCircuit()
    cir2.insert(pq.H(q[1])).insert(pq.X(q[2])).insert(pq.X(q[2])).insert(pq.H(q[1])).insert(pq.X(q[3])).insert(pq.X(q[3]))
    
    cir3 = pq.QCircuit()
    cir3.insert(pq.H(q[1])).insert(pq.H(q[2])).insert(pq.CNOT(q[2], q[1])).insert(pq.H(q[1])).insert(pq.H(q[2]))

    theta_1 = np.pi / 3.0
    cir5 = pq.QCircuit()
    cir5.insert(pq.RZ(q[3], np.pi / 2.0)).insert(pq.CZ(q[3], q[0])).insert(pq.RX(q[3], np.pi / 2.0)).insert(pq.RZ(q[3], theta_1))
    cir5.insert(pq.RX(q[3], -np.pi / 2.0)).insert(pq.CZ(q[3], q[0])).insert(pq.RZ(q[3], -np.pi / 2.0))
    
    prog.insert(cir).insert(cir2).insert(pq.Reset(q[1])).insert(cir3).insert(cir5).insert(pq.measure_all(q, c))
    print("befort optimizered QProg:")
    print(prog)

    # 线路替换优化，会自动读配置文件，从配置文件加载线路替换信息
    #new_prog = pq.circuit_optimizer_by_config(prog)

    # 线路替换优化，mode参数用于优化类型，默认是Merge_H_X（合并抵消连续的H门和x门）
    #new_prog = pq.circuit_optimizer_by_config(prog, mode = pq.QCircuitOPtimizerMode.Merge_RX)

    # u3门转换
    new_prog = pq.circuit_optimizer(prog, mode = pq.QCircuitOPtimizerMode.Merge_U3)

    print("The optimizered QProg:")
    print(new_prog)

    # layer_info = pq.circuit_layer(prog)
    # qcd = MatplotlibDrawer(qregs = layer_info[1], cregs = layer_info[2], ops = layer_info[0], scale=0.7)
    # qcd.draw(filename='D:/test_cir_draw2.jpg')

    #draw_qprog(prog, 'pic', filename='D:/test_cir_draw2.jpg', verbose=True)
    
    #plt.imshow('D:/test_cir_draw2.jpg')
    # plt.show()

if __name__=="__main__":
    test_layer1()
    print("Test over.")