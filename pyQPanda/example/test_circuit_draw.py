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

def plot(list):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    x = [key for key in list.keys()]
    y = [val for val in list.values()]
    y1 = [val/sum(y) for val in list.values()]
    plt.bar(x, y1, align = "center", color = "b", alpha = 0.6)
    plt.ylabel("Probabilities")
    plt.grid(True, axis = "y", ls = ":", color = "r", alpha = 0.3)
    plt.show()

def test_layer1():
    init_machine = InitQMachine()
    machine = init_machine.m_machine
    q = machine.qAlloc_many(6)
    c = machine.cAlloc_many(6)
    prog = pq.QProg()
    cir = pq.QCircuit()
    cir.insert(pq.T(q[0])).insert(pq.SWAP(q[1], q[5])).insert(pq.S(q[1])).insert(pq.CNOT(q[1], q[0]))
    cir.insert(pq.CZ(q[0], q[2])).insert(pq.CU(np.pi/3, 3, 4, 5, q[5], q[2]))
    cir.set_control([q[4],q[3]])
    cir.set_dagger(True)
    
    # prog.insert(pq.T(q[0])).insert(pq.SWAP(q[1], q[2])).insert(pq.X1(q[3]))
    # prog.insert(pq.X(q[0])).insert(pq.CZ(q[1], q[2])).insert(pq.Y(q[3])).insert(pq.Z(q[3]))
    # prog.insert(pq.S(q[3])).insert(pq.CNOT(q[1], q[4]))
    # prog.insert(pq.CR(q[0], q[2], 2.3)).insert(pq.CR(q[0], q[1], np.pi/2.3)).insert(pq.Y1(q[4])).insert(pq.Z1(q[5]))
    # prog.insert(pq.CR(q[5], q[2], np.pi/3))
    # prog.insert(pq.CR(q[2], q[3], np.pi/5))
    # prog.insert(pq.iSWAP(q[5], q[2], np.pi/2.3)).insert(pq.iSWAP(q[4], q[1]))
    # prog.insert(pq.RX(q[2], np.pi/6.2)).insert(pq.RY(q[2], 6.6)).insert(pq.RZ(q[2], np.pi/3))
    # prog.insert(pq.U1(q[0], np.pi/4)).insert(pq.U2(q[1], 1, np.pi/2)).insert(pq.U3(q[2], 1.2, 2, np.pi/3)).insert(pq.U4(1.5, 2.2, np.pi/3, np.pi/4, q[3]))
    # prog.insert(pq.iSWAP(q[5], q[2], np.pi/2.3)).insert(pq.U4(1.5, 2.2, np.pi/3, np.pi/4, q[3]))
    # prog.insert(pq.RX(q[2], np.pi/6.2)).insert(pq.RY(q[2], 6.6)).insert(pq.RZ(q[2], np.pi/3))
    # prog.insert(pq.U1(q[0], np.pi/4)).insert(pq.U2(q[1], 1, np.pi/2)).insert(pq.U3(q[2], 1.2, 2, np.pi/3))
    # prog.insert(pq.CU(np.pi/3, 3, 4, 5, q[5], q[2]))
    # prog.insert(pq.T(q[0])).insert(pq.SWAP(q[1], q[2])).insert(pq.X1(q[3]))
    # prog.insert(pq.X(q[0])).insert(pq.CZ(q[1], q[2])).insert(pq.Y(q[3])).insert(pq.Z(q[3]))
    # prog.insert(pq.S(q[3]))
    # prog.insert(cir)
    # prog.insert(pq.U4(1.5, 2.2, np.pi/3, np.pi/4, q[1]))
    # prog.insert(pq.CU(np.pi/3, 3, 4, 5, q[3], q[2]))
    # prog.insert(pq.CR(q[0], q[2], 2.3)).insert(pq.CR(q[0], q[1], np.pi/2.3)).insert(pq.Y1(q[4])).insert(pq.Z1(q[5]))
    # prog.insert(pq.CR(q[5], q[2], np.pi/3))
    # prog.insert(pq.CR(q[2], q[3], np.pi/5))
    # prog.insert(pq.iSWAP(q[5], q[2], np.pi/2.3)).insert(pq.iSWAP(q[4], q[1]))
    # prog.insert(pq.RX(q[2], np.pi/6.2)).insert(pq.RY(q[2], 6.6)).insert(pq.RZ(q[2], np.pi/3))
    # prog.insert(pq.CR(q[0], q[2], 2.3)).insert(pq.CR(q[0], q[1], np.pi/2.3)).insert(pq.Y1(q[4])).insert(pq.Z1(q[5]))
    # prog.insert(pq.CR(q[5], q[2], np.pi/3))
    # prog.insert(pq.CR(q[2], q[3], np.pi/5))
    # prog.insert(pq.iSWAP(q[5], q[2], np.pi/2.3)).insert(pq.iSWAP(q[4], q[1]))
    # prog.insert(pq.RX(q[2], np.pi/6.2)).insert(pq.RY(q[2], 6.6)).insert(pq.RZ(q[2], np.pi/3))
    # prog.insert(pq.U1(q[0], np.pi/4)).insert(pq.U2(q[1], 1, np.pi/2)).insert(pq.U3(q[2], 1.2, 2, np.pi/3)).insert(pq.U4(1.5, 2.2, np.pi/3, np.pi/4, q[3]))
    # prog.insert(pq.iSWAP(q[5], q[2], np.pi/2.3)).insert(pq.U4(1.5, 2.2, np.pi/3, np.pi/4, q[3]))
    # prog.insert(pq.RX(q[2], np.pi/6.2)).insert(pq.RY(q[2], 6.6)).insert(pq.RZ(q[2], np.pi/3))
    # prog.insert(pq.U1(q[0], np.pi/4)).insert(pq.U2(q[1], 1, np.pi/2)).insert(pq.U3(q[2], 1.2, 2, np.pi/3))
    # prog.insert(pq.CU(np.pi/3, 3, 4, 5, q[5], q[2]))
    # prog.insert(pq.T(q[0])).insert(pq.SWAP(q[1], q[2])).insert(pq.X1(q[3]))
    # prog.insert(pq.X(q[0])).insert(pq.CZ(q[1], q[2])).insert(pq.Y(q[3])).insert(pq.Z(q[3]))
    # prog.insert(pq.S(q[3]))
    # prog.insert(cir)
    # prog.insert(pq.U4(1.5, 2.2, np.pi/3, np.pi/4, q[1]))
    # prog.insert(pq.CU(np.pi/3, 3, 4, 5, q[3], q[2]))
    # prog.insert(pq.S(q[3]))
    # prog.insert(cir)
    # prog.insert(pq.U4(1.5, 2.2, np.pi/3, np.pi/4, q[1]))
    # prog.insert(pq.CU(np.pi/3, 3, 4, 5, q[3], q[2]))
    # prog.insert(pq.CU(np.pi/3, 3, 4, 5, q[5], q[2]))
    # prog.insert(pq.T(q[0])).insert(pq.SWAP(q[1], q[2])).insert(pq.X1(q[3]))
    # prog.insert(pq.X(q[0])).insert(pq.CZ(q[1], q[2])).insert(pq.Y(q[3])).insert(pq.Z(q[3]))
    # prog.insert(pq.S(q[3]))
    prog.insert(cir)
    # prog.insert(pq.U4(1.5, 2.2, np.pi/3, np.pi/4, q[1]))
    # prog.insert(pq.CU(np.pi/3, 3, 4, 5, q[3], q[2]))
    # prog.insert(pq.S(q[3]))
    # prog.insert(cir)
    # prog.insert(pq.U4(1.5, 2.2, np.pi/3, np.pi/4, q[5]))
    # prog.insert(pq.CR(q[0], q[2], 2.3)).insert(pq.CR(q[0], q[1], np.pi/2.3)).insert(pq.Y1(q[4])).insert(pq.Z1(q[5]))
    # prog.insert(pq.CR(q[5], q[2], np.pi/3))
    # prog.insert(pq.CR(q[2], q[3], np.pi/5))
    # prog.insert(pq.iSWAP(q[5], q[2], np.pi/2.3)).insert(pq.iSWAP(q[4], q[1]))
    # prog.insert(pq.RX(q[2], np.pi/6.2)).insert(pq.RY(q[2], 6.6)).insert(pq.RZ(q[2], np.pi/3))
    # prog.insert(pq.U1(q[0], np.pi/4)).insert(pq.U2(q[1], 1, np.pi/2)).insert(pq.U3(q[2], 1.2, 2, np.pi/3)).insert(pq.U4(1.5, 2.2, np.pi/3, np.pi/4, q[3]))
    # prog.insert(pq.iSWAP(q[5], q[2], np.pi/2.3)).insert(pq.U4(1.5, 2.2, np.pi/3, np.pi/4, q[3]))
    # prog.insert(pq.RX(q[2], np.pi/6.2)).insert(pq.RY(q[2], 6.6)).insert(pq.RZ(q[2], np.pi/3))
    # prog.insert(cir)
    # prog.insert(pq.U4(1.5, 2.2, np.pi/3, np.pi/4, q[5]))
    # prog.insert(pq.CR(q[0], q[2], 2.3)).insert(pq.CR(q[0], q[1], np.pi/2.3)).insert(pq.Y1(q[4])).insert(pq.Z1(q[5]))
    # prog.insert(pq.CR(q[5], q[2], np.pi/3))
    # prog.insert(pq.CR(q[2], q[3], np.pi/5))
    # prog.insert(pq.iSWAP(q[5], q[2], np.pi/2.3)).insert(pq.iSWAP(q[4], q[1]))
    # prog.insert(pq.RX(q[2], np.pi/6.2)).insert(pq.RY(q[2], 6.6)).insert(pq.RZ(q[2], np.pi/3))
    # prog.insert(pq.U1(q[0], np.pi/4)).insert(pq.U2(q[5], 1, np.pi/2)).insert(pq.U3(q[5], 1.2, 2, np.pi/3)).insert(pq.U4(1.5, 2.2, np.pi/3, np.pi/4, q[3]))
    # prog.insert(pq.iSWAP(q[5], q[2], np.pi/2.3)).insert(pq.U4(1.5, 2.2, np.pi/3, np.pi/4, q[3]))
    # prog.insert(pq.RX(q[2], np.pi/6.2)).insert(pq.RY(q[2], 6.6)).insert(pq.RZ(q[5], np.pi/3))
    # prog.insert(pq.U1(q[0], np.pi/4)).insert(pq.U2(q[1], 1, np.pi/2)).insert(pq.U3(q[2], 1.2, 2, np.pi/3))
    # prog.insert(pq.CU(np.pi/3, 3, 4, 5, q[5], q[2]))
    # prog.insert(pq.H(q[0])).insert(pq.iSWAP(q[5], q[2], np.pi/2.3)).insert(pq.iSWAP(q[4], q[1]))
    # prog.insert(pq.H(q[0])).insert(pq.H(q[1])).insert(pq.H(q[2])).insert(pq.H(q[3])).insert(pq.H(q[4])).insert(pq.H(q[5])).insert(pq.measure_all(q, c))
    # prog.insert(pq.H(q[0])).insert(pq.H(q[1])).insert(pq.Measure(q[0], c[0]))
    
    print(prog)
    draw_qprog(prog, 'text')
    draw_qprog(prog, 'pic', filename='D:/test_cir_draw.jpg', verbose=True)

if __name__=="__main__":
    test_layer1()
    print("Test over.")