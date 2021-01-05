import pyqpanda.pyQPanda as pq
from pyqpanda.Visualization.circuit_draw import *
import numpy as np
import math

class InitQMachine:
    def __init__(self, machineType = pq.QMachineType.CPU):
        self.m_machine = pq.init_quantum_machine(machineType)

    def __del__(self):
        pq.destroy_quantum_machine(self.m_machine)

def test_grover1():
    init_machine = InitQMachine()
    machine = init_machine.m_machine
    x = machine.cAlloc()

    data=[8, 7, 6, 0, 6, 3, 6, 4, 21, 15, 11, 11, 3, 9, 7]
    measure_qubits = pq.QVec()
    grover_cir = pq.Grover(data, x==6, machine, measure_qubits, 1)
    
    #print(grover_cir)
    draw_qprog(grover_cir, 'pic', filename='D:/cir_grover_1.jpg', verbose=True)
    result = pq.prob_run_dict(grover_cir, measure_qubits)
    print(result)

def test_grover2():
    init_machine = InitQMachine()
    machine = init_machine.m_machine
    x = machine.cAlloc()

    # data=[8, 7, 6, 0, 6, 3, 6, 4, 21, 15, 11, 11, 3, 9, 7]
    data=[2,7,5,1]
    measure_qubits = pq.QVec()
    grover_cir = pq.Grover(data, x==5, machine, measure_qubits, 1)
    grover_prog = pq.QProg()
    grover_prog.insert(grover_cir)
    measure_qubit_num = len(measure_qubits)
    c = machine.cAlloc_many(measure_qubit_num)
    for i in range(0,measure_qubit_num):
        grover_prog.insert(pq.Measure(measure_qubits[i], c[i]))

    draw_qprog(grover_prog, 'pic', filename='D:/prog_grover_1.jpg', verbose=True)
    result = pq.run_with_configuration(grover_prog, c, shots = 1000)
    print(result)

def test_grover_search():
    init_machine = InitQMachine()
    machine = init_machine.m_machine
    x = machine.cAlloc()

    data=[3, 6, 6, 9, 10, 15, 11, 6]
    grover_result = pq.Grover_search(data, x==6, machine, 1)
    print(grover_result[0])
    print(grover_result[1])

if __name__=="__main__":
    #test_grover1()
    test_grover2()
    #test_grover_search()
    print("Test over.")