import pyqpanda.pyQPanda as pq
import numpy as np
import math

class InitQMachine:
    def __init__(self, machineType = pq.QMachineType.CPU):
        self.m_machine = pq.init_quantum_machine(machineType)

    def __del__(self):
        pq.destroy_quantum_machine(self.m_machine)

def test_quantum_walk():
    init_machine = InitQMachine()
    machine = init_machine.m_machine
    x = machine.cAlloc()

    data=[3, 6, 6, 9, 10, 15, 11, 6]
    #data=[3]
    measure_qubits = pq.QVec()
    grover_cir = pq.quantum_walk_alg(data, x==6, machine, measure_qubits, 2)
    result = pq.prob_run_dict(grover_cir, measure_qubits)
    print(result)

def test_quantum_walk_search():
    init_machine = InitQMachine()
    machine = init_machine.m_machine
    x = machine.cAlloc()

    data=[3, 6, 6, 9, 10, 15, 11, 6]
    grover_result = pq.quantum_walk_search(data, x==6, machine, 2)
    print(grover_result[0])
    print(grover_result[1])

if __name__=="__main__":
    test_quantum_walk()
    #test_quantum_walk_search()
    print("Test over.")