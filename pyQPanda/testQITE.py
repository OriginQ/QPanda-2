from pyqpanda import *
from numpy import pi
from numpy import random

if __name__ == "__main__":

    machine = CPUQVM()
    machine.init_qvm()

    q = machine.qAlloc_many(3)
    c = machine.cAlloc_many(3)

    # QGate
    rx_half_pi = RX(q[0], pi/2)
    ry_half_pi = RY(q[0], pi/2)

    # QCircuit
    circuit = QCircuit()
    circuit << rx_half_pi << ry_half_pi

    # AnsatzGate
    ansatz_gate = AnsatzGate(AnsatzGateType.AGT_RX, 1, 1.5)

    # AnsatzGateList
    ansatz_list = [ansatz_gate, ansatz_gate]

    # Ansatz
    ansatz = Ansatz(rx_half_pi)
    ansatz << ry_half_pi
    ansatz << ansatz_gate
    ansatz << ansatz_list
    ansatz.insert(circuit)

    theta_list = []
    for i in range(7):
        theta_list.append(random.rand())

    print(ansatz)
    ansatz.set_thetas(theta_list)
    print(ansatz)

    # pauli = trans_vec_to_Pauli_operator([1, 2, 3, 0])
    # qite = QITE()
    # qite.set_Hamiltonian(pauli)
    # qite.set_ansatz_gate([a, b])
    # qite.exec()
    # result = qite.get_result()
    # print(result)

    # # a.type = pq.AnsatzGateType.AGT_RX
    # # a.target = 1
    # # a.theta = 0
    # # a.
    # print(a.type)
    # print(a.target)
    # print(a.theta)
    # print(a.control)

    # chemiq = py.QITE()
    # chemiq.exec()

    # value = chemiq.getEnergies()
    # print(value)
