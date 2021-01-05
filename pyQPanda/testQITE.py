import pyqpanda as pq

if __name__=="__main__":

    pauli = pq.trans_vec_to_Pauli_operator([1,2,3,0])
    # print(pauli)
    # vec = pq.trans_Pauli_operator_to_vec(pauli)
    # print(vec)

    a = pq.AnsatzGate(pq.AnsatzGateType.AGT_RX, 1, 1.5)
    b = pq.AnsatzGate(pq.AnsatzGateType.AGT_RY, 0, 0.5, 1)
    qite = pq.QITE()
    qite.set_hamiltonian(pauli)
    qite.set_ansatz_gate([a, b])
    # qite.set_para_update_mode(pq.UpdateMode.GD_DIRECTION)
    qite.exec()
    result = qite.get_result()
    print(result)
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
