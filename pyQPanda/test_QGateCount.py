import pyqpanda.pyQPanda as pq

class InitQMachine:
    def __init__(self, quBitCnt, cBitCnt, machineType=pq.QMachineType.CPU):
        self.m_machine = pq.init_quantum_machine(machineType)
        self.m_qlist = self.m_machine.qAlloc_many(quBitCnt)
        self.m_clist = self.m_machine.cAlloc_many(cBitCnt)


if __name__ == "__main__":
    init_machine = InitQMachine(8, 8)
    qlist = init_machine.m_qlist
    clist = init_machine.m_clist
    prog = pq.QProg()

    cir = pq.QCircuit()
    cir.insert( pq.X(qlist[0])).insert( pq.X(qlist[1])).insert(
        pq.Y(qlist[1])).insert(pq.H(qlist[0])).insert(pq.Z(qlist[1])).insert( pq.RX(qlist[0], 3.14))

    prog.insert(cir).insert(pq.T(qlist[0])).insert(pq.CNOT(qlist[1], qlist[2])).insert(pq.H(qlist[3])).insert(
        pq.H(qlist[4])).insert(pq.X(qlist[4])).insert(pq.measure_all(qlist, clist))
        
    total_num = pq.count_qgate_num(prog )
    xnum = pq.count_qgate_num(prog, pq.PAULI_X_GATE);
    hnum = pq.count_qgate_num(prog, pq.HADAMARD_GATE);
    Inum = pq.count_qgate_num(prog, pq.ISWAP_GATE);

    print ("Total num" , total_num)
    print("XGate number " , xnum) 
    print("HGate number " , hnum) 
    print("IGate number " , Inum) 
    