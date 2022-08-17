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
    cir.insert( pq.H(qlist[0])).insert( pq.S(qlist[1])).insert(
        pq.CNOT(qlist[0], qlist[1])).insert(pq.CZ(qlist[1],qlist[2]))

    prog.insert(cir).insert(pq.CU(1, 2, 3, 4,qlist[0],qlist[2])).insert(pq.S(qlist[2])).insert(
        pq.CR(qlist[2],qlist[1],3.14 / 2))


    layerinfo = pq.circuit_layer(prog);
    print( "circuit layer : " , layerinfo[0].size())


        
        
        
