import pyqpanda.pyQPanda as pq
# from pyqpanda import *
def Bell_State():
    machine=pq.init_quantum_machine(pq.QMachineType.CPU)
    qlist=machine.qAlloc_many(2)
    clist=machine.cAlloc_many(2)
    qprog=pq.QProg()
    qprog.insert(pq.H(qlist[0]))\
         .insert(pq.CNOT(qlist[0],qlist[1]))
    qprog.insert(pq.meas_all(qubit_list=qlist,cbit_list=clist))
    machine.load(qprog)
    machine.run()
    result=machine.getResultMap()
    return result
def test():
    machine=pq.init_quantum_machine(pq.QMachineType.CPU)
    qlist=machine.qAlloc_many(4)
    clist = machine.cAlloc_many(4)
    prog=pq.QProg()
    prog.insert(pq.H(qlist[2])).insert(pq.meas_all(qlist,clist))
    data = {'shots':1000}
    result=machine.run_with_configuration(prog,clist,data) 
    pq.destroy_quantum_machine(machine)
    

    machine2=pq.init_quantum_machine(pq.QMachineType.CPU)
    qlist2=machine2.qAlloc_many(3)
    prog2=pq.QProg()
    prog2.insert(pq.H(qlist2[0])).insert(pq.CNOT(qlist2[0],qlist2[1]))
    result2=machine2.prob_run_dict(prog2,qlist2,-1)
    pq.destroy_quantum_machine(machine2)

    pq.init(pq.QMachineType.CPU)
    qlist3=pq.qAlloc_many(5)
    clist3=pq.cAlloc_many(5)
    prog3=pq.QProg()
    prog3.insert(pq.H(qlist3[0])).insert(pq.CNOT(qlist3[0],qlist3[1]))\
         .insert(pq.CNOT(qlist3[1],qlist3[2])).insert(pq.CNOT(qlist3[2],qlist3[3]))\
         .insert(pq.meas_all(qlist3,clist3))
    result3=pq.run_with_configuration(prog3,clist3,100)
    pq.finalize()


    return result,result2,result3
if __name__=="__main__":
    machine=pq.init_quantum_machine(pq.QMachineType.CPU)
    qlist=machine.qAlloc_many(4)
    clist = machine.cAlloc_many(4)
    prog=pq.QProg()
    prog.insert(pq.H(qlist[2])).insert(pq.measure_all(qlist,clist))
    print(pq.to_originir(prog,machine))

    iter = prog.begin()
    type =iter.get_node_type()
    if pq.NodeType.GATE_NODE == type:
        gate = pq.QGate(iter)


    list =pq.get_adjacent_qgate_type(prog,iter)
    print(len(list))
    a = pq.CNOT(qlist[1],qlist[2])

    qif = pq.QIfProg(clist[1]==1,a)

    list = [[1,1,1,1],[1,1,0,1],[1,1,1,1],[1,1,1,1]]
    print(len(list))
    print(pq.is_match_topology(a,list))

    data = {'shots':1000}
    result=machine.run_with_configuration(prog,clist,data) 
    pq.destroy_quantum_machine(machine)



