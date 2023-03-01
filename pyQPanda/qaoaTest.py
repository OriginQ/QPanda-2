import pyqpanda as q
import numpy as np

def parity_check_circuit(qubit_list):
    vqc=q.VariationalQuantumCircuit()
    for i in range(len(qubit_list)-1):
        vqc.insert(q.VariationalQuantumGate_CNOT(qubit_list[i],qubit_list[len(qubit_list)-1]))
    return vqc

def simulateZTerm_VQC(qubit_list,coef,time):
    vqc=q.VariationalQuantumCircuit()
    if 0==len(qubit_list):
        return vqc
    elif 1==len(qubit_list):
        vqc.insert(q.VariationalQuantumGate_RZ(qubit_list[0], -coef * time*2))
    else:
        vqc.insert(parity_check_circuit(qubit_list))\
           .insert(q.VariationalQuantumGate_RZ(qubit_list[len(qubit_list)-1], -coef * time*2))\
           .insert(parity_check_circuit(qubit_list))
    return vqc

def simulatePauliZHamiltonian_VQC(qubit_list,Hamiltonian,time):
    vqc=q.VariationalQuantumCircuit()
    for i in range(len(Hamiltonian)):
        tmp_vec=[]
        item=Hamiltonian[i]
        map=item[0]
        for iter in map:
            if 'Z'!=map[iter]:
                pass
            tmp_vec.append(qubit_list[iter])
        if 0!=len(tmp_vec):
            vqc.insert(simulateZTerm_VQC(qubit_list=tmp_vec,coef=item[1],time=time))
    return vqc

if __name__ == "__main__":
    
    machine=q.init_quantum_machine(q.QMachineType.CPU_SINGLE_THREAD)
   
    H1 = q.PauliOperator({'Z0 Z4':0.73,'Z0 Z5':0.33,'Z0 Z6':0.5,'Z1 Z4':0.69,
    'Z1 Z5':0.36,'Z2 Z5':0.88,'Z2 Z6':0.58,'Z3 Z5':0.67,'Z3 Z6':0.43})
    qlist=machine.qAlloc_many(H1.getMaxIndex()+1)

    step=2
    gamma = q.var(np.ones((step,1), dtype = 'float64')*(0.01))
    beta = q.var(np.ones((step,1), dtype = 'float64')*0.01)

    # vqc2=q.VariationalQuantumCircuit()
    # vqc2.insert(q.RX(qlist[0],3.1415)).insert(q.VariationalQuantumGate_RX(qlist[1],gamma[0])).insert(q.RX(qlist[2],3.1415)).insert(q.RX(qlist[3],3.1415))
    # loss2=q.qop(vqc2, H1, machine, qlist)  
    # print(q.eval(loss2,True))

    vqc=q.VariationalQuantumCircuit()
    for i in qlist:
        vqc.insert(q.VariationalQuantumGate_H(i))
    for i in range(step):
        temp1=gamma[i]
        temp2=beta[i]
        vqc.insert(simulatePauliZHamiltonian_VQC(qlist,H1.toHamiltonian(1),temp1))
        for j in qlist:
            vqc.insert(q.VariationalQuantumGate_RX(j,2.0*temp2))
    grad={gamma:np.zeros((step,1)), beta:np.zeros((step,1))}
    loss1 = q.qop(vqc, H1, machine, qlist)  

    exp=q.expression(loss1)
    leaves=[gamma,beta]
    leaf_set=exp.find_non_consts(leaves)    

    iterations=400
    learning_rate=0.02
    for i in range(iterations):
        print("Loss: ", q.eval(loss1,True))
        q.back(exp,grad,leaf_set)
        print("gamma",gamma.get_value())
        print("gamma grad",grad[gamma])
        print("beta",beta.get_value())
        print("beta grad",grad[beta])
        gamma.set_value(gamma.get_value() - learning_rate * grad[gamma])
        beta.set_value(beta.get_value() - learning_rate * grad[beta])    
    