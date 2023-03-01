from pyqpanda import *
import numpy as np

def parity_check_circuit(qubit_list):
    vqc=VariationalQuantumCircuit()
    for i in range(len(qubit_list)-1):
        vqc.insert(VariationalQuantumGate_CNOT(qubit_list[i],qubit_list[len(qubit_list)-1]))
    return vqc

def simulateZTerm_VQC(qubit_list,coef,time):
    vqc=VariationalQuantumCircuit()
    if 0==len(qubit_list):
        return vqc
    elif 1==len(qubit_list):
        vqc.insert(VariationalQuantumGate_RZ(qubit_list[0], - coef * time * 2))
    else:
        vqc.insert(parity_check_circuit(qubit_list))\
           .insert(VariationalQuantumGate_RZ(qubit_list[-1], - coef * time * 2))\
           .insert(parity_check_circuit(qubit_list))
    return vqc

def simulatePauliZHamiltonian_VQC(qubit_list,Hamiltonian,time):
    vqc=VariationalQuantumCircuit()
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

def variational_qaoa_test():
    machine=QuantumMachine(QMachineType.CPU_SINGLE_THREAD)
   
    H1 = PauliOperator({'Z0 Z4':0.73,'Z0 Z5':0.33,'Z0 Z6':0.5,'Z1 Z4':0.69,
    'Z1 Z5':0.36,'Z2 Z5':0.88,'Z2 Z6':0.58,'Z3 Z5':0.67,'Z3 Z6':0.43})
    qlist=machine.qAlloc_many(H1.getMaxIndex()+1)

    step=2
    gamma = var(np.ones((step,1), dtype = 'float64')*0.01)
    beta  = var(np.ones((step,1), dtype = 'float64')*0.01)

    vqc=VariationalQuantumCircuit()
    for i in qlist:
        vqc.insert(H(i))

    for i in range(step):
        temp1=gamma[i]
        temp2=beta[i]
        vqc.insert(simulatePauliZHamiltonian_VQC(qlist,H1.toHamiltonian(1),temp1))
        for j in qlist:
            vqc.insert(VariationalQuantumGate_RX(j,temp2))
    grad={gamma:np.ones((step,1)), beta:np.ones((step,1))}
    loss = qop(vqc, H1, machine._quantum_machine, qlist)

    exp=expression(loss)
    leaves=[gamma,beta]
    leaf_set=exp.find_non_consts(leaves)
    

    iterations=100
    learning_rate=0.02

    for i in range(iterations):
        print("Loss: ", eval(loss,True))

        back(exp,grad,leaf_set)
        gamma.set_value(gamma.get_value() - learning_rate * grad[gamma])
        beta.set_value(beta.get_value() - learning_rate * grad[beta])