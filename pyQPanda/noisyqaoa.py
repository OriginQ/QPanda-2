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
        vqc.insert(VariationalQuantumGate_RZ(qubit_list[0], -coef * time*2))
    else:
        vqc.insert(parity_check_circuit(qubit_list))\
           .insert(VariationalQuantumGate_RZ(qubit_list[len(qubit_list)-1], -coef * time*2))\
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

def noisy_qaoa(Hamiltonian,noise,step):
    machine=NoiseQVM()
    machine.initQVM(noise)
    qlist=machine.qAlloc_many(H1.getMaxIndex()+1)

    gamma = var(np.ones((step,1), dtype = 'float64')*(0.01))
    beta = var(np.ones((step,1), dtype = 'float64')*0.01)
    vqc=VariationalQuantumCircuit()
    for i in qlist:
        vqc.insert(VariationalQuantumGate_H(i))
    if step is not 1:
        for i in range(step):
            temp1=gamma[i]
            temp2=beta[i]
            vqc.insert(simulatePauliZHamiltonian_VQC(qlist,H1.toHamiltonian(1),temp1))
            for j in qlist:
                vqc.insert(VariationalQuantumGate_RX(j,2.0*temp2))
    else:
        vqc.insert(simulatePauliZHamiltonian_VQC(qlist,H1.toHamiltonian(1),gamma))
        for j in qlist:
            vqc.insert(VariationalQuantumGate_RX(j,2.0*beta))
    grad={gamma:np.zeros((step,1)), beta:np.zeros((step,1))}
    loss1 = qop(vqc, H1, machine, qlist,False)  

    exp=expression(loss1)
    leaves=[gamma,beta]
    leaf_set=exp.find_non_consts(leaves)    

    iterations=100
    learning_rate=0.02
    import time
    now=time.time()
    now=time.localtime(now)
    now=time.strftime("%Y%m%d %H%M%S",now)
    result_fname="result-"+now+".txt"
    with open(result_fname, 'a+') as fp:
        fp.write('Problem Hamiltonian:{}\n'.format(Hamiltonian.toString()))
        fp.write('step={}\n'.format(step))
        fp.write('noise:{}\n\n'.format(noise))

    for i in range(iterations):
        loss=eval(loss1,True)
        with open(result_fname, 'a+') as fp:
            # loss = eval(loss1,True)
            print("Loss: ", loss)
            fp.write('i:{}\nloss:{}\n'.format(i, loss))
            back(exp,grad,leaf_set)
            print("gamma",gamma.get_value())
            print("gamma grad",grad[gamma])
            print("beta",beta.get_value())
            print("beta grad",grad[beta])
            gamma.set_value(gamma.get_value() - learning_rate * grad[gamma])
            beta.set_value(beta.get_value() - learning_rate * grad[beta])    
    machine.finalize()
    destroy_quantum_machine(machine)
    

if __name__=="__main__":
    
    H1 = PauliOperator({'Z0 Z4':0.73,'Z0 Z5':0.33,'Z0 Z6':0.5,'Z1 Z4':0.69,
    'Z1 Z5':0.36,'Z2 Z5':0.88,'Z2 Z6':0.58,'Z3 Z5':0.67,'Z3 Z6':0.43})
    # H2=q.PauliOperator({"Z0 Z4":0.73+0.1j})
    # machine=q.init_quantum_machine(q.QMachineType.CPU_SINGLE_THREAD)
    # qlist=machine.qAlloc_many(H1.getMaxIndex()+1)
    singlenoise=[2,5,2,0.03]

    noise={"RY":[2,5.0,2.0,0.03],
           "RZ":[2,5.0,2.0,0.03],
           "RX":[2,5.0,2.0,0.03],
           "H":[2,5.0,2.0,0.03],
           "CNOT":[6,5.0,2.0,0.06]}
    noisem={"noisemodel":noise}
    noisy_qaoa(H1,noisem,1)
  