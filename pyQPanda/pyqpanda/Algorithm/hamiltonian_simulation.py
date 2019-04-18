'''
Simulating a specific hamitonian (PauliOperator)\n
Copyright (C) Origin Quantum 2017-2018\n
Licensed Under Apache Licence 2.0
'''

from pyqpanda.Hamiltonian.PauliOperator.pyQPandaPauliOperator import PauliOperator
from pyqpanda.pyQPanda import *
from pyqpanda.utils import *
from math import pi
from pyqpanda.Algorithm.fragments import parity_check_circuit
from functools import partial


def simulate_z_term(qubit_list, coef, t):
    '''
    `QPanda Algorithm API`\n
    Simulating z-only term like H=coef * (Z0..Zn-1)\n
    U=exp(-iHt)\n
    \n
    list<Qubit>, float, float -> QCircuit

    Note: Z-Hamiltonian spreads over the qubit_list
    '''
    prog=QCircuit()

    # parity-check
    # RZ(theta)
    # parity-check

    if len(qubit_list) > 1:
        prog.insert(parity_check_circuit(qubit_list))\
            .insert(RZ(qubit_list[-1],coef*t))\
            .insert(parity_check_circuit(qubit_list))
    else:
        if len(qubit_list)!=0:
            prog.insert(RZ(qubit_list[0],coef*t))

    return prog

def simulate_one_term(qubit_list, hamiltonian_term, t):
    '''
    Simulate a single term of Hamilonian like "X0 Y1 Z2" with
    coefficient and time. U=exp(-it*coef*H)
    @param
        qubit_list: qubit needed to simulate the hamiltonian
        hamiltonian_term: tuple like ("X0 Y1 Z2",2.3)
        t: time

    @return: QCircuit
    '''

    prog=QCircuit()

    if not hamiltonian_term:
        return prog

    actual_qlist=list()
    transform=QCircuit()

    for single_term in hamiltonian_term[0]:        
        if hamiltonian_term[0][single_term] is 'X':            
            transform.insert(H(qubit_list[single_term]))            
            actual_qlist.append(qubit_list[single_term])
        elif hamiltonian_term[0][single_term] is 'Y':
            transform.insert(RX(qubit_list[single_term],pi/2))            
            actual_qlist.append(qubit_list[single_term])
        elif hamiltonian_term[0][single_term] is 'Z':               
            actual_qlist.append(qubit_list[single_term])
    prog.insert(transform)\
        .insert(simulate_z_term(actual_qlist, hamiltonian_term[1], t))\
        .insert(transform.dagger())

    return prog

def simulate_pauliZ_hamiltonian(qubit_list,PauliOperator,t):
    '''
    Simulate hamiltonian consists of pauli-Z operators
    @param
        qubit_list: qubit needed to simulate the hamiltonian
        PauliOperator: PauliOperator object from c++
        t: time

    @return: QCircuit
    '''
    prog=QCircuit()
    
    if PauliOperator.isAllPauliZorI():
        Hamiltonian=PauliOperator.toHamiltonian(0)
        for op in Hamiltonian:
            actual_qlist=[]
            for single_term in op[0]:  
                actual_qlist.append(qubit_list[single_term])
            if len(actual_qlist)!=0:
                prog.insert(simulate_z_term(actual_qlist, op[1], t))     
    else:
        throw("unmatched")
    return prog



def simulate_hamiltonian(qubit_list,PauliOperator,t,slices=3):
    '''
    Simulate a general case of hamiltonian by Trotter-Suzuki
    approximation. U=exp(-iHt)=(exp(-i H1 t/n)*exp(-i H2 t/n))^n
    @param:
        qubit_list: the qubit needed to simulate the Hamiltonian
        pauliOperator: the Hamiltonian (PauliOperator type)
        t: time
        slices: the approximate slices.

    @return: QCircuit
    '''

    prog=QCircuit()
    Hamiltonian=PauliOperator.toHamiltonian(0)
    for i in range(slices):
        for op in Hamiltonian:
            prog.insert(simulate_one_term(qubit_list,op,t/slices))

    return prog

def adiabatic_simulation_with_configuration(qn_, Hp_, Hd_, step_, slices_, t_, shots_=1000):
    '''
    simulate hamiltonian with configuration and return result.
    @param
        qn_ : number of qubits
        Hp_ : problem hamiltonian
        Hd_ : driver hamiltonian 
        step_ : number of step in adiabatic simulation
        slices_ : number of slices in trotter-suzuki approximation
        t_ : total time for adiabatic simulation
        shots_ : number of shots
    '''
    machine=init_quantum_machine(QMachineType.CPU)
    prog=QProg()
    q=machine.qAlloc_many(qn_)
    c=machine.cAlloc_many(qn_)
    prog.insert(single_gate_apply_to_all(gate=X, qubit_list=q))
    prog.insert(single_gate_apply_to_all(gate=H, qubit_list=q))

    for i in range(step_+1):
        Ht=Hp_*(i/step_)+Hd_*((step_-i)/step_)
        prog.insert(simulate_hamiltonian(q,Ht,t=t_/step_,slices=slices_))
    machine.directly_run(program=prog)
    result=machine.get_prob_dict(q)
    destroy_quantum_machine(machine)
    return result


def ising_model(qubit_list,graph,gamma_):
    prog=QCircuit()
    length=len(graph)
    for i in range(length):
        prog.insert(CNOT(qubit_list[graph[i][0]],qubit_list[graph[i][1]]))\
            .insert(RZ(qubit_list[graph[i][1]],2*gamma_*graph[i][2]))\
            .insert(CNOT(qubit_list[graph[i][0]],qubit_list[graph[i][1]]))
    return prog

def pauliX_model(qubit_list,beta_):
    prog=QCircuit()
    length=len(qubit_list)
    for i in range(length):
        prog.insert(RX(qubit_list[i],2*beta_))
    return prog

def weight(graph,distribution):
    sum=0
    length=len(graph)
    for i in range(length):
        if distribution[graph[i][0]]!=distribution[graph[i][1]]:
            sum=sum+graph[i][2]
    return sum

def get_qn_from_graph(graph):
    max_qn=0
    for edge in graph:
        if edge[0]>max_qn:
            max_qn=edge[0]
        if edge[1]>max_qn:
            max_qn=edge[1]
    return max_qn+1

def quantum_approximate_optimization_algorithm(
    graph, 
    gamma_,
    beta_,
    use_prob_run=True,
    use_quick_measure=True,
    multiProcessing=False,    
    shots_=100, 
    dataType="list"
):
    '''
    quantum approximate optimization algorithm
    @param
        graph: origin graph
        gamma_: problem hamiltonian parameter
        beta_: driver hamiltonian parameter
        use_prob_run : Use prob_run instead of repeatly measurement
        multiProcessing: no implemented yet
        shot_: execution times

        (the following only enabled when "use_prob_run=True")  
        use_quick_measure : use quick measure instead of output probabilites   
        dataType : chosen data type for the prob_run
    '''
    step_=len(gamma_)
    qn_=get_qn_from_graph(graph)

    init()
    prog=QProg()
    q=qAlloc_many(qn_)
    c=cAlloc_many(qn_)
    prog.insert(single_gate_apply_to_all(gate=H, qubit_list=q))
    for i in range(step_):
        prog.insert(ising_model(q,graph,gamma_[i]))\
            .insert(pauliX_model(q,beta_[i]))    

    if use_prob_run:
        if use_quick_measure:
            directly_run(QProg=prog)
            result=quick_measure(q, shots_)
        else:
            result=prob_run(program=prog,noise=False,select_max=-1,qubit_list=q,dataType=dataType)
    else:
        prog.insert(meas_all(q,c))
        result=run_with_configuration(program=prog, shots=shots_, cbit_list=c)
    
    #print(result)
    finalize()
    target=0

    if not multiProcessing:      
        for outcome in result:
            if weight(graph,outcome)>target:
                target=weight(graph,outcome)
    else:
        raise NotImplementedError()

    return -target

def binding(graph, shots):
    return partial(qaoa_in_list,graph=graph,shots=shots)

def qaoa_in_list(
    arguments,
    graph,
    use_prob_run=True,
    multiProcessing=False,
    shots=100,
    dataType='list',    
):
    #print(arguments)
    step=len(arguments)//2
    beta=list()
    gamma=list()
    for i in range(step):
        beta.append(arguments[i])
        gamma.append(arguments[i+step])

    result= quantum_approximate_optimization_algorithm(
        graph=graph,
        gamma_=gamma,
        beta_=beta,
        use_prob_run=use_prob_run,
        multiProcessing=multiProcessing,
        shots_=shots,
        dataType=dataType)
    f=open("a.txt", 'a')
    f.write(str(arguments)+' '+str(result)+'\n')
    f.close()
    #print(result)
    return result

def qaoa(graph,step_=1,shots_=1000, method="Nelder-Mead"):
    gamma_=[]
    beta_=[]
    for i in range(step_):
        gamma_.append(0)
        beta_.append(0)    
    initial_guess=gamma_+beta_
    result = minimize(binding(graph,shots_), initial_guess, method=method)    
    return result



