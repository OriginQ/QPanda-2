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
    """
    Simulates a z-only term Hamiltonian in a quantum circuit using the QPanda library.

    The function applies a series of RZ gates to the specified qubits in `qubit_list` with
    a phase coefficient `coef` and time `t`. The RZ gates represent the z-component of the
    Hamiltonian. A parity check circuit is inserted before and after the RZ gates to
    ensure correct operation.

        Args:
            qubit_list (list of Qubit): The list of qubits over which the Hamiltonian is spread.
            coef (float): The coefficient for the z-term.
            t (float): The time duration for which the Hamiltonian is applied.

        Returns:
            QCircuit: The resulting quantum circuit after simulating the z-term.

        Note: 
            The parity check circuit is used to verify the integrity of the RZ operations.
    """
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
    """
    Simulates a single term of the Hamiltonian, represented by a string and a coefficient,
    acting on a specified set of qubits over a given time interval.

    The function constructs a quantum circuit that represents the unitary evolution
    of the Hamiltonian term, using the exponential of the negative imaginary time times
    the coefficient times the Hamiltonian matrix.

        Args:
            qubit_list (dict): A dictionary mapping qubit identifiers to their corresponding qubit objects.
            hamiltonian_term (tuple): A tuple containing a string describing the Hamiltonian term
                                      (e.g., "X0 Y1 Z2") and a float representing the coefficient.
            t (float): The time duration over which the Hamiltonian term is applied.

        Returns:
            QCircuit: A quantum circuit object representing the simulated Hamiltonian term.

    The function assumes the existence of the following functions and objects:
        - H: A function to create a Hadamard gate.
        - RX: A function to create a Rotation gate around the X-axis.
        - simulate_z_term: A function to simulate a single Z-term of the Hamiltonian.
        - transform.dagger(): The Hermitian conjugate of the transform quantum circuit.
    """

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
    """
    Constructs a quantum circuit representing the time-evolution of a Hamiltonian composed solely of Pauli-Z operators.

        Args:
            qubit_list (list): The list of qubits over which the Hamiltonian is defined.
            PauliOperator (object): An instance of a PauliOperator from C++ representing the Hamiltonian.
            t (float): The time duration for which the Hamiltonian will act on the system.

        Returns:
            QCircuit: A quantum circuit object representing the simulated Hamiltonian evolution.

        Raises:
            Exception: If the PauliOperator does not consist solely of Pauli-Z or identity operators.

        Notes:
            The function operates within the context of the pyQPanda package, which facilitates quantum computing with quantum circuits,
            gates, and simulators. It is designed to work with quantum circuit simulators or quantum cloud services.
    """
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
    """
    Simulate the evolution of a quantum system described by a Hamiltonian using the Trotter-Suzuki
    approximation method. This method decomposes the time-evolution operator into a series of two-body
    operators, each approximating a part of the Hamiltonian.

        Args:
            qubit_list (list): The qubits involved in the Hamiltonian simulation.
            PauliOperator (PauliOperator): The Hamiltonian expressed as a Pauli operator object.
            t (float): The total time over which the Hamiltonian is applied.
            slices (int, optional): The number of slices to use in the Trotter-Suzuki decomposition.
                Default is 3, which determines the accuracy of the approximation.

        Returns:
            QCircuit: A quantum circuit representing the time-evolved state of the qubits.

    This function is intended for use within the pyQPanda package, a Python library for programming quantum computers.
    It operates on quantum circuits, simulating the effects of Hamiltonians on qubits using quantum gates.
    The function is implemented in the 'pyQPanda.build.lib.pyqpanda.Algorithm.hamiltonian_simulation.py' module.
    """

    prog=QCircuit()
    Hamiltonian=PauliOperator.toHamiltonian(0)
    for i in range(slices):
        for op in Hamiltonian:
            prog.insert(simulate_one_term(qubit_list,op,t/slices))

    return prog

def adiabatic_simulation_with_configuration(qn_, Hp_, Hd_, step_, slices_, t_, shots_=1000):
    """
    Simulates an adiabatic Hamiltonian evolution with specified configuration and returns the resulting probabilities.

        Args:
            qn_ (int): The number of qubits used in the simulation.
            Hp_ (function): The problem Hamiltonian function, which takes a qubit index and returns the corresponding Hamiltonian term.
            Hd_ (function): The driver Hamiltonian function, which takes a qubit index and returns the corresponding Hamiltonian term.
            step_ (int): The number of steps in the adiabatic simulation process.
            slices_ (int): The number of slices used in the Trotter-Suzuki decomposition of the Hamiltonian.
            t_ (float): The total time duration for the adiabatic simulation.
            shots_ (int): The number of times the quantum circuit is executed to average the results.

        Returns:
            dict: A dictionary containing the probability distribution over the qubit states after the simulation.
    """
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
    """
    Constructs a quantum circuit representing the Ising model Hamiltonian.

    The function generates a quantum circuit based on the provided qubit list and graph, which defines the
    interactions between qubits. Each interaction is modeled as a CNOT gate followed by a RZ rotation
    with a phase factor determined by the `gamma_` parameter.

        Args:
            qubit_list (list of Qubit): A list of qubits, where each qubit is an instance from the pyQPanda library.
            graph (list of tuples): A list of tuples, each representing an edge in the graph. Each tuple contains
        two indices of qubits that are connected and an integer representing the interaction strength.
            gamma_ (float): A phase factor for the RZ rotation, determining the strength of the interaction.

        Returns:
            QuantumCircuit: A quantum circuit instance representing the Ising model Hamiltonian.

    This function is designed to be used within the pyQPanda package for simulating quantum algorithms,
    either locally using a quantum virtual machine or remotely via a quantum cloud service.
    """
    prog=QCircuit()
    length=len(graph)
    for i in range(length):
        prog.insert(CNOT(qubit_list[graph[i][0]],qubit_list[graph[i][1]]))\
            .insert(RZ(qubit_list[graph[i][1]],2*gamma_*graph[i][2]))\
            .insert(CNOT(qubit_list[graph[i][0]],qubit_list[graph[i][1]]))
    return prog

def pauliX_model(qubit_list,beta_):
    """
    Generate a quantum circuit with a sequence of Pauli-X gates applied to the specified qubits.

        Args:
            qubit_list (list): A list of integers representing the indices of the qubits on which the gates will be applied.
            beta_ (float): The rotation angle for the Pauli-X gates, with each gate being applied twice the given angle.

        Returns:
            QCircuit: A quantum circuit object with Pauli-X gates inserted at the specified qubits.

    The function operates within the context of the pyQPanda package, which facilitates quantum computing using quantum circuits,
    gates, and simulators. This model is intended to be used in quantum simulations and algorithms where a simple
    sequence of Pauli-X gates is required.
    """
    prog=QCircuit()
    length=len(qubit_list)
    for i in range(length):
        prog.insert(RX(qubit_list[i],2*beta_))
    return prog

def weight(graph,distribution):
    """
    Calculate the total weight of edges in the graph where the nodes have different distributions.

        Args:
            graph (list of tuples): A list where each tuple represents an edge in the graph. Each tuple contains three elements:
          (node1, node2, weight), where node1 and node2 are the nodes connected by the edge, and weight is the edge's weight.
            distribution (dict): A dictionary mapping each node to its associated distribution.

        Returns:
            int: The sum of the weights of all edges where the nodes have different distributions.

    The function traverses the graph and accumulates the weights of the edges where the nodes have distinct distributions,
    which is relevant in quantum computing simulations where the edge weights might represent physical quantities or costs.
    """
    sum=0
    length=len(graph)
    for i in range(length):
        if distribution[graph[i][0]]!=distribution[graph[i][1]]:
            sum=sum+graph[i][2]
    return sum

def get_qn_from_graph(graph):
    """
    Determines the maximum quantum number (qn) from a given quantum circuit graph.

        Args:
            graph (list of tuples): A list of tuples, where each tuple represents an edge in the graph.
                                 Each tuple contains two integers representing the quantum numbers of the nodes connected by the edge.

        Returns:
            int: The next integer after the maximum quantum number found in the graph, representing the next available quantum number.

    This function is intended for use within the pyQPanda package, which facilitates programming quantum computers using quantum circuits and gates.
    It operates on quantum circuit simulators or quantum cloud services.
    """
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
    """
    Perform quantum approximate optimization using a specified graph and Hamiltonian parameters.

        Args:
            graph (object): The origin graph on which the optimization is to be performed.
            gamma_ (list): The problem Hamiltonian parameters corresponding to the graph edges.
            beta_ (list): The driver Hamiltonian parameters.
            use_prob_run (bool): Flag to enable the use of probability run for optimization.
            use_quick_measure (bool): Flag to enable quick measurement instead of output probabilities when use_prob_run is True.
            multiProcessing (bool): Flag to enable multi-processing. Currently not implemented.
            shots_ (int): Number of times the quantum circuit is executed.
            dataType (str): Data type for the probability run output. Only applicable when use_prob_run is True.

        Returns:
            float: The negative of the weight of the optimal outcome, representing the minimized objective.

        Notes:
            The algorithm constructs a quantum circuit based on the provided graph and Hamiltonian parameters.
            It employs quantum gates and measurements to approximate the ground state of the Hamiltonian.
            The result is interpreted to find the outcome with the highest weight, which is then negated and returned.
    """
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
    """
    Constructs a partial application of the `qaoa_in_list` function with pre-defined `graph` and `shots` parameters.
    
    This function is intended for use within the `pyQPanda` package, which facilitates programming quantum computers using quantum circuits and gates.
    It can be executed on a quantum circuit simulator, a quantum virtual machine, or a quantum cloud service.
    
        Args:
            graph (Graph): The quantum graph representing the problem to be solved.
            shots (int): The number of times the quantum circuit should be executed to gather statistical data.
    
        Returns:
            Callable: A partially applied function ready to execute the QAOA algorithm on the specified graph and with the given number of shots.
    """

    return partial(qaoa_in_list,graph=graph,shots=shots)

def qaoa_in_list(
    arguments,
    graph,
    use_prob_run=True,
    multiProcessing=False,
    shots=100,
    dataType='list',    
):
    """
    Executes the Quantum Approximate Optimization Algorithm (QAOA) on a given graph with specified parameters.

        Args:
            arguments (list): A list of real numbers representing the parameters for the QAOA.
            graph (object): The graph on which the QAOA is to be executed, typically a quantum circuit or a set of constraints.
            use_prob_run (bool, optional): Whether to use the probability run mode. Defaults to True.
            multiProcessing (bool, optional): Whether to use multiple processing cores for the computation. Defaults to False.
            shots (int, optional): The number of times the QAOA is run to estimate the expected value. Defaults to 100.
            dataType (str, optional): The data type for the result, either 'list' or 'numpy' array. Defaults to 'list'.

        Returns:
            object: The result of the QAOA execution, which can be a list or a numpy array depending on the dataType.

    The function performs the QAOA by dividing the input arguments into two lists, `beta` and `gamma`, and then calls the
    `quantum_approximate_optimization_algorithm` function with these lists and the provided graph. The result is appended to a
    file named 'a.txt' and returned.
    """

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
    """
    Simulates quantum annealing on a given graph using the Quantum Approximate Optimization Algorithm (QAOA).

        Args:
            graph (object): A graph object representing the quantum system.
            step_ (int, optional): Number of steps in the QAOA process. Default is 1.
            shots_ (int, optional): Number of quantum samples to average over. Default is 1000.
            method (str, optional): Optimization method to use for finding the QAOA parameters. Default is "Nelder-Mead".

        Returns:
            result (object): The result of the optimization process, typically containing the optimized parameters.

    The function performs QAOA by iterating over a specified number of steps, initializing the parameters,
    and using the `binding` function to compute the objective function. It then applies an optimization method
    to find the optimal parameters that minimize the objective function.
    """

    gamma_=[]
    beta_=[]
    for i in range(step_):
        gamma_.append(0)
        beta_.append(0)    
    initial_guess=gamma_+beta_
    result = minimize(binding(graph,shots_), initial_guess, method=method)    
    return result



