from pyqpanda.Hamiltonian.PauliOperator.pyQPandaPauliOperator import PauliOperator
from pyqpanda.Algorithm.QuantumGradient.quantum_gradient import *
from pyqpanda import *
import copy
import numpy as np
from math import pi

def generate_edge_library(dimension):
    """
    Generates a list of edge tuples for a complete graph of the specified dimension.
    
    The function computes all possible edges for a graph where nodes are labeled from 0 to
    (dimension - 1). Each edge is represented as a tuple containing two integers (i, j),
    indicating a connection between nodes i and j. The generated edges form a complete graph,
    where every pair of distinct nodes is connected by exactly one edge.
    
        Args:
            dimension (int): The number of nodes in the graph, which must be a non-negative integer.
    
        Returns:
            list of tuples: A list containing tuples of edge connections.
    
        Usage:
            edges = generate_edge_library(4)
            print(edges)  # Output: [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    """
    edge_lib=[]
    for i in range(dimension):
        for j in range(i+1,dimension):
            edge_lib.append((i,j))
    return edge_lib

def generate_adjacent_matrix(dimension,n_edge):
    """
    Constructs an adjacency matrix for a graph with a specified dimension and number of edges.

    This function generates an adjacency matrix representing a graph. The matrix dimensions are determined by
    the 'dimension' parameter, and the number of edges is defined by 'n_edge'. The matrix elements are initialized
    to zero, and non-zero values are assigned to represent existing edges with a random weight between 0 and 1.

        Args:
            dimension (int): The size of the graph, which corresponds to the number of vertices.
            n_edge (int): The total number of edges to include in the graph.

        Returns:
            numpy.ndarray: An adjacency matrix with shape (dimension, dimension), where non-zero values indicate
                         the presence of an edge between the corresponding vertices with a random weight.

        Note: 
            This function is intended for use within the pyQPanda package, which is designed for programming
            quantum computers using quantum circuits and gates, and may be executed on quantum simulators or
            quantum cloud services.
    """
    edge_lib=generate_edge_library(dimension)
    adjacent_matrix=np.zeros((dimension,dimension))
    number=0
    while number!=n_edge:
        edge_number=np.random.randint(len(edge_lib))
        if adjacent_matrix[edge_lib[edge_number][0]][edge_lib[edge_number][1]]==0:
            adjacent_matrix[edge_lib[edge_number][0]][edge_lib[edge_number][1]]=np.random.uniform(0,1)
            number+=1
    return adjacent_matrix

def generate_maxcut_problem_Hamiltonian(adjacent_matrix):
    """
    Constructs a Hamiltonian representation for the Max-Cut problem suitable for quantum algorithms within the pyQPanda framework.

    This function takes an adjacency matrix representing an undirected graph and generates a Hamiltonian object
    that encapsulates the Max-Cut problem. The Hamiltonian is represented using Pauli operators, where the strength
    of the interaction between nodes is half the value of the corresponding element in the adjacency matrix.

        Args:
            adjacent_matrix (list of list of float): A 2D list representing the adjacency matrix of the graph.

        Returns:
            PauliOperator: An instance of the PauliOperator class that represents the Hamiltonian for the Max-Cut problem.
    """
    str_dict={}
    dimension=len(adjacent_matrix)
    for i in range(dimension):
        for j in range(dimension):
            if adjacent_matrix[i][j]>0:
                key='Z%d Z%d'%(i,j)
                str_dict[key]=adjacent_matrix[i][j]/2
    hamiltonian=PauliOperator(str_dict)
    return hamiltonian

def generate_drive_hamiltonian(qubit_number):
    """
    Constructs a drive Hamiltonian for a quantum circuit with a specified number of qubits.
    
    The Hamiltonian is represented as a Pauli operator, where each qubit is associated with an
    'X' operator with the corresponding qubit index. The 'X' operator is set to have a coefficient of
    1 for each qubit. This function is intended for use within the pyQPanda package for simulating
    quantum circuits and quantum algorithms.
    
        Args:
            qubit_number (int): The number of qubits for which the drive Hamiltonian is to be generated.
    
        Returns:
            PauliOperator: A Pauli operator representing the drive Hamiltonian with the specified number
                           of qubits.
    """
    str_dict={}
    for i in range(qubit_number):
        key='X%d'%i
        str_dict[key]=1
    drive_hamiltonian=PauliOperator(str_dict)
    return drive_hamiltonian    
    
def max_cut(adjacent_matrix):
    """
    Computes the maximum cut in a given adjacency matrix for a quantum algorithm within the pyQPanda framework.
    
    This function performs a maximum cut operation on a weighted graph represented by an adjacency matrix.
    It returns a dictionary containing the maximum cut sum, a list of indices representing the cut, and
    the sum of all possible cuts.
    
        Args:
            adjacent_matrix (list of list of float): A 2D list representing the adjacency matrix of the graph.
    
        Returns:
        dict: A dictionary with the following keys:
                'sum': The maximum cut sum.
                'target_list': A list of indices representing the vertices on one side of the cut.
                'all_cut': A list of sums representing the cut for all possible partitions.
                'all_cut_sum': The sum of all possible cut sums.
    
    The function iterates over all possible partitions of the graph, calculating the cut sum for each
    partition. It keeps track of the maximum cut sum and the corresponding partition indices.
    """
    dimension=len(adjacent_matrix)
    max_sum={}
    max_sum['sum']=0
    target_cut_list=[]
    all_cut=[]
    all_cut_sum=0
    for j in range(dimension):
        for k in range(dimension):
            all_cut_sum+=adjacent_matrix[j][k]
    for i in range(1<<dimension):
        sum=0
        for j in range(dimension):
            for k in range(dimension):
                if (i>>j)%2 != (i>>k)%2:
                    sum+=adjacent_matrix[j][k]
        all_cut.append(sum)
        if sum-max_sum['sum']>1e-6:
            target_cut_list.clear()
            target_cut_list.append(i)
            max_sum['sum']=sum
        elif abs(sum-max_sum['sum'])<1e-6:
            target_cut_list.append(i)
    max_sum['target_list']=target_cut_list
    max_sum['all_cut']=all_cut
    max_sum['all_cut_sum']=all_cut_sum
    return max_sum  

def target_list_to_str_list(target_list,dimension):
    """
    Converts a list of integers to a list of binary strings, each padded to a specified dimension.

    This function is designed to facilitate the creation of binary representations for integers within
    a specified dimension, which is particularly useful in quantum computing applications for constructing
    quantum circuits.

        Args:
            target_list (list of int): The list of integers to be converted.
            dimension (int): The length to which each binary string should be padded.

        Returns:
            list of str: A list containing the binary string representations of the integers in `target_list`,
                       with each string padded to the length specified by `dimension`.
    """
    target_str_list=[]
    for i in target_list:
        temp_str=bin(i)[2:]
        while len(temp_str)<dimension:
            temp_str='0'+temp_str
        target_str_list.append(temp_str)
    return target_str_list

def generate_graph(dimension,n_edge):
    """
    Generates a graph dictionary containing various properties of the graph constructed for the Max-Cut problem.

        Args:
            dimension (int): The number of vertices in the graph.
            n_edge (int): The number of edges in the graph.

        Returns:
            dict: A dictionary with the following keys:
                'adjacent_matrix': The adjacency matrix of the graph.
                'max_value': The maximum cut value.
                'max_cut_str': The sequence of vertices in the maximum cut.
                'n_vertex': The number of vertices.
                'n_edge': The number of edges.
                'all_cut': A list of all cut values.
                'all_cut_sum': The sum of all cut values.

    The function utilizes an adjacency matrix to represent the graph, where '1' indicates an edge between vertices and '0' indicates no edge. It computes the maximum cut and converts the target list of vertices into a string representation.
    """
    graph_dict={}
    adjacent_matrix=generate_adjacent_matrix(dimension=dimension,n_edge=n_edge)
    max_sum=max_cut(adjacent_matrix)
    max_str=target_list_to_str_list(target_list=max_sum['target_list'],dimension=dimension)
    graph_dict['adjacent_matrix']=adjacent_matrix
    graph_dict['max_value']=max_sum['sum']
    graph_dict['max_cut_str']=max_str
    graph_dict['n_vertex']=dimension
    graph_dict['n_edge']=n_edge
    graph_dict['all_cut']=max_sum['all_cut']
    graph_dict['all_cut_sum']=max_sum['all_cut_sum']
    return graph_dict

# def qaoa_maxcut(graph,step,optimize_times):
    
#     qubit_number=graph['n_vertex']
#     target_value=-graph['max_value']
#     target_str_list=graph['max_cut_str']
#     all_cut_list=graph['all_cut']
#     all_cut_list_new=all_cut_list.copy()
#     for i in range(len(all_cut_list)):
#         all_cut_list[i]=-all_cut_list_new[i]
#     del all_cut_list_new
#     adjacent_matrix=graph['adjacent_matrix']
#     hp=generate_maxcut_problem_Hamiltonian(adjacent_matrix=adjacent_matrix)
#     hp=hp+PauliOperator({'':-graph['all_cut_sum']/2})
#     hd=generate_drive_hamiltonian(qubit_number)
#     init(backend_type=pyQPanda.QMachineType.GPU)
#     qlist=qAlloc_many(qubit_number)
#     gamma=np.ones(step)*(-0.01)
#     beta=np.ones(step)*0.01
#     qqat=qaoa(qubitnumber=qubit_number,
#               step=step,
#               gamma=gamma,
#               beta=beta,
#               Hp=hp,
#               Hd=hd,
#               all_cut_value_list=all_cut_list,
#               target_value=target_value,
#               target_str_list=target_str_list)
#     result=qqat.momentum_optimizer(qubit_list=qlist,
#                         max_times=optimize_times, 
#                         threshold_value=0.05,
#                         learning_rate=0.02,
#                         momentum=0.9,
#                         method=1,
#                         delta=1e-6,
#                         is_test=True)
#     finalize()
#     return result

def qaoa_maxcut_gradient_threshold(graph,step,threshold_value=0.05,optimize_times=300,use_GPU=False):
    """
    Solve the maximum cut problem on a quantum circuit using the Quantum Approximate Optimization Algorithm (QAOA)
    with a gradient threshold approach. This function is designed to be used within the pyQPanda package,
    which facilitates quantum computing with quantum circuits and gates, running on quantum simulators or cloud services.

        Args:
            graph (dict): A dictionary containing the graph structure and necessary parameters for the QAOA algorithm.
                    'n_vertex': Number of vertices in the graph.
                    'max_value': Maximum value for the cut problem.
                    'max_cut_str': String representation of the maximum cut.
                    'all_cut': List of all possible cuts in the graph.
                    'adjacent_matrix': Adjacency matrix of the graph.
                    'all_cut_sum': Sum of all cuts in the graph.
            step (int): The number of layers in the quantum circuit.
            threshold_value (float, optional): The threshold for convergence during optimization. Defaults to 0.05.
            optimize_times (int, optional): The maximum number of optimization iterations. Defaults to 300.
            use_GPU (bool, optional): Flag to indicate whether to use GPU acceleration. Defaults to False.

        Returns:
            object: The result object containing the optimized solution and other relevant information.

        Notes:
                The function initializes the quantum machine according to the use_GPU flag.
                It constructs the Hamiltonian for the QAOA algorithm and applies a driving Hamiltonian.
                The learning rate and optimize_times are adjusted based on the number of qubits and the step size.
                The QAOA algorithm is executed with momentum optimization and a specified threshold for convergence.
    """
    qubit_number=graph['n_vertex']
    target_value=-graph['max_value']
    target_str_list=graph['max_cut_str']
    all_cut_list=graph['all_cut']
    all_cut_list_new=all_cut_list.copy()
    for i in range(len(all_cut_list)):
        all_cut_list_new[i]=-all_cut_list[i]
    
    adjacent_matrix=graph['adjacent_matrix']
    hp=generate_maxcut_problem_Hamiltonian(adjacent_matrix=adjacent_matrix)
    hp=hp+PauliOperator({'':-graph['all_cut_sum']/2})
    hd=generate_drive_hamiltonian(qubit_number)
    if use_GPU:
        machine_type=QMachineType.GPU
    else:
        machine_type=QMachineType.CPU
    init(machine_type)
    qlist=qAlloc_many(qubit_number)
    #gamma=(1-2*np.random.random_sample(step))*pi
    #beta=np.random.random_sample(step)*pi/2
    gamma=np.ones(step)*(-0.01)
    beta=np.ones(step)*0.01
    if qubit_number<=20:
        if step<15:
            learning_rate=0.01
        elif step<25:
            learning_rate=0.005
        else:
            learning_rate=0.002
            optimize_times=400
    elif qubit_number<25:
        if step<15:
            learning_rate=0.005
        elif step<25:
            learning_rate=0.002
            optimize_times=400
        else:
            learning_rate=0.001
            optimize_times=400
    elif qubit_number<31:
        learning_rate=0.001
        optimize_times=500
    qqat=qaoa(qubitnumber=qubit_number, 
              step=step,
              gamma=gamma,
              beta=beta,
              Hp=hp,
              Hd=hd,
              all_cut_value_list=all_cut_list_new,
              target_value=target_value,
              target_str_list=target_str_list)
    result=qqat.momentum_optimizer(qubit_list=qlist,
                        max_times=optimize_times, 
                        threshold_value=threshold_value,
                        learning_rate=learning_rate,
                        momentum=0.9,
                        method=1,
                        delta=1e-6,
                        is_test=True)
    finalize()
    return result
# def regenerate_graph(adjacent_matrix):
#     graph_dict={}
#     dimension=len(adjacent_matrix)
#     n_edge=0
#     for i in adjacent_matrix:
#         for j in i:
#             if j>0:
#                 n_edge+=1
#     adjacent_matrix=adjacent_matrix
#     max_sum=max_cut(adjacent_matrix)
#     max_str=target_list_to_str_list(target_list=max_sum['target_list'],dimension=dimension)
#     graph_dict['adjacent_matrix']=adjacent_matrix
#     graph_dict['max_value']=max_sum['sum']
#     graph_dict['max_cut_str']=max_str
#     graph_dict['n_vertex']=dimension
#     graph_dict['n_edge']=n_edge
#     graph_dict['all_cut']=max_sum['all_cut']
#     graph_dict['all_cut_sum']=max_sum['all_cut_sum']
#     return graph_dict

# def add_step_range_test(adjacent_matrix,step_range,optimize_times=300,use_GPU=False):
#     f = open('add_step_range_test.log', 'a')
#     graph=regenerate_graph(adjacent_matrix=adjacent_matrix)
#     adjacent_matrix=graph['adjacent_matrix']
#     print('graph=',adjacent_matrix)
#     f.write(adjacent_matrix.__str__())
#     f.write('\n\n')
#     data=[]
#     for j in range(len(step_range)):
#             result=qaoa_maxcut_gradient_threshold(graph=graph,
#                                                   step=step_range[j],
#                                                   threshold_value=0.05,
#                                                   optimize_times=optimize_times,
#                                                   use_GPU=use_GPU)
#             print('step= ',step_range[j])
#             print(result)
#             data.append(result['target probability'])
#             f.write('step= %d\n'%step_range[j])
#             f.writelines(['result=',result.__str__(),'\n\n'])
#     f.write('data\n')
#     f.write(data.__str__())
#     f.close()
#     return data

# def step_probability(dimension,n_edge,step_range,optimize_times=200,times=10,use_GPU=False):
#     f = open('step_probability_v_%de_%d.log'%(dimension,n_edge), 'a')
#     f.write('{}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
#     data=np.zeros((times,len(step_range)))
#     f.write('vertex number:%d\n'%dimension)
#     f.write('edge number:%d\n'%n_edge)
#     for i in range(times):
#         f.write('graph number:%d\n'%i)
#         graph=generate_graph(dimension=dimension,n_edge=n_edge)
#         adjacent_matrix=graph['adjacent_matrix'].tolist()
#         print('graph=',adjacent_matrix)
#         f.write(adjacent_matrix.__str__())
#         f.write('\n\n')
       
#         for j in range(len(step_range)):
#             result=qaoa_maxcut_gradient_threshold(graph=graph,
#                                                   step=step_range[j],
#                                                   threshold_value=0.05,
#                                                   optimize_times=optimize_times,
#                                                   use_GPU=use_GPU)
#             print('step= ',step_range[j])
#             print(result)
#             data[i][j]=result['target probability']
#             f.write('step= %d\n'%step_range[j])
#             f.writelines(['result=',result.__str__(),'\n\n'])
#     data_list=data.tolist()
#     f.write('data\n')
#     f.write(data_list.__str__())
#     f.close()
#     return data_list
                  
# def qaoa_maxcut_all_connected(dimension,step_range,optimize_times=200,learning_rate=0.005,use_GPU=False):
    
#     f = open('v_%d.log'%(dimension), 'a')
#     f.write('{}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
#     n_edge=dimension*(dimension-1)/2
#     graph=generate_graph(dimension=dimension,n_edge=n_edge)
#     print('graph=',graph['adjacent_matrix'])
#     f.write('graph=')
#     adjacent_matrix=graph['adjacent_matrix'].tolist()
#     f.write(adjacent_matrix.__str__())
#     f.write('\n\n')
#     probability=[]
#     for i in step_range:
#         result=qaoa_maxcut_gradient_threshold(graph=graph,step=i,threshold_value=0.05,optimize_times=optimize_times,use_GPU=use_GPU)
#         print(i)
#         print(result)
#         probability.append(result['target probability'])
#         f.writelines(['step=','%d\n'%i])
#         f.writelines(['result=',result.__str__(),'\n\n'])
#     f.close()
#     return probability

# def scan(dimension,edge_range,step_range,optimize_times=300,learning_rate=0.01,sample_times=10,use_GPU=False):
#     f = open('scan_v_%d.log'%(dimension), 'a')
#     data=[]
#     for n_edge in edge_range:
#         result=step_probability(dimension=dimension,
#             n_edge=n_edge,
#             step_range=step_range,
#             optimize_times=300,
#             times=sample_times,
#             use_GPU=use_GPU)
#         data.append(result)
#         f.write(result.__str__())
#         f.write('\n')
#     f.write("data=")
#     f.write(data.__str__())
#     f.close()
#     return data