from pyqpanda.Hamiltonian.PauliOperator.pyQPandaPauliOperator import PauliOperator
from pyqpanda.Algorithm.QuantumGradient.quantum_gradient import *
from pyqpanda import *
import copy
import numpy as np
from math import pi

def generate_edge_library(dimension):
    edge_lib=[]
    for i in range(dimension):
        for j in range(i+1,dimension):
            edge_lib.append((i,j))
    return edge_lib

def generate_adjacent_matrix(dimension,n_edge):
    '''
    generate adjacent matrix of a graph
    '''
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
    str_dict={}
    for i in range(qubit_number):
        key='X%d'%i
        str_dict[key]=1
    drive_hamiltonian=PauliOperator(str_dict)
    return drive_hamiltonian    
    
def max_cut(adjacent_matrix):
    '''
    to modify
    '''
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
    target_str_list=[]
    for i in target_list:
        temp_str=bin(i)[2:]
        while len(temp_str)<dimension:
            temp_str='0'+temp_str
        target_str_list.append(temp_str)
    return target_str_list

def generate_graph(dimension,n_edge):
    '''
    max_cut_str's sequence: v0v1v2...vd
    '''
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