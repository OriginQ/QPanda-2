import pyqpanda.pyQPanda as pq
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt

g_max_connect_degree = 4

class InitQMachine:
    def __init__(self, machineType = pq.QMachineType.CPU):
        self.m_machine = pq.init_quantum_machine(machineType)

    def __del__(self):
        pq.destroy_quantum_machine(self.m_machine)

def draw_graph(adjacent_matrix):
    G = nx.Graph()
    size = np.shape(adjacent_matrix)
    point = np.arange(0,size[1]).tolist()
    G.add_nodes_from(point)
    total_weight = 0
    edglist=[]
    for i in range(size[1]):
        for j in range(size[1]):
            if 0 != adjacent_matrix[j][i]:
                G.add_edge(i, j, weight=adjacent_matrix[j][i])
                total_weight += adjacent_matrix[j][i]
    # G=nx.Graph(edglist)

    pos = nx.circular_layout(G)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_nodes(G,pos, nodelist=point, node_color="r")
    nx.draw_networkx_edges(G,pos,width=[float(d['weight']*20.0/total_weight+1.0) for (u,v,d) in G.edges(data=True)])
    nx.draw_networkx_labels(G,pos)
    nx.draw_networkx_edge_labels(G,pos,label_pos = 0.2,edge_labels = labels)
    plt.show()

def test_grover_cir_qubit_topology():
    init_machine = InitQMachine()
    machine = init_machine.m_machine
    x = machine.cAlloc()

    data=[0, 6, 7, 9, 4, 5, 13, 6]
    # data=[50, 6, 16, 19]
    measure_qubits = pq.QVec()
    grover_cir = pq.Grover(data, x==6, machine, measure_qubits, 1)
    # print(grover_cir)

    topo_matrix = pq.get_double_gate_block_topology(grover_cir, machine)
    print(topo_matrix)
    print('----------------------------')
    draw_graph(topo_matrix)

    #method 1
    # topo_matrix = pq.del_weak_edge(topo_matrix)
    # print(topo_matrix)
    # print('----------------------------')
    #draw_graph(topo_matrix)

    #method 2
    sub_graph = pq.get_sub_graph(topo_matrix)
    optimizered_topo = pq.del_weak_edge2(topo_matrix, g_max_connect_degree, sub_graph)
    print(optimizered_topo[1])
    draw_graph(optimizered_topo[0])

    #method 3
    # sub_graph = pq.get_sub_graph(topo_matrix)
    # optimizered_topo = pq.del_weak_edge3(topo_matrix, sub_graph, 0.5, 0.5, 0.5)
    # print(optimizered_topo[1])
    # draw_graph(optimizered_topo[0])

    # for test
    # tmp_topo = pq.recover_edges(optimizered_topo[0], g_max_connect_degree, optimizered_topo[2])
    # draw_graph(tmp_topo)

    complex_points = pq.get_complex_points(optimizered_topo[0], g_max_connect_degree)

    complex_point_sub_graph = pq.split_complex_points(complex_points, g_max_connect_degree, optimizered_topo[0], 0)

    new_topology = pq.replace_complex_points(optimizered_topo[0], g_max_connect_degree, complex_point_sub_graph)
    draw_graph(new_topology)

    new_topology = pq.recover_edges(new_topology, g_max_connect_degree, optimizered_topo[2])
    draw_graph(new_topology)
    print(new_topology)

    b = pq.planarity_testing(new_topology)
    print(b)

    draw_graph(new_topology)

    # N = [[0, 3, 5, 1],[0, 0, 4, 3],[0, 0, 0, 5],[0, 0, 0, 0]]
    # draw_graph(topo_matrix)

    # result = pq.prob_run_dict(grover_cir, measure_qubits)
    # print(result)

def test_grover_cir_qubit_topology2():
    init_machine = InitQMachine()
    machine = init_machine.m_machine
    x = machine.cAlloc()

    data=[0, 6, 7, 9, 4, 5, 13, 6]
    measure_qubits = pq.QVec()
    grover_cir = pq.Grover(data, x==6, machine, measure_qubits, 1)

    topo_matrix = pq.get_circuit_optimal_topology(grover_cir, machine, g_max_connect_degree)

    b = pq.planarity_testing(topo_matrix)
    if b:
        print('planarity_testing okkkkkkkkkkk')

    draw_graph(topo_matrix)

if __name__=="__main__":
    #test_grover_cir_qubit_topology()
    test_grover_cir_qubit_topology2()
    print("Test over.")