'''Request realchip topology from origin cloud API and auto select physical qubits 
with high fidelity and fit our circuit topology.
'''
import requests
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def preprocess_qubits(content_dict):
    topology = []
    for qubit in content_dict.keys():
        qubit_str = qubit.split("_")
        qubit_num = list(map(int, qubit_str))
        topology.append(qubit_num)
    return topology

def preprocess_nan(content_dict, keyword):
    new_list = []
    for term in content_dict.values():
        if not term[keyword]:
            term[keyword] = '0'
        term_num = float(term[keyword])
        new_list.append(term_num)
    return new_list


def preprocess_coordinate(content_dict, keyword):
    coordinate_dict = {}
    for key, value in content_dict.items():
        site_data = value[keyword]
        coordinate_dict[key] = (site_data['x'],site_data['y'])
    return coordinate_dict


def preprocess_F0F1(content_dict):
    fidelity = []
    for term in content_dict.values():
        if not term["readFidelity"]:
            term_num = 0
        else:
            term_str = term["readFidelity"].split("/")
            term_num = list(map(float, term_str))
        fidelity.append(term_num)
    return fidelity


def preprocess_readout(content_dict):
    fidelity = []
    for term in content_dict.values():
        if not term['ReadoutFidelity']:
            term_num = 0
        else:    
            term_num = float(term['ReadoutFidelity'])
        fidelity.append(term_num)
    return fidelity


def filter_single_qubits(single_gate):
    """Using list comprehension to filter out qubits with:
    1.gate fidelity/T1/T2 is 0

    Args:
        single_gate (dict): single gate info from HTTP request. \
            keys(str):qubit number; values(dict): info about this qubit.

    Returns:
        (list, list): list of filtered qubits, list of filtered gate fidelity
    """

    single = list(map(int, single_gate.keys()))
    gate_fid = preprocess_nan(single_gate, "averageFidelity")
    t1 = preprocess_nan(single_gate, "T1")
    t2 = preprocess_nan(single_gate, "T2")
    fid_f0f1 = preprocess_F0F1(single_gate)
    fid_readout = preprocess_readout(single_gate)
    
    zipped = zip(single, gate_fid, t1, t2)
    filtered_list = [term for term in zipped if all(term[1:])]
    
    f_qubits, f_fidelity, _, _,= zip(*filtered_list)
    return list(f_qubits), list(f_fidelity)
    
    
def filter_double_qubits(double_gate, single_qubits):
    """Using list comprehension to filter out qubit pair with:
    1.double gate fidelity is 0
    2.qubit pair is not in filtered single qubits.

    Args:
        double_gate (dict): double gate info from HTTP request. \
            keys(str):qubit number pair; values(dict): info about this qubit pair.
        single_qubits (list): filtered single qubits.

    Returns:
        (list, list): list of filtered qubit pair, list of filtered double gate fidelity.
    """

    # Using list comprehension to filter out qubits with fidelity is 0
    double = preprocess_qubits(double_gate)
    double_fidelity = preprocess_nan(double_gate, "fidelity")
    filtered_list = [(q, f) for q, f in zip(double, double_fidelity) if f != 0]
    final_list = []
    for term in filtered_list:
        if term[0][0] in single_qubits and term[0][1] in single_qubits:
            final_list.append(term)
    
    if not final_list:
        raise Exception(f"No available qubit pair is in the qubit list {single_qubits}")
    
    f_qubits, f_fidelity = zip(*final_list)
    return list(f_qubits), list(f_fidelity)


def dict_reform(double, double_fidelity, single, single_fidelity):
    double_str = map(str, double)
    double_dict = dict(zip(double_str, double_fidelity))
    single_dict = dict(sorted(zip(single, single_fidelity)))
    return double_dict, single_dict
    
    
def get_realtime_topology(chip_id: int,
                          verbose=False):
    """get realtime chip topology from origin qcloud.

    Args:
        verbose (bool, optional): If set to False output available topology and qubits. Otherwise output available topology, double fidelity, available qubits, single fidelity. Defaults to False.
        
    Raises:
        Exception: If the status code of HTTP request is not 200, then raise exception.
        
    Returns:
        (list, list): topology and fidelity of real chip.
    """

    url = 'https://console.originqc.com.cn/api/taskApi/getFullConfig.json?chipId={}'.format(chip_id)
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception('Cannot make HTTP request。The status code is {}'.format(response.status_code))
    else:
        # single gate info
        single_gate = response.json()["obj"]["adjJSON"]
        single, single_fidelity = filter_single_qubits(single_gate)
        qubit_coordinate = preprocess_coordinate(single_gate, "site")
        
        # double gate info
        double_gate = response.json()["obj"]["gateJSON"]  
        double, double_fidelity = filter_double_qubits(double_gate, single)
        
        if verbose:
            return double, double_fidelity, single, single_fidelity, qubit_coordinate
        else:
            return double, single
            
            
def find_subgraphs(circuit_topo, chip_topo):
    """find the subgraph isomorphism using networkx

    Args:
        circuit_topo (list(list)): nested list for circuit topology. e.g.: [[0, 1], [0, 3], [2, 3], [1, 2], [1, 5]]
        chip_topo (list(list)): nested list for chip topology. Get from function get_realtime_topology(dict_form=False)

    Returns:
        dict: key(logical qubits of your circuit); value(physical qubits on the chip)
    """
    circuit_topo_graph = nx.from_edgelist(circuit_topo) # own circuit topology
    chip_topo_graph = nx.from_edgelist(chip_topo)
    gm = nx.algorithms.isomorphism.GraphMatcher(chip_topo_graph, circuit_topo_graph)
    subgraphs = gm.subgraph_isomorphisms_iter()
    
    # exchange the keys and values & sort the keys
    new_subgraphs = []
    for term in subgraphs:
        exchange_term = zip(term.values(), term.keys())
        new_subgraphs.append(dict(sorted(exchange_term)))

    # output info
    print("There are {} subgraphs found.".format(len(new_subgraphs)))
    return new_subgraphs

def find_best_subgraph(circuit_topo, chip_topo_dict, subgraphs):
    """make analysis of subgraphs, find the subgraph with the high fidelity by 
    searching the subgraph with the highest minimum fidelity.

    Args:
        circuit_topo (list(list)): nested list for circuit topology.
        chip_topo_dict (dict): topology and fidelity of real chip.
        subgraphs (list(dict)): subgraphs that fit our citcuit topology. key(logical qubits of your circuit); value(physical qubits on the chip)

    Returns:
        list: minimum fidelity(float), new circuit fidelity(list), subgraph(dict), new circuit topo(list(list)).
    """
    best_graph = [0]
    for subgraph in subgraphs:
        new_circuit_topo, new_circuit_fid = [], []
        for i, j in circuit_topo:
            topo_term = sorted([subgraph[i], subgraph[j]])
            topo_term_fid = chip_topo_dict[str(topo_term)]           
            new_circuit_topo.append(topo_term)  # new circuit topology in this subgraph
            new_circuit_fid.append(topo_term_fid) # new circuit fidelity in this subgraph
        
        min_fidelity = min(new_circuit_fid)
        if min_fidelity > best_graph[0]:
            best_graph = [min_fidelity, new_circuit_fid, subgraph, new_circuit_topo]
    return best_graph


from matplotlib.colors import Normalize
def show_chip_topology(chip_id : int):
    double, double_fidelity, single, single_fidelity, qubit_coordinate = get_realtime_topology(72,True)

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1) 
    plt.title(f"{chip_id} qubits chip topology", fontsize=16, color='white', pad=20)

    background_color = (70, 68, 82)
    normalized_background_color = tuple(c / 255 for c in background_color)

    fig.patch.set_facecolor(normalized_background_color)
    ax.set_facecolor(normalized_background_color)

    fidelity_values = double_fidelity + single_fidelity
    norm = mcolors.Normalize(vmin=min(fidelity_values), vmax=max(fidelity_values))
    cmap = plt.cm.Blues

    for i, connection in enumerate(double):
        x1, y1 = qubit_coordinate[str(connection[0])]
        x2, y2 = qubit_coordinate[str(connection[1])]
        color = cmap(norm(double_fidelity[i]))

        # Adjust coordinates to avoid crossing the qubits
        angle = np.arctan2(y2 - y1, x2 - x1)
        offset_x = np.cos(angle) * 25  # Adjust offset distance as needed
        offset_y = np.sin(angle) * 25

        ax.plot([x1 + offset_x, x2 - offset_x], [y1 + offset_y, y2 - offset_y], color=color, linewidth=2 + 5 * (double_fidelity[i] - 0.9))

    def plot_qubits(ax, single, single_fidelity, qubit_coordinate):
        norm = Normalize(vmin=min(single_fidelity), vmax=max(single_fidelity))
        cmap = plt.get_cmap('Blues')

        for node in qubit_coordinate.keys():
            x, y = qubit_coordinate[node]
            if int(node) in single:
                index = single.index(int(node))
                fidelity = single_fidelity[index]
            else:
                fidelity = 0.0
            color = cmap(norm(fidelity))
          
            # Draw a larger circle without edge
            ax.scatter(x, y, s=500, color=color, edgecolor='none')
            
            # Draw the number inside the circle with Consolas font
            ax.text(x, y, str(node), color='black', ha='center', va='center', fontsize=12, fontname='Consolas')

    plot_qubits(ax, single, single_fidelity, qubit_coordinate)
    
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 650)
    ax.set_ylim(0, 1100)
    ax.invert_yaxis()

    cax = fig.add_axes([0.25, 0.05, 0.5, 0.02])  # 参数分别是: [左边缘, 下边缘, 宽度, 高度]
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
    cbar.set_label('Fidelity', labelpad=6)
    cbar.ax.tick_params(labelsize=10)
    cbar.outline.set_visible(False)
    plt.show()

if __name__ == "__main__":

    show_chip_topology(72)
    pass
    
    # circuit_topo = [[0, 1], [0, 3], [2, 3], [1, 2], [1, 5]] # your own circuit topology
    # print(available_qubits)