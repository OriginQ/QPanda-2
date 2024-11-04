from pyqpanda import *
import networkx as nx
import matplotlib.pyplot as plt

gate_type_dict = {2: 'X', 3: 'Y', 4: 'Z', 5: 'X1', 6: 'Y1', 7: 'Z1', 8: 'P', 9: 'H', 10: 'T',
                  11: 'S', 12: 'RX', 13: 'RY', 14: 'RZ', 16: 'U1', 17: 'U2', 18: 'U3', 19: 'U4',
                  21: 'CNOT', 22: 'CZ', 28: 'CR', 39: 'I'}  # 量子门编号和量子门名称对应字典


def build_Qprog(prog):
    """
    Constructs a quantum program with a series of quantum gates on specified qubits.

    Parameters:
        prog (QProg): The quantum program object to be populated with gates.

    Returns:
        QProg: The fully constructed quantum program containing a sequence of gates:
               - H(qubits[0]) - Applies Hadamard gate to the first qubit.
               - X(qubits[2]) - Applies Pauli-X gate to the third qubit.
               - CNOT(qubits[0], qubits[2]) - Applies CNOT gate controlling the first qubit and targeting the third.
               - P(qubits[1], 1) - Applies a phase shift gate by π with respect to the second qubit.
               - CNOT(qubits[1], qubits[2]) - Applies CNOT gate controlling the second qubit and targeting the third.
               - CNOT(qubits[1], qubits[2]) - Applies another CNOT gate controlling the second qubit and targeting the third.

    This function is intended for use within the pyQPanda package, which is designed for quantum circuit programming,
    simulation, and interaction with quantum virtual machines or quantum cloud services.
    """
    prog = QProg()
    prog << H(qubits[0]) << X(qubits[2]) << CNOT(qubits[0], qubits[2]) \
         << P(qubits[1], 1) << CNOT(qubits[1], qubits[2]) << CNOT(qubits[1], qubits[2])

    return prog


def get_vertex_num(vertex_set):
    """
    Extracts the unique mid identifiers from a set of vertices in the quantum circuit.

    Parameters:
    vertex_set (set): A set containing vertices of the quantum circuit, each with an attribute 'm_id'.

    Returns:
    list: A list of unique mid identifiers extracted from the vertices.

    This function is intended for use within the quantum computing framework pyQPanda, specifically within the test_DAG module.
    It is designed to facilitate operations on quantum circuits, aiding in the identification and manipulation of vertices.
    """
    mid = []
    for ver in vertex_set:
        mid.append(ver.m_id)

    return mid


def get_gate_type(vertex_set):
    """
    Extracts the gate types from a set of quantum vertices and returns them as a list of integers.

    Parameters:
    vertex_set (set): A set containing quantum vertices, each with an attribute `m_type` that
                      represents the type of the gate.

    Returns:
    list: A list of integers corresponding to the gate types extracted from the input set.

    This function is intended for use within the pyQPanda package, which facilitates programming
    quantum computers and running quantum circuits on simulators or quantum cloud services.
    """
    mtype = []
    for ver in vertex_set:
        mtype.append(int(ver.m_type))

    return mtype


def id_gate_num(mid, mtype):
    """
    Maps the identifiers from the middle list to their corresponding gate types using a predefined dictionary.

    Parameters:
    mid (list): A list of identifiers corresponding to quantum gates.
    mtype (list): A list of integers representing the gate types.

    Returns:
    dict: A dictionary where each identifier from the `mid` list is associated with its corresponding gate type
          as an integer, then mapped to its string representation using the `gate_type_dict`.

    This function is intended for use within the pyQPanda package, specifically within the test_DAG module,
    and facilitates the translation of gate identifiers to their respective gate types in quantum circuits.
    """
    id_gate = {}
    for k, v in zip(mid, mtype):
        id_gate[k] = v

    for key in id_gate:
        id_gate[key] = gate_type_dict[id_gate[key]]
    return id_gate


def DAG_edges(DAG, get_edges):
    """
    Add edges to a Directed Acyclic Graph (DAG) based on a sequence of edge objects.

    The function iterates over a collection of edge objects and adds each edge to the specified DAG. The edges are provided by the `get_edges` function, which is expected to yield `Edge` instances with attributes `m_from` and `m_to` representing the source and destination nodes, respectively.

    Parameters:
    - DAG: An instance of a directed graph data structure supporting edge addition.
    - get_edges: A callable that generates `Edge` objects. Each `Edge` object should have attributes `m_from` and `m_to`.

    Returns:
    - DAG: The modified DAG with the edges added.

    Example Usage:
        >>> from pyQPanda.test_DAG import DAG_edges
        >>> from pyQPanda.test_DAG import Edge
        >>> DAG = ...
        >>> edges = (Edge(1, 2), Edge(2, 3), Edge(3, 1))
        >>> DAG_edges(DAG, edges)
    """
    for edg in get_edges:
        DAG.add_edge(edg.m_from, edg.m_to)

    return DAG


def DAG_draw(DAG):
    """
    Draws a directed acyclic graph (DAG) using topological layout and assigns a layer to each node.

    Parameters:
    DAG (networkx.DiGraph): A directed acyclic graph (DAG) to be visualized.

    This function assigns a unique layer to each node in the DAG based on its topological order. It then employs a
    multipartite layout to arrange the nodes visually. The resulting graph is displayed with a title and saved as
    'my_networkx_plot.png'. This utility is particularly useful for visualizing the structure of quantum circuits
    within the pyQPanda package, which is designed for programming quantum computers.

    Returns:
    None: This function does not return any value but saves the plot as an image file.
    """

    for layer, nodes in enumerate(nx.topological_generations(DAG)):
        for node in nodes:
            DAG.nodes[node]["layer"] = layer

    pos = nx.multipartite_layout(DAG, subset_key="layer")
    fig, ax = plt.subplots()
    nx.draw_networkx(DAG, pos=pos, ax=ax)
    ax.set_title("DAG layout in topological order")
    fig.tight_layout()
    plt.savefig('my_networkx_plot.png')


if __name__ == "__main__":

    qvm = CPUQVM()
    qvm.init_qvm()
    qnum = 4
    qubits = qvm.qAlloc_many(qnum)
    DAG = nx.DiGraph()
    prog = build_Qprog(qnum)
    print(prog)
    vertex_set = prog_to_dag(prog).get_vertex_set()
    get_edges = prog_to_dag(prog).get_edges()
    gate_id = get_vertex_num(vertex_set)
    gate_type = get_gate_type(vertex_set)
    id_gate = id_gate_num(gate_id, gate_type)
    print(id_gate)
    DAG = DAG_edges(DAG, get_edges)
    DAG_draw(DAG)

