from pyqpanda import *
import networkx as nx
import matplotlib.pyplot as plt

# The relationship between sequence numbers and quantum gates in pyqpanda
gate_type_dict = {2: 'X', 3: 'Y', 4: 'Z', 5: 'X1', 6: 'Y1', 7: 'Z1', 8: 'P', 9: 'H', 10: 'T',
                  11: 'S', 12: 'RX', 13: 'RY', 14: 'RZ', 16: 'U1', 17: 'U2', 18: 'U3', 19: 'U4',
                  21: 'CNOT', 22: 'CZ', 28: 'CR', 39: 'I'}  # 量子门编号和量子门名称对应字典


# Bulid a test circuit
def build_Qprog(prog: QProg) -> QProg:

    """
    Constructs a quantum program by applying Hadamard, X, and CNOT gates to specified qubits.

    Parameters
    ----------
    prog : QProg
        The initial quantum program to which gates are to be added.

    Returns
    -------
    QProg
        A new quantum program with the Hadamard, X, and CNOT gates applied to the specified qubits.

    Notes
    -----
    This function modifies the input quantum program in-place and does not return a copy. The gates are
    applied to qubits 0 and 2 in the following order: Hadamard on qubit 0, X on qubit 2, and a CNOT from
    qubit 0 to qubit 2. This operation is intended for use within the quantum circuit simulator or
    quantum cloud service provided by the pyQPanda package.
    """

    prog = QProg()
    prog << H(qubits[0]) << X(qubits[2]) << CNOT(qubits[0], qubits[2])
    return prog


# Get the gate id from circuit

def get_vertex_num(vertex_set: list) -> list:

    """
    Extracts the gate IDs from a set of quantum program vertices.

    Parameters
    ----------
    vertex_set : list of pyqpanda.QProgDAGVertex
                 A collection of vertices representing quantum program elements.

    Returns
    -------
    list of int
        A list containing the unique gate IDs corresponding to the vertices.

    Usage
    -----
    This function is intended to be used after converting a quantum program to its DAG representation using `prog_to_dag`.
    It returns a list of gate IDs that can be used for further analysis or manipulation within the quantum circuit framework.
    """

    mid = []
    for ver in vertex_set:
        mid.append(ver.m_id)

    return mid


# Get the gate type from circuit

def get_gate_type(vertex_set: list) -> list:
    """
    Extracts and returns the gate types from a list of vertices in a Quantum Programming DAG.

    Parameters
    ----------
    vertex_set : list of pyqpanda.QProgDAGVertex
                 A collection of vertices resulting from the conversion of a quantum program
                 to a Directed Acyclic Graph (DAG) representation.

    Returns
    -------
    list of int
        A list containing the numerical gate types associated with each vertex in the DAG.
    """

    mtype = []

    for ver in vertex_set:
        mtype.append(int(ver.m_type))

    return mtype


# Create a dictionary corresponding to the quantum gate id and the quantum gate
def id_gate_num(mid: list, mtype: list) -> dict:
    """
    Maps gate IDs to their corresponding types in a quantum circuit's Directed Acyclic Graph (DAG).

    Parameters
    ----------
    mid : list[int]
        List of gate IDs in the quantum circuit DAG.
    mtype : list[int]
        List of integer types corresponding to the gates in the quantum circuit DAG.

    Returns
    -------
    dict[int, int]
        A dictionary containing the gate IDs as keys and their respective types as values.

    The function is designed to convert raw gate IDs and types into a more readable format
    based on a predefined dictionary mapping gate types to their respective integers.
    This conversion is essential for interpreting and visualizing quantum circuits within
    the pyQPanda framework, which simulates quantum circuits and gates on quantum virtual
    machines or quantum cloud services.
    """

    id_gate = {}
    for k, v in zip(mid, mtype):
        id_gate[k] = v

    for key in id_gate:
        id_gate[key] = gate_type_dict[id_gate[key]]
    return id_gate


# Add the edge into DAG
def DAG_edges(DAG: nx.DiGraph, get_edges: list) -> nx.DiGraph:

    """
    Adds edges to a directed acyclic graph (DAG) based on a list of edge objects.

    Parameters
    ----------
    DAG : networkx.classes.digraph.DiGraph
        The directed acyclic graph to which edges will be added.
    get_edges : list
        A list of edge objects containing the source and target nodes for each edge.

    Returns
    -------
    nx.DiGraph
        The modified DAG with the specified edges added.

    Notes
    -----
    This function is designed to be used within the pyQPanda package, which facilitates programming quantum computers using quantum circuits and gates. It operates on the quantum circuit simulator or quantum cloud service.
    The function is located in the 'DAG_lib.py' module under the 'pyQPanda' package directory structure.
    """
    for edg in get_edges:
        DAG.add_edge(edg.m_from, edg.m_to)

    return DAG


# Draw the DAG
def DAG_draw(DAG: nx.DiGraph):

    """
    Visualizes a Directed Acyclic Graph (DAG) using a topological layout.

    The function assigns a layer attribute to each node based on its topological
    order, applies a multipartite layout to arrange nodes in these layers, and
    then draws the graph using the `networkx` library.

    Parameters
    ----------
    DAG : networkx.classes.digraph.DiGraph
        The directed acyclic graph to be visualized.

    Returns
    -------
    None
        This function does not return any value; it only displays the graph.

    Notes
    -----
    The `DAG` must be a directed acyclic graph (DAG) as the function relies on
    topological sorting to generate the layout. The visualization is displayed
    using `matplotlib` and is interactive.
    """

    for layer, nodes in enumerate(nx.topological_generations(DAG)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
        for node in nodes:
            DAG.nodes[node]["layer"] = layer

    pos = nx.multipartite_layout(DAG, subset_key="layer")
    fig, ax = plt.subplots()
    nx.draw_networkx(DAG, pos=pos, ax=ax)
    ax.set_title("DAG layout in topological order")
    fig.tight_layout()
    plt.show()
