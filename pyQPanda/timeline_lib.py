from pyqpanda import *
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio

# Set the execution time of a single gate to one unit,
# and the execution time of a double gate to two units. Barrier defaults to 0
timecost = {'X': 1, 'Y': 1, 'Z': 1, 'P': 1, 'H': 1, 'S': 1,
            'CNOT': 2, 'CZ': 2, 'B': 0}

# Create the dictionary of the relationship between sequence numbers and quantum gates in pyqpanda
gate_num_dict = {2: 'X', 3: 'Y', 4: 'Z', 8: 'P', 9: 'H', 11: 'S', 21: 'CNOT', 22: 'CZ'}
num_gate_dict = {'X': 2, 'Y': 3, 'Z': 4, 'P': 8, 'H': 9, 'S': 11, 'CNOT': 21, 'CZ': 22}
# Set the color for Quantum gates
colors_dict = {2: 'yellow', 3: 'yellow', 4: 'yellow', 8: 'yellow', 9: 'yellow', 11: 'yellow',
               21: 'green', 22: 'green'}


# Find the gate with the longest execution time in the current layer and fill the empty part
def find_max_and_fill(lst: list) -> list:

    """
    Replace all occurrences of '0' in the input list with the maximum value found in the list.

    Parameters
    ----------
    lst : list
        A list of integers, where '0' represents an empty or unfilled value.

    Returns
    -------
    list
        A new list with all '0' values replaced by the maximum integer from the original list.

    Notes
    -----
    This function is intended for use within the pyQPanda package, which supports quantum computing
    and quantum circuit simulation. It operates on lists that represent quantum states or circuit
    configurations, filling in missing or empty states with the most frequent state value.
    """

    max_value = max(lst)
    for i in range(len(lst)):
        if lst[i] == '0':
            lst[i] = max_value
    return lst


# Using a matrix to represent quantum gate information
def layer_gate_matrix(prog: QProg, num: int) -> np.ndarray:

    """
    Extracts quantum gate information from a quantum program and constructs a matrix
    representing the gates applied to a specified number of qubits.

    Parameters
    ----------
    prog : QProg
        The quantum program from which to extract gate information.
    num : int
        The total number of qubits to consider for gate information.

    Returns
    -------
    np.ndarray
        A 2D numpy array where each row corresponds to a qubit and each column
        corresponds to a gate layer. The elements of the array represent the
        quantum gates applied at each layer for each qubit.

    Notes
    -----
    This function is intended for use within the pyQPanda package, which supports
    quantum computing programming using quantum circuits and gates. It operates on
    quantum circuits and gates as defined within the pyQPanda framework.
    """

    layer_info = circuit_layer(prog)
    layers = layer_info[0]
    layer_num = 0
    for lay in layers:
        layer_num += 1
    matrix = np.zeros((num, layer_num))
    layer_num = 0
    for lay in layers:
        layer_num += 1
        for node in lay:
            print("m_name:", node.m_name)

            qubits_num = []
            for i in range(len(node.m_target_qubits)):
                qubits_num.append(node.m_target_qubits[i].get_phy_addr())
                matrix[qubits_num, layer_num-1] = num_gate_dict.get(node.m_name)

    return matrix


# Using a matrix to represent quantum gate time information
def layer_time_matrix(prog: QProg, num: int) -> np.ndarray:

    """
    Constructs a time matrix for quantum gate executions within a quantum program.

    This function computes the execution time for each quantum gate in the given quantum program and organizes them into a matrix.
    The matrix is indexed by qubit and layer, where each layer corresponds to a sequence of gates applied to the same set of qubits.

    Parameters
    ----------
    prog : QProg
        The quantum program containing quantum gates and operations.
    num : int
        The total number of qubits in the quantum program.

    Returns
    -------
    np.ndarray
        A two-dimensional numpy array where each row represents a qubit and each column represents a layer of quantum gates.
        The element at position [qubit, layer] contains the execution time of the gate at that layer for the corresponding qubit.
        The data type of the elements is string, which stores the time cost associated with the gate operations.

    Notes
    -----
    The function relies on the `layer_info` and `timecost` components from the quantum circuit simulation framework, which are not
    explicitly defined in this function. The function assumes that the quantum program (`prog`) and the execution time lookup
    (`timecost`) are properly initialized and accessible within the context where this function is called.
    """

    layer_info = circuit_layer(prog)
    layers = layer_info[0]
    layer_num = 0
    for lay in layers:
        layer_num += 1
    matrix_time = np.zeros((num, layer_num), dtype=str)
    layer_num = 0
    for lay in layers:
        layer_num += 1
        for node in lay:
            print("m_name:", node.m_name)

            qubits_num = []
            for i in range(len(node.m_target_qubits)):
                # print("m_target_qubits:", node.m_target_qubits[i].get_phy_addr())
                qubits_num.append(node.m_target_qubits[i].get_phy_addr())
                matrix_time[qubits_num, layer_num-1] = timecost.get(node.m_name)

    return matrix_time


# Add barrier into the matrix
def gate_with_barrier(matrix_time: np.ndarray) -> np.ndarray:
    """
    Processes a matrix of quantum gate execution times by filling blank entries with '0' and
    then reorganizes the matrix to create a final result matrix with barriers filled in.

    Parameters
    ----------
    matrix_time : numpy.ndarray
                  An input matrix representing the execution times of quantum gates.

    Returns
    -------
    numpy.ndarray
        A matrix where the original execution times are preserved, with all blank entries
        replaced by '0', and the matrix is transposed and converted to integer type.

    Notes
    -----
    The function is designed to be used within the pyQPanda package for quantum computing
    applications, particularly for quantum circuit simulation and quantum virtual machine
    operations. It is located in the `pyQPanda.timeline_lib.py` directory.

    The processing involves iterating through the matrix to replace empty strings with '0'
    and then applying a transformation to rearrange the data. The final matrix is
    transposed to align with specific requirements for quantum circuit representation.
    """

    for i in range(matrix_time.shape[0]):
        for j in range(matrix_time.shape[1]):
            if matrix_time[i, j] == '':
                matrix_time[i, j] = '0'
    mat = []
    for j in range(0, len(matrix_time[0])):
        tem = []
        for i in matrix_time[:, j]:
            tem.append(i)
            # print(i)
        find_max_and_fill(tem)
        mat.append(tem)
    mat = np.array(mat)
    mat2 = np.transpose(mat)
    mat_res = mat2.astype(int)
    return mat_res


# Build a matrix
def build_circuit(num: int) -> QProg:

    """
    Constructs a quantum circuit with a specified number of qubits, applying Hadamard, CNOT, and single-qubit gates.

    Parameters
    ----------
    num : int
        The number of qubits in the quantum circuit.

    Returns
    -------
    QProg
        A quantum program representing the constructed circuit.

    Details
    -------
    The function initializes a quantum program, allocates the specified number of qubits, and applies the following operations:
    - A Hadamard gate to the first qubit.
    - Two CNOT gates to create a controlled-NOT between the first two qubits and between the second and third qubits.
    - Single-qubit X, Z, and Y gates to the fourth qubit.

    The resulting quantum program is then returned.
    """
    prog = QProg()
    qbits = qvm.qAlloc_many(num)
    prog << H(qbits[0])
    for i in range(2):
        prog << CNOT(qbits[i], qbits[i+1])
    prog << X(qbits[3]) << Z(qbits[3]) << Y(qbits[3])
    return prog


# Draw the timeline
def timeline_draw(fig: go.Figure, num: int, matrix: np.ndarray, mat_res: np.ndarray):
    """
    Renders a timeline visualization of quantum circuit operations.

    This function takes a Plotly figure, the number of qubits, a matrix
    containing quantum gate information, and the final matrix of the time-evolved
    system. It populates the figure with bars representing the gates' application
    times on specific qubits, creating a visual representation of the quantum
    circuit's timeline.

    Parameters
    ----------
    fig : plotly.graph_objs._figure.Figure
        The Plotly figure object where the timeline will be drawn.
    num : int
        The number of qubits in the quantum circuit.
    matrix : numpy.ndarray
        A 2D numpy array containing the quantum gate information for each gate
        applied to each qubit.
    mat_res : numpy.ndarray
        A 2D numpy array representing the time evolution of the system gates.

    Returns
    -------
    None
        This function modifies the input figure in place and does not return
        anything. The visualization is saved as 'my_plot.png'.

    Notes
    -----
    The function assumes that the input figures have been properly initialized
    and that the gate and color mappings are defined outside this function.
    The visualization includes a title, axis labels, and stacks the bars for
    each qubit horizontally.
    """

    dct = {f'q_{i}': i for i in range(num)}
    qbitnames = [key for key in dct]

    base_time = [0]
    for i in range(len(mat_res[0])-1):
        t = base_time[i]+mat_res[0][i]
        base_time.append(t)
    print("base_time:", base_time)

    for i in range(len(mat_res[:, 0])):   # row
        for j in range(len(mat_res[0, :])):
            if matrix[i][j] == 2:
                fig.add_trace(go.Bar(x=[mat_res[i][j]], y=[qbitnames[i]], base=base_time[j], orientation='h',
                                     text=[gate_num_dict.get(matrix[i][j])], marker_color=colors_dict.get(matrix[i][j])))
            elif matrix[i][j] == 3:
                fig.add_trace(go.Bar(x=[mat_res[i][j]], y=[qbitnames[i]], base=base_time[j], orientation='h', text=[gate_num_dict.get(matrix[i][j])],
                                marker_color=colors_dict.get(matrix[i][j])))
            elif matrix[i][j] == 4:
                fig.add_trace(go.Bar(x=[mat_res[i][j]], y=[qbitnames[i]], base=base_time[j], orientation='h', text=[gate_num_dict.get(matrix[i][j])],
                                marker_color=colors_dict.get(matrix[i][j])))
            elif matrix[i][j] == 8:
                fig.add_trace(go.Bar(x=[mat_res[i][j]], y=[qbitnames[i]], base=base_time[j], orientation='h', text=[gate_num_dict.get(matrix[i][j])],
                                marker_color=colors_dict.get(matrix[i][j])))
            elif matrix[i][j] == 9:
                fig.add_trace(go.Bar(x=[mat_res[i][j]], y=[qbitnames[i]], base=base_time[j], orientation='h', text=[gate_num_dict.get(matrix[i][j])],
                           marker_color=colors_dict.get(matrix[i][j])))
            elif matrix[i][j] == 11:
                fig.add_trace(go.Bar(x=[mat_res[i][j]], y=[qbitnames[i]], base=base_time[j], orientation='h', text=[gate_num_dict.get(matrix[i][j])],
                           marker_color=colors_dict.get(matrix[i][j])))
            elif matrix[i][j] == 21:
                fig.add_trace(go.Bar(x=[mat_res[i][j]], y=[qbitnames[i]], base=base_time[j], orientation='h', text=[gate_num_dict.get(matrix[i][j])],
                           marker_color=colors_dict.get(matrix[i][j])))
            elif matrix[i][j] == 22:
                fig.add_trace(go.Bar(x=[mat_res[i][j]], y=[qbitnames[i]], base=base_time[j], orientation='h', text=[gate_num_dict.get(matrix[i][j])],
                           marker_color=colors_dict.get(matrix[i][j])))
            # else:
            #     fig.add_trace(go.Bar(x=[mat_res[i][j]], y=[qbitnames[i]], base=base_time[j], orientation='h', text=['BA'],
            #                  marker_color='white'))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="circuit", xaxis_title="time", yaxis_title="qwire")
    fig.update_layout(barmode='stack')
    pio.write_image(fig, 'my_plot.png')
