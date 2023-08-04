from pyqpanda import *
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio

timecost = {'X': 1, 'Y': 1, 'Z': 1, 'P': 1, 'H': 1, 'S': 1,
            'CNOT': 2, 'CZ': 2, 'B': 0}

gate_num_dict = {2: 'X', 3: 'Y', 4: 'Z', 8: 'P', 9: 'H', 11: 'S', 21: 'CNOT', 22: 'CZ'}
num_gate_dict = {'X': 2, 'Y': 3, 'Z': 4, 'P': 8, 'H': 9, 'S': 11, 'CNOT': 21, 'CZ': 22}

colors_dict = {2: 'yellow', 3: 'yellow', 4: 'yellow', 8: 'yellow', 9: 'yellow', 11: 'yellow',
               21: 'green', 22: 'green'}


# Find the gate with the longest execution time in the current layer and fill the empty part
def find_max_and_fill(lst):
    """
    Updates all occurrences of '0' in the given list with the maximum value found in the list.

    Parameters:
    lst (list): A list of elements, where '0' indicates a placeholder for the maximum value.

    Returns:
    list: The modified list with all '0' placeholders replaced by the maximum value from the original list.
    """
    max_value = max(lst)
    for i in range(len(lst)):
        if lst[i] == '0':
            lst[i] = max_value
    return lst


def layer_gate_matrix(prog, num):
    """
    Constructs a gate matrix based on the quantum program and the specified number of gates.

    The function first retrieves layer information from the given quantum program. It then
    initializes a zero matrix of the specified size. The matrix is populated by iterating over
    each layer and its nodes, extracting the physical addresses of the target qubits, and
    assigning the corresponding gate count from a dictionary of gate names.

    Parameters:
        prog (object): The quantum program object containing the circuit information.
        num (int): The number of gates to consider when constructing the matrix.

    Returns:
        numpy.ndarray: A matrix where each row represents a gate and each column represents
                       a qubit, with the values indicating the count of each gate at each qubit.

    Notes:
        This function is intended for use within the pyQPanda package, which supports
        programming quantum computers using quantum circuits and gates. It operates within
        the context of a quantum circuit simulator or a quantum cloud service environment.
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
            # print("m_name:", node.m_name)
            qubits_num = []
            for i in range(len(node.m_target_qubits)):
                qubits_num.append(node.m_target_qubits[i].get_phy_addr())
                matrix[qubits_num, layer_num-1] = num_gate_dict.get(node.m_name)

    return matrix


def layer_time_matrix(prog, num):
    """
    Generates a time matrix for quantum circuit layers based on the provided program and number of layers.

    This function computes the time cost for each operation within a quantum circuit, organized into layers.
    It uses the quantum program and the specified number of layers to create a matrix that maps the
    physical addresses of qubits to the time costs of the operations in each layer.

    Parameters:
    - prog: An object representing the quantum program containing the circuit to be analyzed.
    - num: An integer specifying the number of layers to consider.

    Returns:
    - matrix_time: A 2D NumPy array with shape (num, layer_num) where layer_num is the number of layers.
                   The array contains strings representing the time costs of operations for each qubit in each layer.

    The function is part of the pyQPanda package, which facilitates programming quantum computers using
    quantum circuits and gates, supporting simulations on quantum virtual machines or quantum cloud services.
    The function is located within the pyQPanda/test_timeline_draw.py directory.
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
            qubits_num = []
            for i in range(len(node.m_target_qubits)):
                qubits_num.append(node.m_target_qubits[i].get_phy_addr())
                matrix_time[qubits_num, layer_num-1] = timecost.get(node.m_name)

    return matrix_time


def gate_with_barrier(matrix_time):
    """
    Corrects empty strings within a 2D matrix to '0' and processes the matrix to fill columns with the maximum values found in each column.

    Parameters:
    matrix_time (np.ndarray): A 2D NumPy array containing time-related data for quantum circuits.

    Returns:
    np.ndarray: A NumPy array with the corrected and processed data, where empty strings have been replaced with '0' and columns have been filled with the maximum values.

    The function iterates through each element of the input matrix. If an element is an empty string, it is replaced with '0'. It then processes each column of the matrix by finding the maximum value and filling all other entries in that column with that value. The resulting matrix is transposed and converted to integers before being returned.
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


def build_circuit(num):
    """
    Constructs a quantum circuit with a specified number of qubits.

    This function initializes a quantum program, allocates qubits, and applies a series of quantum gates to the qubits. The circuit
    begins by applying a Hadamard gate to the first qubit, followed by two CNOT gates to create entanglement between adjacent qubits.
    Additionally, a Pauli-X, Pauli-Z, and Pauli-Y gate are applied to the fourth qubit in sequence.

    Args:
        num (int): The number of qubits for the quantum circuit.

    Returns:
        QProg: A quantum program representing the constructed quantum circuit.
    """
    prog = QProg()
    qbits = qvm.qAlloc_many(num)
    prog << H(qbits[0])
    for i in range(2):
        prog << CNOT(qbits[i], qbits[i+1])
    prog << X(qbits[3]) << Z(qbits[3]) << Y(qbits[3])
    return prog


def timeline_draw(fig, num, matrix, mat_res):
    """
    Draws a timeline representation of quantum circuit operations using Plotly's Go objects.

    Parameters:
        fig (plotly.graph_objects.Figure): The Plotly figure object to which the timeline bars will be added.
        num (int): The total number of qubits in the quantum circuit.
        matrix (list of list of int): A 2D list representing the operations performed on each qubit at each time step.
        mat_res (list of list of float): A 2D list containing the time durations of each operation for all qubits.

    The function creates a timeline bar chart where each bar represents an operation on a qubit at a specific time.
    It populates the figure with bars that stack horizontally, and each bar's properties are determined by the operation
    type, such as gate type and color, as defined by external dictionaries `gate_num_dict` and `colors_dict`.
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


if __name__ == "__main__":

    qvm = CPUQVM()
    qvm.init_qvm()
    num = 4
    fig = go.Figure()
    prog = build_circuit(4)
    print(prog)
    matrix = layer_gate_matrix(prog, num)
    matrix_time = layer_time_matrix(prog, num)
    mat_res = gate_with_barrier(matrix_time)
    print("mat_res: ", mat_res)
    timeline_draw(fig, num, matrix, mat_res)
