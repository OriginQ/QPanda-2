"""Show quantum circuit info.

    GateType:
        PAULI_X_GATE,       /**< Quantum pauli x  gate */
        PAULI_Y_GATE,       /**< Quantum pauli y  gate */
        PAULI_Z_GATE,       /**< Quantum pauli z  gate */
        X_HALF_PI,          /**< Quantum x half gate */
        Y_HALF_PI,          /**< Quantum y half gate */
        Z_HALF_PI,          /**< Quantum z half gate */
        P_GATE,             /**<Quantum p gate>*/
        HADAMARD_GATE,      /**< Quantum hadamard gate */
        T_GATE,             /**< Quantum t gate */
        S_GATE,             /**< Quantum s gate */
        RX_GATE,            /**< Quantum rotation x gate */
        RY_GATE,            /**< Quantum rotation y gate */
        RZ_GATE,            /**< Quantum rotation z gate */
        RPHI_GATE, 
        U1_GATE,            /**< Quantum u1 gate */
        U2_GATE,            /**< Quantum u2 gate */
        U3_GATE,            /**< Quantum u3 gate */
        U4_GATE,            /**< Quantum u4 gate */
        CU_GATE,            /**< Quantum control-u gate */
        CNOT_GATE,          /**< Quantum control-not gate */
        CZ_GATE,            /**< Quantum control-z gate */
        CP_GATE,            /**<Quantum control-p gate>*/
        RYY_GATE,           /**<Quantum ryy gate>*/
        RXX_GATE,           /**<Quantum rxx gate>*/
        RZZ_GATE,           /**<Quantum rzz gate>*/
        RZX_GATE,           /**<Quantum rzx gate>*/
        CPHASE_GATE,        /**< Quantum control-rotation gate */
        ISWAP_THETA_GATE,   /**< Quantum iswap-theta gate */
        ISWAP_GATE,         /**< Quantum iswap gate */
        SQISWAP_GATE,       /**< Quantum sqiswap gate */
        SWAP_GATE,          /**< Quantum swap gate */
        TWO_QUBIT_GATE,     /**< Quantum two-qubit gate */
        TOFFOLI_GATE, 
        ORACLE_GATE, 
        CORACLE_GATE, 
        I_GATE, 
"""

import pyqpanda as pq


all_gate_types = [
    pq.GateType.PAULI_X_GATE,
    pq.GateType.PAULI_Y_GATE,
    pq.GateType.PAULI_Z_GATE,
    pq.GateType.X_HALF_PI,
    pq.GateType.Y_HALF_PI,
    pq.GateType.Z_HALF_PI,
    pq.GateType.HADAMARD_GATE,
    pq.GateType.T_GATE,
    pq.GateType.S_GATE,
    pq.GateType.CNOT_GATE,
    pq.GateType.CZ_GATE,
    pq.GateType.SWAP_GATE,
    pq.GateType.SQISWAP_GATE,
    pq.GateType.TOFFOLI_GATE,
    pq.GateType.ISWAP_GATE,

    pq.GateType.P_GATE,
    pq.GateType.RX_GATE,
    pq.GateType.RY_GATE,
    pq.GateType.RZ_GATE,
    pq.GateType.U1_GATE,
    pq.GateType.ISWAP_THETA_GATE,
    pq.GateType.RXX_GATE,
    pq.GateType.RYY_GATE,
    pq.GateType.RZX_GATE,
    pq.GateType.RZZ_GATE,
    pq.GateType.CPHASE_GATE,
    pq.GateType.CP_GATE,

    pq.GateType.U2_GATE,

    pq.GateType.U3_GATE,
    
    pq.GateType.U4_GATE,
    pq.GateType.CU_GATE,
]

def get_gate_name(gate_type: pq.GateType):
    """
    Convert a quantum gate type from the pyQPanda package to its corresponding string representation.
    
    Args:
        gate_type (pq.GateType): The enum representing the quantum gate type.
        
    Returns:
        str: The string name of the quantum gate.
        
    Raises:
        TypeError: If an unknown gate type is provided.
        
    Supported gates include Pauli gates, Hadamard, T, S, CNOT, CZ, SWAP, SQISWAP, Toffoli, single-qubit gates,
    rotation gates, controlled gates, and multi-qubit gates. The function uses the enum values from the pq module
    to determine the correct string representation of the gate.
    """
    if gate_type == pq.GateType.PAULI_X_GATE:
        return 'X'
    elif gate_type == pq.GateType.PAULI_Y_GATE:
        return 'Y'
    elif gate_type == pq.GateType.PAULI_Z_GATE:
        return 'Z'
    elif gate_type == pq.GateType.X_HALF_PI:
        return 'X_half'
    elif gate_type == pq.GateType.Y_HALF_PI:
        return 'Y_half'
    elif gate_type == pq.GateType.Z_HALF_PI:
        return 'Z_half'
    elif gate_type == pq.GateType.HADAMARD_GATE:
        return 'H'
    elif gate_type == pq.GateType.T_GATE:
        return 'T'
    elif gate_type == pq.GateType.S_GATE:
        return 'S'
    elif gate_type == pq.GateType.CNOT_GATE:
        return 'CNOT'
    elif gate_type == pq.GateType.CZ_GATE:
        return 'CZ'
    elif gate_type == pq.GateType.SWAP_GATE:
        return 'SWAP'
    elif gate_type == pq.GateType.SQISWAP_GATE:
        return '√SWAP'
    elif gate_type == pq.GateType.TOFFOLI_GATE:
        return 'Toffoli'
    elif gate_type == pq.GateType.P_GATE:
        return 'P'
    elif gate_type == pq.GateType.RX_GATE:
        return 'RX'
    elif gate_type == pq.GateType.RY_GATE:
        return 'RY'
    elif gate_type == pq.GateType.RZ_GATE:
        return 'RZ'
    elif gate_type == pq.GateType.U1_GATE:
        return 'U1'
    elif gate_type == pq.GateType.ISWAP_THETA_GATE:
        return 'iSWAP_theta'
    elif gate_type == pq.GateType.ISWAP_GATE:
        return 'iSWAP'
    elif gate_type == pq.GateType.RXX_GATE:
        return 'RXX'
    elif gate_type == pq.GateType.RYY_GATE:
        return 'RYY'
    elif gate_type == pq.GateType.RZX_GATE:
        return 'RYX'
    elif gate_type == pq.GateType.RZZ_GATE:
        return 'RZZ'
    elif gate_type == pq.GateType.CPHASE_GATE:
        return 'CR'
    elif gate_type == pq.GateType.CP_GATE:
        return 'CR'
    elif gate_type == pq.GateType.U2_GATE:
        return 'U2'
    elif gate_type == pq.GateType.U3_GATE:
        return 'U3'
    elif gate_type == pq.GateType.U4_GATE:
        return 'U4'
    elif gate_type == pq.GateType.CU_GATE:
        return 'CU'
    else:
        raise TypeError("Unknown gate type!")

def get_circuit_info(circ: pq.QCircuit) -> None:
    """
    Analyzes the provided quantum circuit and prints detailed information about it.

    The function takes a quantum circuit object and outputs a string containing:
    - The number of qubits in the circuit.
    - A map of gate types to their respective counts.
    - Total number of gates in the circuit.
    - Count of parameterized gates and total number of parameters across these gates.

    Args:
    - circ (pq.QCircuit): The quantum circuit to analyze.

    Returns:
    - None: This function prints the analysis results directly and does not return a value.
    """
    # number of qubits
    n_qubits = len(pq.get_all_used_qubits(circ))

    # number of gates
    n_gates = pq.count_gate(circ)

    used_parameterized_gates = set()

    n_param_gates = 0
    n_parameters = 0
    single_param_gate_types = [
        pq.GateType.P_GATE,
        pq.GateType.RX_GATE,
        pq.GateType.RY_GATE,
        pq.GateType.RZ_GATE,
        pq.GateType.U1_GATE,
        pq.GateType.ISWAP_THETA_GATE,
        pq.GateType.RXX_GATE,
        pq.GateType.RYY_GATE,
        pq.GateType.RZX_GATE,
        pq.GateType.RZZ_GATE,
        pq.GateType.CPHASE_GATE,
        pq.GateType.CP_GATE,
    ]
    double_param_gate_types = [ pq.GateType.U2_GATE, ]
    triple_param_gate_types = [ pq.GateType.U3_GATE, ]
    quad_param_gate_types = [ pq.GateType.U4_GATE, pq.GateType.CU_GATE, ]

    gate_name_cnt_map = dict()
    for gate_type in all_gate_types:
        n = pq.count_qgate_num(circ, gate_type)
        if n != 0:
            gate_name = get_gate_name(gate_type)
            if gate_name in gate_name_cnt_map:
                gate_name_cnt_map[gate_name] += n
            else:
                gate_name_cnt_map[gate_name] = n

            if gate_type in single_param_gate_types:
                used_parameterized_gates.add(gate_name)
                n_param_gates += n
                n_parameters += n
            elif gate_type in double_param_gate_types:
                used_parameterized_gates.add(gate_name)
                n_param_gates += n
                n_parameters += 2*n
            elif gate_type in triple_param_gate_types:
                used_parameterized_gates.add(gate_name)
                n_param_gates += n
                n_parameters += 3*n
            elif gate_type in quad_param_gate_types:
                used_parameterized_gates.add(gate_name)
                n_param_gates += n
                n_parameters += 4*n
            else:
                pass

    
    head_line = '#' * 30 + ' Circuit Information ' + '#' * 30
    width = len(head_line)
    str_info = head_line + '\n'
    str_info += f'Qubits: {n_qubits}\n'
    str_info += f'Gates info: {gate_name_cnt_map}\n'
    str_info += f'Gates: {n_gates}\n'
    str_info += f'Paramterized gates: {n_param_gates}\n'
    str_info += f'Args: {n_parameters}\n'
    str_info += '#' * width
    return str_info
