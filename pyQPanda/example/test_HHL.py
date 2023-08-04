import pyqpanda as pq
import numpy as np
import unittest


class InitQMachine:
    def __init__(self, machineType=pq.QMachineType.CPU):
        self.m_machine = pq.init_quantum_machine(machineType)
        self.m_machine.set_configure(64, 64)

    def __del__(self):
        pq.destroy_quantum_machine(self.m_machine)


# 筛选出辅助比特测量结果是 1的态
def postselect(statevector, qubit_index, value):
    """
    Extracts a sub-vector from the state vector based on the specified qubit index and value.
    
    This function is designed to be used within the pyQPanda package for quantum computing applications.
    It is intended to facilitate the manipulation of quantum state vectors during quantum circuit simulations.
    
    Parameters:
    - statevector (np.ndarray): The complete quantum state vector from which a sub-vector is to be extracted.
    - qubit_index (int): The index of the qubit for which the value is being postselected.
    - value (int): The specific value to postselect on the qubit; typically 0 or 1.
    
    Returns:
    - np.ndarray: A sub-vector of the original state vector, postselected to only include amplitudes with the specified value for the specified qubit.
    
    Note: This function assumes that the state vector is properly formatted and that the qubit index is within the valid range.
    
    Example usage:
        # Assuming a state vector with a length that is a power of 2, and a qubit index of 2:
        state = np.array([1, 0, 0, 1])
        qubit_idx = 2
        postselect_val = 1
        result = postselect(state, qubit_idx, postselect_val)
        # Result will be the sub-vector [1, 0, 0, 1] if postselect_val is 1.
    """
    array_mask = int((len(statevector)) / 2)

    def normalise(vec: np.ndarray):
        from scipy.linalg import norm
        return vec / norm(vec)

    # return normalise(statevector[array_mask:])
    return (statevector[array_mask:])


# 近似运算
def round_to_zero(vec, tol=2e-6):
    """
    Rounds the real and imaginary parts of a complex number vector to zero if their absolute values are below a specified tolerance.

    Parameters:
    vec (complex): The complex number vector to be rounded.
    tol (float, optional): The tolerance level below which the real and imaginary parts will be set to zero. Default is 2e-6.

    Returns:
    complex: The input vector with its real and imaginary parts rounded to zero if they fall below the specified tolerance.
    """
    vec.real[abs(vec.real) < tol] = 0.0
    vec.imag[abs(vec.imag) < tol] = 0.0
    return vec

class Test_HHL(unittest.TestCase):
    def test_hhl_1(self):
        init_machine = InitQMachine()
        machine = init_machine.m_machine

        x = [15.0 / 4.0, 9.0 / 4.0, 5.0 / 4.0, -3.0 / 4.0,
             9.0 / 4.0, 15.0 / 4.0, 3.0 / 4.0, -5.0 / 4.0,
             5.0 / 4.0, 3.0 / 4.0, 15.0 / 4.0, -9.0 / 4.0,
             -3.0 / 4.0, -5.0 / 4.0, -9.0 / 4.0, 15.0 / 4.0]
        b = [0.5, 0.5, 0.5, 0.5]

        hhl_cir = pq.build_HHL_circuit(x, b, machine)
        pq.directly_run(hhl_cir)
        full_state = np.array(pq.get_qstate())
        # statevector = pq.rround_to_zero(pq.postselect(full_state, 6, True), 1e-6)
        # solution = statevector[: 4]
        # print(solution)


    @unittest.skip("skip")
    def test_hhl_2(self):
        init_machine = InitQMachine()
        machine = init_machine.m_machine

        x = [15.0 / 4.0, 9.0 / 4.0, 5.0 / 4.0, -3.0 / 4.0,
             9.0 / 4.0, 15.0 / 4.0, 3.0 / 4.0, -5.0 / 4.0,
             5.0 / 4.0, 3.0 / 4.0, 15.0 / 4.0, -9.0 / 4.0,
             -3.0 / 4.0, -5.0 / 4.0, -9.0 / 4.0, 15.0 / 4.0]
        b = [0.5, 0.5, 0.5, 0.5]

        hhl_alg = pq.HHLAlg(machine)
        hhl_cir = hhl_alg.get_hhl_circuit(x, b, 2)
        prog = pq.create_empty_qprog()
        prog.insert(pq.build_HHL_circuit(x, machine))
        pq.directly_run(prog)

        result = np.array(machine.get_qstate())[:2]
        pq.destroy_quantum_machine(machine)



        qubit_for_b = hhl_alg.get_qubit_for_b()
        print(qubit_for_b)
        # for _q in qubit_for_b:
        #     print(_q.get_phy_addr())
        qubit_for_qft = hhl_alg.get_qubit_for_QFT()
        # for _q in qubit_for_qft:
        #     print(_q.get_phy_addr())

        amplification_factor = hhl_alg.get_amplification_factor()

        pq.directly_run(hhl_cir)
        full_state = np.array(pq.get_qstate())
        # statevector = pq.round_to_zero(pq.postselect(full_state, 6, True), 1e-6)
        # solution = statevector[: 4]
        # for ii in solution:
        #     print(ii*amplification_factor)


    def test_hhl_solve_linear_equations(self):
        x = [15.0 / 4.0, 9.0 / 4.0, 5.0 / 4.0, -3.0 / 4.0,
             9.0 / 4.0, 15.0 / 4.0, 3.0 / 4.0, -5.0 / 4.0,
             5.0 / 4.0, 3.0 / 4.0, 15.0 / 4.0, -9.0 / 4.0,
             -3.0 / 4.0, -5.0 / 4.0, -9.0 / 4.0, 15.0 / 4.0]

        b = [1, 1, 1, 1]
        result_x = pq.HHL_solve_linear_equations(x, b, 1)
        print(result_x)


if __name__ == "__main__":
    unittest.main(verbosity=2)
