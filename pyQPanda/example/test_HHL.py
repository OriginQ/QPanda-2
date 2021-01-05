import pyqpanda.pyQPanda as pq
import numpy as np

class InitQMachine:
    def __init__(self, machineType = pq.QMachineType.CPU):
        self.m_machine = pq.init_quantum_machine(machineType)

    def __del__(self):
        pq.destroy_quantum_machine(self.m_machine)

# 筛选出辅助比特测量结果是 1的态
def postselect(statevector, qubit_index, value):
    array_mask = int((len(statevector))/2)

    def normalise(vec: np.ndarray):
        from scipy.linalg import norm
        return vec / norm(vec)

    return normalise(statevector[array_mask:])

# 近似运算
def round_to_zero(vec, tol=2e-6):
    vec.real[abs(vec.real) < tol] = 0.0
    vec.imag[abs(vec.imag) < tol] = 0.0
    return vec

def test_hhl():
    init_machine = InitQMachine()
    machine = init_machine.m_machine

    x=[15.0 / 4.0,  9.0 / 4.0,   5.0 / 4.0,   -3.0 / 4.0,  
	        9.0 / 4.0,   15.0 / 4.0,   3.0 / 4.0,   -5.0 / 4.0,  
	        5.0 / 4.0,   3.0 / 4.0,   15.0 / 4.0,   -9.0 / 4.0,  
	       -3.0 / 4.0,   -5.0 / 4.0,   -9.0 / 4.0,   15.0 / 4.0]
    b=[0.5,0.5,0.5,0.5]

    hhl_cir = pq.HHL(x, b, machine)
    pq.draw_qprog(hhl_cir)
    pq.directly_run(hhl_cir)
    full_state = np.array(pq.get_qstate())
    statevector =  round_to_zero( postselect(full_state, 6, True), 1e-6)
    solution = statevector[: 4]
    solution = solution * np.sqrt(340)
    print(solution)

def test_hhl_solve_linear_equations():
    x=[15.0 / 4.0,  9.0 / 4.0,   5.0 / 4.0,   -3.0 / 4.0,  
	        9.0 / 4.0,   15.0 / 4.0,   3.0 / 4.0,   -5.0 / 4.0,  
	        5.0 / 4.0,   3.0 / 4.0,   15.0 / 4.0,   -9.0 / 4.0,  
	       -3.0 / 4.0,   -5.0 / 4.0,   -9.0 / 4.0,   15.0 / 4.0]

    b=[0.5,0.5,0.5,0.5]
    result_x = pq.HHL_solve_linear_equations(x, b)
    result_x = np.array(result_x) * np.sqrt(340)
    print(result_x)

if __name__=="__main__":
    test_hhl()
    #test_hhl_solve_linear_equations()
    print("Test over.")