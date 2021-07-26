import pyqpanda.pyQPanda as pq
import numpy as np

class InitQMachine:
    def __init__(self, machineType = pq.QMachineType.CPU):
        self.m_machine = pq.init_quantum_machine(machineType)
        self.m_machine.set_configure(64, 64)

    def __del__(self):
        pq.destroy_quantum_machine(self.m_machine)

# 筛选出辅助比特测量结果是 1的态
def postselect(statevector, qubit_index, value):
    array_mask = int((len(statevector))/2)

    def normalise(vec: np.ndarray):
        from scipy.linalg import norm
        return vec / norm(vec)

    #return normalise(statevector[array_mask:])
    return (statevector[array_mask:])

# 近似运算
def round_to_zero(vec, tol=2e-6):
    vec.real[abs(vec.real) < tol] = 0.0
    vec.imag[abs(vec.imag) < tol] = 0.0
    return vec

def test_hhl_1():
    init_machine = InitQMachine()
    machine = init_machine.m_machine

    x=[15.0 / 4.0,  9.0 / 4.0,   5.0 / 4.0,   -3.0 / 4.0,  
	        9.0 / 4.0,   15.0 / 4.0,   3.0 / 4.0,   -5.0 / 4.0,  
	        5.0 / 4.0,   3.0 / 4.0,   15.0 / 4.0,   -9.0 / 4.0,  
	       -3.0 / 4.0,   -5.0 / 4.0,   -9.0 / 4.0,   15.0 / 4.0]
    b=[0.5,0.5,0.5,0.5]

    hhl_cir = pq.build_HHL_circuit(x, b, machine)
    pq.directly_run(hhl_cir)
    full_state = np.array(pq.get_qstate())
    statevector =  round_to_zero( postselect(full_state, 6, True), 1e-6)
    solution = statevector[: 4]
    print(solution)

def test_hhl_2():
    init_machine = InitQMachine()
    machine = init_machine.m_machine

    x=[15.0 / 4.0,  9.0 / 4.0,   5.0 / 4.0,   -3.0 / 4.0,  
	        9.0 / 4.0,   15.0 / 4.0,   3.0 / 4.0,   -5.0 / 4.0,  
	        5.0 / 4.0,   3.0 / 4.0,   15.0 / 4.0,   -9.0 / 4.0,  
	       -3.0 / 4.0,   -5.0 / 4.0,   -9.0 / 4.0,   15.0 / 4.0]
    b=[0.5,0.5,0.5,0.5]

    hhl_alg = pq.HHLAlg(machine)
    hhl_cir = hhl_alg.get_hhl_circuit(x, b, 2)

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
    statevector =  round_to_zero( postselect(full_state, 6, True), 1e-6)
    solution = statevector[: 4]
    for ii in solution:
        print(ii*amplification_factor)

def test_hhl_solve_linear_equations():
    x=[15.0 / 4.0,  9.0 / 4.0,   5.0 / 4.0,   -3.0 / 4.0,  
	        9.0 / 4.0,   15.0 / 4.0,   3.0 / 4.0,   -5.0 / 4.0,  
	        5.0 / 4.0,   3.0 / 4.0,   15.0 / 4.0,   -9.0 / 4.0,  
	       -3.0 / 4.0,   -5.0 / 4.0,   -9.0 / 4.0,   15.0 / 4.0]

    b=[1,1,1,1]
    result_x = pq.HHL_solve_linear_equations(x, b, 1)
    print(result_x)

if __name__=="__main__":
    test_hhl_1()
    #test_hhl_2()
    #test_hhl_solve_linear_equations()
    print("Test over.")