
from pyqpanda import *
from pyqpanda.Algorithm.test.qaoa_maxcut_test import generate_graph,qaoa_maxcut_gradient_threshold

def paulioperator_test2():
    from pyqpanda.Hamiltonian.PauliOperator.pyQPandaPauliOperator import PauliOperator
    from pyqpanda.Algorithm.hamiltonian_simulation import simulate_pauliZ_hamiltonian,simulate_hamiltonian
    H1=PauliOperator({"Z0 Z3":1.2,"Z1 Z4":2.2,"X1 Y4":2.2," ":22.2})
    print(H1)
    print(H1.toHamiltonian(0))
    H2=H1.toHamiltonian(0)
    machine=init(QuantumMachine_type.CPU)
    qlist=machine.qAlloc_many(H1.getMaxIndex())
    prog=QProg()
    #prog.insert(simulate_pauliZ_hamiltonian(qlist,H1,1.2))
    prog.insert(simulate_hamiltonian(qlist,H1,2.3,5))
    print(qRunesProg(prog))

def adiabatic_test():
    from pyqpanda.Hamiltonian.PauliOperator.pyQPandaPauliOperator import PauliOperator
    from pyqpanda.Algorithm.hamiltonian_simulation import adiabatic_simulation_with_configuration
    H1=PauliOperator({"Z0 Z3":1.2,"Z1 Z4":2.2,"X1 Y4":2.2," ":22.2})
    H2=PauliOperator({"Y0 Z3":12.2,"Z1 Z4":24.2,"X0 Z4":2.2})
    result=adiabatic_simulation_with_configuration(5,H1,H2,2,3,2)
    print(result)
def qaoa_new_test():
    graph=generate_graph(dimension=9,n_edge=36)
    result=qaoa_maxcut_gradient_threshold(graph=graph,step=5,threshold_value=0.05,optimize_times=300,use_GPU=False)
    return result

if __name__ == "__main__":


    #print("paulioperator_test1")
    #paulioperator_test1()
    #print("paulioperator_test2")
    #paulioperator_test2()
    adiabatic_test()
    #paulioperator_test2()
    # result=qaoa_new_test()
    # print(result)

    

    