import pyqpanda as pq
import numpy as np

a = [1, 2]
b = [3, 4]
c = [5, 6]
d = [7, 8]

array2xd = np.array([a, b])
array3xd = np.array([[a, b], [c, d]])
array4xd = np.array([[[a, b], [c, d]]])

def pyqpanda_circuit():

    machine = pq.CPUQVM()
    machine.init_qvm()
    qubits = machine.qAlloc_many(3)

    # hamiltonian = 0.5 * x(0) * y(1) + 2.2 * x(1) * z(2)
    hamiltonian = 1.

    prog = pq.QProg()
    prog << pq.X(qubits[0:2])\
         << pq.Z(qubits[:2])\
         << pq.H(qubits[:])

    print(prog)

    result = machine.get_expectation(
        prog, hamiltonian.to_hamiltonian(True), [qubits[0], qubits[1]])
    print(result)

if __name__ == "__main__":

    x = np.array([[0., 1.], [1., 0.]])

    op1 = pq.PauliOperator(x)
    op2 = pq.PauliOperator(1.)
    op3 = pq.PauliOperator("Z1",0.8)

    op = op1 + op2 + op3

    print(array3xd)
    tensor = pq.tensor3xd(array4xd)
    print(tensor)

    tensor1 = pq.tensor4xd(array4xd)
    print(tensor1)
