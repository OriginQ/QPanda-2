import pyqpanda as pq
from ..pyqpanda.Visualization.circuit_composer import CircuitComposer


def test_append():
    """
    Constructs a quantum circuit using the CircuitComposer from the pyQPanda package,
    appends various quantum gates and transformations, and then visualizes the circuit
    in different formats.

    Parameters:
    n_qubits (int): The number of qubits to initialize in the circuit.

    The function performs the following steps:
    - Initializes a CircuitComposer object with the specified number of qubits.
    - Creates a quantum circuit and adds a Hadamard gate, two CNOT gates, and a barrier.
    - Appends a Quantum Fourier Transform (QFT) to the circuit.
    - Appends a deep copy of the current circuit to it.
    - Prints the circuit and its underlying quantum circuit representation.
    - Draws the circuit in LaTeX and text formats and saves the LaTeX representation as an image.
    - Prints the LaTeX and text representations of the circuit.
    """
    circ1 = CircuitComposer(n_qubits)
    circuit = pq.QCircuit()
    circuit << pq.H(q[0]) << pq.CNOT(q[0], q[1]) << pq.CNOT(q[1], q[2])
    circ1.append(circuit)
    circ1 << pq.BARRIER(q)
    circ1.append(pq.QFT(q[3:]), "QFT")
    circ1.append(pq.deep_copy(circ1))
    print(circ1)
    print(circ1.circuit)

    a = circ1.draw_circuit("pic", "test.png")
    b = circ1.draw_circuit("latex")
    c = circ1.draw_circuit("text")
    print(b)
    print(c)


if __name__ == '__main__':
    n_qubits = 6
    qvm = pq.CPUQVM()
    qvm.init_qvm()
    q = qvm.qAlloc_many(n_qubits)

    test_append()
