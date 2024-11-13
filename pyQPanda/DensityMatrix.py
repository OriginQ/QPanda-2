from numpy import pi
from pyqpanda import *

def test_probabilities():
    """
    Simulates a quantum circuit using a DensityMatrixSimulator and prints various probabilities and density matrix results.

    The function initializes a quantum virtual machine, allocates qubits, and constructs a quantum circuit with a series of
    gates. It then prints the density matrix and individual probabilities for all possible measurement outcomes of the qubits.
    Additionally, it demonstrates the use of the `get_probabilities` method to retrieve probabilities for multiple qubits.

    Parameters:
        None

    Returns:
        None

    This function is part of the DensityMatrix module within the pyQPanda package, which is designed for programming quantum
    computers using quantum circuits and gates, supporting both quantum circuit simulators and quantum cloud services.
    """

    machine = DensityMatrixSimulator()
    machine.init_qvm()

    q = machine.qAlloc_many(2)
    c = machine.cAlloc_many(2)

    prog = QProg()
    prog.insert(H(q[0]))\
        .insert(Y(q[1]))\
        .insert(RY(q[0], pi / 3))\
        .insert(RX(q[1], pi / 6))\
        .insert(RX(q[1], pi / 9))\
        .insert(CZ(q[0], q[1]))

    print(machine.get_density_matrix(prog))
    print(machine.get_probability(prog, 0))
    print(machine.get_probability(prog, 1))
    print(machine.get_probability(prog, 2))
    print(machine.get_probability(prog, 3))

    print("00 : ", machine.get_probability(prog, "00"))
    print("01 : ", machine.get_probability(prog, "01"))
    print("10 : ", machine.get_probability(prog, "10"))
    print("11 : ", machine.get_probability(prog, "11"))

    print("[0] : ", machine.get_probabilities(prog, [0]))
    print("[0, 1] : ", machine.get_probabilities(prog, [0, 1]))

    machine.finalize()

def test_density_matrix():
    """
    Tests the functionality of the DensityMatrixSimulator by simulating a quantum circuit
    and computing its density matrix and reduced density matrices for specified subsystems.

    Initializes a DensityMatrixSimulator, sets up a quantum circuit with Hadamard, Y, X, and CNOT gates,
    and retrieves the density matrix and reduced density matrices for the entire system and various subsystems.
    Finally, prints the computed matrices and cleans up the simulator.

    Parameters:
        None

    Returns:
        None

    The function operates within the following context:
        - Machine: A DensityMatrixSimulator instance for simulating quantum circuits.
        - QProg: A quantum program that represents the quantum circuit.
        - q: Quantum registers.
        - c: Classical bits associated with the quantum registers.
        - hadamard_circuit: A function that creates a Hadamard gate circuit.
        - Y, X: Quantum gates for Y and X operations.
        - CNOT: A controlled-NOT gate.
        - get_density_matrix: Function to compute the density matrix of the quantum circuit.
        - get_reduced_density_matrix: Function to compute the reduced density matrix for a subsystem.
    """
    
    machine = DensityMatrixSimulator()
    machine.init_qvm()

    prog = QProg()
    q = machine.qAlloc_many(3)
    c = machine.cAlloc_many(3)

    prog.insert(hadamard_circuit(q))\
        .insert(Y(q[1]))\
        .insert(X(q[2]))\
        .insert(CNOT(q[0], q[1]))\
        .insert(CNOT(q[1], q[2]))\

    density_matrix = machine.get_density_matrix(prog)
    reduced_density_matrix1 = machine.get_reduced_density_matrix(prog, [0, 1, 2])
    reduced_density_matrix2 = machine.get_reduced_density_matrix(prog, [q[0], q[1], q[2]])

    reduced_density_matrix3 = machine.get_reduced_density_matrix(prog, [0])
    reduced_density_matrix4 = machine.get_reduced_density_matrix(prog, [0, 1])

    print(density_matrix)
    print(reduced_density_matrix1)
    print(reduced_density_matrix2)
    print(reduced_density_matrix3)
    print(reduced_density_matrix4)
    machine.finalize()

def test_hamitonlian_expval():
    """
    Simulates a quantum circuit with predefined gates and computes the expectation value
    of a Hamiltonian operator using a density matrix simulator.

    This function initializes a quantum virtual machine, constructs a quantum circuit
    with Hadamard, Y, X, and CNOT gates, and then calculates the expectation value
    of a custom Hamiltonian operator. The result is printed out.

    Parameters:
        None

    Returns:
        None - The expectation value is printed directly.

    Notes:
        - The function operates on a quantum virtual machine (QVM) provided by the
          DensityMatrixSimulator class within the pyQPanda package.
        - The quantum circuit consists of three qubits, with Hadamard gates applied
          to the first, a Y gate to the second, and an X gate to the third. CNOT gates
          are applied between the first and second, and the second and third qubits.
        - The Hamiltonian operator is represented by a linear combination of Pauli
          matrices, with coefficients scaled by x, y, and z operators corresponding
          to the respective qubits.
    """
    machine = DensityMatrixSimulator()
    machine.init_qvm()

    prog = QProg()
    q = machine.qAlloc_many(3)
    c = machine.cAlloc_many(3)

    prog.insert(hadamard_circuit(q))\
        .insert(Y(q[1]))\
        .insert(X(q[2]))\
        .insert(CNOT(q[0], q[1]))\
        .insert(CNOT(q[1], q[2]))\

    opt = 0.23 * x(1) + 0.2 * y(2) + 1.6 * z(0)

    expval = machine.get_expectation(prog,opt.to_hamiltonian(False),[0, 1, 2])
    print(expval)

    machine.finalize()

def test_noise_simulate():
    """
    Simulate noise on a quantum circuit using a DensityMatrixSimulator from the pyQPanda package.

    This function initializes a quantum virtual machine, constructs a simple quantum circuit with Hadamard and CNOT gates, and applies noise models to the gates. It then computes and prints the density matrices before and after noise is introduced.

    Parameters:
        None

    Returns:
        None

    Raises:
        None

    The function is designed to be a unit test for the DensityMatrixSimulator class in the pyQPanda package, demonstrating the application of noise to quantum circuits.

    Usage:
        The function can be used to test the behavior of the DensityMatrixSimulator under noise conditions, ensuring that the noise model is correctly applied and that the resulting density matrices reflect the noisy state of the quantum circuit.
    """    
    machine = DensityMatrixSimulator()
    machine.init_qvm()

    prog = QProg()
    q = machine.qAlloc_many(3)
    c = machine.cAlloc_many(3)

    prog.insert(H(q[0]))\
        .insert(CNOT(q[0], q[1]))\
        .insert(CNOT(q[1], q[2]))\


    density_matrix1 = machine.get_density_matrix(prog)

    machine.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.HADAMARD_GATE, 0.3)
    machine.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.CNOT_GATE, 0.3)

    density_matrix2 = machine.get_density_matrix(prog)

    print(density_matrix1)
    print(density_matrix2)
    
    machine.finalize()

def test_multi_noise_simulate():
    """
    Simulates a quantum circuit with multi-noise added and computes the density matrices before and after the noise.
    
    This function initializes a DensityMatrixSimulator instance, allocates quantum and classical bits,
    constructs a quantum circuit with an X gate and a CNOT gate, and then applies bit-flip noise models to
    the CNOT gate. It retrieves the density matrices of the circuit before and after the noise and prints them.
    
    Parameters:
        None
    
    Returns:
        None
    
    Raises:
        No exceptions are raised, but the function assumes that the necessary classes and methods exist
        within the pyQPanda package, such as DensityMatrixSimulator, QProg, qAlloc_many, cAlloc_many,
        insert, get_density_matrix, set_noise_model, and finalize.
    """    
    machine = DensityMatrixSimulator()
    machine.init_qvm()

    prog = QProg()
    q = machine.qAlloc_many(2)
    c = machine.cAlloc_many(2)

    prog.insert(X(q[0]))\
        .insert(CNOT(q[0], q[1]))

    density_matrix1 = machine.get_density_matrix(prog)

    # case 1 expectation: 00 -> 0.42 , 11 -> 0.58
    # machine.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.PAULI_X_GATE, 0.3)
    # machine.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.PAULI_X_GATE, 0.3)
    # density_matrix2 = machine.get_density_matrix(prog)

    # case 1 expectation: 00 -> 0.42 , 11 -> 0.58
    machine.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.CNOT_GATE, 0.2)
    machine.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.CNOT_GATE, 0.2)
    density_matrix2 = machine.get_density_matrix(prog)

    print(density_matrix1)
    print(density_matrix2)
    
    machine.finalize()

if __name__ == "__main__":

    # test_density_matrix()
    # test_probabilities()
    # test_hamitonlian_expval()
    # test_noise_simulate()
    test_multi_noise_simulate()
