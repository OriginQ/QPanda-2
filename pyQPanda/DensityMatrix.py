from numpy import pi
from pyqpanda import *

def test_probabilities():

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

if __name__ == "__main__":

    # test_density_matrix()
    # test_probabilities()
    # test_hamitonlian_expval()
    # test_noise_simulate()
