from pyqpanda import *
from numpy import pi
from matplotlib import pyplot as plt


def plot_state():

    machine = CPUQVM()
    machine.set_configure(50, 50)
    machine.init_qvm()

    q = machine.qAlloc_many(4)
    c = machine.cAlloc_many(4)

    prog = QProg()
    prog.insert(X(q[1]))\
        .insert(H(q[0]))\
        .insert(RX(q[1], pi/2))\
        .insert(RZ(q[0], pi/4))

    machine.directly_run(prog)
    result = machine.get_qstate()

    plot_state_city(result)
    machine.finalize()


def plot_density():

    machine = CPUQVM()
    machine.set_configure(50, 50)
    machine.init_qvm()

    q = machine.qAlloc_many(4)
    c = machine.cAlloc_many(4)

    prog = QProg()
    prog.insert(X(q[1]))\
        .insert(H(q[0]))\
        .insert(H(q[1]))\
        .insert(H(q[2]))\
        .insert(RX(q[1], pi/2))\
        .insert(RY(q[3], pi/3))\
        .insert(RZ(q[0], pi/4))\
        .insert(RZ(q[1], pi))\
        .insert(RZ(q[2], pi))\
        .insert(RZ(q[3], pi))

    machine.directly_run(prog)
    result = machine.get_qstate()

    rho = state_to_density_matrix(result)
    plot_density_matrix(rho)
    machine.finalize()


def plot_bloch_cir():

    machine = CPUQVM()
    machine.set_configure(50, 50)
    machine.init_qvm()

    q = machine.qAlloc_many(1)
    c = machine.cAlloc_many(1)

    cir = QCircuit()
    cir.insert(X(q[0]))\
       .insert(H(q[0]))\
       .insert(RX(q[0], pi/2))\
       .insert(RZ(q[0], pi/4))

    plot_bloch_circuit(cir)
    machine.finalize()


def plot_bloch_vectors():

    machine = CPUQVM()
    machine.set_configure(50, 50)
    machine.init_qvm()

    q = machine.qAlloc_many(2)
    c = machine.cAlloc_many(2)

    prog = QProg()
    prog.insert(X(q[1]))\
        .insert(H(q[0]))\
        .insert(RX(q[1], pi/2))\
        .insert(RY(q[0], pi/4))\
        .insert(RX(q[0], pi/3))

    machine.directly_run(prog)
    result = machine.get_qstate()

    # plot_bloch_vector([0, 1, 0])
    plot_bloch_multivector(result)
    machine.finalize()


if __name__ == "__main__":
    qvm = CPUQVM()
    qvm.initQVM()

    prog = QProg()
    prog1 = QProg()
    cir = QCircuit()

    q = qvm.qAlloc_many(3)
    c = qvm.cAlloc_many(3)

    prog.insert(H(q[0]))\
        .insert(H(q[1]))\
        .insert(H(q[2]))\
        .insert(RZ(q[2], pi / 4))\
        .insert(RZ(q[2], pi / 4))\
        .insert(CNOT(q[1], q[2]))\
        .insert(CNOT(q[2], q[1]))\
        .insert(measure_all(q, c))

    qvm.run_with_configuration(prog, c, 100)
    result = qvm.get_qstate()
    qvm.finalize()

    # test draw_state_city arg :state vector
    draw_state_city(result)

    rho = state_to_density_matrix(result)

    # test draw_density_matrix arg :density matrix
    fig = draw_density_matrix(rho)
    plt.show()
