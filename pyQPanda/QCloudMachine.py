from pyqpanda import NoiseModel
from pyqpanda import GateType
from numpy import pi
from pyqpanda import *
import time
PI = 3.1415926535898

def test_init_state():

    qvm = init_quantum_machine()
    qvm.initQVM()

    prog = QProg()

    q = qvm.qAlloc_many(3)
    c = qvm.cAlloc_many(3)

    prog.insert(H(q[0]))\
        .insert(CNOT(q[0], q[1]))\
        .insert(CNOT(q[1], q[2]))\
        .insert(measure_all(q, c))

    qvm.directly_run(prog)
    state = qvm.get_qstate()



    print(state)

def test_cpu_run_with_no_cbits_args():

    qvm = CPUQVM()
    qvm.initQVM()

    prog = QProg()

    q = qvm.qAlloc_many(4)
    c = qvm.cAlloc_many(4)

    prog.insert(H(q[0]))\
        .insert(CNOT(q[0], q[1]))\
        .insert(CNOT(q[1], q[2]))\
        .insert(CNOT(q[2], q[3]))\
        .insert(measure_all(q, c))

    result0 = qvm.run_with_configuration(prog, c, 10000)
    result1 = qvm.run_with_configuration(prog, 10000)

    print(result0)
    print(result1)

def test_mps_run_with_no_cbits_args():

    qvm = MPSQVM()
    qvm.initQVM()

    prog = QProg()

    q = qvm.qAlloc_many(4)
    c = qvm.cAlloc_many(4)

    prog.insert(H(q[0]))\
        .insert(CNOT(q[0], q[1]))\
        .insert(CNOT(q[1], q[2]))\
        .insert(CNOT(q[2], q[3]))\
        .insert(measure_all(q, c))

    result0 = qvm.run_with_configuration(prog, c, 10000)
    result1 = qvm.run_with_configuration(prog, 10000)

    print(result0)
    print(result1)

def test_noise_run_with_no_cbits_args():

    qvm = NoiseQVM()
    qvm.init_qvm()

    prog = QProg()

    q = qvm.qAlloc_many(4)
    c = qvm.cAlloc_many(4)

    prog.insert(H(q[0]))\
        .insert(CNOT(q[0], q[1]))\
        .insert(CNOT(q[1], q[2]))\
        .insert(CNOT(q[2], q[3]))\
        .insert(measure_all(q, c))

    qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.PAULI_X_GATE, 0.01)
    result0 = qvm.run_with_configuration(prog, c, 100000)
    result1 = qvm.run_with_configuration(prog, 100000)

    print(result0)
    print(result1)


def utilities_fun():

    machine = init_quantum_machine(QMachineType.CPU)

    prog = QProg()
    q = machine.qAlloc_many(6)
    c = machine.cAlloc_many(6)

    prog.insert(H(q[0]))\
        .insert(Y(q[5]))\
        .insert(S(q[2]))\
        .insert(CZ(q[0], q[1]))

    print(to_QRunes(prog, machine))
    print(to_QASM(prog, machine))
    print(to_Quil(prog, machine))
    print(count_gate(prog, machine))
    print(get_clock_cycle(machine, prog))
    print(get_bin_str(prog, machine))

    machine.finalize()


def MPS_fun():

    qvm = MPSQVM()
    qvm.set_configure(64, 64)
    qvm.init_qvm()

    q = qvm.qAlloc_many(10)
    c = qvm.cAlloc_many(10)

    # 构建量子程序
    prog = QProg()
    prog.insert(hadamard_circuit(q))\
        .insert(CZ(q[2], q[4]))\
        .insert(CZ(q[3], q[7]))\
        .insert(CNOT(q[0], q[1]))

    # 量子程序运行1000次，并返回测量结果
    result = qvm.pmeasure_bin_subset(prog, ["0000000001"])

    # 打印量子态在量子程序多次运行结果中出现的次数
    print(result)

    qvm.finalize()


def test_state():

    QCM = CPUQVM()

    QCM.initQVM()

    q = QCM.qAlloc_many(6)
    c = QCM.cAlloc_many(6)

    prog = QProg()

    prog.insert(hadamard_circuit(q))\
        .insert(CZ(q[2], q[4]))\
        .insert(CZ(q[3], q[1]))\
        .insert(CZ(q[0], q[4]))\
        .insert(RZ(q[3], PI / 4))\
        .insert(RX(q[5], PI / 4))\
        .insert(RX(q[4], PI / 4))\
        .insert(RY(q[3], PI / 4))\
        .insert(CZ(q[2], q[4]))\
        .insert(RZ(q[3], PI / 4))\
        .insert(RZ(q[2], PI / 4))\
        .insert(CNOT(q[0], q[1]))\
        .insert(measure_all(q, c))

    result = QCM.run_with_configuration(prog, c, 1000)

    stat = QCM.get_qstate()
    return stat


def test_cpu_state():

    QCM = CPUQVM()

    QCM.initQVM()

    q = QCM.qAlloc_many(6)
    c = QCM.cAlloc_many(6)

    prog = QProg()

    prog.insert(hadamard_circuit(q))\
        .insert(CZ(q[2], q[4]))\
        .insert(CZ(q[3], q[1]))\
        .insert(CZ(q[0], q[4]))\
        .insert(RZ(q[3], PI / 4))\
        .insert(RX(q[5], PI / 4))\
        .insert(RX(q[4], PI / 4))\
        .insert(RY(q[3], PI / 4))\
        .insert(CZ(q[2], q[4]))\
        .insert(RZ(q[3], PI / 4))\
        .insert(RZ(q[2], PI / 4))\
        .insert(CNOT(q[0], q[1]))\
        .insert(RX(q[4], PI / 2))\
        .insert(RX(q[5], PI / 2))\
        .insert(CR(q[0], q[1], PI))\
        .insert(RY(q[1], PI / 2))\
        .insert(RY(q[2], PI / 2))\
        .insert(RZ(q[3], PI / 4))\
        .insert(CR(q[2], q[1], PI))\
        .insert(measure_all(q, c))

    result = QCM.run_with_configuration(prog, c, 1000)
    stat = QCM.get_qstate()
    return stat


def test_init_state():

    ss = test_state()

    QCM = CPUSingleThreadQVM()

    QCM.initQVM()

    q = QCM.qAlloc_many(6)
    c = QCM.cAlloc_many(6)

    QCM.init_state(ss)

    prog = QProg()

    prog.insert(RX(q[4], PI / 2))\
        .insert(RX(q[5], PI / 2))\
        .insert(CR(q[0], q[1], PI))\
        .insert(RY(q[1], PI / 2))\
        .insert(RY(q[2], PI / 2))\
        .insert(RZ(q[3], PI / 4))\
        .insert(CR(q[2], q[1], PI))\
        .insert(measure_all(q, c))

    result = QCM.run_with_configuration(prog, c, 1000)
    stat = QCM.get_qstate()
    stat1 = test_cpu_state()
    for i in range(0, len(stat)):
        print(stat[i])
        print(" : ")
        print(stat1[i])
        print("\n")


def cpu_qvm_fun():

    qvm = CPUQVM()
    qvm.initQVM()
    qubits = qvm.qAlloc_many(4)
    cbits = qvm.cAlloc_many(4)

    # 构建量子程序
    prog = QProg()
    prog.insert(H(qubits[0])).insert(
        CNOT(qubits[0], qubits[1])).insert(Measure(qubits[0], cbits[0]))

    # 量子程序运行1000次，并返回测量结果
    result = qvm.run_with_configuration(prog, cbits, 1000)

    # 打印量子态在量子程序多次运行结果中出现的次数
    print(result)

    qvm.finalize()


def singleAmp_fun():

    machine = SingleAmpQVM()

    machine.initQVM()

    q = machine.qAlloc_many(10)
    c = machine.cAlloc_many(10)

    prog = QProg()

    prog.insert(hadamard_circuit(q))\
        .insert(CZ(q[1], q[5]))\
        .insert(CZ(q[3], q[5]))\
        .insert(CZ(q[2], q[4]))\
        .insert(CZ(q[3], q[7]))\
        .insert(CZ(q[0], q[4]))\
        .insert(RY(q[7], PI / 2))\
        .insert(RX(q[8], PI / 2))\
        .insert(RX(q[9], PI / 2))\
        .insert(CR(q[0], q[1], PI))\
        .insert(CR(q[2], q[3], PI))\
        .insert(RY(q[4], PI / 2))\
        .insert(RZ(q[5], PI / 4))\
        .insert(RX(q[6], PI / 2))\
        .insert(RZ(q[7], PI / 4))\
        .insert(CR(q[8], q[9], PI))\
        .insert(CR(q[1], q[2], PI))\
        .insert(RY(q[3], PI / 2))\
        .insert(RX(q[4], PI / 2))\
        .insert(RX(q[5], PI / 2))\
        .insert(CR(q[9], q[1], PI))\
        .insert(RY(q[1], PI / 2))\
        .insert(RY(q[2], PI / 2))\
        .insert(RZ(q[3], PI / 4))\
        .insert(CR(q[7], q[8], PI))

    machine.run(prog)

    # result1 = machine.pmeasure("6")
    # result2 = machine.pmeasure_bin_index(prog, "0000000000")
    # result3 = machine.pmeasure_dec_index(prog, "1")
    # result = machine.pmeasure("6")
    # print(result3)

    qlist = [q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9]]
    result4 = machine.get_prob_dict(qlist, "3")


def partialAmp_fun():

    PI = 3.141593
    machine = PartialAmpQVM()
    machine.init_qvm()

    q = machine.qAlloc_many(10)
    c = machine.cAlloc_many(10)

    # 构建量子程序
    prog = QProg()
    prog.insert(hadamard_circuit(q))\
        .insert(CZ(q[1], q[5]))\
        .insert(CZ(q[3], q[7]))\
        .insert(CZ(q[0], q[4]))\
        .insert(RZ(q[7], PI / 4))\
        .insert(RX(q[5], PI / 4))\
        .insert(RX(q[4], PI / 4))\
        .insert(RY(q[3], PI / 4))\
        .insert(CZ(q[2], q[6]))\
        .insert(RZ(q[3], PI / 4))\
        .insert(RZ(q[8], PI / 4))\
        .insert(CZ(q[9], q[5]))\
        .insert(RY(q[2], PI / 4))\
        .insert(RZ(q[9], PI / 4))\
        .insert(CZ(q[2], q[3]))

    machine.run(prog)

    result2 = machine.pmeasure_bin_index("0000000000")
    result3 = machine.pmeasure_dec_index("1")

    qlist = ["0", "1", "2"]
    result4 = machine.pmeasure_subset(qlist)

    print(result2, result3, result4)


def graph_match_fun():

    machine = init_quantum_machine(QMachineType.CPU)
    q = machine.qAlloc_many(4)
    c = machine.cAlloc_many(4)

    #           ┌─┐┌────┐┌─┐
    # q_0:  |0>─┤H├┤CNOT├┤H├───────────────
    #           └─┘└──┬─┘└─┘
    # q_1:  |0>───────■─────■──────────────
    #           ┌─┐      ┌──┴─┐┌─┐
    # q_2:  |0>─┤H├──────┤CNOT├┤H├───■─────
    #           ├─┤      └────┘└─┘┌──┴─┐┌─┐
    # q_3:  |0>─┤H├───────────────┤CNOT├┤H├
    #           └─┘               └────┘└─┘

    #           ┌──┐
    # q_0:  |0>─┤CZ├────────
    #           └─┬┘
    # q_1:  |0>───■───■─────
    #               ┌─┴┐
    # q_2:  |0>─────┤CZ├──■─
    #               └──┘┌─┴┐
    # q_3:  |0>─────────┤CZ├
    #                   └──┘

    prog = QProg()
    prog.insert(H(q[0]))\
        .insert(H(q[2]))\
        .insert(H(q[3]))\
        .insert(CNOT(q[1], q[0]))\
        .insert(H(q[0]))\
        .insert(CNOT(q[1], q[2]))\
        .insert(H(q[2]))\
        .insert(CNOT(q[2], q[3]))\
        .insert(H(q[3]))

    query_cir = QCircuit()
    query_cir.insert(H(q[0]))\
             .insert(CNOT(q[1], q[0]))\
             .insert(H(q[0]))

    replace_cir = QCircuit()
    replace_cir.insert(CZ(q[0], q[1]))

    print("before replace")
    print_qprog(prog)

    update_prog = graph_query_replace(prog, query_cir, replace_cir, machine)

    print("after replace")
    # print(to_originir(update_prog,machine))
    print_qprog(update_prog)


def QCloud_fun():

    PI = 3.1416

    QCM = QCloud()
    QCM.init_qvm("E02BB115D5294012AA88D4BE82603984", True)

    q = QCM.qAlloc_many(6)
    c = QCM.cAlloc_many(6)

    prog = QProg()
    prog << hadamard_circuit(q)\
        << RX(q[1], PI / 4)\
        << RX(q[2], PI / 4)\
        << RX(q[1], PI / 4)\
        << CZ(q[0], q[1])\
        << CZ(q[1], q[2])\
        << Measure(q[0], c[0])\
        << Measure(q[1], c[1])

    result5 = QCM.real_chip_measure(
        prog, 1000, real_chip_type.origin_wuyuan_d4)
    print(result5)
    print("real_chip_measure pass !")

    result6 = QCM.get_state_tomography_density(
        prog, 1000, real_chip_type.origin_wuyuan_d4)
    print(result6)
    print("get_state_tomography_density !")

    result6 = QCM.get_state_fidelity(
        prog, 1000, real_chip_type.origin_wuyuan_d4)
    print(result6)
    print("get_state_fidelity !")

    QCM.finalize()


def Cluster_Cloud():

    QCM = QCloud()
    QCM.init_qvm("E02BB115D5294012AA88D4BE82603984", True)

    QCM.set_qcloud_api("https://qcloud.originqc.com.cn")

    qlist = QCM.qAlloc_many(10)
    clist = QCM.cAlloc_many(10)

    prog = QProg()
    prog.insert(H(qlist[0]))\
        .insert(Measure(qlist[0], clist[0]))

    task = QCM.full_amplitude_measure(prog, 100, "123")
    print(task)

    prog1 = QProg()
    prog1.insert(H(qlist[0]))

    task2 = QCM.full_amplitude_pmeasure(prog1, [0, 1, 2], "123")
    print(task2)

    task3 = QCM.single_amplitude_pmeasure(prog1, "1", "123")
    print(task3)

    task4 = QCM.partial_amplitude_pmeasure(prog1, ["1"], "123")
    print(task4)

    QCM.finalize()


def noise_fun():
    qvm = MPSQVM()
    qvm.set_configure(20, 20)

    # default argc
    qubits_num = 1
    shot = 1000

    # 设置噪声模型参数
    # qvm.set_noise_model(NoiseModel.DEPHASING_KRAUS_OPERATOR, GateType.HADAMARD_GATE, [noise_rate])
    # qvm.set_noise_model(NoiseModel.DEPHASING_KRAUS_OPERATOR,GateType.CPHASE_GATE, [2 * noise_rate])

    qvm.add_single_noise_model(
        NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.PAULI_X_GATE, 0.9)
    qvm.add_single_noise_model(
        NoiseModel.DEPHASING_KRAUS_OPERATOR, GateType.PAULI_X_GATE, 0.4)
    qvm.add_single_noise_model(
        NoiseModel.DEPOLARIZING_KRAUS_OPERATOR, GateType.PAULI_X_GATE, 0.999)
    qvm.add_single_noise_model(
        NoiseModel.DECOHERENCE_KRAUS_OPERATOR, GateType.PAULI_X_GATE, 5, 5, 0.9)

    qvm.init_qvm()

    q = qvm.qAlloc_many(qubits_num)
    c = qvm.cAlloc_many(qubits_num)

    prog = QProg()
    prog << X(q[0]) << measure_all(q, c)

    # for i in range(0, qubits_num):
    #     target = q[qubits_num - 1 - i]
    #     prog.insert(H(target))
    #     for j in range(i + 1, qubits_num):
    #         control = q[qubits_num - 1 - j]
    #         prog.insert(CR(control, target, 2 * pi / (1 << (j - i + 1))))
    # prog.insert(measure_all(q, c))

    result = qvm.run_with_configuration(prog, c, shot)
    print(result)

    qvm.finalize()


def jkuqvm_fun():

    machine = JKUQVM()
    machine.set_configure(50, 50)
    machine.init_qvm()

    q = machine.qAlloc_many(1)
    c = machine.cAlloc_many(1)

    prog = QProg()
    prog.insert(X1(q[0]))\
        .insert(Z1(q[0]))\
        .insert(Y1(q[0]))\
        .insert(Measure(q[0], c[0]))

    result = machine.run_with_configuration(prog, c, 100)
    print(result)

    machine.finalize()


def mps_noise():

    mps = MPSQVM()
    mps.set_configure(50, 50)
    mps.init_qvm()

    q = mps.qAlloc_many(6)
    c = mps.cAlloc_many(6)

    prog = QProg()
    prog.insert(X(q[0]))\
        .insert(X(q[1]))\
        .insert(Measure(q[0], c[0]))\
        .insert(Measure(q[1], c[1]))

    mps.set_measure_error(NoiseModel.BITFLIP_KRAUS_OPERATOR, 0.2)
    result = mps.run_with_configuration(prog, c, 1000)
    print(result)

    mps.finalize()


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
    # cir.insert(RX(q[0], pi/2))\
    #    .insert(RZ(q[0], pi/2))\
    #    .insert(RX(q[0], pi / 2))\
    #    .insert(RX(q[0], pi/2))\
    #    .insert(RZ(q[0], pi/4))\
    #    .insert(RZ(q[0], pi/4))

    # cir << X(q[0])
    cir << RY(q[0], pi / 3)
    # cir << RY(q[0], pi / 6)
    # cir << H(q[0])
    # cir << RX(q[0], -pi / 2)

    prog = QProg()
    prog << cir

    # machine.directly_run(prog)
    # result = machine.get_qstate()
    # print(result)

    plot_bloch_circuit(cir)
    plt.show()
    # machine.finalize()


def plot_bloch_vectors():

    machine = CPUQVM()
    machine.set_configure(50, 50)
    machine.init_qvm()

    q = machine.qAlloc_many(2)
    c = machine.cAlloc_many(2)

    prog = QProg()
    prog.insert(X(q[1]))\
        .insert(H(q[0]))\
        .insert(T(q[0]))\
        .insert(Z(q[0]))\
        .insert(RX(q[1], pi/2))\
        .insert(RZ(q[1], pi/2))\
        .insert(RX(q[1], pi/2))\
        .insert(RY(q[0], pi/4))\
        .insert(RX(q[0], pi/3))

    machine.directly_run(prog)
    result = machine.get_qstate()

    # plot_bloch_vector([0, 1, 0])
    plot_bloch_multivector(result)
    machine.finalize()


if __name__ == "__main__":

    test_global_cpu_run_with_no_cbits_args()
    test_cpu_run_with_no_cbits_args()
    test_mps_run_with_no_cbits_args()
    test_noise_run_with_no_cbits_args()
    # partialAmp_fun()
    # MPS_fun()
    # cpu_qvm_fun()
    # singleAmp_fun()
    # partialAmp_fun()
    # Cluster_Cloud()
    # graph_match_fun()
    # noise_fun()
    # jkuqvm_fun()
    # plot_state()
    # plot_density()
    # plot_bloch_vectors()
    # mps_noise()
    # plot_bloch_cir()
    # plot_density()
