from pyqpanda import *
import time
PI = 3.1415926535898


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


def cpu_qvm_fun():

    machine = init_quantum_machine(QMachineType.CPU)
    machine.initQVM()

    q = machine.qAlloc_many(10)
    c = machine.cAlloc_many(10)

    prog = QProg()
    prog.insert(Hadamard_Circuit(q))\
        .insert(T(q[0]).dagger())\
        .insert(Y(q[1]))\
        .insert(RX(q[3], PI / 3))\
        .insert(RY(q[2], PI / 3))\
        .insert(CNOT(q[1], q[5]))
    # .insert(measure_all(q,c))

    print(to_QRunes(prog, machine))
    machine.finalize()


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

    # machine.run(prog)

    # result1 = machine.pmeasure("6")
    result2 = machine.pmeasure_bin_index(prog, "0000000000")
    result3 = machine.pmeasure_dec_index(prog, "1")

    # qlist = [q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9]]
    # result4 = machine.get_prob_dict(qlist, "3")

    print(result2, result3)


def partialAmp_fun():

    machine = PartialAmpQVM()

    machine.initQVM()

    q = machine.qAlloc_many(10)
    c = machine.cAlloc_many(10)

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

    result1 = machine.pmeasure("6")

    result2 = machine.pmeasure_bin_index("0000000000")
    result3 = machine.pmeasure_dec_index("1")

    qlist = [q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9]]
    result4 = machine.get_prob_dict(qlist, "3")

    print(result1, result2, result3, result4)


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

    QCM = QCloud()
    QCM.initQVM()

    qlist = QCM.qAlloc_many(10)
    clist = QCM.qAlloc_many(10)
    prog = QProg()
    for i in qlist:
        prog.insert(H(i))

    prog.insert(CZ(qlist[1], qlist[5]))\
        .insert(CZ(qlist[3], qlist[7]))\
        .insert(CZ(qlist[0], qlist[4]))\
        .insert(RZ(qlist[7], PI / 4))\
        .insert(RX(qlist[5], PI / 4))\
        .insert(RX(qlist[4], PI / 4))\
        .insert(RY(qlist[3], PI / 4))\
        .insert(CZ(qlist[2], qlist[6]))\
        .insert(RZ(qlist[3], PI / 4))\
        .insert(RZ(qlist[8], PI / 4))\
        .insert(CZ(qlist[9], qlist[5]))\
        .insert(RY(qlist[2], PI / 4))\
        .insert(RZ(qlist[9], PI / 4))\
        .insert(CZ(qlist[2], qlist[3]))

    param1 = {"RepeatNum": 1000, "token": "3CD107AEF1364924B9325305BF046FF3",
              "BackendType": QMachineType.CPU}
    param2 = {"token": "3CD107AEF1364924B9325305BF046FF3",
              "BackendType": QMachineType.CPU}

    task = QCM.run_with_configuration(prog, param1)
    print(task)

    time.sleep(3)
    result = QCM.get_result("1904301115021866")

    # print(result)
    # print(QCM.prob_run_dict(prog,qlist,param2))

    # res = QCM.get_result("1904261648207832")
    # print(res)
    QCM.finalize()


def Cluster_Cloud():

    QCM = QCloud()
    QCM.initQVM()

    qlist = QCM.qAlloc_many(10)
    clist = QCM.cAlloc_many(10)

    prog = QProg()
    prog.insert(H(qlist[0]))\
        .insert(Measure(qlist[0], clist[0]))

    # task = QCM.full_amplitude_measure(prog, 100)
    # print(task)

    # time.sleep(3)
    result = QCM.get_cluster_result(ClusterMachineType.Full_AMPLITUDE, "2001061726139435101012920")
    # result = QCM.get_cluster_result(0, "2001061726139435101012920")

    QCM.finalize()


if __name__ == "__main__":

    # cpu_qvm_fun()
    # singleAmp_fun()
    # partialAmp_fun()
    Cluster_Cloud()
    # graph_match_fun()
