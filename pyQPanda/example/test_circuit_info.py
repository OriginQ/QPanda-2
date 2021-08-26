# from pyqpanda import *
import pyqpanda.pyQPanda as pq
import math
import unittest


class InitQMachine:
    def __init__(self, quBitCnt, cBitCnt, machineType=pq.QMachineType.CPU):
        self.m_machine = pq.init_quantum_machine(machineType)
        self.m_qlist = self.m_machine.qAlloc_many(quBitCnt)
        self.m_clist = self.m_machine.cAlloc_many(cBitCnt)
        self.m_prog = pq.QProg()

    def __del__(self):
        pq.destroy_quantum_machine(self.m_machine)

class Test_circuit_info(unittest.TestCase):
    # 测试接口： 获取指定位置前后逻辑门类型

    # @unittest.skip("skip")
    def test_get_adjacent_qgate_type(self):
        init_machine = InitQMachine(8, 8)
        qlist = init_machine.m_qlist
        clist = init_machine.m_clist
        prog = pq.QProg()
        prog.insert(pq.T(qlist[0])).insert(pq.CNOT(qlist[1], qlist[2])).insert(pq.H(qlist[3])).insert(
            pq.H(qlist[4])).insert(pq.measure_all(qlist, clist))
        iter = prog.begin()
        iter = iter.get_next()
        type = iter.get_node_type()
        if pq.NodeType.GATE_NODE == type:
            gate = pq.QGate(iter)
            print(gate.gate_type())
        list = pq.get_adjacent_qgate_type(prog, iter)
        # print(len(list))
        #
        # gateFront = pq.QGate(list[0])
        # print(gateFront.gate_type())
        # gateBack = pq.QGate(list[1])
        # print(gateBack.gate_type())


    # 测试接口： 判断逻辑门是否符合量子拓扑结构
    # @unittest.skip("skip")
    def test_is_match_topology(self):
        init_machine = InitQMachine(8, 8)
        qlist = init_machine.m_qlist
        clist = init_machine.m_clist
        a = pq.CNOT(qlist[1], qlist[3])
        list = [[1, 1, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        print(len(list))

        if (pq.is_match_topology(a, list)) == True:
            print('Match !\n')
        else:
            print('Not match.')


    # 测试接口： Qprog 转 originiR
    # @unittest.skip("skip")
    def test_to_originir(self):
        init_machine = InitQMachine(8, 8)
        qlist = init_machine.m_qlist
        clist = init_machine.m_clist
        machine = init_machine.m_machine
        prog = pq.QProg()
        prog.insert(pq.H(qlist[2])).insert(pq.measure_all(qlist, clist))
        print(pq.to_originir(prog, machine))


    # 测试接口： 判断指定的两个逻辑门是否可以交换位置
    # @unittest.skip("skip")
    def test_is_swappable(self):
        init_machine = InitQMachine(8, 8)
        q = init_machine.m_qlist
        c= init_machine.m_clist
        machine = init_machine.m_machine
        prog = pq.QProg()
        cir = pq.QCircuit()
        cir2 = pq.QCircuit()
        cir2.insert(pq.H(q[0])).insert(pq.RX(q[1], math.pi / 2)).insert(pq.T(q[2])).insert(pq.RY(q[3], math.pi / 2)).insert(
            pq.RZ(q[2], math.pi / 2))
        cir.insert(pq.H(q[1])).insert(cir2).insert(pq.CR(q[1], q[2], math.pi / 2))
        prog.insert(pq.H(q[0])).insert(pq.S(q[2])) \
            .insert(cir) \
            .insert(pq.CNOT(q[0], q[1])).insert(pq.CZ(q[1], q[2])).insert(pq.measure_all(q, c))

        iter_first = cir2.begin()

        iter_second = iter_first.get_next()
        iter_second = iter_second.get_next()
        iter_second = iter_second.get_next()

        type = iter_first.get_node_type()
        if pq.NodeType.GATE_NODE == type:
            gate = pq.QGate(iter_first)
            print(gate.gate_type())

        type = iter_second.get_node_type()
        if pq.NodeType.GATE_NODE == type:
            gate = pq.QGate(iter_second)
            print(gate.gate_type())

        if (pq.is_swappable(prog, iter_first, iter_second)) == True:
            print('Could be swapped !\n')
        else:
            print('Could NOT be swapped.')


    # 测试接口： 获取连续逻辑门的矩阵信息
    # @unittest.skip("skip")
    def test_get_matrix(self):
        init_machine = InitQMachine(8, 8)
        q= init_machine.m_qlist
        c = init_machine.m_clist
        machine = init_machine.m_machine
        prog = pq.QProg()
        prog.insert(pq.H(q[0])).insert(pq.S(q[2])).insert(pq.CNOT(q[0], q[1])).insert(pq.CZ(q[1], q[2])).insert(
            pq.CR(q[1], q[2], math.pi / 2))
        iter_start = prog.begin()
        iter_end = iter_start.get_next()
        iter_end = iter_end.get_next()
        # result_mat = pq.get_matrix(prog, iter_start, iter_end)
        # # pq.print_mat(result_mat)


if __name__ == "__main__":
    unittest.main(verbosity=2)

