import pyqpanda as pq
from circuit_info import get_circuit_info
from circuit_composer import CircuitComposer

if __name__ == '__main__':
    circ = pq.QCircuit()
    qvm = pq.CPUQVM()
    qvm.init_qvm()

    q = qvm.qAlloc_many(4)
    c = qvm.cAlloc_many(4)

    circ1 = CircuitComposer(4)

    circ << pq.H(q[0]) << pq.CNOT(q[0], q[1]) << pq.CNOT(q[1], q[2]) \
        << pq.RX(q[0], 1.0) << pq.U2(q[1], 2.0, 3.0) \
        << pq.U3(q[2], 4.0, 5.0, 6.0) \
        << pq.U4([7, 8, 9, 10], q[3])

    circ1 << circ << circ

    info = get_circuit_info(circ1)
    print(info)
