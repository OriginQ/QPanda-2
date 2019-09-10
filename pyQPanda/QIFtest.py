
from pyqpanda import *

if __name__ == "__main__":

    init(QMachineType.CPU)
    qubits = qAlloc_many(3)
    cbits = cAlloc_many(3)
    cbits[0].setValue(0)
    cbits[1].setValue(1)

    prog = QProg()
    prog_while = QProg()
    prog_while.insert(H(qubits[0])).insert(H(qubits[1])).insert(H(qubits[2]))\
              .insert(assign(cbits[0], cbits[0] + 1)).insert(Measure(qubits[1], cbits[1]))
    qwhile = CreateWhileProg(cbits[1], prog_while)
    prog.insert(qwhile)
    result = directly_run(prog)
    print(cbits[0].eval())
    print(result)
    finalize()
