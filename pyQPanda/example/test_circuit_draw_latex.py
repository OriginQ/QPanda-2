import os.path
dir = os.path.abspath(__file__)
model_path = os.path.abspath(os.path.join(dir, "../.."))
import sys
sys.path.insert(0, model_path)

from pyqpanda import *
from math import pi
import unittest



class Test_Async_Run(unittest.TestCase):
    def test_run(self):
        qvm = CPUQVM()
        qvm.init_qvm()

        qv = qvm.qAlloc_many(4)
        cv = qvm.cAlloc_many(4)

        cir = QCircuit ()

        prog = QProg()
        cir << H(qv[0]).control(qv[1]) << S(qv[2]) << CNOT(qv[0], qv[1]) << \
            CZ(qv[1], qv[2]) << S(qv[1]) << H(qv) << CU(1, 2, 3, 4, qv[0], qv[2]) << \
            S(qv[2]) << CR(qv[2], qv[1], pi / 2) << SWAP(qv[1], qv[2]) 

        cir.set_dagger(True)
        prog << cir
        prog.insert(measure_all(qv, cv))

        tex_str = draw_qprog(prog, 'latex', filename='python_test.tex' )
        # with open

        qvm.finalize()


if __name__ == "__main__":
    unittest.main(verbosity=2)
