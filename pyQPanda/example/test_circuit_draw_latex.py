# code for inplace run test, add \qpanda-2.\pyQPanda to module search paths
import sys
import os.path
dir = os.path.abspath(__file__)
model_path = os.path.abspath(os.path.join(dir, "../.."))
sys.path.insert(0, model_path)

import unittest
from math import pi
from pyqpanda import *

dest_latex_src = r'''\documentclass[border=2px]{standalone}

\usepackage[braket, qm]{qcircuit}
\usepackage{graphicx}

\begin{document}
OriginQ\\
\\
\\
\\
\scalebox{1.0}{
\Qcircuit @C = 1.0em @R = 0.2em @!R{ \\
\nghost{q_{0}  \ket{0}} & \lstick{q_{0}  \ket{0}}&\qw&\qw&\qw&\qw&\ctrl{2}&\gate{\mathrm{H}}&\qw&\ctrl{1} \barrier[0em]{3}&\qw&\gate{\mathrm{H}}&\qw&\meter&\qw&\rstick{}\qw & \nghost{}\\
\nghost{q_{1}  \ket{0}} & \lstick{q_{1}  \ket{0}}&\qswap&\qw&\gate{\mathrm{CR}\,\mathrm{(1.570796)}^\dagger}&\gate{\mathrm{H}}&\qw&\qw&\ctrl{1}&\targ&\qw&\ctrl{-1}&\qw&\qw&\meter&\rstick{}\qw & \nghost{}\\
\nghost{q_{2}  \ket{0}} & \lstick{q_{2}  \ket{0}}&\qswap\qwx[-1] \barrier[0em]{0}&\qw&\ctrl{-1}&\gate{\mathrm{S}^\dagger}&\gate{\mathrm{CU}\,\mathrm{(1.000000,2.000000,3.000000,4.000000)}^\dagger}&\gate{\mathrm{H}}&\gate{\mathrm{CZ}}&\gate{\mathrm{S}^\dagger}&\qw&\meter&\qw&\qw&\qw&\rstick{}\qw & \nghost{}\\
\nghost{q_{3}  \ket{0}} & \lstick{q_{3}  \ket{0}} \barrier[0em]{0}&\qw&\gate{\mathrm{H}}&\qw&\qw&\qw&\qw&\qw&\qw&\qw&\qw&\meter&\qw&\qw&\rstick{}\qw & \nghost{}\\
\nghost{q_{6}  \ket{0}} & \lstick{q_{6}  \ket{0}}&\gate{\mathrm{S}^\dagger}&\qw&\qw&\qw&\qw&\qw&\qw&\qw&\qw&\qw&\qw&\qw&\qw&\rstick{}\qw & \nghost{}\\
\nghost{c_{0}  0} & \lstick{\mathrm{c_{0}  0}}&\cw&\cw&\cw&\cw&\cw&\cw&\cw&\cw&\cw&\cw&\cw&\dstick{_{_{\hspace{0.0em}0}}} \cw \ar @{<=} [-5, 0]&\cw&\rstick{\mathrm{}}\cw & \nghost{}\\
\nghost{c_{1}  0} & \lstick{\mathrm{c_{1}  0}}&\cw&\cw&\cw&\cw&\cw&\cw&\cw&\cw&\cw&\cw&\cw&\cw&\dstick{_{_{\hspace{0.0em}1}}} \cw \ar @{<=} [-5, 0]&\rstick{\mathrm{}}\cw & \nghost{}\\
\nghost{c_{2}  0} & \lstick{\mathrm{c_{2}  0}}&\cw&\cw&\cw&\cw&\cw&\cw&\cw&\cw&\cw&\dstick{_{_{\hspace{0.0em}2}}} \cw \ar @{<=} [-5, 0]&\cw&\cw&\cw&\rstick{\mathrm{}}\cw & \nghost{}\\
\nghost{c_{3}  0} & \lstick{\mathrm{c_{3}  0}}&\cw&\cw&\cw&\cw&\cw&\cw&\cw&\cw&\cw&\cw&\dstick{_{_{\hspace{0.0em}3}}} \cw \ar @{<=} [-5, 0]&\cw&\cw&\rstick{\mathrm{}}\cw & \nghost{}\\
\\ }}
\end{document}
'''

class Test_Draw_Latex(unittest.TestCase):
    def test_run(self):
        qvm = CPUQVM()
        qvm.init_qvm()

        qv = qvm.qAlloc_many(4)
        qv1 = qvm.qAlloc_many(4)
        cv = qvm.cAlloc_many(4)

        cir = QCircuit()

        prog = QProg()
        cir << H(qv[0]).control(qv[1]) << BARRIER(qv) << S(qv[2]) << CNOT(qv[0], qv[1]) \
            << CZ(qv[1], qv[2]) << S(qv1[2]) << H(qv) << CU(1, 2, 3, 4, qv[0], qv[2]) \
            << S(qv[2]) << CR(qv[2], qv[1], pi / 2) << BARRIER(qv[2]) << BARRIER(qv[3]) << SWAP(qv[1], qv[2])

        cir.set_dagger(True)
        prog << cir
        prog.insert(measure_all(qv, cv))

        tex_str = draw_qprog(prog, 'latex', filename='python_test.tex', with_logo=True)
        # check tex content
        self.assertEqual(tex_str, dest_latex_src)
        # check tex file is exported
        self.assertTrue(os.path.isfile('python_test.tex'))

        qvm.finalize()


if __name__ == "__main__":
    unittest.main(verbosity=2)
