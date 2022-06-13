#!/usr/bin/env python
# coding=utf-8
import sys
sys.path.append("../../") 
from pyQPanda.pyqpanda import *

init(QMachineType.CPU)
prog = QProg()
q = qAlloc_many(2)
c = cAlloc_many(2)
prog.insert(H(q[0])).insert(CNOT(q[0],q[1])).insert(measure_all(q, c))
results = run_with_configuration(prog, c, 1000)
print(results)

finalize()
