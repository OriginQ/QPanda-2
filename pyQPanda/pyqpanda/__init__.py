'''
QPanda Python\n
Copyright (C) Origin Quantum 2017-2018\n
Licensed Under Apache Licence 2.0
'''
from pyqpanda import utils
from pyqpanda.pywrapper import *
from math import pi
import numpy as np

One=True
Zero=False

PauliZ=np.array([[1,0],[0,-1]])
PauliX=np.array([[0,1],[1,0]])
PauliY=np.array([[0,-1j],[1j,0]])
I=np.eye(2)
pauli_mat={'I':I,'X':PauliX,'Y':PauliY,'Z':PauliZ}