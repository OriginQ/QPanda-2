'''
QPanda Python\n
Copyright (C) Origin Quantum 2017-2018\n
Licensed Under Apache Licence 2.0
'''
# keep import Operator.pyQPandaOperator first if runtime error occure
# this runtime error is from cases like this: QPanda and pyQPanda is linked with option /Mt instead of /Md
# then if use pybind11 above version 2.5.0, pybind11 will do garbage_collection to free pybind11 internal type info object
# this will cause error of "allocate object in one thread heap but dellocate it from another thread"
# change link option back to /Md will fix this bug, but let QPanda depends on system dynamic thread dll

from .pyQPanda import *
from .Operator.pyQPandaOperator import *
from .utils import *
from .Variational import back
from .Visualization import *
from .OriginService import *
import warnings

One = True
Zero = False
