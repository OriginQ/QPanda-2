'''
QPanda Python\n
Copyright (C) Origin Quantum 2017-2018\n
Licensed Under Apache Licence 2.0
'''

from .utils import *
from .pyQPanda import *
from .Operator.pyQPandaOperator import *
from .Variational import back
import warnings

try:
    from .ChemiQ.pyQPandaChemiQ import *
except ImportError as e:
    warnings.warn("No module named ChemiQ")


One = True
Zero = False
