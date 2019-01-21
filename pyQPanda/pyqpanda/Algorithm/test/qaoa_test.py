from pyqpanda.Hamiltonian import PauliOperator
from pyqpanda.Hamiltonian.QubitOperator import *
from pyqpanda import *
from pyqpanda.utils import *
from pyqpanda.Algorithm.hamiltonian_simulation import *
#define graph


def qaoa_test(graph,step_=1,shots_=100,method='Powell'):

    result=qaoa(graph,step_,shots_, method)
    return result