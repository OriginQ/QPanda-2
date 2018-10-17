from pyqpanda.Hamiltonian import PauliOperator
from pyqpanda.Hamiltonian.QubitOperator import *
from pyqpanda import *
from pyqpanda.utils import *
from pyqpanda.Algorithm.hamiltonian_simulation import *
#define graph
graph=[[0,5,0.18],[0,6,0.49],[1,6,0.59],[1,7,0.44],\
[2,7,0.56],[2,8,0.63],[4,9,0.43],[5,10,0.23],[6,11,0.64],\
[7,12,0.60],[8,13,0.36],[9,14,0.52],[10,15,0.40],[10,16,0.41],\
[11,16,0.57],[11,17,0.50],[12,17,0.71],[12,18,0.40],[13,18,0.72],\
[13,3,0.81],[14,3,0.29]]


result=qaoa(graph=graph,step_=2,shots_=100, method="Nelder-Mead")
print(result)