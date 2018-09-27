from pyqpanda.Hamiltonian import (PauliOperator, 
                                  chem_client)
                                  
from pyqpanda.Hamiltonian.QubitOperator import *
from pyqpanda import *
from pyqpanda.utils import *
from pyqpanda.Algorithm.hamiltonian_simulation import *

if __name__=='__main__':        
    print(get_config_path())
    import pyqpanda.Algorithm.demo.Deustch_Jozsa as Deustch_Jozsa
    Deustch_Jozsa.Two_Qubit_DJ_Demo()

