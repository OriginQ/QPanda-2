from pyqpanda import *
import numpy as np

def Generator_Weight_Circuit(Circuit,weights, qubits, rotation=None):

        rotation = rotation or RX

        shape = np.shape(weights)
        if len(shape) != 2:
            raise ValueError(f"Weights tensor must be 2-dimensional; got shape {shape}")
        if shape[1] != len(qubits):
            raise ValueError(
                f"Weights tensor must have second dimension of length {len(qubits)}; got {shape[1]}"
            )
        for i in weights:
            for k,j in enumerate(i):
                 Circuit << rotation(qubits[k], j)  
            if shape[1]>2:
                 for k,j in enumerate(i):
                      Circuit << CNOT(qubits[k], qubits[(k+1)%shape[1]])
            else:
                Circuit << CNOT(qubits[0], qubits[1])     
        return Circuit
                         

    