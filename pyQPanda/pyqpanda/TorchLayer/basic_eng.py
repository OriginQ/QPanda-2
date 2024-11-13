from pyqpanda import *
import numpy as np

def Generator_Weight_Circuit(Circuit,weights, qubits, rotation=None):
        """
        Construct a quantum circuit based on given weights and qubits, applying single-qubit rotations and possibly CNOT gates.

        The function creates a quantum circuit by applying single-qubit rotations to each qubit according to the provided weights.
        Additionally, it optionally applies CNOT gates to couple qubits based on the structure of the weights tensor.

            Args:
                Circuit (object): The quantum circuit object to be modified.
                weights (numpy.ndarray): A 2-dimensional array representing the weights for the rotations.
                qubits (list): A list of qubit objects or indices corresponding to the qubits in the circuit.
                rotation (callable, optional): A rotation gate function, defaulting to a single-qubit rotation gate (RX).

            Raises:
                ValueError: If the weights tensor is not 2-dimensional or if the second dimension of the weights tensor does not match the number of qubits.

            Returns:
                Circuit (object): The modified quantum circuit object with rotations and optional CNOT gates applied.
        """
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
                         

    