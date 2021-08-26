from pyqpanda import *
import numpy as np
ROT = {"X": RX, "Y": RY, "Z": RZ}


def Generator_Angle_Circuit(Circuit,features, qubits, rotation="X"):

        if rotation not in ROT:
            raise ValueError(f"Rotation option {rotation} not recognized.")
        rotation = ROT[rotation]
        
        shape = np.shape(features)
        if len(shape) != 1:
            raise ValueError(f"Features must be a one-dimensional tensor; got shape {shape}.")
        n_features = shape[0]
        if n_features > len(qubits):
            raise ValueError(
                f"Features must be of length {len(qubits)} or less; got length {n_features}."
            )

        qubits = qubits[:n_features]
        for i,j in enumerate(qubits):
            Circuit << rotation(j, features[i])
        return Circuit 
       

    