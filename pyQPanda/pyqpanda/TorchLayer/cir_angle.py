from pyqpanda import *
import numpy as np
ROT = {"X": RX, "Y": RY, "Z": RZ}


def Generator_Angle_Circuit(Circuit,features, qubits, rotation="X"):
        """
        Generate an angle-based quantum circuit using specified features and rotation gates.

        Constructs a quantum circuit by applying rotation gates to qubits based on provided features. The rotation is
        determined by the specified rotation gate, and the circuit is created for the first 'n_features' qubits, where
        'n_features' is the length of the 'features' array and should not exceed the number of available qubits.

            Args:
                Circuit (object): The quantum circuit object to which the operations will be added.
                features (numpy.ndarray): A one-dimensional array of features that will determine the rotation angles.
                qubits (list of int): A list of qubit indices to apply the rotation gates.
                rotation (str, optional): The type of rotation gate to apply. Must be one of the recognized rotation gate types
            defined in the ROT dictionary. Defaults to "X".

            Raises:
                ValueError: If the specified rotation is not recognized.
                ValueError: If 'features' is not a one-dimensional tensor.
                ValueError: If the length of 'features' exceeds the number of available qubits.

            Returns:
                object: The modified quantum circuit object with the angle-based rotations applied.
        """
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
       

    