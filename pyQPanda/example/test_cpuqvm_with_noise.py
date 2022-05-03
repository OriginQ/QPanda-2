import sys
import os.path
dir = os.path.abspath(__file__)
model_path = os.path.abspath(os.path.join(dir, "../.."))
sys.path.insert(0, model_path)

from typing import Dict
import unittest
from pyqpanda import *

prob_list = [[0.9, 0.1], [0.1, 0.9]]


def runNoiseQVM(shots: int) -> Dict[str, int]:
    qvm = NoiseQVM()
    qvm.init_qvm()

    qbit = qvm.qAlloc_many(3)
    cbit = qvm.cAlloc_many(3)

    qvm.set_noise_model(NoiseModel.BIT_PHASE_FLIP_OPRATOR, GateType.PAULI_X_GATE, 0.7)
    qvm.set_noise_model(NoiseModel.DAMPING_KRAUS_OPERATOR, GateType.PAULI_X_GATE, 0.1)
    qvm.set_noise_model(NoiseModel.DAMPING_KRAUS_OPERATOR, GateType.SWAP_GATE, 0.3)
    qvm.set_noise_model(NoiseModel.DAMPING_KRAUS_OPERATOR, GateType.CNOT_GATE, 0.5)
    qvm.set_readout_error(prob_list, QVec(qbit[2]))
    qvm.set_measure_error(NoiseModel.BITFLIP_KRAUS_OPERATOR, 0.5, QVec(qbit[2]))
    qvm.set_reset_error(0.5, 0.5, QVec(qbit[1]))

    qc = QProg()

    qc << X(qbit[1]) << SWAP(qbit[2], qbit[1]) << CNOT(qbit[2], qbit[0]) << measure_all(qbit, cbit)
    result = qvm.run_with_configuration(qc, cbit, shots)
    qvm.finalize()
    return result

def runCPUQVM(shots: int) -> Dict[str, int]:
    qvm = CPUQVM()
    qvm.init_qvm()

    qbit = qvm.qAlloc_many(3)
    cbit = qvm.cAlloc_many(3)

    noise = Noise()
    noise.add_noise_model(NoiseModel.BIT_PHASE_FLIP_OPRATOR, GateType.PAULI_X_GATE, 0.7)
    noise.add_noise_model(NoiseModel.DAMPING_KRAUS_OPERATOR, GateType.PAULI_X_GATE, 0.1)
    noise.add_noise_model(NoiseModel.DAMPING_KRAUS_OPERATOR, GateType.SWAP_GATE, 0.3)
    noise.add_noise_model(NoiseModel.DAMPING_KRAUS_OPERATOR, GateType.CNOT_GATE, 0.5)
    noise.add_readout_error(prob_list, QVec(qbit[2]))
    noise.add_measure_error(NoiseModel.BITFLIP_KRAUS_OPERATOR, 0.5, QVec(qbit[2]))
    noise.add_reset_error(0.5, 0.5, QVec(qbit[1]))

    qc = QProg()

    qc << X(qbit[1]) << SWAP(qbit[2], qbit[1]) << CNOT(qbit[2], qbit[0]) << measure_all(qbit, cbit)
    result = qvm.run_with_configuration(qc, cbit, shots, noise)
    qvm.finalize()
    return result

class Test_CPUQVM_with_noise(unittest.TestCase):
    def test_compare_cpuqvm_noiseqvm(self):
        shot = 5000
        noiseqvm_result = runNoiseQVM(shot)
        cpuqvm_result = runCPUQVM(shot)

        self.assertEqual(len(noiseqvm_result), len(cpuqvm_result), "CPUQVM result not match NoiseQVM result")

        for k in noiseqvm_result.keys():
            self.assertLessEqual(abs(float(noiseqvm_result[k]) - float(cpuqvm_result[k]))/shot, 0.03)



if __name__ == "__main__":
    unittest.main(verbosity=2)