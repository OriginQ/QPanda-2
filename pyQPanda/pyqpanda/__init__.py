'''
QPanda Python\n
Copyright (C) Origin Quantum 2017-2018\n
Licensed Under Apache Licence 2.0
'''

from .utils import *
from .pyQPanda import *
from .Variational import back
from .Visualization import *
import warnings

#  classes

# pyQPanda
from .pyQPanda import AbstractOptimizer
from .pyQPanda import AdaGradOptimizer
from .pyQPanda import AdamOptimizer
from .pyQPanda import ArchType
from .pyQPanda import CBit
from .pyQPanda import ClassicalCondition
from .pyQPanda import ClassicalProg
from .pyQPanda import QuantumMachine
from .pyQPanda import CPUQVM
from .pyQPanda import MPSQVM
from .pyQPanda import CPUSingleThreadQVM
from .pyQPanda import DoubleGateTransferType
from .pyQPanda import GateType
from .pyQPanda import QCircuit
from .pyQPanda import hadamard_circuit
from .pyQPanda import NodeInfo
from .pyQPanda import NodeIter
from .pyQPanda import NodeType
from .pyQPanda import NoiseModel
from .pyQPanda import NoiseQVM
from .pyQPanda import OriginCollection
from .pyQPanda import PartialAmpQVM
from .pyQPanda import PhysicalQubit
from .pyQPanda import QCodarGridDevice
from .pyQPanda import QError
from .pyQPanda import QGate
from .pyQPanda import QIfProg
from .pyQPanda import QMachineType
from .pyQPanda import QMeasure
from .pyQPanda import QProg
from .pyQPanda import QReset
from .pyQPanda import QResult
from .pyQPanda import Qubit
from .pyQPanda import QVec
from .pyQPanda import QWhileProg
from .pyQPanda import SingleAmpQVM
from .pyQPanda import SingleGateTransferType
from .pyQPanda import SwapQubitsMethod

from .pyQPanda import complex_var
from .pyQPanda import expression
from .pyQPanda import MomentumOptimizer
from .pyQPanda import Optimizer
from .pyQPanda import OptimizerFactory
from .pyQPanda import OptimizerMode
from .pyQPanda import OptimizerType
from .pyQPanda import QOptimizationResult
from .pyQPanda import NodeSortProblemGenerator
from .pyQPanda import RMSPropOptimizer
from .pyQPanda import VanillaGradientDescentOptimizer
from .pyQPanda import var
from .pyQPanda import VariationalQuantumCircuit
from .pyQPanda import VariationalQuantumGate
from .pyQPanda import VariationalQuantumGate_CNOT
from .pyQPanda import VariationalQuantumGate_CRX
from .pyQPanda import VariationalQuantumGate_CRY
from .pyQPanda import VariationalQuantumGate_CRZ
from .pyQPanda import VariationalQuantumGate_CZ
from .pyQPanda import VariationalQuantumGate_H
from .pyQPanda import VariationalQuantumGate_RX
from .pyQPanda import VariationalQuantumGate_RY
from .pyQPanda import VariationalQuantumGate_RZ
from .pyQPanda import VariationalQuantumGate_X

from .Operator.pyQPandaOperator import *
# pyQPandaOperator class
from .Operator.pyQPandaOperator import FermionOperator
from .Operator.pyQPandaOperator import PauliOperator
from .Operator.pyQPandaOperator import VarFermionOperator
from .Operator.pyQPandaOperator import VarPauliOperator
# pyQPandaOperator function
from .Operator.pyQPandaOperator import trans_vec_to_Pauli_operator
from .Operator.pyQPandaOperator import trans_Pauli_operator_to_vec

# funtions

from .pyQPanda import accumulateProbability
from .pyQPanda import accumulate_probabilities
from .pyQPanda import accumulate_probability
from .pyQPanda import add
from .pyQPanda import all_cut_of_graph
from .pyQPanda import amplitude_encode
from .pyQPanda import apply_QGate
from .pyQPanda import assign
from .pyQPanda import bin_to_prog
from .pyQPanda import cAlloc
from .pyQPanda import cAlloc_many
from .pyQPanda import cast_qprog_qcircuit
from .pyQPanda import cast_qprog_qgate
from .pyQPanda import cast_qprog_qmeasure
from .pyQPanda import cFree
from .pyQPanda import cFree_all
from .pyQPanda import circuit_optimizer
from .pyQPanda import circuit_optimizer_by_config
from .pyQPanda import CNOT
from .pyQPanda import convert_binary_data_to_qprog
from .pyQPanda import convert_originir_to_qprog
from .pyQPanda import convert_originir_str_to_qprog
from .pyQPanda import convert_qasm_to_qprog
from .pyQPanda import convert_qasm_string_to_qprog
from .pyQPanda import convert_qprog_to_binary
from .pyQPanda import convert_qprog_to_originir
from .pyQPanda import convert_qprog_to_qasm
from .pyQPanda import convert_qprog_to_quil
from .pyQPanda import count_gate
from .pyQPanda import CR
from .pyQPanda import CreateEmptyCircuit
from .pyQPanda import CreateEmptyQProg
from .pyQPanda import CreateIfProg
from .pyQPanda import CreateWhileProg
from .pyQPanda import create_empty_circuit
from .pyQPanda import create_empty_qprog
from .pyQPanda import create_if_prog
from .pyQPanda import create_while_prog
from .pyQPanda import crossEntropy
from .pyQPanda import CU
from .pyQPanda import CZ
from .pyQPanda import destroy_quantum_machine
from .pyQPanda import directly_run
from .pyQPanda import div
from .pyQPanda import dot
from .pyQPanda import circuit_layer
from .pyQPanda import draw_qprog_text
from .pyQPanda import draw_qprog_text_with_clock
from .pyQPanda import fit_to_gbk
from .pyQPanda import quantum_chip_adapter
from .pyQPanda import get_all_used_qubits
from .pyQPanda import get_all_used_qubits_to_int
from .pyQPanda import decompose_multiple_control_qgate
from .pyQPanda import dropout
from .pyQPanda import equal
from .pyQPanda import eval
from .pyQPanda import exp
from .pyQPanda import fill_qprog_by_I
from .pyQPanda import finalize
from .pyQPanda import flatten
from .pyQPanda import getAllocateCMem
from .pyQPanda import getAllocateQubitNum
from .pyQPanda import getstat
from .pyQPanda import get_adjacent_qgate_type
from .pyQPanda import get_allocate_cmem_num
from .pyQPanda import get_allocate_qubit_num
from .pyQPanda import get_bin_data
from .pyQPanda import get_bin_str
from .pyQPanda import get_clock_cycle
from .pyQPanda import get_matrix
from .pyQPanda import get_prob_dict
from .pyQPanda import get_prob_list
from .pyQPanda import get_qgate_num
from .pyQPanda import get_qprog_clock_cycle
from .pyQPanda import get_qstate
from .pyQPanda import get_tuple_list
from .pyQPanda import get_unsupport_qgate_num
from .pyQPanda import graph_query_replace
from .pyQPanda import Grover
from .pyQPanda import Grover_search
from .pyQPanda import quantum_walk_alg
from .pyQPanda import quantum_walk_search
from .pyQPanda import H
from .pyQPanda import HHL
from .pyQPanda import HHL_solve_linear_equations
from .pyQPanda import I
from .pyQPanda import init
from .pyQPanda import init_quantum_machine
from .pyQPanda import inverse
from .pyQPanda import isCarry
from .pyQPanda import iSWAP
from .pyQPanda import is_match_topology
from .pyQPanda import is_supported_qgate_type
from .pyQPanda import is_swappable
from .pyQPanda import log
from .pyQPanda import MAJ
from .pyQPanda import MAJ2
from .pyQPanda import matrix_decompose
from .pyQPanda import Measure
from .pyQPanda import measure_all
from .pyQPanda import mul
from .pyQPanda import originir_to_qprog
from .pyQPanda import PMeasure
from .pyQPanda import pmeasure
from .pyQPanda import PMeasure_no_index
from .pyQPanda import pmeasure_no_index
from .pyQPanda import poly
from .pyQPanda import print_matrix
from .pyQPanda import prob_run_dict
from .pyQPanda import prob_run_list
from .pyQPanda import prob_run_tuple_list
from .pyQPanda import QAdder
from .pyQPanda import QAdderIgnoreCarry
from .pyQPanda import qAlloc
from .pyQPanda import qAlloc_many
from .pyQPanda import qcodar_match
from .pyQPanda import QFT
from .pyQPanda import qop
from .pyQPanda import qop_pmeasure
from .pyQPanda import QPE
from .pyQPanda import quick_measure
from .pyQPanda import Reset
from .pyQPanda import run_with_configuration
from .pyQPanda import RX
from .pyQPanda import RY
from .pyQPanda import RZ
from .pyQPanda import S
from .pyQPanda import sigmoid
from .pyQPanda import softmax
from .pyQPanda import stack
from .pyQPanda import sub
from .pyQPanda import sum
from .pyQPanda import SWAP
from .pyQPanda import T
from .pyQPanda import topology_match
from .pyQPanda import to_originir
from .pyQPanda import to_Quil
from .pyQPanda import transform_binary_data_to_qprog
from .pyQPanda import transform_originir_to_qprog
from .pyQPanda import transform_qprog_to_binary
from .pyQPanda import transform_qprog_to_originir
from .pyQPanda import transform_qprog_to_quil
from .pyQPanda import transform_to_base_qgate
from .pyQPanda import transpose
from .pyQPanda import U1
from .pyQPanda import U2
from .pyQPanda import U3
from .pyQPanda import U4
from .pyQPanda import UMA
from .pyQPanda import validate_double_qgate_type
from .pyQPanda import validate_single_qgate_type
from .pyQPanda import vector_dot
from .pyQPanda import X
from .pyQPanda import X1
from .pyQPanda import Y
from .pyQPanda import Y1
from .pyQPanda import Z
from .pyQPanda import Z1
from .pyQPanda import BARRIER
from .pyQPanda import _back
from .pyQPanda import state_fidelity

from .pyQPanda import AnsatzGateType
from .pyQPanda import AnsatzGate
from .pyQPanda import UpdateMode
from .pyQPanda import QITE

from .pyQPanda import Shor_factorization

from .pyQPanda import get_circuit_optimal_topology
from .pyQPanda import planarity_testing

from .utils import Toffoli

# pyQPandaOperator

try:
    from .pyQPanda import QCloud
    from .pyQPanda import ClusterMachineType
except ImportError as e:
    warnings.warn("No module named QCloud")

try:
    # classes

    from .ChemiQ.pyQPandaChemiQ import ChemiQ
    from .ChemiQ.pyQPandaChemiQ import TransFormType
    from .ChemiQ.pyQPandaChemiQ import UccType

    # funtions

    from .ChemiQ.pyQPandaChemiQ import getCCSD_N_Trem
    from .ChemiQ.pyQPandaChemiQ import getCCSD_Var
    from .ChemiQ.pyQPandaChemiQ import getCCS_Normal
    from .ChemiQ.pyQPandaChemiQ import getCCS_N_Trem
    from .ChemiQ.pyQPandaChemiQ import getCCS_Var
    from .ChemiQ.pyQPandaChemiQ import getElectronNum
    from .ChemiQ.pyQPandaChemiQ import get_ccsd_normal
    from .ChemiQ.pyQPandaChemiQ import get_ccsd_n_trem
    from .ChemiQ.pyQPandaChemiQ import get_ccsd_var
    from .ChemiQ.pyQPandaChemiQ import get_ccs_normal
    from .ChemiQ.pyQPandaChemiQ import get_ccs_n_trem
    from .ChemiQ.pyQPandaChemiQ import get_ccs_var
    from .ChemiQ.pyQPandaChemiQ import get_electron_num
    from .ChemiQ.pyQPandaChemiQ import JordanWignerTransform
    from .ChemiQ.pyQPandaChemiQ import JordanWignerTransformVar
    from .ChemiQ.pyQPandaChemiQ import jordan_wigner_transform
    from .ChemiQ.pyQPandaChemiQ import jordan_wigner_transform_var
    from .ChemiQ.pyQPandaChemiQ import parsePsi4DataToFermion
    from .ChemiQ.pyQPandaChemiQ import parse_psi4_data_to_fermion
    from .ChemiQ.pyQPandaChemiQ import simulateHamiltonian_Var
    from .ChemiQ.pyQPandaChemiQ import simulate_hamiltonian_var
    from .ChemiQ.pyQPandaChemiQ import transCC2UCC_Normal
    from .ChemiQ.pyQPandaChemiQ import transCC2UCC_Var
    from .ChemiQ.pyQPandaChemiQ import trans_cc_2_ucc_normal
    from .ChemiQ.pyQPandaChemiQ import trans_cc_2_ucc_var

except ImportError as e:
    warnings.warn("No module named ChemiQ")


One = True
Zero = False
