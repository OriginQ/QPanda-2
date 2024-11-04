from pyqpanda.Hamiltonian import chem_client
#from pyqpanda.Hamiltonian.QubitOperator import *
from pyqpanda import *
#from pyqpanda.utils import *
from pyqpanda.Algorithm.hamiltonian_simulation import *
from pyqpanda.Algorithm.fragments import *
from scipy.optimize import minimize
from functools import partial

import numpy as np
from numpy.linalg import eig
# definition of to_dense



def convert_operator(str):
    """
    Converts a string representation of a molecule into a PauliOperator object.

    This function processes a string containing molecule information and constructs a
    PauliOperator object, which is used to represent the Hamiltonian in quantum
    circuits. The input string is expected to be formatted with terms separated by ' +\n',
    where each term consists of a coefficient and a Pauli operator expression. The
    coefficient is assigned to the corresponding Pauli operator in the resulting
    PauliOperator object.

        Args:
        str (str): A string representing the molecule's Hamiltonian in the format:
                    'coefficient PauliOperator', where PauliOperator is a string
                    representing a Pauli gate (e.g., 'X', 'Y', 'Z', 'I').

        Returns:
        PauliOperator: An instance of PauliOperator containing the molecule's Hamiltonian
                       terms, with coefficients and corresponding Pauli operators.

        Raises:
            ValueError: If the input string is not properly formatted.

        Example:
            >>> convert_operator('1.0 X + 2.5 Y + 3 Z')
            PauliOperator({'X': 1.0, 'Y': 2.5, 'Z': 3.0})
    """
    terms=str.split(' +\n')
    tuplelist=dict()
    for term in terms:
        data=term.split(' ',1)        
        tuplelist[data[1][1:-1]]=eval(data[0])

    operator=PauliOperator(tuplelist)    
    return operator

def H2_energy_from_distance(distance):
    """
    Calculate the ground-state energy of a hydrogen molecule (H2) given the distance between its two atoms.

        Args:
            distance (float): The distance between the two hydrogen atoms in angstroms.

        Returns:
            float: The computed energy of the H2 molecule in Hartrees.

    The function uses the chem_client from the pyQPanda package to perform quantum chemistry calculations
    with specified methods such as MP2, CISD, CCSD, and FCI. The calculations are based on the STO-3G basis set
    and are run on a quantum circuit simulator or quantum cloud service. The minimum eigenvalue of the Hamiltonian
    matrix is returned, which represents the ground-state energy of the H2 molecule.
    """
    geometry=[['H',[0,0,0]], ['H',[0,0,distance]]]
    
    basis="sto-3g"
    multiplicity=1
    charge=0
    run_mp2=True
    run_cisd=True
    run_ccsd=True
    run_fci=True

    str1=chem_client(
            geometry=geometry,
            basis=basis,
            multiplicity=multiplicity,
            charge=charge,
            run_mp2=run_mp2,
            run_cisd=run_cisd,
            run_ccsd=run_ccsd,
            run_fci=run_fci,
            hamiltonian_type="pauli")
    
    pauli_op=convert_operator(str1)
    matrix_=get_matrix(pauli_op)
    eigval,_=eig(matrix_)
    return min(eigval).real

Atom_Name=['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'He', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl']
Atom_Electron=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
Atom_Dict=dict()
for i in range(len(Atom_Name)):
    Atom_Dict[Atom_Name[i]]=Atom_Electron[i]
    
def get_electron_count(geometry):
    """
    Calculate the total number of electrons in a given molecular geometry by summing the electrons of each atom.

        Args:
            geometry (list of str): A list of strings representing the atoms in the molecular geometry. Each string starts with the
                                chemical symbol of an atom, which is a key in the Atom_Dict dictionary.

        Returns:
            int: The total electron count for the provided molecular geometry.

            Atom_Dict (dict): A predefined dictionary mapping chemical symbols to their respective electron counts, e.g., {'H': 1, 'He': 2, ...,
                           'Cl': 17}. This dictionary is used to look up the electron count for each atom in the geometry.

    This function is designed to be utilized within the pyQPanda package, which facilitates programming quantum computers and quantum
    circuits, including running simulations on quantum virtual machines or quantum cloud services.
    """
    n_electron = 0
    for atom in geometry:
        n_electron+=Atom_Dict[atom[0]]
    return n_electron


def get_fermion_jordan_wigner(fermion_type, op_qubit):
    """
    Generates a Pauli operator representing the Jordan-Wigner transformation of a fermionic operator.

        Args:
            fermion_type (str): Specifies the fermionic operation type. It must be either 'a' for annihilation or 'c' for creation.
            op_qubit (int): The qubit index for the operation.

        Returns:
            PauliOperator: A PauliOperator instance representing the fermionic operator after applying the Jordan-Wigner
          transformation.

        Raises:
            AssertionError: If `fermion_type` is neither 'a' nor 'c'.

    The function constructs a string `opstr` that represents the Pauli operators for the qubits up to `op_qubit - 1`.
    It then appends 'X' and 'Y' operators for the `op_qubit`. Depending on the `fermion_type`, it returns a
    PauliOperator with the appropriate signs for the 'X' and 'Y' operators.
    """
    opstr=''
    for i in range(op_qubit):
        opstr=opstr+'Z'+str(i)+' '
    
    opstr1=opstr+'X'+str(op_qubit)
    opstr2=opstr+'Y'+str(op_qubit)
    if fermion_type == 'a':
        return PauliOperator({opstr1:1,opstr2:1j})
    elif fermion_type == 'c':
        return PauliOperator({opstr1:1,opstr2:-1j})
    else:
        assert False
        
def get_ccs_n_term(n_qubit, n_electron):
    """
    Calculate the number of single-particle excitation terms in the coupled cluster single (CCS) model for a given number of qubits
    and electrons.

        Args:
            n_qubit (int): The total number of qubits in the system.
            n_electron (int): The total number of electrons in the system.

        Returns:
            int: The count of single-particle excitation terms.

        Raises:
            AssertionError: If the number of electrons exceeds the number of qubits.

    The function assumes that the occupied orbitals are the first `n_electron` qubits. It then computes the number of
    single-particle excitations by considering all possible transitions from occupied orbitals to virtual orbitals. The
    virtual orbitals are defined as the qubits from `n_electron` to `n_qubit`. For example, with 4 qubits and 2 electrons,
    the occupied orbitals are qubits 0 and 1, and the virtual orbitals are qubits 2 and 3. The function counts the transitions
    from the occupied orbitals to the virtual orbitals, such as 0->2, 0->3, 1->2, and 1->3.

    This function is intended for use within the pyQPanda package, which supports quantum computing programming with quantum
    circuits and gates, and is compatible with quantum circuit simulators or quantum cloud services.
    """
    if n_electron>n_qubit:
        assert False
    elif n_electron==n_qubit:
        return 0
    param_n=0
    # ccsd is each electron jump to the excited level, and also each two
    result_op=PauliOperator(dict())
    for i in range(n_electron):
        for ex in range(n_electron, n_qubit):
            param_n+=1
                    
    return param_n

def get_ccsd_n_term(n_qubit, n_electron):
    """
    Calculate the number of coupled cluster single and double (CCSD) terms for a given number of qubits and electrons.

        Args:
            n_qubit (int): The total number of qubits in the quantum system.
            n_electron (int): The total number of electrons in the quantum system.

        Returns:
            int: The total number of CCSD terms.

        Raises:
            AssertionError: If the number of electrons exceeds the number of qubits.

        Examples:
            For a system with 4 qubits and 2 electrons, the function calculates the terms considering the following:
            - Single excitations: 0->2, 0->3, 1->2, 1->3
            - Double excitations: 01->23

        Note: 
            This function is designed for use within the pyQPanda package, which is utilized for programming quantum computers using quantum circuits and gates. It operates on quantum circuit simulators or quantum cloud services.
    """

    if n_electron>n_qubit:
        assert False
    elif n_electron==n_qubit:
        return 0
    param_n=0
    # ccsd is each electron jump to the excited level, and also each two
    result_op=PauliOperator(dict())
    for i in range(n_electron):
        for ex in range(n_electron, n_qubit):
            param_n+=1
    
    for i in range(n_electron):
        for j in range(i+1,n_electron):
            for ex1 in range(n_electron,n_qubit):
                for ex2 in range(ex1+1,n_qubit):
                    param_n+=1
                    
    return param_n

def get_ccs(n_qubit, n_electron, param_list):
    """
    Computes the coupled cluster single (CCS) model's Pauli operator using the Jordan-Wigner transformation.

    This function implements the CCS model by applying the J-W transformation to the model and obtaining the
    corresponding Pauli operator. It ensures that the number of electrons does not exceed the number of qubits.
    If the number of electrons equals the number of qubits, it returns an empty Pauli operator. Otherwise, it
    constructs the Pauli operator by considering each electron's jump to an excited level and interactions
    between electrons.

        Args:
            n_qubit (int): The total number of qubits in the quantum system.
            n_electron (int): The number of electrons in the quantum system.
            param_list (list): A list of parameters corresponding to the CCS model.

        Returns:
            PauliOperator: The Pauli operator resulting from the CCS model's J-W transformation.

        Raises:
            AssertionError: If the number of electrons exceeds the number of qubits.

    This function is intended for use within the pyQPanda package, which facilitates quantum programming
    and quantum circuit simulations.
    """
    if n_electron>n_qubit:
        assert False
    elif n_electron==n_qubit:
        return PauliOperator(dict())
    param_n=0
    # ccsd is each electron jump to the excited level, and also each two
    result_op=PauliOperator(dict())
    for i in range(n_electron):
        for ex in range(n_electron, n_qubit):
            result_op+=get_fermion_jordan_wigner('c',ex)*get_fermion_jordan_wigner('a',i)*param_list[param_n]
            param_n+=1
            
    return result_op
        
def get_ccsd(n_qubit, n_electron, param_list):
    """
    Constructs the coupled cluster single and double (CCSD) model's Pauli operator using the J-W transform.

        Args:
            n_qubit (int): The total number of qubits in the quantum system.
            n_electron (int): The number of electrons in the quantum system.
            param_list (list): A list of parameters used to scale the contributions from the fermion Jordan-Wigner transformations.

        Returns:
            PauliOperator: A Pauli operator representing the CCSD model, constructed using the provided parameters.

        Raises:
            AssertionError: If the number of electrons exceeds the number of qubits.

    This function is intended for use within the pyQPanda package, which is designed for programming quantum computers using quantum circuits and gates. It operates on quantum circuit simulators, quantum virtual machines, or quantum cloud services. The function is located within the directory `pyQPanda.build.lib.pyqpanda.Algorithm.VariationalQuantumEigensolver.vqe.py`.

    The function performs the following steps:
    1. Checks if the number of electrons exceeds the number of qubits, raising an assertion error if true.
    2. For systems with equal numbers of qubits and electrons, returns an empty Pauli operator.
    3. Iterates over pairs of electrons to calculate the single-excitation and double-excitation terms, applying the fermion Jordan-Wigner transformations and scaling by the parameters from `param_list`.
    4. Constructs and returns the resulting Pauli operator.
    """
    if n_electron>n_qubit:
        assert False
    elif n_electron==n_qubit:
        return PauliOperator(dict())
    param_n=0
    # ccsd is each electron jump to the excited level, and also each two
    result_op=PauliOperator(dict())
    for i in range(n_electron):
        for ex in range(n_electron, n_qubit):
            result_op+=get_fermion_jordan_wigner('c',ex)*get_fermion_jordan_wigner('a',i)*param_list[param_n]
            param_n+=1
    
    for i in range(n_electron):
        for j in range(i+1,n_electron):
            for ex1 in range(n_electron,n_qubit):
                for ex2 in range(ex1+1,n_qubit):
                    result_op+=get_fermion_jordan_wigner('c',ex2)* \
                               get_fermion_jordan_wigner('c',ex1)* \
                               get_fermion_jordan_wigner('a',j)*   \
                               get_fermion_jordan_wigner('a',i)* param_list[param_n]
                    param_n+=1
                    
    return result_op



def cc_to_ucc_hamiltonian(cc_op):
    """
    Constructs the Hamiltonian for the unitary coupled cluster (UCC) method by applying the
    coupled cluster transformation, H = 1j * (T† - T), where T† denotes the conjugate transpose
    of the cluster operator T. The resulting exponential of the Hamiltonian, exp(-jHt),
    corresponds to the action of the transformed cluster operator on the system.

        Args:
            cc_op (numpy.ndarray): The cluster operator T from the UCC method, expected to be a
                               complex numpy array.

        Returns:
            numpy.ndarray: The Hamiltonian matrix as a complex numpy array, calculated as 1j * (T† - T).
    """
    return 1j*(cc_op-cc_op.dagger())

def flatten(pauliOperator):
    """
    Converts the coefficients of a Pauli operator to floating-point numbers if they are effectively real numbers within a specified tolerance.
    
    This function processes the coefficients of the input `pauliOperator` object. If the imaginary part of any coefficient is below the
    `pauliOperator.m_error_threshold`, it replaces the complex coefficient with its real part as a float. If any coefficient cannot be
    considered real within the threshold, the function asserts an error, indicating that the operation cannot proceed as expected.
    
        Args:
            pauliOperator (object): An instance of a Pauli operator from the pyQPanda package.

        Returns:
            PauliOperator: A new Pauli operator object with all coefficients converted to floats if they are effectively real within the
                           specified tolerance, or raises an assertion error if any coefficient cannot be considered real.
    """
    new_ops=deepcopy(pauliOperator.ops)
    for term in new_ops:
        if abs(new_ops[term].imag)<pauliOperator.m_error_threshold:
            new_ops[term]=float(new_ops[term].real)
        else:
            assert False
    return PauliOperator(new_ops)

def transform_base(qubitlist,base):
    """
    Transforms the measurement basis of a quantum circuit by rotating all axes to the z-axis.

    This function parses a specified measurement basis string into Pauli operators and applies
    appropriate single-qubit gates to rotate the qubits accordingly. It supports the Pauli operators
    'X', 'Y', and 'Z', where 'X' is rotated to the z-axis with a Hadamard gate, 'Y' is rotated to the
    z-axis with a half-pi RX gate, and 'Z' is ignored as it is already aligned with the z-axis.

        Args:
            qubitlist: A list of qubits, representing the quantum system.
            base: A string representing the measurement basis to be transformed.

        Returns:
            qcircuit: A QuantumCircuit object with the transformed measurement basis.

        Raises:
            AssertionError: If an unsupported Pauli operator is encountered.

    This function is intended for use within the pyQPanda package, which is designed for programming
    quantum computers with quantum circuits and gates. It can be executed on a quantum circuit simulator,
    quantum virtual machine, or quantum cloud service.
    """
    tuple_list=PauliOperator.parse_pauli(base)
    qcircuit=QCircuit()
    for i in tuple_list:
        if i[0]=='X':
            qcircuit.insert(H(qubitlist[i[1]]))
        elif i[0]=='Y':
            qcircuit.insert(RX(qubitlist[i[1]],pi/2))
        elif i[0]=='Z':
            pass
        else:
            assert False
    return qcircuit

# def transform_base(qubitlist,base):
#     '''
#     choose measurement basis,
#     it means rotate all axis to z-axis
#     '''
#     tuple_list=PauliOperator.parse_pauli(base)
#     qcircuit=QCircuit()
#     for i in tuple_list:
#         if i[0]=='X':
#             qcircuit.insert(H(qubitlist[i[1]])
#         elif i[0]=='Y':
#             qcircuit.insert(RX(qubitlist[i[1]],pi/2))
#         elif i[0]=='Z':
#             pass
#         else:
#             assert False
#     return qcircuit


#e.g. component=('X0 Y1 Z2',0.33)
def get_expectation(qubit_number,unitaryCC,component,shots_):
    """
    Calculate the expectation value of a Pauli operator within a quantum circuit.

        Args:
            qubit_number (int): The number of qubits in the quantum system.
            unitaryCC (list): A list representing the unitary coupled cluster operator.
            component (tuple): A tuple containing a Pauli operator string and its coefficient.
                           Format: ('PauliOperatorString', Coefficient).
            shots_ (int): The number of times the quantum circuit is run to collect data.

        Returns:
            float: The expectation value of the specified Pauli operator, adjusted by its coefficient.

    This function initializes a quantum program, sets up the circuit with the given qubits,
    applies the unitary coupled cluster operator, and then performs the transformation
    of the base if required. The circuit is executed, and the expectation value is computed
    by evaluating the probability amplitudes of the Pauli operator within the circuit.
    """
    init()
    prog=QProg()
    q=qAlloc_many(qubit_number)
    c=cAlloc_many(qubit_number)
    prog.insert(X(q[0])).insert(X(q[2]))
    #print(unitaryCC)
    prog.insert(simulate_hamiltonian(qubit_list=q,pauliOperator=unitaryCC,t=1,slices=3))
    if component[0]!='':
        prog.insert(transform_base(q,component[0]))
    #print(to_qrunes(prog))
    directly_run(QProg=prog)
    result=get_probabilites(q, select_max=-1, dataType="dict")
    finalize()
    expectation=0
    for i in result:
        if parity_check(i, component[0]):
            expectation-=result[i]
        else:
            expectation+=result[i]       
    #print(result)
    return expectation*component[1]

def vqe_subroutine(qubit_number_,electron_number,Hamiltonian,unitaryCC_,shots):
    """
    Calculate the expectation value of a Hamiltonian using a unitary coupled cluster operator.

        Args:
            qubit_number_: (int) The number of qubits in the quantum circuit.
            electron_number: (int) The number of electrons involved in the quantum system.
            Hamiltonian: (object) A Hamiltonian object represented by Pauli operators.
            unitaryCC_: (object) The unitary coupled cluster operator for the quantum system.
            shots: (int) The number of times the quantum circuit is executed to estimate the expectation value.

        Returns:
            float: The calculated expectation value of the Hamiltonian.

        Notes:
            This function is part of the Variational Quantum Eigensolver (VQE) subroutine within the pyQPanda package.
            It iterates over the Pauli operators in the Hamiltonian, computes the expectation value for each,
            and aggregates these values to produce the total expectation value.
    """
    expectation=0
    for component in Hamiltonian.ops:
        #print(component)
        temp=(component,Hamiltonian.ops[component])
        expectation+=get_expectation(qubit_number=qubit_number_,unitaryCC=unitaryCC_,component=temp,shots_=shots)
    expectation=float(expectation.real)
    return expectation


def binding(qubit_number,electron_number,Hamiltonian,shots):
    """
    Generates a Variational Quantum Eigensolver (VQE) instance configured for the specified number of qubits,
    electrons, and quantum Hamiltonian, to be executed with a given number of shots.

        Args:
            qubit_number (int): The number of qubits to be used in the quantum circuit.
            electron_number (int): The number of electrons involved in the quantum system.
            Hamiltonian (object): The quantum Hamiltonian describing the system.
            shots (int): The number of times the quantum circuit will be run to estimate the expectation value.

        Returns:
            partial (callable): A callable object representing the VQE instance with parameters partially applied.
    """
    return partial(vqe_in_list,
                    qubit_number=qubit_number,
                    electron_number=electron_number,
                    Hamiltonian=Hamiltonian,
                    shots=shots)

def vqe_in_list(arguments,qubit_number,electron_number,Hamiltonian,shots):
    """
    Computes the expected energy of a variational quantum eigensolver (VQE) given a set of parameters,
    a number of qubits, a number of electrons, a Hamiltonian, and a number of shots.

        Args:
            arguments (list): A list of variational parameters for the quantum circuit.
            qubit_number (int): The total number of qubits in the quantum circuit.
            electron_number (int): The number of electrons in the system.
            Hamiltonian (object): An object representing the Hamiltonian of the system.
            shots (int): The number of times the quantum circuit is run to estimate the expectation value.

        Returns:
            float: The expected energy of the system, computed as the average of the expectation values
                 obtained from running the quantum circuit the specified number of times.

    The function utilizes helper functions `get_ccs`, `cc_to_ucc_hamiltonian`, and `flatten` to construct
    and simplify the unitary coupled cluster (UCC) Hamiltonian. It then calculates the expectation value
    for each component of the Hamiltonian and sums them to obtain the total expectation value.
    """
    op1=get_ccs(qubit_number,electron_number,arguments)
    ucc=cc_to_ucc_hamiltonian(op1)
    ucc=flatten(ucc)
    expectation=0
    for component in Hamiltonian.ops:
        #print(component)
        temp=(component,Hamiltonian.ops[component])
        expectation+=get_expectation(qubit_number=qubit_number,unitaryCC=ucc,component=temp,shots_=shots)
    expectation=float(expectation.real)
    return expectation



# def H2_vqe_subprocess(distance,Hamiltonian,n_electron,initial_guess,method='Powell'):
#     # geometry=[['H',[0,0,0]], ['H',[0,0,distance]]]
    
#     # basis="sto-3g"
#     # multiplicity=1
#     # charge=0
#     # run_mp2=True
#     # run_cisd=True
#     # run_ccsd=True
#     # run_fci=True
#     # str1=chem_client(
#     #         geometry=geometry,
#     #         basis=basis,
#     #         multiplicity=multiplicity,
#     #         charge=charge,
#     #         run_mp2=run_mp2,
#     #         run_cisd=run_cisd,
#     #         run_ccsd=run_ccsd,
#     #         run_fci=run_fci,
#     #         hamiltonian_type="pauli")
#     # Hamiltonian=convert_operator(str1)
#     n_qubit = Hamiltonian.get_qubit_count()
#     #n_electron=get_electron_count(geometry)
#     #n_param=get_ccs_n_term(n_qubit,n_electron)
#     #initial_guess
#     #initial_guess=np.ones(n_param)*0.5
#     result=minimize(binding(n_qubit,n_electron,Hamiltonian,shots=1000),initial_guess,method=method)
#     return result

def H2_vqe(
    distance_range,
    initial_guess,
    basis='sto-3g',
    multiplicity=1,
    charge=0,
    run_mp2=True,
    run_cisd=True,
    run_ccsd=True,
    run_fci=True,
    method='Powell'
):
    """
    Compute the energy of the H2 molecule for a range of bond distances using variational quantum eigensolver (VQE).

        Args:
            distance_range (list): A list of bond distances to evaluate the energy at.
            initial_guess (numpy.ndarray): Initial guess for the variational parameters of the quantum circuit.
            basis (str, optional): The basis set to use for the molecular orbitals. Default is 'sto-3g'.
            multiplicity (int, optional): The electronic multiplicity of the system. Default is 1.
            charge (int, optional): The net charge of the system. Default is 0.
            run_mp2 (bool, optional): Whether to run the MP2 correction. Default is True.
            run_cisd (bool, optional): Whether to run the CISD correction. Default is True.
            run_ccsd (bool, optional): Whether to run the CCSD correction. Default is True.
            run_fci (bool, optional): Whether to run the Full Configuration Interaction (FCI). Default is True.
            method (str, optional): The optimization method to use. Default is 'Powell'.

        Returns:
            numpy.ndarray: The energy of the H2 molecule at each distance in the distance_range.

    The function utilizes a quantum circuit simulator to compute the energy of the H2 molecule by iteratively optimizing the variational parameters of the quantum circuit. The energy is calculated using various quantum chemistry methods, including MP2, CISD, CCSD, and FCI, if specified. The optimization is performed using the specified method (e.g., Powell) to minimize the energy.

    Note: This function requires the pyQPanda package for quantum computing and associated quantum chemistry tools.
    """

    energy=np.zeros(len(distance_range))
    for i in range(len(distance_range)):
        print(i)
        geometry=[['H',[0,0,0]], ['H',[0,0,distance_range[i]]]]
        str1=chem_client(
            geometry=geometry,
            basis=basis,
            multiplicity=multiplicity,
            charge=charge,
            run_mp2=run_mp2,
            run_cisd=run_cisd,
            run_ccsd=run_ccsd,
            run_fci=run_fci,
            hamiltonian_type="pauli")
        Hamiltonian=convert_operator(str1)
        n_qubit = Hamiltonian.get_qubit_count()
        n_electron=get_electron_count(geometry)
        result=minimize(binding(n_qubit,n_electron,Hamiltonian,shots=1000),initial_guess,method=method)
        energy[i]=result.fun
        print(energy[i])
    return energy



# def vqe_subprocess(distance)

# def vqe_algrithm(
#     Hamiltonian,
#     qubit_number,
#     electron_number,
#     initial_guess,
#     distance_range,
#     method='Powell'
# ):
    
#     energy=np.zeros(len(distance_range))
#     for i in range(len(distance_range)):
#         print(i)
#         result=H2_vqe_subprocess(distance_range[i],method)
#         energy[i]=result.fun
#         print(energy[i])
#     return energy 





