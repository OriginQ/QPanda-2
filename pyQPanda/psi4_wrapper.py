'''
QPanda Utilities\n
Copyright (C) Origin Quantum 2017-2019\n
Licensed Under Apache Licence 2.0
'''

from psi4 import core
from psi4.driver.molutil import geometry
from psi4.driver.driver import energy

import copy                                                                      
import itertools                                                                 
import numpy                                                                     
import traceback        
                                                                                                                           
def general_basis_change(general_tensor, rotation_matrix, key):                  
    """Change the basis of an general interaction tensor.                     
                                                                                 
    M'^{p_1p_2...p_n} = R^{p_1}_{a_1} R^{p_2}_{a_2} ...                          
                        R^{p_n}_{a_n} M^{a_1a_2...a_n} R^{p_n}_{a_n}^T ...       
                        R^{p_2}_{a_2}^T R_{p_1}_{a_1}^T                          
                                                                                 
    where R is the rotation matrix, M is the general tensor, M' is the           
    transformed general tensor, and a_k and p_k are indices. The formula uses    
    the Einstein notation (implicit sum over repeated indices).                  
                                                                                 
    In case R is complex, the k-th R in the above formula need to be conjugated  
    if key has a 1 in the k-th place (meaning that the corresponding operator    
    is a creation operator).                                                     
                                                                                 
    Args:                                                                        
        general_tensor: A square numpy array or matrix containing information    
            about a general interaction tensor.                                  
        rotation_matrix: A square numpy array or matrix having dimensions of     
            n_qubits by n_qubits. Assumed to be unitary.                         
        key: A tuple indicating the type of general_tensor. Assumed to be        
            non-empty. For example, a tensor storing coefficients of             
            :math:`a^\dagger_p a_q` would have a key of (1, 0) whereas a tensor 
            storing coefficients of :math:`a^\dagger_p a_q a_r a^\dagger_s`     
            would have a key of (1, 0, 0, 1).                                    
                                                                                 
    Returns:                                                                     
        transformed_general_tensor: general_tensor in the rotated basis.         
    """                                                                       
   
    n_orbitals = rotation_matrix.shape[0]                                        
    if general_tensor.shape[0] == 2 * n_orbitals:                                
        rotation_matrix = numpy.kron(rotation_matrix, numpy.eye(2))              
                                                                                 
    order = len(key)                                                             
    if order > 26:                                                               
        raise ValueError('Order exceeds maximum order supported (26).')          
                                                                                 

    subscripts_first = ''.join(chr(ord('a') + i) for i in range(order))          
                                                                                 
   
    subscripts_rest = ','.join(chr(ord('a') + i) +                               
                               chr(ord('A') + i) for i in range(order))          
                                                                                 
    subscripts = subscripts_first + ',' + subscripts_rest                        
                                                                                 
   
    rotation_matrices = [rotation_matrix.conj() if x else                        
                         rotation_matrix for x in key]                           
                                                                                 
   
    transformed_general_tensor = numpy.einsum(subscripts,                        
                                              general_tensor,                    
                                              *rotation_matrices)                     
    return transformed_general_tensor                                            
                                                                                 
class MyPolynomialTensor(object):                                                
    def __init__(self, n_body_tensors):                                          
        self.n_body_tensors = n_body_tensors                                     
        key_iterator = iter(n_body_tensors.keys())                               
        key = next(key_iterator)                                                 
        if key == ():                                                            
            key = next(key_iterator)                                             
        self.n_qubits = n_body_tensors[key].shape[0]                             
                                                                                 
    def __getitem__(self, args):                                                 
        """Look up matrix element.                                            
                                                                                 
        Args:                                                                    
            args: Tuples indicating which coefficient to get. For instance,      
                `my_tensor[(6, 1), (8, 1), (2, 0)]`                              
                returns                                                          
                `my_tensor.n_body_tensors[1, 1, 0][6, 8, 2]`                     
        """                                                                   
        if len(args) == 0:                                                       
            return self.n_body_tensors[()]                                       
        else:                                                                    
            index = tuple([operator[0] for operator in args])                    
            key = tuple([operator[1] for operator in args])                      
            return self.n_body_tensors[key][index]                               
                                                                                 
    def __iter__(self):                                                          
        """Iterate over non-zero elements of PolynomialTensor."""          
        def sort_key(key):                                                       
            """This determines how the keys to n_body_tensors                 
            should be sorted."""                                              
           
            if key == ():                                                        
                return 0, 0                                                      
            else:                                                                
                key_int = int(''.join(map(str, key)))                            
                return len(key), key_int                                         
                                                                                 
        for key in sorted(self.n_body_tensors.keys(), key=sort_key):             
            if key == ():                                                        
                yield ()                                                         
            else:                                                                
                n_body_tensor = self.n_body_tensors[key]                         
                for index in itertools.product(                                  
                        range(self.n_qubits), repeat=len(key)):                  
                    if n_body_tensor[index]:                                     
                        yield tuple(zip(index, key))                             
                                                                                 
    def __str__(self):                                                           
        """Print out the non-zero elements of PolynomialTensor."""         
        strings = []                                                             
        for key in self:                                                         
            strings.append('{} : {}\n'.format(key, self[key]))                  
        return ''.join(strings) if strings else '0'       

def get_molecular_hamiltonian(mints, canonical_orbitals, nuclear_repulsion,EQ_TOLERANCE = 1e-8):
        """Output arrays of the second quantized Hamiltonian coefficients.

        Returns:
            molecular_hamiltonian: An instance of the MolecularOperator class.

        Note:
            The indexing convention used is that even indices correspond to
            spin-up (alpha) modes and odd indices correspond to spin-down
            (beta) modes.
        """

        one_body_integrals = general_basis_change(                                   
            numpy.asarray(mints.ao_kinetic()), canonical_orbitals, (1, 0))           
        one_body_integrals += general_basis_change(                                  
            numpy.asarray(mints.ao_potential()), canonical_orbitals, (1, 0))         
        two_body_integrals = numpy.asarray(mints.ao_eri())   
        n_orbitals = canonical_orbitals.shape[0]                          
        two_body_integrals.reshape((n_orbitals, n_orbitals,                          
                                    n_orbitals, n_orbitals))                         
        two_body_integrals = numpy.einsum('psqr', two_body_integrals)                
        two_body_integrals = general_basis_change(                                   
            two_body_integrals, canonical_orbitals, (1, 1, 0, 0))    

        n_qubits = 2 * one_body_integrals.shape[0]

        # Initialize Hamiltonian coefficients.
        one_body_coefficients = numpy.zeros((n_qubits, n_qubits))
        two_body_coefficients = numpy.zeros((n_qubits, n_qubits,
                                             n_qubits, n_qubits))
        # Loop through integrals.
        for p in range(n_qubits // 2):
            for q in range(n_qubits // 2):

                # Populate 1-body coefficients. Require p and q have same spin.
                one_body_coefficients[2 * p, 2 * q] = one_body_integrals[
                    p, q]
                one_body_coefficients[2 * p + 1, 2 *
                                      q + 1] = one_body_integrals[p, q]
                # Continue looping to prepare 2-body coefficients.
                for r in range(n_qubits // 2):
                    for s in range(n_qubits // 2):

                        # Mixed spin
                        two_body_coefficients[2 * p, 2 * q + 1,
                                              2 * r + 1, 2 * s] = (
                            two_body_integrals[p, q, r, s] / 2.)
                        two_body_coefficients[2 * p + 1, 2 * q,
                                              2 * r, 2 * s + 1] = (
                            two_body_integrals[p, q, r, s] / 2.)

                        # Same spin
                        two_body_coefficients[2 * p, 2 * q,
                                              2 * r, 2 * s] = (
                            two_body_integrals[p, q, r, s] / 2.)
                        two_body_coefficients[2 * p + 1, 2 * q + 1,
                                              2 * r + 1, 2 * s + 1] = (
                            two_body_integrals[p, q, r, s] / 2.)

        # Truncate.
        one_body_coefficients[
            numpy.absolute(one_body_coefficients) < EQ_TOLERANCE] = 0.
        two_body_coefficients[
            numpy.absolute(two_body_coefficients) < EQ_TOLERANCE] = 0.

        operator = MyPolynomialTensor(                                                   
                {(): nuclear_repulsion,                                             
                (1, 0): one_body_coefficients,                                      
                (1, 1, 0, 0): two_body_coefficients})    
        return operator.__str__()                   

def run_psi4(para:dict):
    try:
        if (not para.__contains__("multiplicity")):
            para["multiplicity"] = 1
    
        if (not para.__contains__("charge")):
            para["charge"] = 0

        if (not para.__contains__("basis")):
            para["basis"] = 0

        if (not para.__contains__("EQ_TOLERANCE")):
            para["EQ_TOLERANCE"] = 1e-8

        str_mol = para["mol"]
        str_mol = str_mol + "\n symmetry c1"

        mol = geometry(str_mol, "mol")

        core.IO.set_default_namespace("mol")                                                                                
        mol.set_multiplicity(para["multiplicity"])
        mol.set_molecular_charge(para["charge"])

        core.set_global_option("BASIS", para["basis"])

        core.set_global_option("SCF_TYPE", "pk")

        core.set_global_option("SOSCF", "false")

        core.set_global_option("FREEZE_CORE", "false")

        core.set_global_option("DF_SCF_GUESS", "false")

        core.set_global_option("OPDM", "true")

        core.set_global_option("TPDM", "true")

        core.set_global_option("MAXITER", 1e6)

        core.set_global_option("NUM_AMPS_PRINT", 1e6)

        core.set_global_option("R_CONVERGENCE", 1e-6)

        core.set_global_option("D_CONVERGENCE", 1e-6)

        core.set_global_option("E_CONVERGENCE", 1e-6)

        core.set_global_option("DAMPING_PERCENTAGE", 0)
                                                                                    
        if mol.multiplicity == 1:                                                        
            core.set_global_option("REFERENCE", "rhf")
            core.set_global_option("GUESS", "sad")
        else:                                                                            
            core.set_global_option("REFERENCE", "rohf")
            core.set_global_option("GUESS", "gwh")
        hf_energy, hf_wavefunction = energy('scf', return_wfn=True)     
    except Exception as exception:                                                       
        return (-1, traceback.format_exc())                                                                                                             
    else:                                                                            
        nuclear_repulsion = mol.nuclear_repulsion_energy()                           
        canonical_orbitals = numpy.asarray(hf_wavefunction.Ca())                     

        mints = core.MintsHelper(hf_wavefunction.basisset())      
        fermion_str = get_molecular_hamiltonian(mints, canonical_orbitals, nuclear_repulsion, para["EQ_TOLERANCE"]) 
        return (0,fermion_str)
                                        
                                                                                            
                                                                
