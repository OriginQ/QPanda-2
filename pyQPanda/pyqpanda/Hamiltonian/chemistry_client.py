'''
Client for Quantum Chemistry Simulation.
Copyright (C) Origin Quantum 2017-2018\n
Licensed Under Apache Licence 2.0
'''

import json
import requests

def chem_client(geometry,
                basis,
                multiplicity,
                charge,
                run_mp2=True,
                run_cisd=True,
                run_ccsd=True,
                run_fci=True,
                hamiltonian_type='pauli',
                url='http://117.71.57.182:2222'
):    
    """
    Interface for quantum chemistry simulations, designed to compute the Hamiltonian of a molecular system.

    Parameters:
        geometry (str): The molecular geometry data.
        basis (str): The basis set information.
        multiplicity (int): The electronic multiplicity of the molecule.
        charge (int): The overall charge of the molecule.
        run_mp2 (bool): Flag to indicate whether to run the MP2 calculation.
        run_cisd (bool): Flag to indicate whether to run the CISD calculation.
        run_ccsd (bool): Flag to indicate whether to run the CCSD calculation.
        run_fci (bool): Flag to indicate whether to run the FCI calculation.
        hamiltonian_type (str): Type of Hamiltonian representation:
                                'pauli' for PauliOperator,
                                'fermion' for FermionOperator,
                                'raw' for FermionOperator without simplification.
        url (str): The URL endpoint for the quantum chemistry simulation service.

    Returns:
        str: The JSON response from the quantum chemistry simulation service.

    The function constructs a molecule dictionary from the input parameters,
    serializes it to a JSON string, and sends it to a specified URL to perform
    quantum chemistry calculations. The result is then returned as a string.
    """
    molecule_dict={
        'molecule':geometry,
        'basis':basis,
        'multiplicity':multiplicity,
        'charge':charge,
        'run_mp2':run_mp2,
        'run_cisd':run_cisd,
        'run_ccsd':run_ccsd,
        'run_fci':run_fci,
        'hamiltonian_type':hamiltonian_type
    }
    molecule_str=json.dumps(molecule_dict)
    response = requests.get(url+'/?'+molecule_str)
    return response.text