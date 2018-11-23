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
    '''
    client for the quantum chemistry simulation.
    Used to calculate the hamiltonian for a molecule.

    hamiltonian_type:
        'pauli' for PauliOperator representation.
        'fermion' for FermionOperator representation.
        'raw' for FermionOperator representation without simplification
    '''
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