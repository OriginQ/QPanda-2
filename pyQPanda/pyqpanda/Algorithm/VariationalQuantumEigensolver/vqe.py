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
    '''
    construct Hamiltonian based on str,str has molecule information
    '''
    terms=str.split(' +\n')
    tuplelist=dict()
    for term in terms:
        data=term.split(' ',1)        
        tuplelist[data[1][1:-1]]=eval(data[0])

    operator=PauliOperator(tuplelist)    
    return operator

def H2_energy_from_distance(distance):
    '''
    compute base energy of H2.
    distance: distance between two atoms of H2
    '''
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
    '''
    get electron number based on Atom_Dict,
    Atom_Dict={'H':1,'He':2,...,'CI':17}
    '''
    n_electron = 0
    for atom in geometry:
        n_electron+=Atom_Dict[atom[0]]
    return n_electron


def get_fermion_jordan_wigner(fermion_type, op_qubit):
    '''
    fermion_op = ('a',1) or ('c',1)
    'a' is for annihilation
    'c' is for creation
    '''
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
    '''
    coupled cluster single model.
    e.g. 4 qubits, 2 electrons
    then 0 and 1 are occupied,just consider 0->2,0->3,1->2,1->3
    '''
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
    '''
    coupled cluster single and double model.
    e.g. 4 qubits, 2 electrons
    then 0 and 1 are occupied,just consider 0->2,0->3,1->2,1->3,01->23
    '''

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
    '''
    coupled cluster single model.
    J-W transform on CCS, get paulioperator 
    '''
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
    '''
    coupled cluster single and double model.
    J-W transform on CCSD, get paulioperator 
    '''
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
    '''
    generate Hamiltonian form of unitary coupled cluster based on coupled cluster,H=1j*(T-dagger(T)),
    then exp(-jHt)=exp(T-dagger(T))
    '''
    return 1j*(cc_op-cc_op.dagger())

def flatten(pauliOperator):
    '''
    if all coefficients of paulioperator can be written as C+0j,
    transform coefficients to float style,
    else, return error
    '''
    new_ops=deepcopy(pauliOperator.ops)
    for term in new_ops:
        if abs(new_ops[term].imag)<pauliOperator.m_error_threshold:
            new_ops[term]=float(new_ops[term].real)
        else:
            assert False
    return PauliOperator(new_ops)

def transform_base(qubitlist,base):
    '''
    choose measurement basis,
    it means rotate all axis to z-axis
    '''
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
    '''
    get expectation of one paulioperator.
    qubit_number:qubit number
    unitaryCC: unitary coupled cluster operator
    component: paolioperator and coefficient,e.g. ('X0 Y1 Z2',0.33)
    '''
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
    '''
    get expectation of Hamitonian.
    qubit_number:qubit number
    electron_number:electron number
    Hamiltonian:Hamiltonian expressed by paulioperator
    unitaryCC: unitary coupled cluster operator
    '''
    expectation=0
    for component in Hamiltonian.ops:
        #print(component)
        temp=(component,Hamiltonian.ops[component])
        expectation+=get_expectation(qubit_number=qubit_number_,unitaryCC=unitaryCC_,component=temp,shots_=shots)
    expectation=float(expectation.real)
    return expectation


def binding(qubit_number,electron_number,Hamiltonian,shots):
    return partial(vqe_in_list,
                    qubit_number=qubit_number,
                    electron_number=electron_number,
                    Hamiltonian=Hamiltonian,
                    shots=shots)
def vqe_in_list(arguments,qubit_number,electron_number,Hamiltonian,shots):
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





