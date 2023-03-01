from pyqpanda import *
from psi4_wrapper import *
import numpy as np
from functools import partial
from math import pi
import matplotlib.pyplot as plt
#损失函数
def loss_func(para_list, qubit_number, electron_number, Hamiltonian):
    '''
    <𝜓^∗|𝐻|𝜓>, Calculation system expectation of Hamiltonian in experimental state.
    para_list: parameters to be optimized
    qubit_number: qubit number
    electron_number: electron number
    Hamiltonian: System Hamiltonian
    '''
    fermion_cc =get_ccsd(qubit_number, electron_number, para_list)
    pauli_cc = JordanWignerTransform(fermion_cc)
    ucc = cc_to_ucc_hamiltonian(pauli_cc)
    expectation=0
    for component in Hamiltonian:
        expectation+=get_expectation(qubit_number, electron_number, ucc, component)
    expectation=float(expectation.real)
    print(expectation)
    return ("", expectation)
#获取CCSD模型的参数个数
def get_ccsd_n_term(qn, en):
    '''
    coupled cluster single and double model.
    e.g. 4 qubits, 2 electrons
    then 0 and 1 are occupied,just consider 0->2,0->3,1->2,1->3,01->23
    '''

    if n_electron>n_qubit:
        assert False
    
    return int((qn - en) * en + (qn - en)* (qn -en - 1) * en * (en - 1) / 4)
#构造普通参数的CCSD模型的哈密顿量
def get_ccsd(qn, en, para):
    '''
    get Coupled cluster single and double model.
    e.g. 4 qubits, 2 electrons
    then 0 and 1 are occupied,just consider 0->2,0->3,1->2,1->3,01->23.
    returned FermionOperator like this:
    {{"2+ 0":var[0]},{"3+ 0":var[1]},{"2+ 1":var[2]},{"3+ 1":var[3]},
    {"3+ 2+ 1 0":var[4]}}

    '''
    if n_electron>n_qubit:
        assert False
    if n_electron==n_qubit:
        return FermionOperator()

    if get_ccsd_n_term(qn, en) != len(para):
        assert False

    cnt = 0
    fermion_op = FermionOperator()
    for i in range(en):
        for ex in range(en, qn):
            fermion_op += FermionOperator(str(ex) + "+ " + str(i), para[cnt])
            cnt += 1   
    
    for i in range(n_electron):
        for j in range(i+1,n_electron):
            for ex1 in range(n_electron,n_qubit):
                for ex2 in range(ex1+1,n_qubit):
                    fermion_op += FermionOperator(
                        str(ex2)+"+ "+str(ex1)+"+ "+str(j)+" "+str(i),
                        para[cnt]
                    )
                    cnt += 1
                    
    return fermion_op
#JW变换，将FermionOperator转换成PauliOperator
def JordanWignerTransform(fermion_op):
    data = fermion_op.data()
    pauli = PauliOperator()
    for i in data:
        pauli += get_fermion_jordan_wigner(i[0][0])*i[1]
    return pauli
#JordanWigner变换
def get_fermion_jordan_wigner(fermion_item):
    pauli = PauliOperator("", 1)

    for i in fermion_item:
        op_qubit = i[0]
        op_str = ""
        for j in range(op_qubit):
            op_str += "Z" + str(j) + " "

        op_str1 = op_str + "X" + str(op_qubit)
        op_str2 = op_str + "Y" + str(op_qubit)

        pauli_map = {}
        pauli_map[op_str1] = 0.5

        if i[1]:
            pauli_map[op_str2] = -0.5j
        else:
            pauli_map[op_str2] = 0.5j

        pauli *= PauliOperator(pauli_map)

    return pauli
#CC到UCC变换
def cc_to_ucc_hamiltonian(cc_op):
    '''
    generate Hamiltonian form of unitary coupled cluster 
    based on coupled cluster,H=1j*(T-dagger(T)),
    then exp(-iHt)=exp(T-dagger(T))
    '''
    return 1j*(cc_op-cc_op.dagger())
#计算期望
def get_expectation(n_qubit, n_en, ucc,component):
    '''
    get expectation of one hamiltonian.
    n_qubit: qubit number
    n_en: electron number
    ucc: unitary coupled cluster operator
    component: paolioperator and coefficient,e.g. ('X0 Y1 Z2',0.2)
    '''

    machine=init_quantum_machine(QMachineType.CPU)
    q = machine.qAlloc_many(n_qubit)
    prog=QProg()

    prog.insert(prepareInitialState(q, n_en))
    prog.insert(simulate_hamiltonian(q, ucc, 1.0, 4))
    
    for i, j in component[0].items():
        if j=='X':
            prog.insert(H(q[i]))
        elif j=='Y':
            prog.insert(RX(q[i],pi/2))
    
    machine.directly_run(prog)
    result=machine.get_prob_dict(q, select_max=-1)
    machine.qFree_all(q)
    
    expectation=0
    #奇负偶正
    for i in result:
        if parity_check(i, component[0]):
            expectation-=result[i]
        else:
            expectation+=result[i]       
    return expectation*component[1]
#制备初态
def prepareInitialState(qlist, en):
    '''
    prepare initial state. 
    qlist: qubit list
    en: electron number
    return a QCircuit
    '''
    circuit = QCircuit()
    if len(qlist) < en:
        return circuit

    for i in range(en):
        circuit.insert(X(qlist[i]))

    return circuit;
#哈密顿量模拟
def simulate_hamiltonian(qubit_list,pauli,t,slices=3):
    '''
    Simulate a general case of hamiltonian by Trotter-Suzuki
    approximation. U=exp(-iHt)=(exp(-i H1 t/n)*exp(-i H2 t/n))^n
    '''
    circuit =QCircuit()

    for i in range(slices):
        for op in pauli.data():
            term = op[0][0]
            circuit.insert(
                simulate_one_term(
                    qubit_list, 
                    term, op[1].real, 
                    t/slices
                )
            )

    return circuit
#哈密顿量模拟——对单个子项进行模拟
def simulate_one_term(qubit_list, hamiltonian_term, coef, t):
    '''
    Simulate a single term of Hamilonian like "X0 Y1 Z2" with
    coefficient and time. U=exp(-it*coef*H)
    '''
    circuit =QCircuit()

    if not hamiltonian_term:
        return circuit

    transform=QCircuit()
    tmp_qlist = []
    for q, term in hamiltonian_term.items():        
        if term is 'X':            
            transform.insert(H(qubit_list[q]))            
        elif term is 'Y':
            transform.insert(RX(qubit_list[q],pi/2))              

        tmp_qlist.append(qubit_list[q])     

    circuit.insert(transform)

    size = len(tmp_qlist)
    if size == 1:
        circuit.insert(RZ(tmp_qlist[0], 2*coef*t))
    elif size > 1:
        for i in range(size - 1):
            circuit.insert(CNOT(tmp_qlist[i], tmp_qlist[size - 1]))   
        circuit.insert(RZ(tmp_qlist[size-1], 2*coef*t))
        for i in range(size - 1):
            circuit.insert(CNOT(tmp_qlist[i], tmp_qlist[size - 1]))  

    circuit.insert(transform.dagger())

    return circuit
#奇偶校验
def parity_check(number, terms):
    '''
    pairty check 
    number: quantum state
    terms: a single term of PauliOperator, like"[(0, X), (1, Y)]"
    '''
    check=0
    number=number[::-1]
    for i in terms:
        if number[i]=='1':
            check+=1
    return check%2
#非梯度下降优化算法
def optimize_by_no_gradient(mol_pauli, n_qubit, n_en, iters):
    n_para = get_ccsd_n_term(n_qubit, n_electron)

    para_vec = []
    for i in range(n_para):
        para_vec.append(0.5)

    no_gd_optimizer = OptimizerFactory.makeOptimizer(OptimizerType.NELDER_MEAD)
    no_gd_optimizer.setMaxIter(iters)
    no_gd_optimizer.setMaxFCalls(iters)
    no_gd_optimizer.registerFunc(partial(
        loss_func, 
        qubit_number = n_qubit, 
        electron_number = n_en,
        Hamiltonian=mol_pauli.toHamiltonian(1)), 
        para_vec)

    no_gd_optimizer.exec()
    result = no_gd_optimizer.getResult()
    print(result.fun_val)

    return result.fun_val
#获取原子对应的电子数
def getAtomElectronNum(atom):
    atom_electron_map = {
        'H':1, 'He':2, 'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Ne':10, 
        'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'Ar':18
    }

    if (not atom_electron_map.__contains__(atom)):
        return 0

    return atom_electron_map[atom]
#主函数
if __name__=="__main__":    
    distances = [x * 0.1 for x in range(2, 25)]
    molecule = "H 0 0 0\nH 0 0 {0}"

    molecules = []
    for d in distances:
        molecules.append(molecule.format(d))

    chemistry_dict = {
        "mol":"",
        "multiplicity":1,
        "charge":0,
        "basis":"sto-3g",
    }

    energies = []

    for d in distances:
        mol = molecule.format(d)

        chemistry_dict["mol"] = molecule.format(d)
        data = run_psi4(chemistry_dict)
        #get molecule electron number
        n_electron = 0
        mol_splits = mol.split()
        cnt = 0
        while (cnt < len(mol_splits)):
            n_electron += getAtomElectronNum(mol_splits[cnt])
            cnt += 4

        fermion_op = parsePsi4DataToFermion(data[1])
        pauli_op = JordanWignerTransform(fermion_op)

        n_qubit = pauli_op.getMaxIndex()+1

        energies.append(optimize_by_no_gradient(pauli_op, n_qubit, n_electron, 200))

    plt.plot(distances , energies, 'r')
    plt.xlabel('distance')
    plt.ylabel('energy')
    plt.title('VQE PLOT')
    plt.show()