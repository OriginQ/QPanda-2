#综合示例——H2基态能量计算
#2.使用梯度下降优化器进行实现
#准备工作
from pyqpanda import *
from psi4_wrapper import *
import numpy as np
from functools import partial
from math import pi
import matplotlib.pyplot as plt
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
#构造可变参数的CCSD模型的哈密顿量
def get_ccsd_var(qn, en, para):
    '''
    get Coupled cluster single and double model with variational parameters.
    e.g. 4 qubits, 2 electrons
    then 0 and 1 are occupied,just consider 0->2,0->3,1->2,1->3,01->23.
    returned FermionOperator like this:
    {{"2+ 0":var[0]},{"3+ 0":var[1]},{"2+ 1":var[2]},{"3+ 1":var[3]},
    {"3+ 2+ 1 0":var[4]}}

    '''
    if en > qn:
        assert False
    if en == qn:
        return VarFermionOperator()
    
    if get_ccsd_n_term(qn, en) != len(para):
        assert False
    
    cnt = 0
    var_fermion_op = VarFermionOperator()
    for i in range(en):
        for ex in range(en, qn):
            var_fermion_op += VarFermionOperator(str(ex) + "+ " + str(i), para[cnt])
            cnt += 1
            
    return var_fermion_op
    
    #for i in range(en):
    #    for j in range(i+1, en):
    #        for ex1 in range(en, qn):
    #            for ex2 in range(ex1+1, qn):
    #                fermion_op += VarFermionOperator(
    #                    str(ex2)+"+ "+str(ex1)+"+ "+str(j)+" "+str(i),
    #                    para[cnt]
    #                )
    #                cnt += 1
                    
    #return fermion_op

#JordanWigner变换，将费米子哈密顿量子项转换成泡利哈密顿量
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
#JordanWigner变换，将VarFermionOperator转换成VarPauliOperator
def JordanWignerTransformVar(var_fermion_op):
    data = var_fermion_op.data()
    var_pauli = VarPauliOperator()
    for i in data:
        one_pauli = get_fermion_jordan_wigner(i[0][0])
        for j in one_pauli.data():
            var_pauli += VarPauliOperator(j[0][1], complex_var(
                i[1].real()*j[1].real-i[1].imag()*j[1].imag,
                i[1].real()*j[1].imag+i[1].imag()*j[1].real))
    
    return var_pauli
#CC到UCC变换
def cc_to_ucc_hamiltonian_var(cc_op):
    '''
    generate Hamiltonian form of unitary coupled cluster based on coupled cluster,H=1j*(T-dagger(T)),
    then exp(-jHt)=exp(T-dagger(T))
    '''
    pauli = VarPauliOperator()
    for i in cc_op.data():
        pauli += VarPauliOperator(i[0][1], complex_var(var(-2)*i[1].imag(), var(0)))

    return pauli
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
#哈密顿量模拟——对单个子项进行模拟
def simulate_one_term_var(qubit_list, hamiltonian_term, coef, t):
    '''
    Simulate a single term of Hamilonian like "X0 Y1 Z2" with
    coefficient and time. U=exp(-it*coef*H)
    '''
    vqc = VariationalQuantumCircuit()

    if len(hamiltonian_term) == 0:
        return vqc

    tmp_qlist = []
    for q, term in hamiltonian_term.items():        
        if term is 'X':            
            vqc.insert(H(qubit_list[q]))            
        elif term is 'Y':
            vqc.insert(RX(qubit_list[q],pi/2))              

        tmp_qlist.append(qubit_list[q])     

    size = len(tmp_qlist)
    if size == 1:
        vqc.insert(VariationalQuantumGate_RZ(tmp_qlist[0], 2*coef*t))
    elif size > 1:
        for i in range(size - 1):
            vqc.insert(CNOT(tmp_qlist[i], tmp_qlist[size - 1]))   
        vqc.insert(VariationalQuantumGate_RZ(tmp_qlist[size-1], 2*coef*t))
        for i in range(size - 1):
            vqc.insert(CNOT(tmp_qlist[i], tmp_qlist[size - 1]))   
    
    # dagger
    for q, term in hamiltonian_term.items():        
        if term is 'X':            
            vqc.insert(H(qubit_list[q]))            
        elif term is 'Y':
            vqc.insert(RX(qubit_list[q],-pi/2))  
    
    return vqc
#哈密顿量模拟使用特洛特变换
def simulate_hamiltonian_var(qubit_list,var_pauli,t,slices=3):
    '''
    Simulate a general case of hamiltonian by Trotter-Suzuki
    approximation. U=exp(-iHt)=(exp(-i H1 t/n)*exp(-i H2 t/n))^n
    '''
    vqc = VariationalQuantumCircuit()

    for i in range(slices):
        for j in var_pauli.data():
            term = j[0][0]
            vqc.insert(simulate_one_term_var(qubit_list, term, j[1].real(), t/slices))

    return vqc
#梯度下降优化算法
def GradientDescent(mol_pauli, n_qubit, n_en, iters):
    n_para = get_ccsd_n_term(n_qubit, n_electron)

    var_para = []
    for i in range(n_para):
        var_para.append(var(0.5, True))
    
    fermion_cc = get_ccsd_var(n_qubit, n_en, var_para)
    pauli_cc = JordanWignerTransformVar(fermion_cc)
    ucc = cc_to_ucc_hamiltonian_var(pauli_cc)

    machine=init_quantum_machine(QMachineType.CPU)
    qlist = machine.qAlloc_many(n_qubit)

    vqc = VariationalQuantumCircuit()
    vqc.insert(prepareInitialState(qlist, n_en))
    vqc.insert(simulate_hamiltonian_var(qlist, ucc, 1.0, 3))

    loss = qop(vqc, mol_pauli, machine, qlist)
    gd_optimizer = MomentumOptimizer.minimize(loss, 0.1, 0.9)
    leaves = gd_optimizer.get_variables()

    min_energy=float('inf')
    for i in range(iters):
        gd_optimizer.run(leaves, 0)
        loss_value = gd_optimizer.get_loss()
    
        print(loss_value)
        if loss_value < min_energy:
            min_energy = loss_value
    
    return min_energy
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

        n_qubit = pauli_op.getMaxIndex()

        energies.append(GradientDescent(pauli_op, n_qubit, n_electron, 30))

    plt.plot(distances , energies, 'r')
    plt.xlabel('distance')
    plt.ylabel('energy')
    plt.title('VQE PLOT')
    plt.show()