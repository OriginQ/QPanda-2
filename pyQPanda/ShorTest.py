from pyqpanda import *
import matplotlib.pyplot as plt
import math as m 

def plotBar(xdata, ydata):
    fig, ax = plt.subplots()
    fig.set_size_inches(6,6)
    fig.set_dpi(100)
    
    rects =  ax.bar(xdata, ydata, color='b')

    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha="center", va="bottom")
    
    plt.rcParams['font.sans-serif']=['Arial']
    plt.title("Origin Q", loc='right', alpha = 0.5)
    plt.ylabel('Times')
    plt.xlabel('States')
        
    plt.show()
    
def reorganizeData(measure_qubits, quick_meausre_result):
    xdata = []
    ydata = []

    for i in quick_meausre_result:
        xdata.append(i)
        ydata.append(quick_meausre_result[i])
    
    return xdata, ydata
   
def gcd(m,n):
    if not n:
        return m
    else:
        return gcd(n, m%n)
  
def MAJ(a, b, c):
    circ = QCircuit()
    circ.insert(CNOT(c,b))
    circ.insert(CNOT(c,a))
    circ.insert(Toffoli(a, b, c))

    return circ	
# def Adder(a, b, c):
#     circuit = CreateEmptyCircuit()
#     nbit = len(a)
#     circuit.insert(MAJ(c, a[0], b[0]))
#     for i in range(1,nbit,1):
#         circuit.insert(MAJ(b[i - 1], a[i], b[i]))
#     for i in range(nbit-1,0, - 1):
#         circuit.insert(MAJ(b[i - 1], a[i], b[i]))
#     circuit.insert(UMA(c, a[0], b[0]))
#     return circuit
    
def UMA(a, b, c):
    circ = QCircuit()
    circ.insert(Toffoli(a, b, c)).insert(CNOT(c, a)).insert(CNOT(a, b))  

    return circ   
    
def MAJ2(a, b, c):
    if ((len(a) == 0) or (len(a) != (len(b)))):
        raise RuntimeError('a and b must be equal, but not equal to 0!')

    nbit = len(a)
    circ = QCircuit()
    circ.insert(MAJ(c, a[0], b[0]))

    for i in range(1, nbit):
        circ.insert(MAJ(b[i-1], a[i], b[i]))
    
    return circ

def Adder(a, b, c):
    if ((len(a) == 0) or (len(a) != (len(b)))):
        raise RuntimeError('a and b must be equal, but not equal to 0!')

    nbit = len(a)
    circ = QCircuit()
    circ.insert(MAJ(c, a[0], b[0]))

    for i in range(1, nbit):
        circ.insert(MAJ(b[i-1], a[i], b[i]))

    for i in range(nbit-1, 0, -1):
        circ.insert(UMA(b[i-1], a[i], b[i]))

    circ.insert(UMA(c, a[0], b[0]))
    
    return circ

def isCarry(a, b, c, carry):
    if ((len(a) == 0) or (len(a) != (len(b)))):
        raise RuntimeError('a and b must be equal, but not equal to 0!')

    circ = QCircuit()

    circ.insert(MAJ2(a, b, c))
    circ.insert(CNOT(b[-1], carry))
    circ.insert(MAJ2(a, b, c).dagger())

    return circ
    
def bindData(qlist, data):
    check_value = 1 << len(qlist)
    if (data >= check_value):
        raise RuntimeError('data >= check_value')

    circ = QCircuit()
    i = 0
    while (data >= 1):
        if (data % 2) == 1:
            circ.insert(X(qlist[i]))
        
        data = data >> 1
        i = i+1
    
    return circ

def isCarry(a, b, c, carry):
    if ((len(a) == 0) or (len(a) != (len(b)))):
        raise RuntimeError('a and b must be equal, but not equal to 0!')

    circ = QCircuit()

    circ.insert(MAJ2(a, b, c))
    circ.insert(CNOT(b[-1], carry))
    circ.insert(MAJ2(a, b, c).dagger())

    return circ
    
def constModAdd(qa, C, M, qb, qs1):
    circ = QCircuit()
    
    q_num = len(qa)

    tmp_value = (1 << q_num) - M + C

    circ.insert(bindData(qb, tmp_value))
    circ.insert(isCarry(qa, qb, qs1[1], qs1[0]))
    circ.insert(bindData(qb, tmp_value))
    
    tmp_circ = QCircuit()
    tmp_circ.insert(bindData(qb, tmp_value))
    tmp_circ.insert(Adder(qa, qb, qs1[1]))
    tmp_circ.insert(bindData(qb, tmp_value))
    tmp_circ = tmp_circ.control([qs1[0]])
    circ.insert(tmp_circ)

    circ.insert(X(qs1[0]))

    tmp2_circ = QCircuit()
    tmp2_circ.insert(bindData(qb, C))
    tmp2_circ.insert(Adder(qa, qb, qs1[1]))
    tmp2_circ.insert(bindData(qb, C))
    tmp2_circ = tmp2_circ.control([qs1[0]])
    circ.insert(tmp2_circ)

    circ.insert(X(qs1[0]))

    tmp_value = (1 << q_num) - C
    circ.insert(bindData(qb, tmp_value))
    circ.insert(isCarry(qa, qb, qs1[1], qs1[0]))
    circ.insert(bindData(qb, tmp_value))
    circ.insert(X(qs1[0]))

    return circ
    
def modreverse(c, m):
    if (c == 0):
        raise RecursionError('c is zero!')
    
    if (c == 1):
        return 1
    
    m1 = m
    quotient = []
    quo = m // c
    remainder = m % c

    quotient.append(quo)

    while (remainder != 1):
        m = c
        c = remainder
        quo = m // c
        remainder = m % c
        quotient.append(quo)

    if (len(quotient) == 1):
        return m - quo

    if (len(quotient) == 2):
        return 1 + quotient[0]*quotient[1]

    rev1 = 1
    rev2 = quotient[-1]
    reverse_list = quotient[0:-1]
    reverse_list.reverse()
    for i in reverse_list:
        rev1 = rev1 + rev2 * i
        temp = rev1
        rev1 = rev2
        rev2 = temp

    if ((len(quotient) % 2) == 0):
        return rev2

    return m1 - rev2
    

def constModMul(qa, const_num, M, qs1, qs2, qs3):
    circ = QCircuit()
    
    q_num = len(qa)

    for i in range(0, q_num):
        tmp_circ = QCircuit()
        tmp = const_num * pow(2, i) %M
        tmp_circ.insert(constModAdd(qs1, tmp, M, qs2, qs3))
        tmp_circ = tmp_circ.control([qa[i]])
        circ.insert(tmp_circ)

    #state swap
    for i in range(0, q_num):
        circ.insert(CNOT(qa[i], qs1[i]))
        circ.insert(CNOT(qs1[i], qa[i]))
        circ.insert(CNOT(qa[i], qs1[i]))

    Crev = modreverse(const_num, M)
    
    tmp2_circ = QCircuit()
    for i in range(0, q_num):
        tmp = Crev* pow(2, i)
        tmp = tmp % M
        tmp_circ = QCircuit()
        tmp_circ.insert(constModAdd(qs1, tmp, M, qs2, qs3))
        tmp_circ = tmp_circ.control([qa[i]])
        tmp2_circ.insert(tmp_circ)
    
    circ.insert(tmp2_circ.dagger())

    return circ
    
def constModExp(qa, qb, base, M, qs1, qs2, qs3):
    circ = QCircuit()

    cqnum = len(qa)

    temp = base

    for i in range(0, cqnum):    
        circ.insert(constModMul(qb, temp, M, qs1, qs2, qs3).control([qa[i]]))
        temp = temp * temp
        temp = temp % M

    return circ
    
def qft(qlist):
    circ = QCircuit()
    
    qnum = len(qlist)
    for i in range(0, qnum):
        circ.insert(H(qlist[qnum-1-i]))
        for j in range(i + 1, qnum):
            circ.insert(CR(qlist[qnum-1-j], qlist[qnum-1-i], m.pi/(1 << (j-i))))

    for i in range(0, qnum//2):
        circ.insert(CNOT(qlist[i], qlist[qnum-1-i]))
        circ.insert(CNOT(qlist[qnum-1-i], qlist[i]))
        circ.insert(CNOT(qlist[i], qlist[qnum-1-i]))

    return circ

def shorAlg(base, M):
    if ((base < 2) or (base > M - 1)):
        raise('Invalid base!')

    if (gcd(base, M) != 1):
        raise('Invalid base! base and M must be mutually prime')
    
    binary_len = 0
    while M >> binary_len != 0 :
        binary_len = binary_len + 1
    
    machine = init_quantum_machine(QMachineType.CPU_SINGLE_THREAD)

    qa = machine.qAlloc_many(binary_len*2)
    qb = machine.qAlloc_many(binary_len)

    qs1 = machine.qAlloc_many(binary_len)
    qs2 = machine.qAlloc_many(binary_len)
    qs3 = machine.qAlloc_many(2)

    prog = QProg()

    prog.insert(X(qb[0]))
    prog.insert(single_gate_apply_to_all(H, qa))
    prog.insert(constModExp(qa, qb, base, M, qs1, qs2, qs3))
    prog.insert(qft(qa).dagger())

    directly_run(prog)
    result = quick_measure(qa, 100)

    print(result)

    xdata, ydata = reorganizeData(qa, result)
    plotBar(xdata, ydata)

    return result
    
if __name__=="__main__":
    base = 2
    N = 15
    shorAlg(base, N)