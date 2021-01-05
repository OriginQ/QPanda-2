from pyqpanda import *

qvm = init_quantum_machine(QMachineType.CPU_SINGLE_THREAD)

a = qvm.qAlloc_many(3)
b = qvm.qAlloc_many(3)
k = qvm.qAlloc_many(len(a)+2)

c = qvm.qAlloc()
is_carry = qvm.qAlloc()

prog = QProg()

# prog.insert(X(a[0])).insert(X(a[1])) ## Preparation of addend a = |011>
# prog.insert(X(b[0])).insert(X(b[1])).insert(X(b[2])) ## Preparation of addend b = |111>
#prog.insert(isCarry(a, b, c, is_carry)) ## Return carry item of a + b 
#prog.insert(QAdderIgnoreCarry(a, b, c)) ## Return a + b (ignore carry item of a + b)
# prog.insert(QAdder(a, b, c, is_carry)) ## Return a + b 
prog.insert(bind_data(3, a))
prog.insert(bind_data(3, b))
prog.insert(QSub(a, b, k))

# originir = convert_qprog_to_originir(prog, qvm) # Return the quantum circuit
# print(originir) 

directly_run(prog)

# result_is_carry = quick_measure([is_carry], 1000) 
result_a = quick_measure(a+b+k, 1000)
# result_b = quick_measure(b, 1000)

# print(" The carry item of a + b : ", result_is_carry)
print(" The result of a + b minus the carry term : ", result_a)
# print(" The invariant addend b : ", result_b)





