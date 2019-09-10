from pyqpanda import *

def test_qprog_to_originir():
    machine = init_quantum_machine(QMachineType.CPU)
    f = open('testfile.txt', mode='w',encoding='utf-8')  
    f.write("""QINIT 4
        CREG 4
        DAGGER
        X q[1]
        X q[2]
        CONTROL q[1], q[2]
        RY q[0], (1.047198)
        ENDCONTROL
        ENDDAGGER
        MEASURE q[0], c[0]
        QIF c[0]
        H q[1]
        H q[2]
        RZ q[2], (2.356194)
        CU q[2], q[3], (3.141593, 4.712389, 1.570796, -1.570796)
        CNOT q[2], q[1]
        ENDQIF
        """)
        
    f.close()
    prog_trans = originir_to_qprog("testfile.txt", machine)
    print(to_originir(prog_trans,machine))

    destroy_quantum_machine(machine)

def test_originir_to_qprog():
    machine = init_quantum_machine(QMachineType.CPU)
    qlist = machine.qAlloc_many(4)
    clist = machine.cAlloc_many(4)
            
    prog = CreateEmptyQProg()
    prog_cir = CreateEmptyCircuit()
    prog_cir.insert(Y(qlist[2])).insert(H(qlist[2])).insert(CNOT(qlist[0],qlist[1]))
    qwhile = CreateWhileProg(clist[1], prog_cir)
            
    prog.insert(H(qlist[2])).insert(measure(qlist[1],clist[1])).insert(qwhile)
            
    print(to_originir(prog,machine))
            
    destroy_quantum_machine(machine)

if __name__=="__main__":
    print('测试量子程序转换OriginIR程序示例：')
    test_qprog_to_originir()
    #print('测试OriginIR程序转换量子程序示例：')
    #test_originir_to_qprog()