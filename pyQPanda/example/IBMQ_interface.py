from pyqpanda import *
import unittest

import pyqpanda.backends.IBM.executeOnIBMQ as IBMQexecute
import math
class Test_IBMQ(unittest.TestCase):
    def testIBMQOnPypanda(outputLog = True):
        ibmExecute = IBMQexecute.CExecuteOnIBMQ()
        ibmExecute.outputTestLog = outputLog

        #save account to local side
        try:
            import IBMQconfig
            #ibmExecute.save_IBMQ_account(IBMQconfig.APItoken, IBMQconfig.config['url'])
        except:
            print("""WARNING: There's no connection with the API for remote backends.
                    Have you initialized a IBMQconfig.py file with your personal token?
                    For now, there's only access to local simulator backends...
                    By the way, you can create a IBMQconfig.py like this:
    
                        APItoken = 'PUT_YOUR_API_TOKEN_HERE'
                        config = {'url': 'https://auth.quantum-computing.ibm.com/api}
    
                    if you have created the IBMQconfig.py, please try again. """)
            return

        #load account
        ibmExecute.load_IBMQ_account()

        machine = init_quantum_machine(QMachineType.CPU)
        prog = QProg()
        q = machine.qAlloc_many(4)
        c = machine.cAlloc_many(4)
        cir = QCircuit()
        cir2 = QCircuit()
        cir2.insert(H(q[0])).insert(RX(q[2], math.pi/2))
        cir.insert(H(q[1])).insert(cir2.dagger()).insert(CR(q[1], q[2], math.pi/2))

        prog.insert(H(q[0])).insert(S(q[2]))\
            .insert(cir.dagger())\
            .insert(CNOT(q[0], q[1])).insert(CZ(q[1], q[2])).insert(measure_all(q,c))

        #print(type((IBMQ_BACKENDS)))
        #print(type((IBMQ_BACKENDS.IBMQ_QASM_SIMULATOR)))
        qasmData = to_QASM(prog, IBMQBackends.IBMQ_QASM_SIMULATOR)
        #qasmStr = to_QASM(prog)  .insert(CNOT(q[0], q[1])).insert(CZ(q[1], q[2]))
        qasmStr = qasmData[0]

        # temp test
        testQasm = '''OPENQASM 2.0;
                    include "qelib1.inc";
                    qreg q[4];
                    creg c[4];
                    h q[0];
                    s q[2];
                    measure q[0] -> c[0];
                    measure q[1] -> c[1];
                    measure q[2] -> c[2];
                    measure q[3] -> c[3];'''
        #qasmStr = QuantumCircuit.from_qasm_str(testQasm)
        #qasmStr = testQasm

        print(qasmStr)


        # run on qpanda2
        result = run_with_configuration(prog, cbit_list = c, shots = 1024)
        print(">>>>>>>>>>> On qpanda execute result:", end=" ")
        print(result)
        print("===============================================")
        machine.finalize()

        #run on IBMQ
        print('>>>> run_qprog_on_IBMQ >>>>')
        ibmExecute.executeOntagBackend(qasmStr, qasmData[1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
