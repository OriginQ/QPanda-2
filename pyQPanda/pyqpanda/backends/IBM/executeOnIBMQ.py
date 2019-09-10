# Import the Qiskit SDK

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, IBMQ, QiskitError
from qiskit.providers.ibmq import least_busy
from enum import IntEnum

class CExecuteOnIBMQ:
    #if true the program will output test log, or else not.
    outputTestLog = True

    def save_IBMQ_account(self, token, url):
        print('update IBMQ account now.')
        IBMQ.save_account(token, url, overwrite=True)
        IBMQ.load_account()
        IBMQ.enable_account(token, url)
        if self.outputTestLog:
            print('The actived account:')
            print(IBMQ.active_account())

    def load_IBMQ_account(self):
        try:
            IBMQ.load_account()
        except:
            print('Failed to load account! \n\
                Please make sure you have save IBMQ account to local side, \n\
                    if you have not, call save_IBMQ_account to save account to local side.')
           


    def executeOnLeastBusyBackend(self, QASMStr, qubitsCnt, allowSimulatorBackends = True, repeatTimes = 1024, credits = 5):
        #a real device backend
        provider = IBMQ.get_provider()
        ibmq_available_backends = provider.backends(\
            filters = (lambda b: b.configuration().n_qubits >= qubitsCnt and b.configuration().simulator == allowSimulatorBackends))
        least_busy_device = least_busy(ibmq_available_backends)
        #least_busy_device = least_busy(provider.backends(simulator=allowSimulatorBackends))
        print("got a least busy device: ", least_busy_device)

        self.__execute(QASMStr, least_busy_device, repeatTimes, credits)

    def executeOntagBackend(self, QASMStr, backendStr, repeatTimes = 1024, credits = 5):
        provider = IBMQ.get_provider()
        print("To get backend by name: %s" % backendStr)
        ibmq_available_backends = provider.backends(backendStr)
        least_busy_device = least_busy(ibmq_available_backends)
        print("Will be Running on current least busy device: ", least_busy_device)
        self.__execute(QASMStr, least_busy_device, repeatTimes, credits)

    def __execute(self, QASMStr, backend, repeatTimes, credits):
        #QASM str to IBM quantum circuit
        print("str to circuit ...")
        IBMQCirc = QuantumCircuit.from_qasm_str(QASMStr)
        if self.outputTestLog:
            #test output to check circuit
            print("+++++++++++++++++ circuit print test +++++++++++++++++")
            print(IBMQCirc)
        
        job_exp = execute(IBMQCirc, backend, shots = repeatTimes, max_credits = credits)
        result_exp = job_exp.result()
        if self.outputTestLog:
            # Show the results
            print("execution details: ", result_exp)

        print('execution result:')
        print(result_exp.get_counts(IBMQCirc))