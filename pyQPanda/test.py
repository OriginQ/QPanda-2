import pyqpanda.pyQPanda as pq
import math
import numpy as np

class InitQMachine:
    def __init__(self, quBitCnt, cBitCnt, machineType = pq.QMachineType.CPU):
        self.m_machine = pq.init_quantum_machine(machineType)
        self.m_qlist = self.m_machine.qAlloc_many(quBitCnt)
        self.m_clist = self.m_machine.cAlloc_many(cBitCnt)
        self.m_prog = pq.QProg()

    def __del__(self):
        pq.destroy_quantum_machine(self.m_machine)

#测试接口： 量子态振幅编码
if __name__=="__main__":
    init_machine = InitQMachine(10, 15)
    qlist = init_machine.m_qlist
    machine = init_machine.m_machine
    #这里实现用2个qubit，编码4个double数据[2.2, 1, 4.5, 3.7]
    mat=[0.15311858100051695,0.0961350374871273,0.3859320687001368,0.5634457467385428,0.1474901012487757,0.45185782723129864,0.32284355187278985,0.4132085412578166]
    #data=[["000",0.6793113376921358+0.1376859100584252j],["111",0.720435424880283+0.02348393561289133j]]
    input_vector = 2*np.random.rand(8)-1
    input_vector = input_vector / np.linalg.norm(input_vector)
    cir_encode=pq.Encode()
    data = {"001": 0.6793113376921358+0.1376859100584252j,"111":0.720435424880283+0.02348393561289133j}
    #data = {"000": -0.4012058758884066,"011":0.9121413556170931,"111":0.08385697660676902}
    data1=[np.pi,np.pi,np.pi]
    #cir_encode.ds_quantum_state_preparation(qlist,data)
    # cir_encode.ds_quantum_state_preparation(qlist,data1)
    #cir_encode.angle_encode(qlist,data1)
    # cir_encode.angle_encode(qlist,data1,"x")
    # cir_encode.angle_encode(qlist,data1,"z")
    # cir_encode.angle_encode(qlist,input_vector,"w")
    #cir_encode.dense_angle_encode(qlist,data1)
    # cir_encode.dc_amplitude_encode(qlist,input_vector)
    cir_encode.amplitude_encode(qlist,input_vector)
    # cir_encode.bid_amplitude_encode(qlist,input_vector)
    # cir_encode.bid_amplitude_encode(qlist,input_vector,1)
    # cir_encode.bid_amplitude_encode(qlist,input_vector,2)
    # cir_encode.bid_amplitude_encode(qlist,input_vector,3)
    # cir_encode.bid_amplitude_encode(qlist,input_vector,4)
    # cir_encode.iqp_encode(qlist,input_vector)
    # cir_encode.iqp_encode(qlist,input_vector,repeats=2)
    # cir_encode.iqp_encode(qlist,input_vector,inverse=True)
    #cir_encode.iqp_encode(qlist,input_vector,gate_type="y")
    #输出编码量子线路
    #print(cir_encode.get_circuit())
    prog=pq.QProg()
    prog<<cir_encode.get_circuit()
    print(prog)
    res=pq.prob_run_dict(prog,cir_encode.get_out_qubits(),-1)
    #res=prob_run_list(prog,encode.quantum_data,-1)
    print(res)
    output_vector=[]
    for i in input_vector:
        output_vector.append(i**2)
    print(output_vector)
    print("Test over.")