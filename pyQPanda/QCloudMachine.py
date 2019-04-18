from pyqpanda import *
import QCloudJsonConfig
PI = 3.1415926535898


if __name__ == "__main__":

    QCM = QCloud()

    QCM.initQVM()

    qlist = QCM.qAlloc_many(10)
    clist = QCM.qAlloc_many(10)
    prog = QProg();
    for i in qlist:
        prog.insert(H(i))
    
    prog.insert(CZ(qlist[1], qlist[5]))\
        .insert(CZ(qlist[3], qlist[7]))\
        .insert(CZ(qlist[0], qlist[4]))\
        .insert(RZ(qlist[7], PI / 4))\
        .insert(RX(qlist[5], PI / 4))\
        .insert(RX(qlist[4], PI / 4))\
        .insert(RY(qlist[3], PI / 4))\
        .insert(CZ(qlist[2], qlist[6]))\
        .insert(RZ(qlist[3], PI / 4))\
        .insert(RZ(qlist[8], PI / 4))\
        .insert(CZ(qlist[9], qlist[5]))\
        .insert(RY(qlist[2], PI / 4))\
        .insert(RZ(qlist[9], PI / 4))\
        .insert(CZ(qlist[2], qlist[3]));

    param1 = {"RepeatNum": 1000,"token":"E5CD3EA3CB534A5A9DA60280A52614E1", "BackendType": QMachineType.NOISE}
    param2 = {"token":"E5CD3EA3CB534A5A9DA60280A52614E1", "BackendType": QMachineType.CPU}
    print(QCM.run_with_configuration(prog,param1))
    print(QCM.prob_run_dict(prog,qlist,param2))

    QCM.finalize()
    




