'''
QPanda Python Wrapper\n
Copyright (C) Origin Quantum 2017-2018\n
Licensed Under Apache Licence 2.0
'''

# add the source path into sys.path
# otherwise importing pyQPanda will fail
import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

from pyqpanda import pyQPanda

def init():
    """
    `QPanda Basic API` \n
    Initialize the environment \n

    None -> None    
    """
    pyQPanda.init()
    return

def qAlloc():
    """
    `QPanda Basic API` \n
    Allocate *1* qubit \n

    None -> Qubit    
    """
    return pyQPanda.qAlloc()

def cAlloc():
    """
    `QPanda Basic API` \n
    Allocate *1* cbit \n

    None -> CBit    
    """
    return pyQPanda.cAlloc()

def qFree(qubit):
    """
    `QPanda Basic API` \n
    Free the qubit \n

    Qubit -> None    
    """
    pyQPanda.qFree(qubit)
    return

def cFree(cbit):
    """
    `QPanda Basic API` \n
    Free the cbit \n

    CBit -> None    
    """
    pyQPanda.cFree(cbit)
    return

def run():
    """
    `QPanda Basic API` \n
    Run loaded QProg in the machine \n
    This is often used after loading the program \n

    None -> None    
    """
    pyQPanda.run()
    return

def getResultMap():
    """
    `QPanda Basic API` \n
    Get the result as a dict \n
    This is often used after running the program \n

    None -> Dict<string,bool>    
    """
    return pyQPanda.getResultMap()

def getCBitValue(cbit):
    """
    `QPanda Basic API` \n
    Get the value of a specified CBit \n
    This is often used after running the program \n

    CBit -> bool  
    """
    return pyQPanda.getCBitValue(cbit)

def CreateIfProg(classical_condition, 
                true_branch,
                false_branch=None):
    """
    `QPanda Basic API` \n
    Create a q-if subprogram \n
    (ClassicalCondition,QNode,QNode) -> QProg 
    QNode = QProg | QCircuit | QGate 
    """
    if false_branch is None:
        return pyQPanda.CreateIfProg(classical_condition,
            true_branch)
    else:
        return pyQPanda.CreateIfProg(classical_condition,
            true_branch, false_branch)

def CreateWhileProg(classical_condition,
                    loop_body):
    """
    `QPanda Basic API` \n
    Create a q-while subprogram \n
    (ClassicalCondition,QNode) -> QProg 
    QNode = QProg | QCircuit | QGate  
    """                    
    return CreateWhileProg(classical_condition,loop_body)

def getAllocateQubitNum():
    """
    `QPanda Basic API` \n
    Get number of allocated qubits \n
    None -> int     
    """
    return pyQPanda.getAllocateQubitNum()

def bind_a_cbit(cbit):
    """
    `QPanda Basic API` \n
    Get number of allocated qubits \n
    None -> int     
    """
    return pyQPanda.bind_a_cbit(cbit)

def QProg():
    """
    `QPanda Basic API` \n
    Create an empty QProg object \n
    
    None -> QProg\n

    Comment: A QProg can insert any kind of QNode, such as QProg,
    QCircuit, QGate, QMeasure  QIfNode or QWhileNode. 
    This can be done by: \n
    `some_qprog.insert(another_qnode)`
    
    """
    return pyQPanda.QProg()

def QCircuit():
    """
    `QPanda Basic API`
        Create an empty QCircuit object
    
        None -> QCircuit
        Comment: A QCircuit can insert a QCircuit or QGate. 
        
        This can be done by:
            some_qprog.insert(another_QGateNode)`
    """
    return pyQPanda.QCircuit()

def H(qubit):
    """
    `QPanda Basic API` \n
    Create a Hadamard gate \n
    Qubit -> QGate  
    """
    return pyQPanda.H(qubit)

def X(qubit):
    """
    `QPanda Basic API` \n
    Create an X gate \n
    Qubit -> QGate  
    """
    return pyQPanda.X(qubit)

def NOT(qubit):
    """
    `QPanda Basic API` \n
    Create an X gate \n
    Qubit -> QGate  
    """
    return pyQPanda.X(qubit)

def T(qubit):
    """
    `QPanda Basic API` \n
    Create a T gate \n
    Qubit -> QGate  
    """
    return pyQPanda.T(qubit)

def S(qubit):
    """
    `QPanda Basic API` \n
    Create a S gate \n
    Qubit -> QGate  
    """
    return pyQPanda.S(qubit)

def Y(qubit):
    """
    `QPanda Basic API` \n
    Create a Y gate \n
    Qubit -> QGate  
    """
    return pyQPanda.Y(qubit)

def Z(qubit):
    """
    `QPanda Basic API` \n
    Create a Y gate \n
    Qubit -> QGate  
    """
    return pyQPanda.Z(qubit)

def X1(qubit):
    """
    `QPanda Basic API` \n
    Create an X(pi/2) gate \n
    Qubit -> QGate  
    """
    return pyQPanda.X1(qubit)

def Y1(qubit):
    """
    `QPanda Basic API` \n
    Create a Y(pi/2) gate \n
    Qubit -> QGate  
    """
    return pyQPanda.Y1(qubit)

def Z1(qubit):
    """
    `QPanda Basic API` \n
    Create a Z(pi/2) gate \n
    Qubit -> QGate  
    """
    return pyQPanda.Z1(qubit)

def RX(qubit,angle):
    """
    `QPanda Basic API` \n
    Create a RX gate \n
    Qubit,double -> QGate  
    """
    return pyQPanda.RX(qubit,angle)

def RY(qubit,angle):
    """
    `QPanda Basic API` \n
    Create a RY gate \n
    Qubit,double -> QGate  
    """
    return pyQPanda.RY(qubit,angle)

def RZ(qubit,angle):
    """
    `QPanda Basic API` \n
    Create a RZ gate \n
    Qubit,double -> QGate  
    """
    return pyQPanda.RZ(qubit,angle)

def CNOT(control_qubit,target_qubit):
    """
    `QPanda Basic API` \n
    Create a CNOT gate \n
    Qubit(Controller),Qubit(Target) -> QGate  
    """
    return pyQPanda.CNOT(control_qubit,target_qubit)

def CZ(control_qubit,target_qubit):
    """
    `QPanda Basic API` \n
    Create a CZ gate \n
    Qubit(Controller),Qubit(Target) -> QGate  
    """
    return pyQPanda.CZ(control_qubit,target_qubit)

def U4(qubit,alpha,beta,gamma,delta):
    """
    `QPanda Basic API` \n
    Create a U4(alpha,beta,gamma,delta) gate \n
    Qubit,double,double,double,double -> QGate
    """
    return pyQPanda.U4(alpha,beta,gamma,delta,qubit)

def CU(control_qubit,target_qubit,alpha,beta,gamma,delta):
    """
    `QPanda Basic API` \n
    Create a C-U4(alpha,beta,gamma,delta) gate \n
    Qubit(Controller),Qubit(Target),double,beta,gamma,delta -> QGate  
    """
    return pyQPanda.CU(alpha,beta,gamma,delta,control_qubit,target_qubit)

def iSWAP(qubit1,qubit2,theta=None):
    """
    QPanda Basic API` \n
    Create an iSWAP(theta) gate \n
    theta is optional \n
    Qubit,Qubit,double(Optional) -> QGate
    """
    if theta is None:
        return pyQPanda.iSWAP(qubit1,qubit2)
    else:
        return pyQPanda.iSWAP(qubit1,qubit2,theta)

def Measure(qubit,cbit):
    """
    `QPanda Basic API` \n
    Perform a measurement \n
    Qubit,CBit -> QMeasure  
    """
    return pyQPanda.Measure(qubit,cbit)

def load(QProg):
    """
    `QPanda Basic API` \n
    Load the QProg into the machine \n
    This must be used after calling init()\n
    QProg -> None  
    """
    pyQPanda.load(QProg)
    return

def append(QProg):
    """
    `QPanda Basic API` \n
    Append a QProg into the machine \n
    QProg -> None
    """
    pyQPanda.append(QProg)

def finalize():
    """
    `QPanda Basic API` \n
    Finalize machine. Do this after you finish all your work\n
    
    None -> None  
    """
    pyQPanda.finalize()
    return

def to_qrunes(qprog):
    '''
    `QPanda Basic API`
    convert a quantum program into QRunes

    QProg -> str
    '''
    return pyQPanda.qRunesProg(qprog)

def get_prob(qlist, select_max=-1, listonly=False):
    '''
    `QPanda Basic API`
    get the probabilities distribution
    select_max(optional): select the biggest `select_max` probs.

    list<Qubit>, int -> list<tuple<int,double> (listonly=False)
    list<Qubit>, int -> list<double> (listonly=True)
    '''
    if listonly is False:
        return pyQPanda.PMeasure(qlist,select_max)
    else:
        return pyQPanda.PMeasure_no_index(qlist)

def accumulate_probabilities(problist):
    """
    `QPanda Basic API`
    Transform a list of probabilites into its accumulation.
    Ready for simulating sample from a distribution.
    
    list<double> -> list<double>
    """
    return pyQPanda.accumulateProbability(problist)

def quick_measure_cpp(qlist, shots, accumulate_probabilities):
    """
    `QPanda Basic API`
    C++ version for quick_measure
    
    list<Qubit>, int, list<double> -> dict<string, int>
    """
    return pyQPanda.quick_measure(qlist, shots, accumulate_probabilities)

def get_config_path():
    return pyQPanda.getConfigFilePath()