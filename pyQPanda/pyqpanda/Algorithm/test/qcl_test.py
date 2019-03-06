from matplotlib.pyplot import *
from pyqpanda import *
from pyqpanda.utils import *
from pyqpanda.Algorithm.VariationalQuantumEigensolver import *
from pyqpanda.Algorithm.QuantumCircuitLearning.quantum_circuit_learning import (qcl,generate_train_data)

def quantum_circuit_learning_test(function_kind):   

    test=qcl(qubit_number=6,
            layer=2,
            coef=1,
            t=10)
    print('qubit number is:',test.qnum)
    train_data=generate_train_data(function_kind,np.linspace(-1,1,10))
    if function_kind==1:
        print("target function is: f(x)=x*x")
    elif function_kind==2:
        print("target function is: f(x)=exp(x)")
    elif function_kind==3:
        print("target function is: f(x)=sin(x)")
    elif function_kind==4:
        print("target function is: f(x)=abs(x)")
    else:
        print("undefined")
    x=np.linspace(-1,1,10)
    init(QMachineType.CPU_SINGLE_THREAD)
    qlist=qAlloc_many(test.qnum)
    cost=test.function_learning(qlist,train_data,step=10,velocity=0.1)
    print('cost is:',cost)
    y1=[]
    for i in x:
        y1.append(test.get_function_value(qlist,i))
    print('second')
    print(test.theta3d)
    print(test.coef)    
    plot(x,x*x,'g',x,y1,'b')
    show()
    finalize()
    