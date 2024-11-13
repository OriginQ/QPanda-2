from matplotlib.pyplot import *
from pyqpanda import *
from pyqpanda.utils import *
from pyqpanda.Algorithm.VariationalQuantumEigensolver import *
from pyqpanda.Algorithm.QuantumCircuitLearning.quantum_circuit_learning import (qcl,generate_train_data)

def quantum_circuit_learning_test(function_kind):   
    """
    Executes a quantum circuit learning test with specified parameters and visualizes the results.

        Args:
            function_kind (int): Specifies the target function to learn.
                1: f(x) = x*x
                2: f(x) = exp(x)
                3: f(x) = sin(x)
                4: f(x) = abs(x)
                Any other value: Prints 'undefined'

        The function performs the following steps:
            1. Initializes a quantum circuit with 6 qubits, 2 layers, and a coefficient of 1.
            2. Generates training data for the specified target function.
            3. Prints the target function based on the `function_kind`.
            4. Initializes the quantum virtual machine.
            5. Allocates quantum lists and computes the cost of learning the function.
            6. Prints the cost and the function values at various points.
            7. Visualizes the learned function and the target function.
            8. Finalizes the quantum virtual machine session.

        Returns:
            None

    The function is part of the pyQPanda package, designed for programming quantum computers using quantum circuits and gates.
    """
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
    