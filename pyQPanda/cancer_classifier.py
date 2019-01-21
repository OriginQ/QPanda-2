from pyqpanda import *
from scipy.optimize import minimize
from functools import partial
from matplotlib.pyplot import *
import sklearn.datasets as datasets
import numpy as np
import math

def initial_state(qubit_list,x):
    qcir=QCircuit()
    sum=0
    for i in x:
        sum=sum+i
    for i in range(len(x)):
        x[i]=x[i]/sum
    for i in range(len(qubit_list)):
        qcir.insert(RY(qubit_list[i],x[3*i]))\
            .insert(RZ(qubit_list[i],x[3*i+1]))\
            .insert(RY(qubit_list[i],x[3*i+2])) 
    return qcir

def step_1(qubit_list):
    qcir=QCircuit()
    for i in range(len(qubit_list)-1):
        qcir.insert(CZ(qubit_list[i],qubit_list[-1]))
    return qcir

def variational_circuit(qubit_list,theta_y,theta_z):
    vqc=VariationalQuantumCircuit()
    for i in range(len(qubit_list)):
        vqc.insert(VariationalQuantumGate_RY(qubit_list[i],theta_y[i]))\
           .insert(VariationalQuantumGate_RZ(qubit_list[i],theta_z[i]))
    return vqc

def classifier_circuit(qubit_list,x,layer,theta_y,theta_z):
    vqc=VariationalQuantumCircuit()
    vqc.insert(initial_state(qubit_list,x))
    for i in range(layer):
        vqc.insert(step_1(qubit_list)).insert(variational_circuit(qubit_list,theta_y[i],theta_z[i]))
    return vqc

def data_shuffle_and_make_train_test(data, label, train_percentage = 0.8):
    n_data = len(label)
    permutation = np.random.permutation(range(n_data))
    n_train = int(n_data * train_percentage)
    n_test = n_data - n_train   

    train_permutation = permutation[0:n_train]
    test_permutation = permutation[n_train:]

    return data[train_permutation], label[train_permutation], data[test_permutation], label[test_permutation]

def cross_entropy(x,y):
    entropy=[]
    for i in range(len(x)):
        temp=0
        for j in range(len(x[i])):
            temp=temp-y[i][j]*log(x[i][j])
        entropy.append(temp)
    return entropy

def reduce_sum(y):
    sum=0
    for i in y:
        sum=sum+i
    return sum       

def classifier():
    qubit_num=10
    layer=2
    machine=init_quantum_machine(QuantumMachine_type.CPU_SINGLE_THREAD)
    qlist=machine.qAlloc_many(qubit_num)
    H1=PauliOperator({'Z0':1})
    data, label = datasets.load_breast_cancer(return_X_y=True)

    data, label, test_d , test_l = data_shuffle_and_make_train_test(data,label)
    x=data
    y=label
    y_one_hot=[]
    for i in y:
        if i==0:
            y_one_hot.append([1,0])
        elif i==1:
            y_one_hot.append([0,1])

    theta_y=var(np.random.random((layer,qubit_num)))
    theta_z=var(np.random.random((layer,qubit_num)))
    loss=[]

    for data in x:
        vqc=classifier_circuit(qlist,data,layer,theta_y,theta_z)
        loss_temp=qop(vqc,H1,machine,qlist)
        loss.append([(loss_temp+1)/2.0,(1-loss_temp)/2.0])
    entropy=cross_entropy(loss,y_one_hot)
    cost_function=reduce_sum(entropy)
    exp=expression(cost_function)
    leaves=[theta_y,theta_z]
    leaf_set=exp.find_non_consts(leaves)
    grad={theta_y:np.zeros((layer,qubit_num)), theta_z:np.zeros((layer,qubit_num))}
    iterations=10000

    # adam optimizer

    learning_rate=0.001
    beta1=0.9
    beta2=0.999
    epsilon=1e-08

    # first moment vector
    m = {theta_y:np.zeros((layer,qubit_num)), theta_z:np.zeros((layer,qubit_num))}

    # first moment vector estimate    
    m_estimate = {theta_y:np.zeros((layer,qubit_num)), theta_z:np.zeros((layer,qubit_num))}

    # second moment vector
    v = {theta_y:np.zeros((layer,qubit_num)), theta_z:np.zeros((layer,qubit_num))}

    # second moment vector
    v_estimate = {theta_y:np.zeros((layer,qubit_num)), theta_z:np.zeros((layer,qubit_num))}

    open("result.txt",'w+').close()

    for i in range(1, iterations+1):            
        with open("result.txt", 'a+') as fp:
            print("i:",i)
            loss = eval(cost_function,True)
            print(loss)
            back(exp,grad,leaf_set)

            fp.write('i:{}\nloss:{}\n'.format(i, loss))
            for variable in m:
                m[variable] = beta1 * m[variable] + (1-beta1) * grad[variable]
                v[variable] = beta2 * v[variable] + (1-beta2) * (grad[variable] ** 2)
                m_estimate[variable] = m[variable] / (1-beta1 ** i)
                v_estimate[variable] = v[variable] / (1-beta2 ** i)
                raw_value = variable.get_value()
                bias = learning_rate * m_estimate[variable] / (np.sqrt(v_estimate[variable])+ epsilon)
                new_value =  raw_value - bias
                variable.set_value(new_value)  
                fp.write('variable:{}\nm:{}\nv:{}\n'.format(variable.get_value(), m[variable],v[variable]))
            
if __name__=="__main__":
    print("start")
    classifier()
    print("end")



