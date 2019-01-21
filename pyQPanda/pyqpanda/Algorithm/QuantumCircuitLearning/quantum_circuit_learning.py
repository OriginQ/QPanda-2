from pyqpanda import *
from pyqpanda.utils import *
import copy
import numpy as np
import math
from math import pi
import random
from matplotlib.pyplot import *


def pauliX(qubit_list,coef,t):
    qcir=QCircuit()
    for i in range(len(coef)):
        qcir.insert(RX(qubit_list[i],2*coef[i]*t))
    return qcir
def pauliZjZk(qubit_list,coef,t):
    qcir=QCircuit()
    for i in range(len(coef)):
        for j in range(i):
            qcir.insert(CNOT(qubit_list[i],qubit_list[j])) \
                .insert(RZ(qubit_list[j],coef[i][j]*2*t)) \
                .insert(CNOT(qubit_list[i],qubit_list[j]))
    return qcir

def initial_state(qubit_list,x):
    '''
    input x 
    '''
    qcir=QCircuit()
    for qubit in qubit_list:
        qcir.insert(RY(qubit,np.arcsin(x))).insert(RZ(qubit,np.arccos(x*x)))
    return qcir


def ising_model_simulation(qubit_list,hamiltonian_coef2d,step,t):
    '''
    simulation of Ising model Hamiltonian: H=aiXi+JjkZjZk
    qubit_list: qubit list
    single_coef: coefficients of Xi,ai[i]
    double_coef:coefficients of ZjZk, Jjk[j][k]
    '''
    single_coef=[]
    for i in range(len(qubit_list)):
        single_coef.append(hamiltonian_coef2d[i][i])
    qcir=QCircuit()
    for i in range(step):
        qcir.insert(pauliX(qubit_list,single_coef,t/step)) \
            .insert(pauliZjZk(qubit_list,hamiltonian_coef2d,t/step))
    return qcir
def unitary(qubit,theta):
    qcir=QCircuit()
    qcir.insert(RX(qubit,theta[2])).insert(RZ(qubit,theta[1])).insert(RX(qubit,theta[0]))
    return qcir
def one_layer(qubit_list,theta2d,hamiltonian_coef2d,step=100,t=10):
    '''
    theta2d:[n][3],n is qubit number
    hamiltonian_coef2d:[n,n],n is qubit number
    
    '''
    qcir=QCircuit()
    qcir.insert(ising_model_simulation(qubit_list,hamiltonian_coef2d,step,t))
    for i in range(len(qubit_list)):
        qcir.insert(unitary(qubit_list[i],theta2d[i]))
    return qcir



def learning_circuit(qubit_list,layer,theta3d,hamiltonian_coef3d,x,t=10):
    
    '''
    qubit_list: qubit list
    theta: parameters to be optimized, [layer,qubit_number,3]
    hamiltonian_coef:coefficients of fully connected transverse Ising model hamiltonian,[layer,qubit_num,qubit_num],
    C[i,i]is coefficients of pauli Xi;
    C[i,j]is coefficients of ZiZj when i>j,C[i,j]=0 when i<j
    
    '''
    qcir=QCircuit()
    qcir.insert(initial_state(qubit_list,x))
    for i in range(layer):
        qcir.insert(one_layer(qubit_list,theta3d[i],hamiltonian_coef3d[i],t))
    return qcir
    
def get_expectation(program,qubit_list):
    expect=0
    result=prob_run(program=program,noise=False,select_max=-1,qubit_list=qubit_list,dataType='list')
    for i in range(len(result)):
        if parity(i):
            expect-=result[i]
        else:
            expect+=result[i]
    return expect   
def parity(number):
    check=0
    for i in bin(number):
        if i=='1':
            check+=1
    return check%2
class qcl:
    def __init__(self,
                 qubit_number,
                 layer,
                 coef=1,
                 t=10
    ):
        self.qnum=qubit_number
        self.layer=layer
        self.theta3d=np.random.random_sample((layer,qubit_number,3))*2*pi
        self.hamiltonian_coef3d=1-2*np.random.random_sample((layer,qubit_number,qubit_number))
        self.coef=1
        self.t=10
    def learning_circuit(self,qubit_list,x):
        qcir=QCircuit()
        qcir.insert(initial_state(qubit_list,x))
        for i in range(self.layer):
            qcir.insert(one_layer(qubit_list,self.theta3d[i],self.hamiltonian_coef3d[i],self.t))
        return qcir
   
    def get_function_value(self,qubit_list,x):
        
        prog=QProg()
        prog.insert(learning_circuit(qubit_list,self.layer,self.theta3d,self.hamiltonian_coef3d,x,self.t))
        return get_expectation(prog,[qubit_list[0]])*self.coef
        
    def get_expectation(self,qubit_list,x):
        prog=QProg()
        prog.insert(learning_circuit(qubit_list,self.layer,self.theta3d,self.hamiltonian_coef3d,x,self.t))
        return get_expectation(prog,[qubit_list[0]])
    
    def cost_funciton(self,qubit_list,train_data):
        cost=0
        for data in train_data:
            cost+=(data[1]-self.get_function_value(qubit_list,data[0]))*(data[1]-self.get_function_value(qubit_list,data[0]))
        return cost
    
    def optimize(self,qubit_list,m_qulist,train_data,velocity=0.01):
        '''
        parameter optimization:optimize theta3d and coef
    
        '''
    
        
        new_theta3d=copy.copy(self.theta3d)
        grad1=0
        for data in train_data:
            grad1+=(self.get_function_value(qubit_list,data[0])-data[1])*self.get_expectation(qubit_list,data[0])*2
        
        for i in range(self.layer):
            for j in range(self.qnum):
                for k in range(3):
                    grad=0
                    for data in train_data:
                        
                        self.theta3d[i][j][k]+=pi/2
                        qprog2=QProg()
                        qprog2.insert(learning_circuit(qubit_list,self.layer,self.theta3d,self.hamiltonian_coef3d,data[0],10))
                        
                        qprog3=QProg()
                        self.theta3d[i][j][k]-=pi
                        qprog3.insert(learning_circuit(qubit_list,self.layer,self.theta3d,self.hamiltonian_coef3d,data[0],10))
                        
                        self.theta3d[i][j][k]+=pi/2
                        
                        result3=get_expectation(qprog2,m_qulist)
                        result4=get_expectation(qprog3,m_qulist)
                        grad+=self.coef*(self.get_function_value(qubit_list,data[0])-data[1])*(result3-result4)
                    new_theta3d[i][j][k]-=grad*velocity
        #print(new_theta3d)
        self.theta3d=copy.copy(new_theta3d)
        self.coef-=grad1*velocity*0.05    
        
    def function_learning(self,qubit_list,train_data,step=1000,velocity=0.01):
        '''
        train quantum circuit
        
        '''
        for i in range(step):
            self.optimize(qubit_list,[qubit_list[0]],train_data,velocity)
        cost=self.cost_funciton(qubit_list,train_data)
        return cost

        
def generate_train_data(type,range):
    '''
    generate train data
    type:function type,include 4 types,1 means x^2,2 means exp(x),3 means sin(x),4 means |x|
    range:range of variety x
    
    '''
    train_data=[]
    for i in range:
        if type==1:
            train_data.append((i,i*i))
        elif type==2:
            train_data.append((i,np.exp(i)))
        elif type==3:
            train_data.append((i,np.sin(i)))
        elif type==4:
            train_data.append((i,np.abs(i)))
        else:
            print('undefined')
    return train_data
        
    