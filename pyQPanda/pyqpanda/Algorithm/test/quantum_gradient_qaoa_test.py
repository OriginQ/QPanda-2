from pyqpanda.Algorithm.QuantumGradient.quantum_gradient import *
from pyqpanda.Algorithm.VariationalQuantumEigensolver.vqe import flatten
from pyqpanda.Hamiltonian.QubitOperator import PauliOperator
from pyqpanda.pywrapper import *
from pyqpanda.utils import *
import numpy as np
from math import pi

Hp_19=PauliOperator({'Z0 Z5':0.18,'Z0 Z6':0.49,'Z1 Z6':0.59,'Z1 Z7':0.44,'Z2 Z7':0.56,'Z2 Z8':0.63,'Z5 Z10':0.23,
                 'Z6 Z11':0.64,'Z7 Z12':0.60,'Z8 Z13':0.36,'Z9 Z14':0.52,'Z10 Z15':0.40,'Z10 Z16':0.41,'Z11 Z16':0.57,
                 'Z11 Z17':0.50,'Z12 Z17':0.71,'Z12 Z18':0.40,'Z13 Z18':0.72,'Z13 Z3':0.81,'Z14 Z3':0.29})
Hd_19=PauliOperator({'X0':1,'X1':1,'X2':1,'X3':1,'X4':1,'X5':1,'X6':1,
                    'X7':1,'X8':1,'X9':1,'X10':1,'X11':1,'X12':1,'X13':1,
                    'X14':1,'X15':1,'X16':1,'X17':1,'X18':1})
Hp_7=PauliOperator({'Z0 Z4':0.73,'Z0 Z5':0.33,'Z0 Z6':0.5,'Z1 Z4':0.69,'Z1 Z5':0.36,'Z2 Z5':0.88,'Z2 Z6':0.58,
                 'Z3 Z5':0.67,'Z3 Z6':0.43})
Hd_7=PauliOperator({'X0':1,'X1':1,'X2':1,'X3':1,'X4':1,'X5':1,'X6':1})

Hp_13=PauliOperator({'Z0 Z7':0.33,'Z0 Z8':0.61,'Z0 Z9':0.55,'Z1 Z7':0.46,'Z1 Z8':0.40,'Z1 Z11':0.94,
                 'Z2 Z8':0.42,'Z2 Z9':0.43,'Z3 Z9':0.81,'Z3 Z10':0.45,'Z3 Z12':0.90,'Z4 Z10':0.67,
                 'Z4 Z11':0.77,'Z5 Z10':0.84,'Z5 Z11':0.76,'Z5 Z12':0.83,'Z6 Z7':0.50,'Z6 Z11':0.49,'Z6 Z12':0.69})
Hd_13=PauliOperator({'X0':1,'X1':1,'X2':1,'X3':1,'X4':1,'X5':1,'X6':1,
                    'X7':1,'X8':1,'X9':1,'X10':1,'X11':1,'X12':1})



def quantum_gradient_qaoa_test(Hp,Hd,target_value,target_str_list,step,gamma,beta,times_=100,optimizer=('Momentum',0.02,0.9),method=1,delta=1e-7,is_test=True):

    
    qubit_number=Hp.get_qubit_count()
    output={}
    init()
    qubit_list_=qAlloc_many(qubit_number)
    qaoa_obj=qaoa(qubit_number,step,gamma,beta,Hp,Hd,target_value=target_value,target_str_list=target_str_list)
    if optimizer[0]=='Momentum':
        output=qaoa_obj.momentum_optimizer(qubit_list=qubit_list_,
            max_times=times_,
            threshold_value=0.01,
            learning_rate=optimizer[1],
            momentum=optimizer[2],
            method=method,
            delta=delta,
            is_test=is_test)
        # final_cost_value=qaoa_obj.get_cost_value(qubit_list_)
        # output={'cost value':final_cost_value,'gamma':qaoa_obj.gamma,'beta':qaoa_obj.beta}
    elif optimizer[0]=='GradientDescent':
        output=qaoa_obj.momentum_optimizer(qubit_list=qubit_list_,
            max_times=times_,
            threshold_value=0.1,
            learning_rate=optimizer[1],
            method=method,
            delta=delta,
            is_test=is_test)
        # final_cost_value=qaoa_obj.get_cost_value(qubit_list_)
        # output={'cost value':final_cost_value,'gamma':qaoa_obj.gamma,'beta':qaoa_obj.beta}
    elif optimizer[0]=='Adam':
        output=qaoa_obj.Adam_optimizer(qubit_list=qubit_list_,
            max_times=times_,
            threshold_value=0.01,
            learning_rate=optimizer[1],
            decay_rate_1=optimizer[2],
            decay_rate_2=optimizer[3],
            method=1,
            delta=1e-8,
            is_test=is_test)
    elif optimizer[0]=='Powell':
        output=qaoa_obj.bulit_in_optimizer(qubit_list=qubit_list_,method='Powell')
    else:
        assert(False)
        raise Exception("undefined")
    return output

def generate_factor_hamiltonian(n_bit,offset=0):
    str_dict={}
    for i in range(n_bit):
        key='Z%d'%(i+offset)
        str_dict[key]=-2**(i-1)
    str_dict[' ']=(1<<(n_bit-1))-0.5
    hamiltonian=PauliOperator(str_dict)
    return hamiltonian

def generate_drive_hamiltonian(qubit_number):
    str_dict={}
    for i in range(qubit_number):
        key='X%d'%i
        str_dict[key]=1
    drive_hamiltonian=PauliOperator(str_dict)
    return drive_hamiltonian

def quantum_gradient_qaoa_test_factorize(number,factor1,factor2,step,gamma,beta,times_=100,optimizer=('Momentum',0.02,0.9),method=1,delta=1e-7,is_test=True):
    '''
    number:target number,assume it is a pseudoprime,such as 35,77
    suppose:M=PQ
    M:m bits, M: m-1 bits, N:[m/2]
    '''
    target_bin=bin(number)[2:]
    factor1_bin=bin(factor1)[2:]
    factor2_bin=bin(factor2)[2:]
    n_bit_factor1=len(target_bin)-1
    n_bit_factor2=int(len(target_bin)/2)
    while len(factor1_bin)!=n_bit_factor1:
        factor1_bin='0'+factor1_bin
    while len(factor2_bin)!=n_bit_factor2:
        factor2_bin='0'+factor2_bin
    target_str=factor2_bin+factor1_bin
    print('target_str',target_str)
    print('n_bit_factor1',n_bit_factor1)
    print('n_bit_factor2',n_bit_factor2)
    qubit_number=n_bit_factor1+n_bit_factor2
    h_factor1=generate_factor_hamiltonian(n_bit=n_bit_factor1,offset=0)
    print('h_factor1',h_factor1.ops)
    h_factor2=generate_factor_hamiltonian(n_bit=n_bit_factor2,offset=n_bit_factor1)
    print('h_factor2',h_factor2.ops)
    h_target=PauliOperator({' ':number})
    hp=h_factor1*h_factor2-h_target
    hp=hp*hp*(1/number/number)
    hp=flatten(hp)
    hd=generate_drive_hamiltonian(qubit_number)
    #optimize parameter
    init()
    qubit_list_=qAlloc_many(qubit_number)
    qaoa_obj=qaoa(qubit_number,step,gamma,beta,hp,hd)
    if optimizer[0]=='Momentum':
        qaoa_obj.momentum_optimizer(qubit_list=qubit_list_,
        times=times_,
        learning_rate=optimizer[1],
        momentum=optimizer[2],
        method=method,
        delta=delta,
        is_test=is_test)
    elif optimizer[0]=='GradientDescent':
        qaoa_obj.momentum_optimizer(qubit_list=qubit_list_,
        times=times_,
        learning_rate=optimizer[1],
        method=method,
        delta=delta,
        is_test=is_test)
    else:
        print("undefined")
    #output outcome
    prog=QProg()
    prog.insert(qaoa_obj.prog_generation(qubit_list_))
    result=prob_run(program=prog,noise=False,select_max=100,qubit_list=qubit_list_,dataType='dict')
    final_cost_value=qaoa_obj.get_cost_value(qubit_list_)
    finalize()
    qaoa_obj.gamma
    #output=(final_cost_value,result[target_str],qaoa_obj.gamma,qaoa_obj.beta)
    output={"cost value":final_cost_value,"target probability":result[target_str],
    "gamma":qaoa_obj.gamma,"beta":qaoa_obj.beta}
    return output
def quantum_gradient_qaoa_test_factorize1(number=77,step=5):
    '''
    number:target number,assume it is a pseudoprime,such as 35,77
    suppose:M=PQ
    M:m bits, M: m-1 bits, N:[m/2]
    '''
    h1=PauliOperator({'Z2':-0.2,"Z1":-1,'Z0':-0.5,'':3.5})
    h2=PauliOperator({'Z6':-4,'Z5':-2,"Z4":-1,"Z3":-0.5,'':7.5})
    h3=PauliOperator({'':77})
    hp=h1*h2-h3
    hp=hp*hp*(1/77/77)
    hx=PauliOperator({'X0':1,"X1":1,'X2':1,"X3":1,'X4':1,'X5':1,"X6":1})
    hp=flatten(hp)
    gamma=(1-2*np.random.random_sample(step))*2
    beta=(1-2*np.random.random_sample(step))*pi/4
    cost_value=quantum_gradient_qaoa_test(Hp=hp,Hd=hx,step=step,gamma=gamma,beta=beta,times_=100,optimizer=('Momentum',0.02,0.9),method=1,delta=1e-7,is_test=True)
    return cost_value
