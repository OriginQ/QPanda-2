from pyqpanda.Hamiltonian import PauliOperator
from pyqpanda.Hamiltonian.QubitOperator import *
from pyqpanda import *
from pyqpanda.utils import *
#from pyqpanda.Algorithm.hamiltonian_simulation import *
from pyqpanda.Algorithm.hamiltonian_simulation import simulate_pauliZ_hamiltonian,pauliX_model,simulate_one_term

from pyqpanda.Algorithm.VariationalQuantumEigensolver.vqe import flatten,transform_base
from math import pi
import copy



def parity_check(number):
    check=0
    for i in bin(number):
        if i=='1':
            check+=1
    return check%2
def get_one_expectation_component(program,qubit_list):
    '''
    get expectation of operator ZiZj....Zm
    '''
    expectation=0
    result=prob_run(program=program,noise=False,select_max=-1,qubit_list=qubit_list,dataType='list')
    #print(result)
    for i in range(len(result)):
        if parity_check(i):
            expectation-=result[i]
        else:
            expectation+=result[i]
    return expectation   

def get_expectation(qubit_list,program,Hamiltonian):
    '''
    get Hamiltonian's expectation
    '''
    expectation=0
    for op in Hamiltonian.ops:
        if op=='':
            expectation+=Hamiltonian.ops[op]
        else:
            actual_qlist=[]
            tuplelist=PauliOperator.parse_pauli(op)
            
            for i in tuplelist:
                actual_qlist.append(qubit_list[i[1]])
            prog=QProg()
            prog.insert(program).insert(transform_base(actual_qlist,op))
            expectation+=get_one_expectation_component(prog,actual_qlist)*Hamiltonian.ops[op]
    return expectation     


class qaoa:
    def __init__(self,
                 qubitnumber,
                 step,
                 gamma,
                 beta,
                 Hp,
                 Hd
    ):
        self.qnum=qubitnumber
        self.step=step
        self.Hp=Hp
        self.Hd=Hd
        self.gamma=gamma
        self.beta=beta
    
    def prog_generation(self,qubit_list):
        prog=QProg()
        prog.insert(single_gate_apply_to_all(gate=H, qubit_list=qubit_list))
        for i in range(self.step):
            prog.insert(simulate_pauliZ_hamiltonian(qubit_list,self.Hp,2*self.gamma[i]))\
            .insert(pauliX_model(qubit_list,self.beta[i]))
        return prog


    def get_cost_value(self,qubit_list):
        prog=QProg()
        prog.insert(single_gate_apply_to_all(gate=H, qubit_list=qubit_list))
        for i in range(self.step):
            prog.insert(simulate_pauliZ_hamiltonian(qubit_list,self.Hp,2*self.gamma[i]))\
            .insert(pauliX_model(qubit_list,self.beta[i]))
        expect=get_expectation(qubit_list,prog,self.Hp)
        return expect
        
    def get_one_component_of_partial_derivative(self,qubit_list,label):
        '''
        label[0]:position of hamiltonian's component
        label[1]:gamma or beta
        label[2]:position of gamma or beta
        label[3]:Hp's component ,such as 'Z0 Z4'
        <E>=f1(theta)f2(theta)...fn(theta)
        <E>=<E1>+<E2>...+<Em>
        return: d<Ei>/d(theta)中的一项
        get_one_component_of_partial_derivative

        ''' 
        prog0=QProg()
        prog1=QProg()
        prog0.insert(single_gate_apply_to_all(gate=H, qubit_list=qubit_list))
        prog1.insert(single_gate_apply_to_all(gate=H, qubit_list=qubit_list))
        coef=0
        if label[1]==0:
            coef=2*self.Hp.ops[label[0]]
        else:
            coef=2
        for i in range(len(self.gamma)):
            if label[2]!=i:
                prog0.insert(simulate_pauliZ_hamiltonian(qubit_list,self.Hp,2*self.gamma[i]))\
                .insert(pauliX_model(qubit_list,self.beta[i]))
                prog1.insert(simulate_pauliZ_hamiltonian(qubit_list,self.Hp,2*self.gamma[i]))\
                .insert(pauliX_model(qubit_list,self.beta[i]))
            else:
                if label[1]==0:
                    '''
                    Hp:gamma
                    '''
                    for j in self.Hp.ops:
                        if j!=label[0]:
                            prog0.insert(simulate_one_term(qubit_list,j, self.Hp.ops[j],2*self.gamma[i]))
                            prog1.insert(simulate_one_term(qubit_list,j, self.Hp.ops[j],2*self.gamma[i]))
                        else:
                            prog0.insert(simulate_one_term(qubit_list,j, self.Hp.ops[j],2*self.gamma[i]+pi/2/self.Hp.ops[j]))
                            prog1.insert(simulate_one_term(qubit_list,j, self.Hp.ops[j],2*self.gamma[i]-pi/2/self.Hp.ops[j]))
                    prog0.insert(pauliX_model(qubit_list,self.beta[i]))
                    prog1.insert(pauliX_model(qubit_list,self.beta[i]))
                
                elif label[1]==1:
                    '''
                    Hd:beta
                    '''
                    prog0.insert(simulate_pauliZ_hamiltonian(qubit_list,self.Hp,2*self.gamma[i]))
                    prog1.insert(simulate_pauliZ_hamiltonian(qubit_list,self.Hp,2*self.gamma[i]))
                    for j in range(self.qnum):
                        if j!=label[0]:
                            prog0.insert(RX(qubit_list[j],self.beta[i]*2))
                            prog1.insert(RX(qubit_list[j],self.beta[i]*2))
                        else:
                            prog0.insert(RX(qubit_list[j],self.beta[i]*2+pi/2))
                            prog1.insert(RX(qubit_list[j],self.beta[i]*2-pi/2))
                    
        actual_qlist=[]
        tuplelist=PauliOperator.parse_pauli(label[3])
        for i in tuplelist:
            actual_qlist.append(qubit_list[i[1]])
        expectation0=get_one_expectation_component(prog0,actual_qlist)
        expectation1=get_one_expectation_component(prog1,actual_qlist)
        return (expectation0-expectation1)/2*coef

    def get_one_component_of_one_parameter_partial_derivative(self,qubit_list,label):
        '''
        label[0]:gamma or beta
        label[1]:step number
        label[2]:Ei
        return:d<Ei>/d(theta_i)
        
        '''
        partial_derivative=0
        if label[0]==0:
            for op in self.Hp.ops:
                partial_derivative+=self.get_one_component_of_partial_derivative(qubit_list,(op,label[0],label[1],label[2]))
        elif label[0]==1:
            for op in range(self.qnum):
                partial_derivative+=self.get_one_component_of_partial_derivative(qubit_list,(op,label[0],label[1],label[2]))
            
        return partial_derivative
    
    


    def get_one_parameter_partial_derivative(self,qubit_list,label,method=0,delta=1e-6):
        '''
        label[0]:gamma or beta
        label[1]:step number
        return:d<E>/d(theta_i)
        '''
        partial_derivative=0
        if method==0:
            for op in self.Hp.ops:
                partial_derivative+=self.get_one_component_of_one_parameter_partial_derivative(qubit_list,(label[0],label[1],op))*self.Hp.ops[op]
        elif method==1:
            if label[0]==0:
                self.gamma[label[1]]+=delta
                cost1=self.get_cost_value(qubit_list)
                self.gamma[label[1]]-=(2*delta)
                cost2=self.get_cost_value(qubit_list)
                self.gamma[label[1]]+=delta
                partial_derivative=(cost1-cost2)/2/delta
            elif label[0]==1:
                self.beta[label[1]]+=delta
                cost1=self.get_cost_value(qubit_list)
                self.beta[label[1]]-=(2*delta)
                cost2=self.get_cost_value(qubit_list)
                self.beta[label[1]]+=delta
                partial_derivative=(cost1-cost2)/2/delta
        return partial_derivative  

    def get_gradient(self,qubit_list,method=0,delta=1e-6):
        '''
        compute gradient
        '''
        gradient=0
        for i in range(self.step):
            gamma_partial_derivative=self.get_one_parameter_partial_derivative(qubit_list,(0,i),method,delta)
            beta_partial_derivative=self.get_one_parameter_partial_derivative(qubit_list,(1,i),method,delta)
            gradient+=(gamma_partial_derivative*gamma_partial_derivative+beta_partial_derivative*beta_partial_derivative)
        gradient=np.sqrt(gradient)
        return gradient
    
    def get_partial_derivative(self,qubit_list,method=0,delta=1e-6):
        
        partial_derivative_list=np.zeros((self.step,2))

        for i in range(self.step):
            partial_derivative_list[i][0]=self.get_one_parameter_partial_derivative(qubit_list,(0,i),method,delta)
            partial_derivative_list[i][1]=self.get_one_parameter_partial_derivative(qubit_list,(1,i),method,delta)
        return partial_derivative_list
            

    


    def momentum_optimizer(self,qubit_list,times,learning_rate=0.01,momentum=0.001,method=0,delta=1e-6,is_test=False):
        '''
        momentum algorithm
        method=0: pi/2 and -pi/2
        method=1: (f(x+delta)-f(x-delta))/2*delta
        '''
        momentum_list=np.zeros((self.step,2))
        for j in range(times):
            
            partial_derivative_list=self.get_partial_derivative(qubit_list,method,delta)
            for i in range(self.step):

                momentum_list[i][0]=momentum*momentum_list[i][0]-learning_rate*partial_derivative_list[i][0]
                self.gamma[i]+=momentum_list[i][0]
                momentum_list[i][1]=momentum*momentum_list[i][1]-learning_rate*partial_derivative_list[i][1]
                self.beta[i]+=momentum_list[i][1]
            if is_test:
                print("momentum_optimizer:beta  ",self.beta)
                print("momentum_optimizer:gamma  ",self.gamma)
                cost_value=self.get_cost_value(qubit_list)
                print(j,'cost',cost_value)
        cost_value=self.get_cost_value(qubit_list)
        return cost_value

    # def Adam_optimize(self,qubit_list,times,velocity=0.01,decay_rate_1=0.9,decay_rate_1=0.999,method=1,delta=1e-8):
    #     '''
    #     Adam algorithm
        
    #     '''
    #     first_order=np.zeros((self.step,2))
    #     second_order=np.zeros((self.step,2))
    #     for j in range(times):
    #         new_gamma=copy.copy(self.gamma)
    #         new_beta=copy.copy(self.beta)
    #         grad1=0
    #         grad2=0
            
    #         for i in range(self.step):

    #             grad1=self.get_grad(qubit_list,(0,i),method,delta)
    #             first_order[i][0]=decay_rate_1*first_order[i][0]+(1-decay_rate_1)*grad1
    #             second_order[i][0]=decay_rate_2*second_order[i][0]+(1-decay_rate_2)*grad1
    #             first_order[i][0]=first_order[i][0]/(1-decay_rate_1)
    #             second_order[i][0]=second_order[i][0]/(1-decay_rate_2)
    #             delta_theta=-velocity*first_order[i][0]/()


    #             new_gamma[i]+=vv[i][0]
    #             grad2=self.get_grad(qubit_list,(1,i),method,delta)
    #             first_order[i][1]=momentum_coef*vv[i][1]-velocity*grad2
    #             new_beta[i]+=vv[i][1]
    #             print('gamma',i,vv[i][0])
    #             print('beta',i,vv[i][1])
    #         self.gamma=copy.copy(new_gamma)
    #         self.beta=copy.copy(new_beta)
    #         print(self.beta,self.gamma)
    #         if (j+1)%2==0:
    #             cost=self.get_cost_value(qubit_list)
    #             print(j,'cost',cost)
    #     cost=self.get_cost_value(qubit_list)
    #     return cost
        
    def gradient_descent_optimizer(self,qubit_list,times,learning_rate=0.001,method=0,delta=1e-6,is_test=False):
        for i in range(times):
            partial_derivative_list=self.get_partial_derivative(qubit_list,method,delta)
            for j in range(self.step):

                self.gamma-=partial_derivative_list[j][0]*learning_rate
                self.beta-=partial_derivative_list[j][1]*learning_rate
            if is_test:
                print('gamma',self.gamma)
                print('beta',self.beta)
                cost_value=self.get_cost_value(qubit_list)
                print(i,'cost',cost_value)   
        cost_value=self.get_cost_value(qubit_list)    
        return cost_value

