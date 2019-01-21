from pyqpanda import *
from pyqpanda.Algorithm.hamiltonian_simulation import simulate_pauliZ_hamiltonian,pauliX_model,simulate_one_term
from math import pi
from scipy.optimize import minimize
import copy
import numpy as np
from pyqpanda.pyQPanda import *

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
    result=prob_run_list(program=prog,qubit_list=qubit_list,select_max=-1) 
    #print(result)
    for i in range(len(result)):
        if parity_check(i):
            expectation-=result[i]
        else:
            expectation+=result[i]
    return expectation   

def get_expectation(qubit_list,program,PauliOperator):
    '''
    get Hamiltonian's expectation
    '''
    expectation=0
    Hamiltonian=PauliOperator.toHamiltonian(0)
    for op in Hamiltonian:
        if op[0]=={}:
            expectation+=op[1]
        else:

            prog=QProg()
            prog.insert(program)
            actual_qlist=[]
            for i in op[0]:
                if i=='X':
                    prog.insert(H(qubit_list[op[0][i]]))
                    actual_qlist.append(qubit_list[op[0][i]])
                elif i=='Y':
                    prog.insert(RX(qubit_list[op[0][i]],pi/2))
                    actual_qlist.append(qubit_list[op[0][i]])
                elif i=="Z":
                    actual_qlist.append(qubit_list[op[0][i]])
            expectation+=get_one_expectation_component(prog,actual_qlist)*op[1]
    return expectation     

class qaoa:
    def __init__(self,
                 qubitnumber,
                 step,
                 gamma,
                 beta,
                 Hp,
                 Hd,
                 all_cut_value_list=[],
                 target_value=0,
                 target_str_list=[]
    ):
        self.qnum=qubitnumber
        self.step=step
        self.Hp=Hp
        self.Hd=Hd
        self.gamma=gamma
        self.beta=beta
        self.all_cut_value_list=all_cut_value_list
        self.target_value=target_value
        self.target_str_list=target_str_list
    
    def prog_generation(self,qubit_list):
        prog=QProg()
        prog.insert(single_gate_apply_to_all(gate=H, qubit_list=qubit_list))
        for i in range(self.step):
            prog.insert(simulate_pauliZ_hamiltonian(qubit_list,self.Hp,2*self.gamma[i]))\
            .insert(pauliX_model(qubit_list,self.beta[i]))
        return prog

    def get_cost_value(self,qubit_list):
        '''
        compute Hp's expectation,
        all_cut_value_list:eigenvalues of Hp
        '''
        prog=QProg()
        prog.insert(single_gate_apply_to_all(gate=H, qubit_list=qubit_list))
        for i in range(self.step):
            prog.insert(simulate_pauliZ_hamiltonian(qubit_list,self.Hp,2*self.gamma[i]))\
            .insert(pauliX_model(qubit_list,self.beta[i]))
        cost_value=0

        if len(self.all_cut_value_list):
            result=prob_run_list(program=prog,qubit_list=qubit_list,select_max=-1) 
            cost_value=vector_dot(result,self.all_cut_value_list)
            
    
            # cost_value=np.sum(np.array(result)*np.array(self.all_cut_value_list))

        else:
            cost_value=get_expectation(qubit_list,prog,self.Hp)

        return cost_value





    def get_one_component_of_partial_derivative(self,qubit_list,label):
        '''
        label[0]:({"Z":0,"Z":1},1.2)
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


        for i in label[3]:
            actual_qlist.append(qubit_list[i[1]])
        expectation0=get_one_expectation_component(prog0,actual_qlist)
        expectation1=get_one_expectation_component(prog1,actual_qlist)
        return (expectation0-expectation1)/2*coef

    def get_one_component_of_one_parameter_partial_derivative(self,qubit_list,label):
        '''
        label[0]:gamma or beta
        label[1]:step number
        label[2]:Ei={"Z":0,"Z":1}
        return:d<Ei>/d(theta_i)
        
        '''
        # partial_derivative=0
        # #gamma
        # if label[0]==0:


        #     for op in self.Hp.toHamiltonian(0):
        #         partial_derivative+=self.get_one_component_of_partial_derivative(qubit_list,(op,label[0],label[1],label[2]))
        # #beta
        # elif label[0]==1:
        #     for op in self.Hd.toHamiltonian(0):
        #         partial_derivative+=self.get_one_component_of_partial_derivative(qubit_list,(op,label[0],label[1],label[2]))
            
        # return partial_derivative
        return 0
    
    


    def get_one_parameter_partial_derivative(self,qubit_list,label,method=1,delta=1e-6):
        '''
        label[0]:gamma or beta
        label[1]:step number
        return:d<E>/d(theta_i)
        '''
        partial_derivative=0
        if method==0:
            pass
            #d<E>/dtheta=d<E0>/dtheta+d<E1>/dtheta+...+d<En>/dtheta
            # for op in self.Hp.toHamiltonian(0):
            #     partial_derivative+=self.get_one_component_of_one_parameter_partial_derivative(qubit_list,(label[0],label[1],op[0]))*op[1]
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
    
    def target_state_proability(self,qubit_list):

        
        sum=0
        qprog=QProg()
        qprog.insert(self.prog_generation(qubit_list))
        result=prob_run_list(program=qprog,qubit_list=qubit_list,select_max=-1) 
        for i in self.target_str_list:
            sum+=result[int(i,2)]
        return sum




    def bulit_in_optimizer(self,qubit_list,method='Powell'):
        def cost_function(parameter):
            self.gamma=parameter[0:self.step]
            self.beta=parameter[self.step:]
            cost_value=self.get_cost_value(qubit_list)
            return cost_value
        output={}
        initial_guess=np.zeros(2*self.step)
        initial_guess[0:self.step]=self.gamma
        initial_guess[self.step:]=self.beta
        result=minimize(cost_function,initial_guess,method=method)
        output['optimizer']={'opt':method}
        output['target probability']=self.target_state_proability(qubit_list)
        output['times']=result['nfev']
        output['target cost value']=self.target_value
        output['cost value']=result['fun']
        output['qaoa step']=self.step
        output['gamma']=self.gamma
        output['beta']=self.beta
        return output


       


    def momentum_optimizer(self,qubit_list,max_times=200, threshold_value=0.01,learning_rate=0.01,momentum=0.9,method=1,delta=1e-6,is_test=False):
        '''
        momentum algorithm
        method=0: pi/2 and -pi/2
        method=1: (f(x+delta)-f(x-delta))/2*delta
        '''
        momentum_list=np.zeros((self.step,2))
        optimize_times=0
        cost_value=0
        #target_probability=0
        output={}
        gradient=1
        while (gradient > threshold_value) and (optimize_times < max_times):

            gradient=0
            partial_derivative_list=self.get_partial_derivative(qubit_list,method,delta)
            for i in partial_derivative_list:
                for j in i:
                    gradient+=j*j
            gradient=np.sqrt(gradient)
            print('optimize_times:',optimize_times)
            print('gradient:',gradient)
            for i in range(self.step):

                momentum_list[i][0]=momentum*momentum_list[i][0]-learning_rate*partial_derivative_list[i][0]
                #one step can not bigger than 0.01
                if momentum_list[i][0]>0.01:
                    self.gamma[i]+=0.01
                elif momentum_list[i][0]<-0.01:
                    self.gamma[i]-=0.01
                else:
                    self.gamma[i]+=momentum_list[i][0]
                momentum_list[i][1]=momentum*momentum_list[i][1]-learning_rate*partial_derivative_list[i][1]
                if momentum_list[i][1]>0.01:
                    self.beta[i]+=0.01
                elif momentum_list[i][1]<-0.01:
                    self.beta[i]-=0.01
                else:
                    self.beta[i]+=momentum_list[i][1]
            if is_test:
                print("momentum_optimizer:beta  ",self.beta)
                print("momentum_optimizer:gamma  ",self.gamma)
                cost_value=self.get_cost_value(qubit_list)
                print(optimize_times,'cost',cost_value)
            optimize_times+=1
            #target_probability=self.target_state_proability(qubit_list)['sum']
        output['optimizer']={'opt':'momentum','learning_rate':learning_rate,'momentum':momentum}
        output['target probability']=self.target_state_proability(qubit_list)
        #parameter optimization once needs run quantum program 4*step times
        output['times']=optimize_times*4*self.step
        output['target cost value']=self.target_value
        output['cost value']=cost_value
        output['qaoa step']=self.step
        output['gamma']=self.gamma
        output['beta']=self.beta
        return output

       
    def Adam_optimizer(self,qubit_list,max_times=200, threshold_value=0.01,learning_rate=0.01,decay_rate_1=0.9,decay_rate_2=0.999,method=1,delta=1e-8,is_test=False):
        '''
        Adam algorithm
        
        '''
        first_order=np.zeros((self.step,2))
        second_order=np.zeros((self.step,2))
        optimize_times=0
        target_probability=0
        output={}
        while (target_probability < threshold_value) and (optimize_times < max_times):
            
            partial_derivative_list=self.get_partial_derivative(qubit_list,method,delta)
            
            for i in range(self.step):

                #grad1=self.get_grad(qubit_list,(0,i),method,delta)
                if optimize_times==0:
                    first_order[i][0]=partial_derivative_list[i][0]
                    second_order[i][0]=partial_derivative_list[i][0]*partial_derivative_list[i][0]
                else:
                    first_order[i][0]=decay_rate_1*first_order[i][0]+(1-decay_rate_1)*partial_derivative_list[i][0]
                    second_order[i][0]=decay_rate_2*second_order[i][0]+(1-decay_rate_2)*partial_derivative_list[i][0]*partial_derivative_list[i][0]
                first_order_gamma=first_order[i][0]/(1-decay_rate_1**(optimize_times+1))
                second_order_gamma=second_order[i][0]/(1-decay_rate_2**(optimize_times+1))
                delta_gamma=-learning_rate*first_order_gamma/(np.sqrt(second_order_gamma)+delta)
                self.gamma[i]+=delta_gamma

                if optimize_times==0:
                    first_order[i][1]=partial_derivative_list[i][1]
                    second_order[i][1]=partial_derivative_list[i][1]*partial_derivative_list[i][1]
                else:
                    first_order[i][1]=decay_rate_1*first_order[i][1]+(1-decay_rate_1)*partial_derivative_list[i][1]
                    second_order[i][1]=decay_rate_2*second_order[i][1]+(1-decay_rate_2)*partial_derivative_list[i][1]*partial_derivative_list[i][1]
                first_order_beta=first_order[i][1]/(1-decay_rate_1**(optimize_times+1))
                second_order_beta=second_order[i][1]/(1-decay_rate_2**(optimize_times+1))
                delta_beta=-learning_rate*first_order_beta/(np.sqrt(second_order_beta)+delta)
                self.beta[i]+=delta_beta
                optimize_times+=1
                # print('f',j,i,first_order[i][0],first_order[i][1])
                # print('s',j,i,second_order[i][0],second_order[i][1])
                if is_test:


                    print('Adam_optimize  gamma:',i,self.beta)
                    print('Adam_optimize  beta:',i,self.gamma)
            if is_test:
                cost_value=self.get_cost_value(qubit_list)
                print(optimize_times,'cost',cost_value)
        output['optimizer']={'opt':'Adam','learning_rate':learning_rate,'decay_rate_1':decay_rate_1,'decay_rate_2':decay_rate_2}
        output['target probability']=self.target_state_proability(qubit_list)
        output['times']=optimize_times*4*self.step
        output['target cost value']=self.target_value
        output['cost value']=cost_value
        output['qaoa step']=self.step
        output['gamma']=self.gamma
        output['beta']=self.beta
        return output


        
    def gradient_descent_optimizer(self,qubit_list,max_times=200, threshold_value=0.01,learning_rate=0.001,method=0,delta=1e-6,is_test=False):




        optimize_times=0
        target_probability=0
        output={}
        while (target_probability < threshold_value) and (optimize_times < max_times):

            partial_derivative_list=self.get_partial_derivative(qubit_list,method,delta)
            for i in range(self.step):

                self.gamma-=partial_derivative_list[i][0]*learning_rate
                self.beta-=partial_derivative_list[i][1]*learning_rate
            if is_test:
                print("gradient_descent_optimizer:beta  ",self.beta)
                print("gradient_descent_optimizer:gamma  ",self.gamma)
                cost_value=self.get_cost_value(qubit_list)
                print(optimize_times,'cost',cost_value)
            optimize_times+=1
            target_probability=self.target_state_proability(qubit_list)
        output['optimizer']={'opt':'Gradient descent','learning_rate':learning_rate}
        output['target probability']=self.target_state_proability(qubit_list)
        output['times']=optimize_times*4*self.step
        output['target cost value']=self.target_value
        output['cost value']=cost_value
        output['qaoa step']=self.step
        output['gamma']=self.gamma
        output['beta']=self.beta
        return output


