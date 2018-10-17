'''
QPanda Utilities\n
Copyright (C) Origin Quantum 2017-2018\n
Licensed Under Apache Licence 2.0
'''
import pyqpanda.pywrapper as pywrap

def qAlloc_many(alloc_num):
    """
    `Extended QPanda API` \n
    Allocate a list of qubit \n
    int -> list<Qubit>  
    """
    qubit_list=list()
    for q in range(0,alloc_num):
        qubit_list.append(pywrap.qAlloc())
    return qubit_list    

def cAlloc_many(alloc_num):
    """
    `Extended QPanda API` \n
    Allocate a list of cbit \n
    int -> list<CBit>  
    """
    cbit_list=list()
    for i in range(0,alloc_num):
        cbit_list.append(pywrap.cAlloc())
    return cbit_list
    
def qFree_all(qubit_list):
    """
    `Extended QPanda API` \n
    Free a list of qubit \n
    list<Qubit> -> None
    """
    for q in qubit_list:
        pywrap.qFree(q)   

def cFree_all(cbit_list):
    """
    `Extended QPanda API` \n
    Free a list of cbit \n
    list<CBit> -> None
    """
    for c in cbit_list:
        pywrap.cFree(c)

def directly_run(QProg=None):
    """
    `Extended QPanda API` \n
    Load, Run and Fetch Result \n
    QProg(optional) -> Dict<string,bool> \n
    Comment: if called by `directly_run()`, it will use the loaded program,
    if called by `directly_run(qprog)`, the input QProg will be loaded first
    """
    if QProg is None:
        pywrap.run()
        return pywrap.getResultMap()
    else:
        pywrap.load(QProg)
        pywrap.run()
        return pywrap.getResultMap()


def result_to_binary_string(cbit_list):
    """
    `Extended QPanda API` \n
    Turn the cbits (after running) into 0/1 string \n
    list<CBit> -> string
    """
    str=''
    for c in cbit_list:
        if (pywrap.getCBitValue(c)):
            str=str+'1'
        else:
            str=str+'0'
    
    return str

def dec2bin(dec_num, max_digit):
    binstr=bin(dec_num)[2:]
    if len(binstr)<max_digit:
        binstr='0'*(max_digit-len(binstr))+binstr
    return binstr

def get_probabilites(qlist, select_max=-1, dataType="dict"):
    '''
    `QPanda Extended API`
    Get the select_max (default -1 for all distribution) largest
    component from the probability distribution with qubits (qlist)

    list<Qubit>, int(optional) ->  dict<string,double> (dataType="dict")
    list<Qubit>, int(optional) ->  list<tuple<int,double> (dataType="tuplelist")
    list<Qubit>, int(optional) ->  list<double> (dataType="list")
    '''
    max_digit=len(qlist)
    if dataType == "dict":        
        prob=pywrap.get_prob(qlist,select_max)
        probabilites=dict()
        for pair in prob:
            probabilites[(dec2bin(pair[0],max_digit))]=pair[1]
        return probabilites
    elif dataType == "tuplelist":        
        prob=pywrap.get_prob(qlist,select_max)
        return prob
    elif dataType == "list":
        prob=pywrap.get_prob(qlist,select_max, True)
        return prob
    else:
        assert False
    

def prob_run(
    program=None,
    noise=False,
    select_max=-1,
    qubit_list=[],
    dataType= 'tuplelist'
):
    '''
    `Extended QPanda API
    '''
    if noise:
        raise("Noisy simulation is not implemented yet")
    
    if program is not None:
        pywrap.load(program)

    pywrap.run()
    return get_probabilites(qlist=qubit_list,select_max=select_max,dataType=dataType)

def run_with_configuration(
    program=None,
    shots=100,
    noise=False,
    cbit_list={}
):
    """
    `Extended QPanda API`
        Run the program with some configuration
        Args:
            prog(optional): assign the program to run
            shots: repeated time of running
            noise:  True=Simulation with noise
                    False=Noise Free Simulation
            cbit_list: the CBit value you care
        Return:
            (shots=0) dict<string,bool>
            (otherwise) dict<string,int>
        Comments:
            1. If shots=0, it is same directly_run
            2. Noisy simulation is not implemented yet
            3. Assigning a prog means the loaded program will be overwritten
    """
    if noise:
        raise("Noisy simulation is not implemented yet")

    if program is not None:
        pywrap.load(program)
    
    if shots==0:
        if cbit_list=={}:
            return directly_run()
        else:
            print("**************************************************************************")
            print("WARNING: In 'run_with_configuration' the param 'cbit_list' will be ignored")
            print("  CAUSE: shots==0")
            print("**************************************************************************")
            return directly_run()
    else:
        result_dict=dict()
        for i in range(0,shots):
            pywrap.run()
            s=result_to_binary_string(cbit_list)
            if not s in result_dict:
                result_dict[s]=1
            else:
                result_dict[s]=result_dict[s]+1
        return result_dict         

def single_gate_apply_to_all(gate,qubit_list):
    '''
    `Extended QPanda API`\n
    Apply single gates to all qubits in qubit_list
    QGate(callback), list<Qubit> -> QCircuit
    '''
    qcirc=pywrap.QCircuit()
    for q in qubit_list:
        qcirc.insert(gate(q))
    return qcirc

def single_gate(gate,qubit,angle=None):
    '''
    `Extended QPanda API`\n
    Apply a single gate to a qubit\n
    Gate(callback), Qubit, angle(optional) -> QGate
    '''
    if angle is None:
        return gate(qubit)
    else:
        return gate(qubit,angle)

def meas_all(qubits, cbits):
    '''
    `Extended QPanda API`\n
    Measure qubits mapping to cbits\n
    list<Qubit>, list<CBit> -> QProg
    '''
    prog=pywrap.QProg()
    for i in range(len(qubits)):
        prog.insert(pywrap.Measure(qubits[i],cbits[i]))

    return prog

def Toffoli(control1,control2,target):
    '''
    `Extended QPanda API`\n
    Create foffoli gate\n
    Qubit(control1), Qubit(control2), Qubit(target) -> QGate
    '''

    return pywrap.X(target).control([control1,control2])

def get_fidelity(result, shots, target_result):
    correct_shots=0
    for term in target_result:
        if term in result:
            correct_shots+=result[term]
    return correct_shots/shots

import time
import random

def add_up_a_dict(dict_, key):
    if key in dict_:
        dict_[key] +=1
    else:
        dict_[key] =1
    

def quick_measure(qlist,shots,seed=None, use_cpp=True):
    '''
    a fast way for sampling.
    after run the program without measurement, call this to fetch
    a simulated result of many shots.

    qlist: qubits to measure
    shots: execution times
    seed(optional) : choose random seed
    use_cpp: use C++ version instead of a python one

    list<Qubit>, int -> dict<string, int>
    '''
    prob_list=pywrap.get_prob(qlist,listonly=True)    
    accumulate_prob=pywrap.accumulate_probabilities(prob_list)
    
    if use_cpp is False:
        meas_result=dict()

        for i in range(shots):    
            rnd=random.random()        
            # direct search
            if rnd<accumulate_prob[0]:            
                add_up_a_dict(meas_result,
                            dec2bin(0,len(qlist)))
                continue

            for i in range(1,len(accumulate_prob)):
                if rnd<accumulate_prob[i] and rnd>=accumulate_prob[i-1]:                
                    add_up_a_dict(meas_result,dec2bin(i,len(qlist)))
                    break  
    else:
        meas_result=pywrap.quick_measure_cpp(qlist, shots, accumulate_prob)

    return meas_result    
