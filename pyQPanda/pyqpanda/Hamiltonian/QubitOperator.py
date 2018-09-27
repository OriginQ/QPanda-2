'''
Definition of PauliOperator class\n
Copyright (C) Origin Quantum 2017-2018\n
Licensed Under Apache Licence 2.0
'''

import json
from copy import deepcopy

class PauliOperator:
    '''
    `QPanda Hamiltonian API`\n
    Hamiltonian expressed in Pauli Operator\n
    @OPTION:
        op_dict(necessary): `dict`, e.g.: {'X2 Y1':1, 'X1 Z2': -0.5}
        The key of op_dict can be parsed by `PauliOperator.parse_pauli`. Further more please refer to it.
        error_threshold(default=1e-6): `int`, absolute coeffients under the threshold will be eliminated

    @note:
        Because in a dict there cannot have two same keys. In python, only the last one will remain.
        e.g.: {'X1':2, 'X1':3} is equal to {'X1':3}
        However, {'X1 Y2':2, 'Y2 X1':3} can be automatically transformed into {'X1 Y2':5}
        At all the time, each key should be arranged well without manually calling `arrange()`.
    '''

    @staticmethod
    def parse_pauli(input_str):
        '''
        `Extended QPanda API`\n
        Parse the input string to rearrange & parse\n
        e.g.:
            'X5 Y1 Z1' or ''

        Return: tuplelist(the converted) \n
        tuple in tuplelist:
            tuple[0] is its type: 'X'/'Y'/'Z'
            tuple[1] is its qubit index

        tuplelist is None iff input_str=''
        '''
        wordlist=input_str.split() # split by space
        tuplelist=list()           # tuple[0] is its type; tuple[1] is qubit index

        for term in wordlist:
            if term[0] is 'X' or term[0] is 'Y' or term[0] is 'Z':
                tuplelist.append((term[0],int(term[1:])))            
            elif term[0] is 'x' or term[0] is 'y' or term[0] is 'z':
                tuplelist.append((term[0].capitalize(),int(term[1:])))
            elif term[0] is '':
                pass
            else:
                # should not come here
                assert(False)    
        return tuplelist
    
    @staticmethod
    def tuple_list_sort_by_qubit(tuplelist):
        tuplelist.sort(key=lambda tuple:tuple[1])

    @staticmethod
    def tuplelist_to_str(tuplelist):
        retStr=''
        for term in tuplelist:
            retStr=retStr+term[0]+str(term[1])
            if term is not tuplelist[-1]:
                # No ' ' in the last
                retStr=retStr+' '
        return retStr

    @staticmethod
    def tuplelist_merge_single(tuplelist, tuple_right):
        '''
        merge a tuple into a tuplelist.
        '''
        retval=1        # the coefficient
        for i,tuple_left in enumerate(tuplelist):
            if tuple_left[1]<tuple_right[1]:            
                if tuple_left is tuplelist[-1]:       
                    tuplelist.append(tuple_right)
                    retval=1
                    break
                else:                    
                    continue
            elif tuple_left[1]==tuple_right[1]:            
                # decide what [PAULI] should be
                if tuple_left[0] == tuple_right[0]:                    
                    tuplelist.remove(tuple_left)
                    retval=1
                    break
                elif tuple_left[0] == 'X':
                    if tuple_right[0] == 'Y':                        
                        tuplelist[i]='Z',tuple_left[1] # XY=iZ
                        retval=1j                 
                    else:
                        tuplelist[i]='Y',tuple_left[1] # XZ=-iY
                        retval=-1j
                    break
                elif tuple_left[0] == 'Y':
                    if tuple_right[0] == 'X':
                        tuplelist[i]='Z',tuple_left[1] # YX=-iZ
                        retval=-1j
                    else:
                        tuplelist[i]='X',tuple_left[1] # YZ=iX
                        retval=1j
                    break
                elif tuple_left[0] == 'Z':
                    if tuple_right[0] == 'X':
                        tuplelist[i]='Y',tuple_left[1] # ZX=iY
                        retval=1j
                    else:
                        tuplelist[i]='X',tuple_left[1] # ZY=-iX
                        retval=-1j
                    break
                else:
                    assert False
            elif tuple_left[1]>tuple_right[1]:
                tuplelist.insert(tuplelist.index(tuple_left),tuple_right)
                retval=1
                break  

        return retval      

    @staticmethod
    def tuplelist_merge(tuplelist1,tuplelist2):
        retval=1
        for tuple2 in tuplelist2:
            retval*=PauliOperator.tuplelist_merge_single(tuplelist1,tuple2)
        return retval

    def eliminate(self):
        '''
        Remove all terms with value under the threshold.
        '''
        for term in list(self.m_ops):
            if abs(self.m_ops[term]) < self.m_error_threshold:
                self.m_ops.pop(term)  

    def arrange(self):
        '''
        arrange terms in the order of the qubit index and merge all the same term.
        Finally eliminate the 0 value (under the threshold)
        e.g.:
            'X5 Y1' -> 'Y1 X5'
            'X5 Y1':0.5, 'Y1 X5': -0.5 -> None
        '''
        new_ops=dict()

        for term in self.m_ops:
            tuplelist=PauliOperator.parse_pauli(term)
            PauliOperator.tuple_list_sort_by_qubit(tuplelist)
            opstr=PauliOperator.tuplelist_to_str(tuplelist)
            if opstr in new_ops:
                new_ops[opstr]+=self.m_ops[term]
            else:
                new_ops[opstr]=self.m_ops[term]

        self.m_ops=new_ops
        self.eliminate()
        
    def __init__(self, op_dict, error_threshold=1e-6):
        self.m_ops=op_dict
        self.m_error_threshold=error_threshold
        self.arrange()
        
    def __str__(self):
        return str(self.m_ops)    

    @property
    def ops(self):
        return self.m_ops

    @ops.setter
    def ops(self,value):
        self.m_ops=value
    
    def remove_I(self):
        '''
        remove the Identity term and return a new Operator and its value      
        '''        
        ops=deepcopy(self.m_ops)
        if '' in ops:
            value=ops['']
            ops.pop('')
        
        return PauliOperator(ops),

    def get_qubit_count(self):
        '''
        get the max_qubit_index, and then return max_qubit_index+1
        '''
        max_qubit_index=0
        for term in self.m_ops:
            tuplelist=PauliOperator.parse_pauli(term)
            for _tuple_ in tuplelist:
                if _tuple_[1]>max_qubit_index:
                    max_qubit_index=_tuple_[1]
        
        return max_qubit_index+1

    def __add__(self, roprand):    
        '''
        self + roprand
        roprand has two possibilities:
            1. Constant
            2. Another QubitOperator
        ''' 
        if isinstance(roprand, PauliOperator):
            #Case 2                       
            new_ops=deepcopy(self.ops)
            for term in roprand.m_ops:
                if term in new_ops:
                    new_ops[term]+=roprand.m_ops[term]
                else:
                    new_ops[term]=roprand.m_ops[term]
            
            return PauliOperator(new_ops)
        else:
            # is case 1 ?
            try:
                value=float(roprand)
            except ValueError(e):
                raise(e)
            newdict={'':value}
            return self+PauliOperator(newdict)                

    def __neg__(self):    
        '''
        -self
        '''      
        new_ops=deepcopy(self.ops)
        for term in new_ops:
            new_ops[term]=-new_ops[term]
        return PauliOperator(new_ops)

    def __sub__(self, roprand):
        '''
        self-roprand
        '''  
        return self+(-roprand)    
    
    def __rsub__(self, roprand):
        '''
        roprand-self
        '''  
        return -self+roprand   
    
    def __mul__(self, roprand):
        '''
        self * roprand
        roprand has two possibilities:
            1. Constant
            2. Another QubitOperator
        '''     
        if isinstance(roprand,PauliOperator):
            # Case 2
            self.arrange()
            roprand.arrange()
            dict1=self.ops
            dict2=roprand.ops
            new_dict=dict()
            new_paulioperator=PauliOperator(new_dict)

            for term1 in dict1:
                for term2 in dict2:
                    tuplelist1=PauliOperator.parse_pauli(term1)
                    tuplelist2=PauliOperator.parse_pauli(term2)
                    tuplelist1_tmp=deepcopy(tuplelist1)
                    tuplelist2_tmp=deepcopy(tuplelist2)
                    coef=PauliOperator.tuplelist_merge(tuplelist1_tmp,tuplelist2_tmp)
                    coef=dict1[term1]*dict2[term2]*coef
                    one_term=dict({PauliOperator.tuplelist_to_str(tuplelist1_tmp):coef})                
                    new_paulioperator+=PauliOperator(one_term)

            return new_paulioperator
        else:
            # Case 1
            try:
                value=complex(roprand)
            except ValueError(e):
                raise(e)
            newdict={'':value}
            return self*PauliOperator(newdict)   

    def __rmul__(self, loprand): 
        '''
        loprand * self
        '''       
        value=complex(loprand)
        newdict={'':value}
        return self*PauliOperator(newdict)        

def create_x_driver_hamiltonian(qubit_count):
    hamiltonian_dict=dict()
    for i in range(qubit_count):
        hamiltonian_dict["X"+str(i)]=1

    return PauliOperator(hamiltonian_dict)
    
