'''
State Preparation Test\n
Copyright (C) Origin Quantum 2017-2018\n
Licensed Under Apache Licence 2.0
'''

from pyqpanda import *
from pyqpanda.utils import *
from .test_utils import test_begin_str, test_end_str

def entanglement_test():
    """
    Test a two qubit entanglement state
    """
    print(test_begin_str('Entanglement Test'))
    # init the environment
    init()

    # allocate qubit/cbit
    q_list=qAlloc_many(20)
    c_list=cAlloc_many(20)

    # Create empty program
    qprog=QProg()

    # insert gates
    qprog.insert(single_gate_apply_to_all(H,q_list)) \
        .insert(CZ(q_list[1],q_list[0])) \
        .insert(H(q_list[1])) \
        .insert(meas_all(q_list[0:2],c_list[0:2]))
    
    shots_num=1000
    result=run_with_configuration(program=qprog,shots=shots_num,cbit_list=c_list[0:2])
  
    print("Shots:",shots_num,"Results:",result)
    # finalize
    finalize()

    print(test_end_str('Entanglement Test'))

def qif_test():
    """
    The the qif module test
    """
    print(test_begin_str('Q-If Test'))
    
    #init the environment
    init()

    #allocate resources
    q=qAlloc_many(10)
    c=cAlloc_many(10)
    
    #set the true branch and false branch
    #ready for qif
    true_branch=X(q[2])
    false_branch=X(q[1])

    qprog=QProg().insert(H(q[0])) \
        .insert(Measure(q[0],c[0])) \
        .insert(CreateIfProg(bind_a_cbit(c[0]),true_branch,false_branch)) \
        .insert(meas_all(q[1:3],c[1:3]))

    shots_num=1000
    result=run_with_configuration(program=qprog,shots=shots_num,cbit_list=c[0:3])
  
    print("Shots:",shots_num,"Results:",result)
    # finalize
    finalize()

    print(test_end_str('Q-If Test'))


def full_test():
    test_list=[entanglement_test, qif_test]

    for i in range(len(test_list)):
        test_list[i]()
        print('State_preparation test: %d/%d pass' % (i+1,len(test_list)))
        print(" ")