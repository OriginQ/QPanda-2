'''
Test Module\n
Copyright (C) Origin Quantum 2017-2018\n
Licensed Under Apache Licence 2.0
'''
from pyqpanda import *
from . import state_preparation_test

def full_test():
    
    test_list=[state_preparation_test]

    for i in range(len(test_list)):
        test_list[i].full_test()
        print(" ")
        print('Full test: %d/%d pass' % (i+1,len(test_list)))
        print(" ")
