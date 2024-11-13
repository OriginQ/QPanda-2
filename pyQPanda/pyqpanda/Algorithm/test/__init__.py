'''
Test Module\n
Copyright (C) Origin Quantum 2017-2024\n
Licensed Under Apache Licence 2.0
'''
from pyqpanda import *
from . import state_preparation_test

def full_test():
    """
    Executes a series of full tests on the quantum algorithms within the test list.

    The function iterates over a list of test objects, each of which should have a
    `full_test()` method to perform the actual test. After each test, it prints the
    result and the current test number out of the total number of tests.

    This function is intended for internal use within the pyQPanda package to validate
    the functionality of quantum algorithms when running on a quantum circuit simulator
    or quantum cloud service.
    """
    test_list=[state_preparation_test]

    for i in range(len(test_list)):
        test_list[i].full_test()
        print(" ")
        print('Full test: %d/%d pass' % (i+1,len(test_list)))
        print(" ")
