# coding=utf-8

import unittest
from unittest.runner import TextTestRunner
from unittest.suite import TestSuite


global TESTOPT
TESTOPT = "single"


if TESTOPT == "all":
    def test():
        # Run all test cases
        case_dir = ".\\"
        discover = unittest.defaultTestLoader.discover(case_dir, pattern="test_*.py", top_level_dir=None)
        runner = TextTestRunner(verbosity=2)
        runner.run(discover)

elif TESTOPT == "single":
    def test():
        # Run a Single Test Case
        suite = unittest.TestSuite()
        suite.addTests(unittest.TestLoader().loadTestsFromName("test_Grover.Test_Grover.test_grover2"))
        runner = TextTestRunner(verbosity=2)
        runner.run(suite)

else:
    def test():
        print("TEST  OPTION  ERROR!")

if __name__ == '__main__':
    test()
