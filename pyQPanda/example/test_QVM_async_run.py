# code for inplace run test, add \qpanda-2.\pyQPanda to module search paths
import sys
import os.path
dir = os.path.abspath(__file__)
model_path = os.path.abspath(os.path.join(dir, "../.."))
sys.path.insert(0, model_path)

from pyqpanda import *
import unittest


class Test_Async_Run(unittest.TestCase):
    def test_run(self):
        qvm = CPUQVM()
        qvm.init_qvm()

        qv = qvm.qAlloc_many(4)
        cv = qvm.cAlloc_many(4)

        prog = QProg()
        for i in range(0, 1001):
            prog.insert(X(qv))

        total_num = get_qgate_num(prog)

        self.assertEqual(qvm.get_processed_qgate_num(), 0,
                         "qvm async task not start, gate processed num should be 0")

        # aysnc run prog
        qvm.async_run(prog)

        # get aysnc run progress
        while not qvm.is_async_finished():
            processed_gate_num = qvm.get_processed_qgate_num()
            print("processed_gate_num : {}/{}".format(processed_gate_num, total_num))
            self.assertGreaterEqual(processed_gate_num, 0)
            self.assertLessEqual(processed_gate_num, total_num)

        self.assertEqual(qvm.get_processed_qgate_num(), 0)

        # compare aysnc result with directly run
        result = qvm.get_async_result()

        # aysnc run prog

        qvm2 = CPUQVM()
        qvm2.init_qvm()

        qv2 = qvm2.qAlloc_many(4)
        cv2 = qvm2.cAlloc_many(4)
        
        prog2 = QProg()
        for i in range(0, 1001):
            prog2.insert(X(qv2))

        result2 = qvm2.directly_run(prog2)
        self.assertEqual(result, result2)

        qvm.finalize()
        qvm2.finalize()
    
if __name__ == "__main__":
    unittest.main(verbosity=2)
