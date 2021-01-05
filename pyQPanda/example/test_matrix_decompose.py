import pyqpanda.pyQPanda as pq
from pyqpanda.Visualization.circuit_draw import *
import numpy as np

def test_decompose():
    machine = pq.init_quantum_machine(pq.QMachineType.CPU)
    q = machine.qAlloc_many(2)
    c = machine.cAlloc_many(2)

    x = [(0.6477054522122977+0.1195417767870219j), (-0.16162176706189357-0.4020495632468249j), (-0.19991615329121998-0.3764618308248643j), (-0.2599957197928922-0.35935248873007863j),
     (-0.16162176706189363-0.40204956324682495j), (0.7303014482204584-0.4215172444390785j), (-0.15199187936216693+0.09733585496768032j), (-0.22248203136345918-0.1383600597660744j),
      (-0.19991615329122003-0.3764618308248644j), (-0.15199187936216688+0.09733585496768032j), (0.6826630277354306-0.37517063774206166j), (-0.3078966462928956-0.2900897445133085j),
       (-0.2599957197928923-0.3593524887300787j), (-0.22248203136345912-0.1383600597660744j), (-0.30789664629289554-0.2900897445133085j), (0.6640994547408099-0.338593803336005j)]
    cir = pq.matrix_decompose(q, x)
    result_mat = pq.get_matrix(cir)
    #draw_qprog(cir, 'text')
    draw_qprog(cir, 'pic', filename='d:/test_cir_draw-88.jpg')
    x1 = np.round(np.array(x), 3)
    print('x1')
    print(x1)
    mat2 = np.round(np.array(result_mat), 3)
    print('mat2')
    print(mat2)
    if np.all(x1 == mat2):
        print('ok')
    else:
        print('falsh')
        
    print('The decomposed cir matrix:')
    pq.print_matrix(result_mat, 15)

if __name__=="__main__":
    test_decompose()
    print("Test over.")