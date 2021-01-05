import pyqpanda.pyQPanda as pq
from pyqpanda.Visualization.circuit_draw import *
from pyqpanda.Visualization.draw_probability_map import *

if __name__=="__main__":
    list2 = {'000': 506, '010': 300, '001': 400, '110': 300, '111': 508}
    # list2 = {'000': 506}
    draw_probaility(list2)

    print("Test over.")