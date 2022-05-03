import sys
import os.path
dir = os.path.abspath(__file__)
model_path = os.path.abspath(os.path.join(dir, "../.."))
sys.path.insert(0, model_path)

import pyqpanda.pyQPanda as pq
from pyqpanda.Visualization.circuit_draw import *
from pyqpanda.Visualization.draw_probability_map import *

if __name__=="__main__":
    list2 = {'000': 506, '010': 300, '001': 400, '110': 300, '111': 508}
    # list2 = {'000': 506}
    draw_probaility(list2)
