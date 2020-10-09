import pyqpanda.pyQPanda as pq
import matplotlib.pyplot as plt
from .matplotlib_draw import *

def plot(list):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    x = [key for key in list.keys()]
    y = [val for val in list.values()]
    y1 = [val/sum(y) for val in list.values()]
    plt.bar(x, y1, align = "center", color = "b", alpha = 0.6)
    plt.ylabel("Probabilities")
    plt.grid(True, axis = "y", ls = ":", color = "r", alpha = 0.3)
    plt.show()

def draw_circuit_pic(prog, pic_name, verbose=True):
    layer_info = pq.circuit_layer(prog)
    qcd = MatplotlibDrawer(qregs = layer_info[1], cregs = layer_info[2], ops = layer_info[0], scale=0.7)
    qcd.draw(pic_name, verbose)