import matplotlib.pyplot as plt
import matplotlib as mpl

def draw_probaility(list):
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    x = [key for key in list.keys()]
    y = [val for val in list.values()]
    y1 = [val/sum(y) for val in list.values()]
    plt.bar(x, y1, align = "center", color = "b", alpha = 0.6)
    plt.ylabel("Probabilities")
    plt.grid(True, axis = "y", ls = ":", color = "r", alpha = 0.3)
    #plt.show()