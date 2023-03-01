import matplotlib.pyplot as plt
import matplotlib as mpl

plt.switch_backend('agg')


def draw_probaility(list):
    """Draw a quantum state probaility dict

    Args:
        list : the quantum state probaility dict

    Returns: 
        no return

    """
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    x = [key for key in list.keys()]
    y = [val for val in list.values()]
    y1 = [val/sum(y) for val in list.values()]
    plt.bar(x, y1, align="center", color="b", alpha=0.6)
    plt.ylabel("Probabilities")
    plt.grid(True, axis="y", ls=":", color="r", alpha=0.3)
    plt.show()


def draw_probaility_dict(prob_dict):
    """Draw a quantum state probaility dict

    Args:
        list : the quantum state probaility dict

    Returns: 
        no return

    """
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False

    x = [item[0] for item in prob_dict.items() if item[1] > 0.]
    y = [item[1] for item in prob_dict.items() if item[1] > 0.]
    y1 = [val/sum(y) for val in y]

    plt.bar(x, y1, align="center", color="b", alpha=0.6)
    plt.ylabel("Probabilities", fontsize=18)
    plt.grid(True, axis="y", ls=":", color="r", alpha=0.3)
    plt.yticks(size=15)
    plt.xticks(size=15)

    plt.show()
