import matplotlib.pyplot as plt
import matplotlib as mpl

plt.switch_backend('agg')


def draw_probability(list):
    """
    Generate a bar plot visualizing the probabilities of a quantum state.

    The function takes a dictionary containing quantum state keys and their corresponding
    probabilities, normalizes the probabilities, and plots them using a bar chart.

    Args:
        probability_dict (dict): A dictionary where keys represent the quantum states and
                                 values represent their respective probabilities.

    Returns:
        None: This function does not return a value. It plots the probabilities directly.

    Notes:
        - The bar plot uses the Chinese font 'SimHei' and ensures that negative symbols are
          not displayed in the axes.
        - The x-axis represents the quantum states, while the y-axis represents the
          probabilities.
        - The function includes a grid on the y-axis for better readability.
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

def draw_probability_dict(prob_dict):
    """
    Generate a bar plot representing the probabilities of a quantum state from a given dictionary.

    Args:
        prob_dict (dict): A dictionary where keys are quantum state labels and values are their corresponding probabilities.

    Returns:
        None: This function does not return a value; it displays a bar plot.

    The function configures Matplotlib to use SimHei font for better readability in Chinese characters.
    It filters out non-positive probabilities, calculates normalized probabilities, and then plots them as a bar chart.
    The plot includes a grid and labeled axes for clarity.
    This utility is intended for use within the pyQPanda package, which is designed for programming quantum computers.
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
