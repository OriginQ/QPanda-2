from pyqpanda.Variational import pyQPandaVariational
import numpy as np




if __name__ == "__main__":
    a = var(np.ones((3,3), dtype = 'float64'))
    b = var(np.ones((3,3), dtype = 'float64'))
    c = stack(0, a, b)
    s = eval(c)
    print(s)
    