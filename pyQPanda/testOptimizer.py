# from pyqpanda import *
import pyqpanda as pq
import numpy as np

x = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59,
             2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27,3.1])
y = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53,
             1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94,1.3])

# def lossFunc(para):
#     y_ = np.zeros(len(y))

#     for i in range(len(y)):
#         y_[i] = para[0] * x[i] + para[1]

#     loss = 0
#     for i in range(len(y)):
#         loss += (y_[i] - y[i])**2/len(y)

#     return ("", loss)

def test_f(x):
    return ("", x[0]**2+x[1]**2+x[2]**2)

optimizer = pq.OptimizerFactory.makeOptimizer('NELDER_MEAD')

# init_para = [0, 0]
# optimizer.registerFunc(lossFunc, init_para)
init_para=[100.0,100.0,100.0]
optimizer.registerFunc(test_f, init_para)
optimizer.setXatol(1e-6)
optimizer.setFatol(1e-6)
optimizer.setMaxIter(200)
# optimizer.setCacheFile("testLiYe1014.txt")
# optimizer.setRestoreFromCacheFile(True)
optimizer.exec()

result = optimizer.getResult()
print(result.message)
print(" Current function value: ", result.fun_val)
print(" Iterations: ", result.iters)
print(" Function evaluations: ", result.fcalls)
print(" Optimized para: W: ", result.para[0], " b: ", result.para[1])



