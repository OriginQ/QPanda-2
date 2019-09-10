from pyqpanda import *
from scipy.optimize import minimize
from functools import partial
from pyqpanda.Algorithm.test.qaoa_maxcut_test import qaoa_maxcut_gradient_threshold,generate_graph
import time

if __name__=="__main__":

    dimension=6
    n_edge=dimension*(dimension-1)/2
    graph=generate_graph(dimension=dimension,n_edge=n_edge)
    step=4
    start=time.clock()
    result=qaoa_maxcut_gradient_threshold(graph=graph,
    step=step,
    threshold_value=0.05,
    optimize_times=300,
    use_GPU=False)
    print(result)
    end=time.clock()
    print("time is: {}".format(end-start))