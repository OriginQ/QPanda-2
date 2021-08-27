from pyqpanda import *
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from functools import partial

import numpy as np
def _fit_func(x, *params):
    A, B, p = params
    y = A * p ** x + B
    return y

def _residuals(p, y, x, func):
    return y-func(x, *p)

def RMSE(x, y1, y2):
    variances = list(map(lambda x, y : (x-y)**2, y1, y2))
    variance = np.sum(variances)
    rmse = np.sqrt(variance / len(x))
    return rmse

def _least_sq_fit_step(xdata, ydata, p0, func):
    x = xdata
    y = ydata
    num = 0
    _residuals_attach = partial(_residuals, func=func)
    plsq = leastsq(_residuals_attach, p0, args=(y, x))
    p = plsq[0]
    y2 = func(x, *p)
    rmse = RMSE(x, y, y2)
    while(rmse > 1e-3) and (num < 1000):
        plsq = leastsq(_residuals, p0, args=(y, x, func))
        p = plsq[0]
        y2 = func(x, *p)
        rmse = RMSE(x, y, y2)
        num+=1

    return p, rmse, y2

def _get_fidelity(x, y):
    # fit data
    #B = np.mean(y[-5:])
    B = np.mean(y)
    p = 0.9
    A = y[0] - B
    param, rmse, fit_y = _least_sq_fit_step(x, y, [A, B, p], _fit_func)

    #calc fidelity
    p = param[2]
    rc = (1 - p) * (1 - 1 / 2)  # clifford错误率
    rg = rc / 1.875  # 每个门错误率
    fidelity = 1-rg
    #return p, rc, rg, fit_y
    return fidelity


if __name__=="__main__":
    # qvm = init_quantum_machine(QMachineType.NOISE)

    # q = qvm.qAlloc_many(1)
    # qvm.set_noise_model(NoiseModel.DEPOLARIZING_KRAUS_OPERATOR, GateType.PAULI_X_GATE, 0.1, [q[0]])
    # qvm.set_noise_model(NoiseModel.DEPOLARIZING_KRAUS_OPERATOR, GateType.PAULI_Y_GATE, 0.1, [q[0]])
    # qvm.set_noise_model(NoiseModel.DEPOLARIZING_KRAUS_OPERATOR, GateType.RX_GATE, 0.1, [q[0]])
    # qvm.set_noise_model(NoiseModel.DEPOLARIZING_KRAUS_OPERATOR, GateType.RY_GATE, 0.1, [q[0]])
    #qvm.set_noise_model(NoiseModel.DEPOLARIZING_KRAUS_OPERATOR, GateType.PAULI_Z_GATE, 0.1, [q[0]])

    #clifford_range = range(5, 46, 10)
    #clifford_range = range(2, 10, 2),
    qvm = QCloud()
    qvm.init_qvm("898D47CF515A48CEAA9F2326394B85C6")
    q = qvm.qAlloc_many(2)
    #clifford_range = [2,4,6,8,10,12,14,16,18,20]
    clifford_range = [2,4,6,8]
    #res = single_qubit_rb(qvm, q[0],clifford_range, 10, 5000, [])
    res = double_gate_xeb(qvm, GateType.CZ_GATE, q[0], q[1], clifford_range, 10, 5000)

    x , y = list(res.keys()), list(res.values())
    fidelity = _get_fidelity(x, y)
    print(fidelity)
    #destroy_quantum_machine(qvm)
    
