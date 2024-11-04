from pyqpanda import *
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from functools import partial

import numpy as np
def _fit_func(x, *params):
    """
    Evaluates a polynomial function with a base exponent of the first parameter and a constant term.

    This function computes the value of a polynomial of the form A * p^x + B, where:
    - A is the coefficient of the polynomial term with the highest exponent.
    - B is the constant term of the polynomial.
    - p is the base of the exponentiation, which is derived from the additional parameters.

    Parameters:
    x (int): The exponent to which the base `p` is raised.
    *params: Variable length argument list, where the first element is `A` and the second is `B`.

    Returns:
    float: The computed value of the polynomial for the given `x` and `params`.

    This function is utilized within the pyQPanda package, specifically in the testRB module, for various
    computations related to quantum circuits and quantum computation simulations.
    """
    A, B, p = params
    y = A * p ** x + B
    return y

def _residuals(p, y, x, func):
    """
    Calculate the residuals for a given model and data points.

    This function computes the residuals by subtracting the model prediction from the observed data.
    It is designed to be used within the pyQPanda package, which is dedicated to programming quantum computers
    using quantum circuits and gates. The residuals can be utilized for various purposes, such as
    assessing the accuracy of the model or for further analysis in quantum computing applications.

    Parameters:
    p (tuple): A tuple containing the parameters of the model function.
    y (array_like): The observed data points.
    x (array_like): The corresponding input values for the data points.
    func (callable): The model function that takes the input values and parameters to compute predictions.

    Returns:
    array_like: The residuals, which are the differences between the observed data and the model predictions.
    """
    return y-func(x, *p)

def RMSE(x, y1, y2):
    """
    Calculate the Root Mean Square Error (RMSE) between two sequences of values.

    This function computes the RMSE between two lists or arrays, `y1` and `y2`, by
    first determining the variances of their corresponding elements. It then averages
    these variances and takes the square root of the result to obtain the RMSE.

    Parameters:
    x (list or array): A sequence of values used to align `y1` and `y2` during variance calculation.
    y1 (list or array): The first sequence of values for which the variance is computed.
    y2 (list or array): The second sequence of values for which the variance is computed.

    Returns:
    float: The RMSE value, which is a measure of the differences between the two sequences.

    Notes:
    - The function assumes that `x`, `y1`, and `y2` are of the same length and are properly aligned.
    - The function uses NumPy's sum and sqrt functions for efficient computation.
    """
    variances = list(map(lambda x, y : (x-y)**2, y1, y2))
    variance = np.sum(variances)
    rmse = np.sqrt(variance / len(x))
    return rmse

def _least_sq_fit_step(xdata, ydata, p0, func):
    """
    Perform a least squares fit to a given function using initial parameter estimates.

    This function computes the best fit parameters for a given function `func` using the least squares method.
    It starts with an initial guess `p0` for the parameters and iteratively refines the fit until the root mean
    squared error (RMSE) falls below a threshold of 1e-3 or a maximum of 1000 iterations is reached.

    Parameters:
    xdata (array-like): The independent variable data points.
    ydata (array-like): The dependent variable data points.
    p0 (array-like): Initial guess for the parameters of the function `func`.
    func (callable): The function to fit the data points to.

    Returns:
    tuple: A tuple containing the optimized parameters `p`, the final RMSE value `rmse`, and the
           fitted values `y2` for the given data points.

    Notes:
    - The function `func` must accept a sequence of arguments, where the first argument is `xdata` and
      subsequent arguments are the optimized parameters `p`.
    - This function uses the `leastsq` function from the `scipy.optimize` module to perform the least squares
      fit.
    - The RMSE calculation is performed using a separate function `RMSE`, which must be defined elsewhere
      in the package.
    """
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
    """
    Calculate the fidelity of a quantum circuit based on the provided data points.

    The function fits the data using a least squares method, evaluates certain parameters,
    and computes the fidelity based on the fitting results.

    Parameters:
    x (list): The x-coordinates corresponding to the data points.
    y (list): The y-coordinates corresponding to the data points.

    Returns:
    float: The calculated fidelity of the quantum circuit.
    """
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
    
