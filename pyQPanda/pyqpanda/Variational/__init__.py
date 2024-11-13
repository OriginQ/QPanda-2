from pyqpanda.pyQPanda import _back
def back(exp, grad, leaf_set = None):
    """
    Updates the gradients for each variable in the given gradient dictionary based on the
    results of the `_back` function. The function is designed to propagate gradients through
    a quantum circuit, considering only the variables specified in `leaf_set` if provided.

    Parameters:
    - exp: A quantum expression (typically a quantum circuit) from which to calculate gradients.
    - grad: A dictionary mapping variables to their corresponding gradients.
    - leaf_set (optional): A set of variable names to consider when calculating gradients.
        If not provided, all variables are considered.

    Returns:
    - None. The gradients are updated in-place within the `grad` dictionary.
    """
    vars = [_var for _var in grad]
    _grad = None
    if leaf_set == None:
        _grad = _back(exp, grad)
    else:
        _grad = _back(exp, grad, leaf_set)
    for _var in vars:
        for _key in _grad:
            if _key == _var:
                grad[_var] = _grad[_key]