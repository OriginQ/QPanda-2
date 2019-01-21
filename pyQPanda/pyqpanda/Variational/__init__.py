from pyqpanda.pyQPanda import _back
def back(exp, grad, leaf_set = None):
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