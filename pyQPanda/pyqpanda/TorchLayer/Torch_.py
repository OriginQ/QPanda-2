

# pylint: disable=protected-access, attribute-defined-outside-init, arguments-differ, no-member, import-self, too-many-statements
import numpy as np
import semantic_version
import torch
from pyqpanda import *

COMPLEX_SUPPORT = semantic_version.match(">=1.6.0", torch.__version__)

class TorchModel(torch.autograd.Function):
    @staticmethod
    def forward(ctx,func,input_):
        res = func(**input_)
        res = torch.as_tensor(torch.from_numpy(np.array(res)))
        ctx.save_for_backward(res)
        return res
    @staticmethod
    def backward(ctx, grad_outputs):  # pragma: no cover
        outputs, = ctx.saved_tensors
        return grad_outputs * outputs
       


