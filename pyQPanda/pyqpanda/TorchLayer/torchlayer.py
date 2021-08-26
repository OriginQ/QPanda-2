import functools
import inspect
import math
from collections.abc import Iterable
from typing import Callable, Optional
from .Torch_  import TorchModel

try:
    import torch
    from torch.nn import Module

    TORCH_IMPORTED = True
except ImportError:
    # The following allows this module to be imported even if PyTorch is not installed. Users
    # will instead see an ImportError when instantiating the TorchLayer.
    from unittest.mock import Mock

    Module = Mock
    TORCH_IMPORTED = False
class TorchLayer(Module):
    def __init__(self, func, weight_shapes: dict, init_method: Optional[Callable] = None):
        self.qweight={}
        self.func=func 
        if not TORCH_IMPORTED:
            raise ImportError(
                "TorchLayer requires PyTorch. PyTorch can be installed using:\n"
                "pip install torch\nAlternatively, "
                "visit https://pytorch.org/get-started/locally/ for detailed "
                "instructions."
            )
        super().__init__()
        weight_shapes = {
            weight: (tuple(size) if isinstance(size, Iterable) else (size,) if size > 1 else ())
            for weight, size in weight_shapes.items()
        }

        # validate the Circuit signature, and convert to a Torch Circuit.
        # TODO: update the docstring regarding changes to restrictions when tape mode is default.
        self._signature_validation(func, weight_shapes)

        if not init_method:
            init_method = functools.partial(torch.nn.init.uniform_, b=2 * math.pi)
        

        for name, size in weight_shapes.items():
            if len(size) == 0:
                self.qweight[name] = torch.nn.Parameter(init_method(torch.Tensor(1))[0])
            else:
                self.qweight[name] = torch.nn.Parameter(init_method(torch.Tensor(*size)))

    def _signature_validation(self, func, weight_shapes):
        sig = inspect.signature(func).parameters

        if self.input_arg not in sig:
            raise TypeError(
                "Circuit must include an argument with name {} for inputting data".format(
                    self.input_arg
                )
            )

        if self.input_arg in set(weight_shapes.keys()):
            raise ValueError(
                "{} argument should not have its dimension specified in "
                "weight_shapes".format(self.input_arg)
            )

        param_kinds = [p.kind for p in sig.values()]

        if inspect.Parameter.VAR_POSITIONAL in param_kinds:
            raise TypeError("Cannot have a variable number of positional arguments")

        if inspect.Parameter.VAR_KEYWORD not in param_kinds:
            if set(weight_shapes.keys()) | {self.input_arg} != set(sig.keys()):
                raise ValueError("Must specify a shape for every non-input parameter in the Circuit")

    def forward(self, inputs):  # pylint: disable=arguments-differ
        if len(inputs.shape) > 1:
            reconstructor = []
            x=torch.unbind(inputs)[len(inputs.shape)-1]
            reconstructor.append(self._evaluate_qnode(x))
            return torch.stack(reconstructor)

        return self._evaluate_qnode(inputs)

    def _evaluate_qnode(self, x):

        kwargs = {
            **{self.input_arg: x},
            **{arg: weight.to(x) for arg, weight in self.qweight.items()},
        }
        res=TorchModel.apply(self.func,kwargs)
        return res
    _input_arg = "inputs"

    @property
    def input_arg(self):
        """Name of the argument to be used as the input to the Torch layer. Set to ``"inputs"``."""
        return self._input_arg