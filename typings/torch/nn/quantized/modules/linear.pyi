"""
This type stub file was generated by pyright.
"""

import torch
import torch.nn as nn
from typing import Optional

"""
This type stub file was generated by pyright.
"""
class LinearPackedParams(torch.nn.Module):
    _version = ...
    def __init__(self, dtype=...) -> None:
        ...
    
    @torch.jit.export
    def set_weight_bias(self, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> None:
        ...
    
    def forward(self, x):
        ...
    
    @torch.jit.export
    def __getstate__(self):
        ...
    
    @torch.jit.export
    def __setstate__(self, state):
        ...
    
    def __repr__(self):
        ...
    


class Linear(torch.nn.Module):
    r"""
    A quantized linear module with quantized tensor as inputs and outputs.
    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear for documentation.

    Similar to :class:`~torch.nn.Linear`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        weight (Tensor): the non-learnable quantized weights of the module of
                         shape :math:`(\text{out\_features}, \text{in\_features})`.
        bias (Tensor): the non-learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized to zero.
        scale: `scale` parameter of output Quantized Tensor, type: double
        zero_point: `zero_point` parameter for output Quantized Tensor, type: long

    Examples::

        >>> m = nn.quantized.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> input = torch.quantize_per_tensor(input, 1.0, 0, torch.quint8)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    _version = ...
    _FLOAT_MODULE = ...
    def __init__(self, in_features, out_features, bias_=..., dtype=...) -> None:
        ...
    
    def extra_repr(self):
        ...
    
    def __repr__(self):
        ...
    
    def forward(self, x):
        ...
    
    def weight(self):
        ...
    
    def bias(self):
        ...
    
    def set_weight_bias(self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None:
        ...
    
    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a float module or qparams_dict

        Args:
            mod (Module): a float module, either produced by torch.quantization
                          utilities or provided by the user
        """
        ...
    

