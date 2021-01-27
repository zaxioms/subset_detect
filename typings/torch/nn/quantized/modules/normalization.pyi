"""
This type stub file was generated by pyright.
"""

import torch

"""
This type stub file was generated by pyright.
"""
class LayerNorm(torch.nn.LayerNorm):
    r"""This is the quantized version of :class:`~torch.nn.LayerNorm`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """
    def __init__(self, normalized_shape, weight, bias, scale, zero_point, eps=..., elementwise_affine=...) -> None:
        ...
    
    def forward(self, input):
        ...
    
    @classmethod
    def from_float(cls, mod):
        ...
    


class GroupNorm(torch.nn.GroupNorm):
    r"""This is the quantized version of :class:`~torch.nn.GroupNorm`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """
    __constants__ = ...
    def __init__(self, num_groups, num_channels, weight, bias, scale, zero_point, eps=..., affine=...) -> None:
        ...
    
    def forward(self, input):
        ...
    
    @classmethod
    def from_float(cls, mod):
        ...
    


class InstanceNorm1d(torch.nn.InstanceNorm1d):
    r"""This is the quantized version of :class:`~torch.nn.InstanceNorm1d`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """
    def __init__(self, num_features, weight, bias, scale, zero_point, eps=..., momentum=..., affine=..., track_running_stats=...) -> None:
        ...
    
    def forward(self, input):
        ...
    
    @classmethod
    def from_float(cls, mod):
        ...
    


class InstanceNorm2d(torch.nn.InstanceNorm2d):
    r"""This is the quantized version of :class:`~torch.nn.InstanceNorm2d`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """
    def __init__(self, num_features, weight, bias, scale, zero_point, eps=..., momentum=..., affine=..., track_running_stats=...) -> None:
        ...
    
    def forward(self, input):
        ...
    
    @classmethod
    def from_float(cls, mod):
        ...
    


class InstanceNorm3d(torch.nn.InstanceNorm3d):
    r"""This is the quantized version of :class:`~torch.nn.InstanceNorm3d`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """
    def __init__(self, num_features, weight, bias, scale, zero_point, eps=..., momentum=..., affine=..., track_running_stats=...) -> None:
        ...
    
    def forward(self, input):
        ...
    
    @classmethod
    def from_float(cls, mod):
        ...
    


