"""
This type stub file was generated by pyright.
"""

import torch

"""
This type stub file was generated by pyright.
"""
class ConvReLU1d(torch.nn.Sequential):
    r"""This is a sequential container which calls the Conv 1d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, relu) -> None:
        ...
    


class ConvReLU2d(torch.nn.Sequential):
    r"""This is a sequential container which calls the Conv 2d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, relu) -> None:
        ...
    


class ConvReLU3d(torch.nn.Sequential):
    r"""This is a sequential container which calls the Conv 3d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, relu) -> None:
        ...
    


class LinearReLU(torch.nn.Sequential):
    r"""This is a sequential container which calls the Linear and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, linear, relu) -> None:
        ...
    


class ConvBn1d(torch.nn.Sequential):
    r"""This is a sequential container which calls the Conv 1d and Batch Norm 1d modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn) -> None:
        ...
    


class ConvBn2d(torch.nn.Sequential):
    r"""This is a sequential container which calls the Conv 2d and Batch Norm 2d modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn) -> None:
        ...
    


class ConvBnReLU1d(torch.nn.Sequential):
    r"""This is a sequential container which calls the Conv 1d, Batch Norm 1d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn, relu) -> None:
        ...
    


class ConvBnReLU2d(torch.nn.Sequential):
    r"""This is a sequential container which calls the Conv 2d, Batch Norm 2d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn, relu) -> None:
        ...
    


class ConvBn3d(torch.nn.Sequential):
    r"""This is a sequential container which calls the Conv 3d and Batch Norm 3d modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn) -> None:
        ...
    


class ConvBnReLU3d(torch.nn.Sequential):
    r"""This is a sequential container which calls the Conv 3d, Batch Norm 3d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn, relu) -> None:
        ...
    


class BNReLU2d(torch.nn.Sequential):
    r"""This is a sequential container which calls the BatchNorm 2d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, batch_norm, relu) -> None:
        ...
    


class BNReLU3d(torch.nn.Sequential):
    r"""This is a sequential container which calls the BatchNorm 3d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, batch_norm, relu) -> None:
        ...
    


