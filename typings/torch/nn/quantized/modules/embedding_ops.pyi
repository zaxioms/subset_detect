"""
This type stub file was generated by pyright.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch._jit_internal import Optional

"""
This type stub file was generated by pyright.
"""
class EmbeddingPackedParams(torch.nn.Module):
    _version = ...
    def __init__(self, num_embeddings, embedding_dim, dtype=...) -> None:
        ...
    
    @torch.jit.export
    def set_weight(self, weight: torch.Tensor) -> None:
        ...
    
    def forward(self, x):
        ...
    
    def __repr__(self):
        ...
    


class Embedding(torch.nn.Module):
    r"""
    A quantized Embedding module with quantized packed weights as inputs.
    We adopt the same interface as `torch.nn.Embedding`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding for documentation.

    Similar to :class:`~torch.nn.Embedding`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        weight (Tensor): the non-learnable quantized weights of the module of
                         shape :math:`(\text{num\_embeddings}, \text{embedding\_dim})`.

    Examples::
        >>> m = nn.quantized.Embedding(num_embeddings=10, embedding_dim=12)
        >>> indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8])
        >>> output = m(indices)
        >>> print(output.size())
        torch.Size([9, 12]

    """
    _version = ...
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = ..., max_norm: Optional[float] = ..., norm_type: float = ..., scale_grad_by_freq: bool = ..., sparse: bool = ..., _weight: Optional[Tensor] = ..., dtype=...) -> None:
        ...
    
    def forward(self, indices: Tensor) -> Tensor:
        ...
    
    def __repr__(self):
        ...
    
    def extra_repr(self):
        ...
    
    def set_weight(self, w: torch.Tensor) -> None:
        ...
    
    def weight(self):
        ...
    
    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized embedding module from a float module

        Args:
            mod (Module): a float module, either produced by torch.quantization
                          utilities or provided by user
        """
        ...
    


class EmbeddingBag(Embedding):
    r"""
    A quantized EmbeddingBag module with quantized packed weights as inputs.
    We adopt the same interface as `torch.nn.EmbeddingBag`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.EmbeddingBag for documentation.

    Similar to :class:`~torch.nn.EmbeddingBag`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        weight (Tensor): the non-learnable quantized weights of the module of
                         shape :math:`(\text{num\_embeddings}, \text{embedding\_dim})`.

    Examples::
        >>> m = nn.quantized.EmbeddingBag(num_embeddings=10, embedding_dim=12, include_last_offset=True, mode='sum')
        >>> indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8, 3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3])
        >>> offsets = torch.tensor([0, 19, 20, 28, 28, 32])
        >>> output = m(indices, offsets)
        >>> print(output.size())
        torch.Size([5, 12]

    """
    _version = ...
    def __init__(self, num_embeddings: int, embedding_dim: int, max_norm: Optional[float] = ..., norm_type: float = ..., scale_grad_by_freq: bool = ..., mode: str = ..., sparse: bool = ..., _weight: Optional[Tensor] = ..., include_last_offset: bool = ..., dtype=...) -> None:
        ...
    
    def forward(self, indices: Tensor, offsets: Optional[Tensor] = ..., per_sample_weights: Optional[Tensor] = ..., compressed_indices_mapping: Optional[Tensor] = ...) -> Tensor:
        ...
    
    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized embedding_bag module from a float module

        Args:
            mod (Module): a float module, either produced by torch.quantization
                          utilities or provided by user
        """
        ...
    


