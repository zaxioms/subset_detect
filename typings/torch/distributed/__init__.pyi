"""
This type stub file was generated by pyright.
"""

import torch
from .distributed_c10d import *
from .distributed_c10d import _backend

def is_available():
    """
    Returns ``True`` if the distributed package is available. Otherwise,
    ``torch.distributed`` does not expose any other APIs. Currently,
    ``torch.distributed`` is available on Linux, MacOS and Windows. Set
    ``USE_DISTRIBUTED=1`` to enable it when building PyTorch from source.
    Currently, the default value is ``USE_DISTRIBUTED=1`` for Linux and Windows,
    ``USE_DISTRIBUTED=0`` for MacOS.
    """
    ...

if is_available() and not torch._C._c10d_init():
    ...
if is_available():
    ...