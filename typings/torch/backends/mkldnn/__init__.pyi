"""
This type stub file was generated by pyright.
"""

import sys
import torch
from contextlib import contextmanager
from torch.backends import ContextProp, PropModule, __allow_nonbracketed_mutation

"""
This type stub file was generated by pyright.
"""
def is_available():
    r"""Returns whether PyTorch is built with MKL-DNN support."""
    ...

def set_flags(_enabled):
    ...

@contextmanager
def flags(enabled=...):
    ...

class MkldnnModule(PropModule):
    def __init__(self, m, name) -> None:
        ...
    
    enabled = ...


