"""
This type stub file was generated by pyright.
"""

import torch
from torch._six import with_metaclass

"""
This type stub file was generated by pyright.
"""
class VariableMeta(type):
    def __instancecheck__(cls, other):
        ...
    


class Variable(with_metaclass(VariableMeta, torch._C._LegacyVariableBase)):
    ...


