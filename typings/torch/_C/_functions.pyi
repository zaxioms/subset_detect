"""
This type stub file was generated by pyright.
"""

from torch import Tensor
from typing import AnyStr, List

"""
This type stub file was generated by pyright.
"""
class UndefinedGrad:
    def __init__(self) -> None:
        ...
    
    def __call__(self, inputs: List[Tensor]) -> List[Tensor]:
        ...
    


class DelayedError:
    def __init__(self, msg: AnyStr, num_inputs: int) -> None:
        ...
    
    def __call__(self, inputs: List[Tensor]) -> List[Tensor]:
        ...
    


