"""
This type stub file was generated by pyright.
"""

from typing import Tuple
from .optimizer import Optimizer, _params_t

"""
This type stub file was generated by pyright.
"""
class Adamax(Optimizer):
    def __init__(self, params: _params_t, lr: float = ..., betas: Tuple[float, float] = ..., eps: float = ..., weight_decay: float = ...) -> None:
        ...
    

