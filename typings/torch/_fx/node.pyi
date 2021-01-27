"""
This type stub file was generated by pyright.
"""

import torch
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Tuple, Union
from .graph import Graph

if TYPE_CHECKING:
    ...
BaseArgumentTypes = Union[str, int, float, bool, torch.dtype, torch.Tensor]
base_types = BaseArgumentTypes.__args__
Target = Union[Callable[..., Any], str]
Argument = Optional[Union[Tuple[Any, ...], List[Any], Dict[str, Any], slice, 'Node', BaseArgumentTypes]]
class Node:
    def __init__(self, graph: Graph, name: str, op: str, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Argument]) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    


